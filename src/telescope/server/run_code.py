from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path
from typing import Any

from .db import _get_data_dir


def _code_dir() -> Path:
    return _get_data_dir() / "code"
MAX_TREE_NODES = 20_000
MAX_DIFF_MATRIX_CELLS = 2_000_000
MAX_DIFF_SUMMARY_FILE_BYTES = 4 * 1024 * 1024

_INVALID_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_path_part(value: str) -> str:
    sanitized = _INVALID_PATH_CHARS.sub("_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "part"


def run_storage_slug(run_path: str) -> str:
    parts = [part for part in run_path.split("/") if part]
    safe_parts = [_sanitize_path_part(part) for part in parts]
    stem = "__".join(safe_parts) if safe_parts else "run"
    digest = hashlib.sha1(run_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{digest}"


def get_run_code_run_dir(run_path: str) -> Path:
    return _code_dir() / run_storage_slug(run_path)


def get_run_code_metadata_path(run_path: str) -> Path:
    return get_run_code_run_dir(run_path) / "metadata.json"


def get_run_code_dir(run_path: str) -> Path:
    return get_run_code_run_dir(run_path) / "source"


def ensure_run_storage_dirs() -> None:
    _code_dir().mkdir(parents=True, exist_ok=True)


def resolve_code_file_path(run_path: str, relative_file_path: str) -> Path:
    code_dir = get_run_code_dir(run_path).resolve()
    candidate = (code_dir / relative_file_path).resolve()
    if candidate != code_dir and code_dir not in candidate.parents:
        raise ValueError("File path escapes code directory")
    return candidate


def _sorted_entries(directory: Path) -> list[Path]:
    try:
        entries = list(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    entries.sort(key=lambda path: (path.is_file(), path.name.lower()))
    return entries


def _file_sha1(path: Path) -> str | None:
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _flatten_tree_file_hashes(nodes: list[dict[str, Any]]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    stack: list[dict[str, Any]] = list(nodes)
    while stack:
        node = stack.pop()
        node_type = node.get("type")
        if node_type == "file":
            file_path = node.get("path")
            if isinstance(file_path, str):
                file_hash = node.get("hash")
                out[file_path] = file_hash if isinstance(file_hash, str) else None
            continue
        if node_type == "directory":
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        stack.append(child)
    return out


def _content_to_diff_lines(content: str) -> list[str]:
    if not content:
        return []
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _read_text_lines_for_diff(path: Path) -> list[str] | None:
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_DIFF_SUMMARY_FILE_BYTES + 1)
    except (FileNotFoundError, PermissionError, OSError):
        return None

    if len(raw) > MAX_DIFF_SUMMARY_FILE_BYTES:
        raw = raw[:MAX_DIFF_SUMMARY_FILE_BYTES]

    if b"\x00" in raw[:8192]:
        return None

    content = raw.decode("utf-8", errors="replace")
    return _content_to_diff_lines(content)


def _count_line_diff_fallback(left_lines: list[str], right_lines: list[str]) -> tuple[int, int]:
    added = 0
    removed = 0
    i = 0
    j = 0
    while i < len(left_lines) or j < len(right_lines):
        left_text = left_lines[i] if i < len(left_lines) else None
        right_text = right_lines[j] if j < len(right_lines) else None

        if left_text is not None and right_text is not None and left_text == right_text:
            i += 1
            j += 1
            continue

        if left_text is not None:
            removed += 1
            i += 1
        if right_text is not None:
            added += 1
            j += 1

    return added, removed


def _count_line_diff_lcs(left_lines: list[str], right_lines: list[str]) -> tuple[int, int]:
    added = 0
    removed = 0
    matcher = difflib.SequenceMatcher(None, left_lines, right_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def _count_line_diff(left_lines: list[str], right_lines: list[str]) -> tuple[int, int]:
    matrix_cells = (len(left_lines) + 1) * (len(right_lines) + 1)
    if matrix_cells <= MAX_DIFF_MATRIX_CELLS:
        return _count_line_diff_lcs(left_lines, right_lines)
    return _count_line_diff_fallback(left_lines, right_lines)


def build_code_diff_summary(left_code_dir: Path, right_code_dir: Path) -> dict[str, int]:
    left_tree, _, _ = build_code_tree(left_code_dir)
    right_tree, _, _ = build_code_tree(right_code_dir)
    left_files = _flatten_tree_file_hashes(left_tree)
    right_files = _flatten_tree_file_hashes(right_tree)

    total_added = 0
    total_removed = 0
    changed_files = 0

    for relative_path in sorted(set(left_files.keys()) | set(right_files.keys())):
        in_left = relative_path in left_files
        in_right = relative_path in right_files
        if not in_left and not in_right:
            continue

        if in_left and in_right:
            left_hash = left_files.get(relative_path)
            right_hash = right_files.get(relative_path)
            if left_hash is not None and right_hash is not None and left_hash == right_hash:
                continue

        changed_files += 1

        left_path = left_code_dir / relative_path
        right_path = right_code_dir / relative_path

        if in_left and not in_right:
            left_lines = _read_text_lines_for_diff(left_path)
            if left_lines is not None:
                total_removed += len(left_lines)
            continue

        if in_right and not in_left:
            right_lines = _read_text_lines_for_diff(right_path)
            if right_lines is not None:
                total_added += len(right_lines)
            continue

        left_lines = _read_text_lines_for_diff(left_path)
        right_lines = _read_text_lines_for_diff(right_path)
        if left_lines is None or right_lines is None:
            continue

        added, removed = _count_line_diff(left_lines, right_lines)
        total_added += added
        total_removed += removed

    return {
        "changed_files": changed_files,
        "added_lines": total_added,
        "removed_lines": total_removed,
    }


def build_code_tree(
    code_dir: Path, max_nodes: int = MAX_TREE_NODES
) -> tuple[list[dict[str, Any]], bool, int]:
    if not code_dir.exists() or not code_dir.is_dir():
        return [], False, 0

    truncated = False
    node_count = 0

    def walk(current: Path, relative_prefix: Path) -> list[dict[str, Any]]:
        nonlocal truncated, node_count

        nodes: list[dict[str, Any]] = []
        for entry in _sorted_entries(current):
            if truncated:
                break
            if entry.is_symlink():
                continue

            node_count += 1
            if node_count > max_nodes:
                truncated = True
                break

            relative = relative_prefix / entry.name
            relative_posix = relative.as_posix()

            if entry.is_dir():
                nodes.append(
                    {
                        "name": entry.name,
                        "path": relative_posix,
                        "type": "directory",
                        "children": walk(entry, relative),
                    }
                )
            elif entry.is_file():
                file_hash = _file_sha1(entry)
                nodes.append(
                    {
                        "name": entry.name,
                        "path": relative_posix,
                        "type": "file",
                        "hash": file_hash,
                    }
                )

        return nodes

    tree = walk(code_dir, Path())
    return tree, truncated, node_count

