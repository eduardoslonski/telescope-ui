# server/app/ingest.py
from __future__ import annotations
import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Suppress wandb's own chatty stderr output (e.g. "Loaded credentials…").
# Our own log.debug calls provide the relevant info when --debug is on.
os.environ.setdefault("WANDB_SILENT", "true")

import pyarrow.parquet as pq
import wandb

from .db import (
    connect,
    transaction,
    get_ingest_state,
    update_ingest_state,
    insert_prompts,
    insert_rollouts,
    insert_samples_data,
    insert_rollouts_metrics,
    insert_golden_answers,
    insert_step_metrics,
    insert_events_orchestrator,
    insert_events_trainer,
    insert_events_inference,
    insert_system_metrics_gpu,
    insert_system_metrics_cpu,
    insert_vllm_metrics,
    insert_prompts_discarded,
    insert_rollouts_discarded,
    insert_samples_data_discarded,
    insert_rollouts_metrics_discarded,
    insert_golden_answers_discarded,
    insert_info_turns,
    insert_info_turns_discarded,
    insert_prompts_eval,
    insert_rollouts_eval,
    insert_samples_data_eval,
    insert_rollouts_metrics_eval,
    insert_golden_answers_eval,
    insert_info_turns_eval,
    upsert_run,
    get_ingested_tails,
    insert_ingested_tails,
    get_ingested_steps,
    insert_ingested_steps,
    get_ingested_step_metrics,
    insert_ingested_step_metrics,
    get_ingested_evals_after_training,
    insert_ingested_evals_after_training,
    get_wandb_api_key,
    insert_known_project,
    delete_known_project,
    list_known_projects,
)
from .run_code import (
    ensure_run_storage_dirs,
    get_run_code_dir,
    get_run_code_metadata_path,
    get_run_code_run_dir,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

POLL_SECONDS = 5
TAGGED_RUNS_POLL_SECONDS = 5
TAGGED_RUNS_TAG = "telescope"
INGEST_STATE_RECONCILE_SECONDS = 60
# When there are no known projects yet (fresh DB), periodically re-run full
# discovery so new tagged runs can still be found later.
EMPTY_KNOWN_PROJECTS_DISCOVERY_SECONDS = 60

# Schema version required for runs to be ingested.
# Runs must have a schema_version tag that exactly matches this value.
# Runs without a schema_version tag or with a different value are skipped
# during initial discovery and polling (similar to telescope-ignore).
SCHEMA_VERSION = "0.1.7"

# Current schema version for each table.
TABLE_SCHEMA_VERSIONS: dict[str, str] = {
    "events_orchestrator": "0.1",
    "events_trainer": "0.2",
    "events_inference": "0.5",
    "prompts": "0.1",
    "rollouts": "0.1",
    "samples_data": "0.2",
    "rollouts_metrics": "0.1",
    "golden_answers": "0.1",
    "system_metrics_gpu": "0.1",
    "system_metrics_cpu": "0.1",
    "vllm_metrics": "0.1",
    "step_metrics": "0.2",
    "prompts_discarded": "0.1",
    "rollouts_discarded": "0.1",
    "samples_data_discarded": "0.2",
    "rollouts_metrics_discarded": "0.1",
    "golden_answers_discarded": "0.1",
    "info_turns": "0.1",
    "info_turns_discarded": "0.1",
    "prompts_eval": "0.1",
    "rollouts_eval": "0.1",
    "samples_data_eval": "0.1",
    "rollouts_metrics_eval": "0.1",
    "golden_answers_eval": "0.1",
    "info_turns_eval": "0.1",
    "ingest_state": "0.1",
    "runs": "0.1",
    "custom_metrics_layout": "0.1",
    "ingested_tails": "0.1",
    "ingested_steps": "0.1",
    "ingested_step_metrics": "0.1",
    "ingested_evals_after_training": "0.1",
}

# Number of parallel downloads
PARALLEL_DOWNLOADS = 20
# Number of projects to scan in parallel during initial tagged-run discovery.
DISCOVERY_PROJECT_SCAN_WORKERS = 8
# Number of concurrent sync workers pulling from the sync queue.
SYNC_QUEUE_WORKERS = 4
SOURCE_ZIP_REMOTE_PATH = "code/source.zip"
SOURCE_METADATA_REMOTE_PATH = "code/metadata.json"

# Global state for active runs being tracked
_active_runs: dict[str, dict] = {}

# Global state for sync status
_sync_status: dict[str, dict] = {}

# Global state for evals-after-training sync status (separate from main sync)
_evals_after_training_status: dict[str, dict] = {}

# Global state for run discovery progress
_discovery_status: dict = {"status": "idle"}
_discovery_status_lock = threading.Lock()

# Queue for syncs triggered by config/poll.
# Tuple shape: (run_path, api_key, force_sync)
_sync_queue: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue()

# Set of run paths currently in the sync queue or being processed by a worker.
# Prevents the reconcile loop / poll loop from re-enqueuing runs that are
# already queued or actively syncing, which caused duplicate sync rounds.
_sync_queue_pending: set[str] = set()

# Projects known to contain telescope-tagged runs (populated from DB on startup).
_known_projects: set[str] = set()

# Compaction pause — when True, all background loops skip their work.
_compaction_paused: bool = False

# Cached wandb API client — avoids the overhead of re-creating (and
# login-verifying) a new client on every 5-second poll iteration.
_cached_api: Any | None = None
_cached_api_key: str | None = None
_cached_api_lock = threading.Lock()


def _get_wandb_api(api_key: str) -> Any:
    """Return a cached wandb.Api instance, creating one only if needed."""
    global _cached_api, _cached_api_key
    with _cached_api_lock:
        if _cached_api is not None and _cached_api_key == api_key:
            log.debug("[WANDB] Reusing cached wandb.Api client")
            return _cached_api
        t = time.time()
        log.debug("[WANDB] Creating new wandb.Api client...")
        _cached_api = wandb.Api(api_key=api_key)
        _cached_api_key = api_key
        log.debug(f"[WANDB] wandb.Api client created in {time.time() - t:.2f}s")
        return _cached_api


def _get_fresh_run(api: Any, run_path: str) -> Any:
    """Get a fresh Run object, bypassing the wandb SDK's internal run cache.

    wandb.Api caches Run objects in api._runs, so repeated calls to
    api.run(path) return the same object with a stale .summary.
    Evicting the cache entry forces the SDK to create a new Run with
    up-to-date summary data from the server.
    """
    api._runs.pop(run_path, None)
    return api.run(run_path)

# Per-run ingestion locks to prevent sync + tracking from ingesting the same run
# concurrently (which can lead to duplicates without UNIQUE constraints).
_run_ingest_locks: dict[str, asyncio.Lock] = {}


def _get_run_ingest_lock(run_path: str) -> asyncio.Lock:
    lock = _run_ingest_locks.get(run_path)
    if lock is None:
        lock = asyncio.Lock()
        _run_ingest_locks[run_path] = lock
    return lock


def _extract_zip_safely(zip_path: Path, destination: Path) -> int:
    """Extract zip file while rejecting entries that escape destination."""
    extracted_files = 0
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename
            if not member_name:
                continue

            target_path = (destination / member_name).resolve()
            if target_path != destination_resolved and destination_resolved not in target_path.parents:
                log.warning(f"[CODE] Skipping unsafe zip path: {member_name}")
                continue

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1

    return extracted_files


def _sync_source_artifacts(run: Any, run_path: str) -> None:
    """Download run code artifacts (source.zip + metadata.json) once per local run snapshot."""
    ensure_run_storage_dirs()
    source_dir = get_run_code_dir(run_path)
    metadata_path = get_run_code_metadata_path(run_path)
    run_code_dir = get_run_code_run_dir(run_path)

    # Code artifacts are immutable for a run. If both files exist locally, skip.
    if (
        source_dir.exists()
        and source_dir.is_dir()
        and metadata_path.exists()
        and metadata_path.is_file()
    ):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            source_zip_file = run.file(SOURCE_ZIP_REMOTE_PATH).download(root=tmpdir, replace=True)
        except wandb.errors.CommError:
            log.info(f"[CODE] {SOURCE_ZIP_REMOTE_PATH} not found for {run_path}")
            return
        except Exception as e:
            log.warning(
                f"[CODE] Failed to download {SOURCE_ZIP_REMOTE_PATH} for {run_path}: {repr(e)}"
            )
            return

        if source_zip_file is None:
            log.info(f"[CODE] Empty download for {SOURCE_ZIP_REMOTE_PATH} in {run_path}")
            return

        try:
            metadata_file = run.file(SOURCE_METADATA_REMOTE_PATH).download(root=tmpdir, replace=True)
        except wandb.errors.CommError:
            log.info(f"[CODE] {SOURCE_METADATA_REMOTE_PATH} not found for {run_path}")
            return
        except Exception as e:
            log.warning(
                f"[CODE] Failed to download {SOURCE_METADATA_REMOTE_PATH} for {run_path}: {repr(e)}"
            )
            return

        if metadata_file is None:
            log.info(f"[CODE] Empty download for {SOURCE_METADATA_REMOTE_PATH} in {run_path}")
            return

        zip_path = Path(source_zip_file.name)
        metadata_local_path = Path(metadata_file.name)
        if not metadata_local_path.exists() or not metadata_local_path.is_file():
            log.warning(f"[CODE] Missing local metadata file after download for {run_path}")
            return

        extract_dir = Path(tmpdir) / "source_extract"
        try:
            extracted_files = _extract_zip_safely(zip_path, extract_dir)
        except zipfile.BadZipFile:
            log.warning(f"[CODE] Invalid zip archive for {run_path}: {SOURCE_ZIP_REMOTE_PATH}")
            return
        except Exception as e:
            log.warning(f"[CODE] Failed extracting source zip for {run_path}: {repr(e)}")
            return

        if extracted_files == 0:
            log.info(f"[CODE] No files extracted from {SOURCE_ZIP_REMOTE_PATH} for {run_path}")
            return

        run_code_dir.parent.mkdir(parents=True, exist_ok=True)
        if run_code_dir.exists():
            shutil.rmtree(run_code_dir)
        run_code_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extract_dir, source_dir)
        shutil.copy2(metadata_local_path, metadata_path)
        log.info(
            f"[CODE] Downloaded source+metadata for {run_path} "
            f"({extracted_files} files from {SOURCE_ZIP_REMOTE_PATH})"
        )


def _extract_commit_and_schema_version(tags: list[str]) -> tuple[str | None, str | None, dict[str, str]]:
    """Extract commit hash, schema version, and per-table schema versions from W&B tags.

    Looks for tags in the format ``commit:<hash>``, ``schema_version:<value>``,
    and ``schema_version_<table>:<value>``.
    Returns (commit, schema_version, table_schema_versions).
    """
    commit = None
    schema_version = None
    table_schema_versions: dict[str, str] = {}
    for tag in tags:
        if tag.startswith("commit:"):
            commit = tag[len("commit:"):]
        elif tag.startswith("schema_version_"):
            # schema_version_<table>:<value>
            rest = tag[len("schema_version_"):]
            if ":" in rest:
                table, value = rest.split(":", 1)
                table_schema_versions[table] = value
        elif tag.startswith("schema_version:"):
            schema_version = tag[len("schema_version:"):]
    return commit, schema_version, table_schema_versions


def _schema_version_mismatch(tags: list[str]) -> bool:
    """Return True if the run's schema_version tag is missing or doesn't match SCHEMA_VERSION."""
    _, schema_version, _ = _extract_commit_and_schema_version(tags)
    return schema_version != SCHEMA_VERSION


def build_run_metadata(run, run_path: str) -> dict:
    """Extract run metadata from a W&B run object.

    Only reads fields already present on the lightweight GraphQL fragment
    (no extra API calls).
    """
    from datetime import datetime

    # Access _attrs directly where possible to avoid __getattr__ overhead
    # and to sidestep the Attrs special-case that raises AttributeError for 'user'.
    attrs = getattr(run, "_attrs", {})

    run_data = {
        "run_id": run_path,
        "wandb_id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "url": getattr(run, "url", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "state": getattr(run, "state", None),
        "notes": attrs.get("notes"),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # Get created_at - W&B returns it as a datetime object or string
    created_at = attrs.get("createdAt") or getattr(run, "created_at", None)
    if created_at:
        if hasattr(created_at, "isoformat"):
            run_data["created_at"] = created_at.isoformat()
        else:
            run_data["created_at"] = str(created_at)

    # Get user info directly from the attrs dict (lightweight fragment
    # includes user { name, username }).
    user_data = attrs.get("user")
    if isinstance(user_data, dict):
        run_data["user_name"] = user_data.get("username") or user_data.get("name")
    elif hasattr(user_data, "username"):
        run_data["user_name"] = getattr(user_data, "username", None) or getattr(user_data, "name", None)

    # Get tags
    tags = getattr(run, "tags", None)
    if tags:
        tags_list = list(tags) if not isinstance(tags, list) else tags
        run_data["tags"] = tags_list
        commit, schema_version, table_schema_versions = _extract_commit_and_schema_version(tags_list)
        if commit:
            run_data["trainer_commit"] = commit
        if schema_version:
            run_data["schema_version"] = schema_version
        if table_schema_versions:
            run_data["table_schema_versions"] = table_schema_versions

    # Get lastHistoryStep / historyLineCount if available (in lightweight fragment)
    run_data["last_history_step"] = (
        attrs.get("historyLineCount")
        or attrs.get("lastHistoryStep")
        or getattr(run, "lastHistoryStep", None)
    )

    return run_data


def save_run_metadata(run, run_path: str, config_json: str | None = None):
    """Extract and save run metadata from a W&B run object to the database."""
    log.info(f"[METADATA] Extracting metadata for run: {run_path}")

    try:
        run_data = build_run_metadata(run, run_path)
        if config_json is not None:
            run_data["config_json"] = config_json

        # Store in database
        con = connect()
        upsert_run(con, run_data)
        con.close()

        log.info(f"[METADATA] Saved metadata for run: {run_path} (created: {run_data.get('created_at')})")

    except Exception as e:
        log.error(f"[METADATA] Error extracting metadata for {run_path}: {repr(e)}")


def set_active_run(run_path: str, api_key: str):
    """Set the active run to track."""
    log.info(f"[TRACKING] Adding run to active tracking: {run_path}")
    _active_runs[run_path] = {"api_key": api_key}


def get_active_runs() -> dict[str, dict]:
    """Get all active runs being tracked."""
    return _active_runs


def clear_active_run(run_path: str):
    """Stop tracking a run (only if it was being tracked)."""
    if run_path in _active_runs:
        log.info(f"[TRACKING] Removing run from active tracking: {run_path}")
        del _active_runs[run_path]


def is_tracking(run_path: str) -> bool:
    """Check if a run is being tracked."""
    return run_path in _active_runs


def get_all_known_projects() -> list[dict]:
    """Return all known projects with source info (user-added vs derived from runs)."""
    con = connect()
    user_projects = list_known_projects(con)
    con.close()

    user_set = {p["project"] for p in user_projects}
    result = []
    for p in user_projects:
        result.append({"project": p["project"], "source": "user", "added_at": p["added_at"]})
    for project in sorted(_known_projects):
        if project not in user_set:
            result.append({"project": project, "source": "derived", "added_at": None})
    return result


def has_known_projects() -> bool:
    """Return True if there are any known projects (user-added or derived)."""
    return bool(_known_projects)


def add_project_to_known(project: str) -> bool:
    """Add a project to known projects (in-memory + DB). Returns True if newly added."""
    con = connect()
    is_new = insert_known_project(con, project)
    con.close()
    _known_projects.add(project)
    return is_new


def remove_project_from_known(project: str) -> bool:
    """Remove a user-added project from known projects. Returns True if removed."""
    con = connect()
    removed = delete_known_project(con, project)
    con.close()
    if removed:
        _known_projects.discard(project)
    return removed


async def pause_for_compaction():
    """Pause all background sync/ingestion loops and wait for active syncs to drain."""
    global _compaction_paused
    _compaction_paused = True
    log.info("[COMPACT] Pausing all background loops...")
    # Wait for any in-flight sync workers to finish (up to 120s)
    for _ in range(1200):
        if not _sync_queue_pending and _active_sync_workers == 0:
            break
        await asyncio.sleep(0.1)
    log.info("[COMPACT] All background loops paused")


def resume_after_compaction():
    """Resume all background sync/ingestion loops."""
    global _compaction_paused
    _compaction_paused = False
    log.info("[COMPACT] Background loops resumed")


def restore_active_runs_from_db(api_key: str) -> int:
    """Re-populate _active_runs from the DB for runs last known to be running.

    Called at startup so that tracking resumes immediately without waiting
    for a full W&B project scan.

    Also pre-populates _known_projects so the very first poll can use the
    fast path instead of scanning every project in the entity.
    """
    con = connect()
    rows = con.execute("""
        SELECT run_id FROM runs
        WHERE COALESCE(removed, FALSE) = FALSE
          AND state = 'running'
    """).fetchall()

    # Pre-populate _known_projects from ALL run_ids in the DB.
    # run_id format is "entity/project/run_id" — we extract "entity/project".
    all_run_ids = con.execute("SELECT run_id FROM runs").fetchall()

    for (run_id,) in all_run_ids:
        parts = run_id.split("/")
        if len(parts) >= 3:
            _known_projects.add(f"{parts[0]}/{parts[1]}")

    # Also load user-added projects from the persistent known_projects table.
    user_projects = list_known_projects(con)
    con.close()
    for p in user_projects:
        _known_projects.add(p["project"])
    if user_projects:
        log.debug(f"[WANDB] Loaded {len(user_projects)} user-added project(s) from DB")

    if _known_projects:
        log.debug(f"[WANDB] Pre-populated {len(_known_projects)} known project(s) from DB: {sorted(_known_projects)}")

    count = 0
    for (run_id,) in rows:
        set_active_run(run_id, api_key)
        count += 1

    if count:
        log.info(f"[TRACKING] Restored {count} running run(s) from DB for immediate tracking")
    return count


def get_sync_status(run_path: str) -> dict | None:
    """Get sync status for a run."""
    return _sync_status.get(run_path)


def is_syncing(run_path: str) -> bool:
    """Check if a run is currently syncing."""
    status = _sync_status.get(run_path)
    return status is not None and status.get("status") == "syncing"


def is_syncing_evals_after_training(run_path: str) -> bool:
    """Check if evals-after-training sync is in progress for a run."""
    status = _evals_after_training_status.get(run_path)
    return status is not None and status.get("status") == "syncing"


def get_discovery_status() -> dict:
    """Get the current run discovery status."""
    with _discovery_status_lock:
        return _discovery_status.copy()


def set_discovery_status_discovering() -> None:
    """Set discovery status to 'discovering' synchronously.

    Called from the /add-project endpoint *before* the background task starts
    so the very next /runs poll sees the status immediately.
    """
    with _discovery_status_lock:
        _discovery_status.update({
            "status": "discovering",
            "runs_found": 0,
            "projects_scanned": 0,
            "total_projects": 1,
        })


def _list_files_with_prefix(run: Any, prefix: str) -> list[str]:
    """List all files in the run with a given prefix."""
    try:
        log.info(f"[WANDB] Listing files with prefix: {prefix}")
        files = [f.name for f in run.files() if f.name.startswith(prefix)]
        log.info(f"[WANDB] Found {len(files)} files with prefix {prefix}")
        return files
    except Exception as e:
        log.error(f"[WANDB] Error listing files with prefix {prefix}: {e}")
        return []


def _to_serializable(obj: Any) -> Any:
    """Recursively convert wandb objects (like SummarySubDict) to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if hasattr(obj, "items"):
        try:
            return {k: _to_serializable(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None


class EventZipData:
    """Container for all data extracted from an event zip file."""
    def __init__(self):
        self.orchestrator: list[dict] | None = None
        self.trainer: list[dict] | None = None
        self.inference: list[dict] | None = None
        self.gpu: list[dict] | None = None
        self.cpu: list[dict] | None = None
        self.vllm: list[dict] | None = None
        # Discarded data (from event zips)
        self.prompts_discarded: list[dict] | None = None
        self.rollouts_discarded: list[dict] | None = None
        self.samples_data_discarded: list[dict] | None = None
        self.rollouts_metrics_discarded: list[dict] | None = None
        self.golden_answers_discarded: list[dict] | None = None
        self.info_turns_discarded: list[dict] | None = None
        # Eval data (from event zips, same tail_idx lifecycle as discarded)
        self.prompts_eval: list[dict] | None = None
        self.rollouts_eval: list[dict] | None = None
        self.samples_data_eval: list[dict] | None = None
        self.rollouts_metrics_eval: list[dict] | None = None
        self.golden_answers_eval: list[dict] | None = None
        self.info_turns_eval: list[dict] | None = None
        # Metadata from metadata.json inside the zip
        self.metadata: dict | None = None
    
    @property
    def min_tail_idx(self) -> int | None:
        """Get min_tail_idx from metadata."""
        if self.metadata:
            return self.metadata.get("min_tail_idx")
        return None
    
    @property
    def max_tail_idx(self) -> int | None:
        """Get max_tail_idx from metadata."""
        if self.metadata:
            return self.metadata.get("max_tail_idx")
        return None
    
    def has_any_data(self) -> bool:
        return any([
            self.orchestrator,
            self.trainer,
            self.inference,
            self.gpu,
            self.cpu,
            self.vllm,
            self.prompts_discarded,
            self.rollouts_discarded,
            self.samples_data_discarded,
            self.rollouts_metrics_discarded,
            self.golden_answers_discarded,
            self.info_turns_discarded,
            self.prompts_eval,
            self.rollouts_eval,
            self.samples_data_eval,
            self.rollouts_metrics_eval,
            self.golden_answers_eval,
            self.info_turns_eval,
        ])


def _download_event_zip_sync(run: Any, file_path: str) -> tuple[str, EventZipData]:
    """Download an event zip file and extract all parquet files plus metadata.
    
    Each zip contains:
    - metadata.json (contains min_tail_idx, max_tail_idx)
    - orchestrator.parquet (instant events with tail_idx column)
    - trainer.parquet (duration events with tail_idx column)
    - inference.parquet (duration events with tail_idx column)
    - gpu.parquet (GPU metrics with tail_idx column)
    - cpu.parquet (CPU metrics with tail_idx column)
    - vllm.parquet (vLLM inference server metrics with tail_idx column)
    - rollouts_discarded.parquet (discarded rollouts with tail_idx column)
    - rollouts_metrics_discarded.parquet (discarded rollout metrics with tail_idx column)
    - golden_answers_discarded.parquet (discarded golden answers with tail_idx column)
    
    Returns (file_path, EventZipData).
    """
    data = EventZipData()
    try:
        start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = run.file(file_path).download(root=tmpdir, replace=True)
            if local_file:
                # Extract zip file
                with zipfile.ZipFile(local_file.name, 'r') as zf:
                    zf.extractall(tmpdir)
                    
                    # Read metadata.json if it exists
                    metadata_path = f"{tmpdir}/metadata.json"
                    try:
                        with open(metadata_path, 'r') as f:
                            data.metadata = json.load(f)
                        log.info(f"[WANDB] Extracted metadata: {data.metadata}")
                    except Exception as e:
                        log.debug(f"[WANDB] No metadata.json in {file_path}: {e}")
                    
                    # Read orchestrator.parquet if it exists
                    orchestrator_path = f"{tmpdir}/orchestrator.parquet"
                    try:
                        table = pq.read_table(orchestrator_path)
                        data.orchestrator = table.to_pylist()
                        log.info(f"[WANDB] Extracted orchestrator: {len(data.orchestrator)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No orchestrator.parquet in {file_path}: {e}")
                    
                    # Read trainer.parquet if it exists
                    trainer_path = f"{tmpdir}/trainer.parquet"
                    try:
                        table = pq.read_table(trainer_path)
                        data.trainer = table.to_pylist()
                        log.info(f"[WANDB] Extracted trainer: {len(data.trainer)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No trainer.parquet in {file_path}: {e}")
                    
                    # Read inference.parquet if it exists
                    inference_path = f"{tmpdir}/inference.parquet"
                    try:
                        table = pq.read_table(inference_path)
                        data.inference = table.to_pylist()
                        log.info(f"[WANDB] Extracted inference: {len(data.inference)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No inference.parquet in {file_path}: {e}")
                    
                    # Read gpu.parquet if it exists
                    gpu_path = f"{tmpdir}/gpu.parquet"
                    try:
                        table = pq.read_table(gpu_path)
                        data.gpu = table.to_pylist()
                        log.info(f"[WANDB] Extracted gpu: {len(data.gpu)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No gpu.parquet in {file_path}: {e}")
                    
                    # Read cpu.parquet if it exists
                    cpu_path = f"{tmpdir}/cpu.parquet"
                    try:
                        table = pq.read_table(cpu_path)
                        data.cpu = table.to_pylist()
                        log.info(f"[WANDB] Extracted cpu: {len(data.cpu)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No cpu.parquet in {file_path}: {e}")
                    
                    # Read vllm.parquet if it exists
                    vllm_path = f"{tmpdir}/vllm.parquet"
                    try:
                        table = pq.read_table(vllm_path)
                        data.vllm = table.to_pylist()
                        log.info(f"[WANDB] Extracted vllm: {len(data.vllm)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No vllm.parquet in {file_path}: {e}")
                    
                    # Read prompts_discarded.parquet if it exists
                    prompts_discarded_path = f"{tmpdir}/prompts_discarded.parquet"
                    try:
                        table = pq.read_table(prompts_discarded_path)
                        data.prompts_discarded = table.to_pylist()
                        log.info(f"[WANDB] Extracted prompts_discarded: {len(data.prompts_discarded)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No prompts_discarded.parquet in {file_path}: {e}")
                    
                    # Read rollouts_discarded.parquet if it exists
                    gen_discarded_path = f"{tmpdir}/rollouts_discarded.parquet"
                    try:
                        table = pq.read_table(gen_discarded_path)
                        data.rollouts_discarded = table.to_pylist()
                        log.info(f"[WANDB] Extracted rollouts_discarded: {len(data.rollouts_discarded)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No rollouts_discarded.parquet in {file_path}: {e}")
                    
                    # Read samples_data_discarded.parquet if it exists
                    samples_data_discarded_path = f"{tmpdir}/samples_data_discarded.parquet"
                    try:
                        table = pq.read_table(samples_data_discarded_path)
                        data.samples_data_discarded = table.to_pylist()
                        log.info(f"[WANDB] Extracted samples_data_discarded: {len(data.samples_data_discarded)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No samples_data_discarded.parquet in {file_path}: {e}")
                    
                    # Read rollouts_metrics_discarded.parquet if it exists
                    metrics_discarded_path = f"{tmpdir}/rollouts_metrics_discarded.parquet"
                    try:
                        table = pq.read_table(metrics_discarded_path)
                        data.rollouts_metrics_discarded = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted rollouts_metrics_discarded: "
                            f"{len(data.rollouts_metrics_discarded)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No rollouts_metrics_discarded.parquet in {file_path}: {e}"
                        )

                    # Read golden_answers_discarded.parquet if it exists
                    golden_answers_discarded_path = f"{tmpdir}/golden_answers_discarded.parquet"
                    try:
                        table = pq.read_table(golden_answers_discarded_path)
                        data.golden_answers_discarded = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted golden_answers_discarded: "
                            f"{len(data.golden_answers_discarded)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No golden_answers_discarded.parquet in {file_path}: {e}"
                        )

                    # Read info_turns_discarded.parquet if it exists
                    info_turns_discarded_path = f"{tmpdir}/info_turns_discarded.parquet"
                    try:
                        table = pq.read_table(info_turns_discarded_path)
                        data.info_turns_discarded = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted info_turns_discarded: "
                            f"{len(data.info_turns_discarded)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No info_turns_discarded.parquet in {file_path}: {e}"
                        )
                    
                    # Read eval parquet files
                    for eval_name, attr in [
                        ("prompts_eval", "prompts_eval"),
                        ("rollouts_eval", "rollouts_eval"),
                        ("samples_data_eval", "samples_data_eval"),
                        ("rollouts_metrics_eval", "rollouts_metrics_eval"),
                        ("golden_answers_eval", "golden_answers_eval"),
                        ("info_turns_eval", "info_turns_eval"),
                    ]:
                        eval_path = f"{tmpdir}/{eval_name}.parquet"
                        try:
                            table = pq.read_table(eval_path)
                            setattr(data, attr, table.to_pylist())
                            log.info(
                                f"[WANDB] Extracted {eval_name}: "
                                f"{len(getattr(data, attr))} rows"
                            )
                        except Exception as e:
                            log.debug(
                                f"[WANDB] No {eval_name}.parquet in {file_path}: {e}"
                            )
                
                elapsed = time.time() - start
                log.info(f"[WANDB] Downloaded {file_path} in {elapsed:.2f}s")
                return (file_path, data)
    except wandb.errors.CommError:
        log.debug(f"[WANDB] File not found: {file_path}")
    except Exception as e:
        log.error(f"[WANDB] Error downloading {file_path}: {e}")
    return (file_path, data)


def _download_event_zips_parallel(run: Any, block_indices: list[int]) -> dict[int, EventZipData]:
    """Download multiple event block zip files in parallel.
    Returns dict mapping index to EventZipData containing all parquet data."""
    results = {}
    
    if not block_indices:
        return results
    
    log.info(f"[EVENTS] Downloading {len(block_indices)} event blocks in parallel: {block_indices}")
    
    # Create file paths for all blocks
    files_to_download = [(idx, f"events/block_{idx}.zip") for idx in block_indices]
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as executor:
        # Submit all download tasks
        future_to_idx = {
            executor.submit(_download_event_zip_sync, run, file_path): idx
            for idx, file_path in files_to_download
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                file_path, data = future.result()
                if data.has_any_data():
                    results[idx] = data
            except Exception as e:
                log.error(f"[EVENTS] Error downloading zip {idx}: {e}")
    
    return results


class RolloutZipData:
    """Container for data extracted from a rollout zip file."""
    def __init__(self):
        self.prompts: list[dict] | None = None
        self.rollouts: list[dict] | None = None
        self.samples_data: list[dict] | None = None
        self.rollouts_metrics: list[dict] | None = None
        self.golden_answers: list[dict] | None = None
        self.info_turns: list[dict] | None = None
        self.metrics: list[dict] | None = None
        # Metadata from metadata.json inside the zip
        self.metadata: dict | None = None
    
    @property
    def start_step(self) -> int | None:
        """Get start_step from metadata (used in finalized blocks)."""
        if self.metadata:
            return self.metadata.get("start_step")
        return None
    
    @property
    def end_step(self) -> int | None:
        """Get end_step from metadata (only present in finalized blocks)."""
        if self.metadata:
            return self.metadata.get("end_step")
        return None
    
    @property
    def min_step(self) -> int | None:
        """Get min_step from metadata (used in tail.zip and block_live.zip)."""
        if self.metadata:
            return self.metadata.get("min_step")
        return None
    
    @property
    def max_step(self) -> int | None:
        """Get max_step from metadata (used in tail.zip and block_live.zip)."""
        if self.metadata:
            return self.metadata.get("max_step")
        return None
    
    def has_any_data(self) -> bool:
        return (
            self.prompts is not None
            or self.rollouts is not None
            or self.samples_data is not None
            or self.rollouts_metrics is not None
            or self.golden_answers is not None
            or self.info_turns is not None
            or self.metrics is not None
        )


def _download_rollout_zip_sync(run: Any, file_path: str) -> tuple[str, RolloutZipData]:
    """Download a rollout zip file and extract prompts.parquet, rollouts.parquet,
    rollouts_metrics.parquet, golden_answers.parquet, and metadata.json.
    Returns (file_path, RolloutZipData)."""
    data = RolloutZipData()
    try:
        start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = run.file(file_path).download(root=tmpdir, replace=True)
            if local_file:
                # Extract zip file
                with zipfile.ZipFile(local_file.name, 'r') as zf:
                    zf.extractall(tmpdir)
                    
                    # Read metadata.json if it exists
                    metadata_path = f"{tmpdir}/metadata.json"
                    try:
                        with open(metadata_path, 'r') as f:
                            data.metadata = json.load(f)
                        log.info(f"[WANDB] Extracted rollout metadata: {data.metadata}")
                    except Exception as e:
                        log.debug(f"[WANDB] No metadata.json in {file_path}: {e}")
                    
                    # Read prompts.parquet if it exists (new schema)
                    prompts_path = f"{tmpdir}/prompts.parquet"
                    try:
                        table = pq.read_table(prompts_path)
                        data.prompts = table.to_pylist()
                        log.info(f"[WANDB] Extracted prompts: {len(data.prompts)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No prompts.parquet in {file_path}: {e}")
                    
                    # Read rollouts.parquet if it exists
                    rollouts_path = f"{tmpdir}/rollouts.parquet"
                    try:
                        table = pq.read_table(rollouts_path)
                        data.rollouts = table.to_pylist()
                        log.info(f"[WANDB] Extracted rollouts: {len(data.rollouts)} rows")
                    except Exception as e:
                        log.warning(f"[WANDB] No rollouts.parquet in {file_path}: {e}")
                    
                    # Read samples_data.parquet if it exists
                    samples_data_path = f"{tmpdir}/samples_data.parquet"
                    try:
                        table = pq.read_table(samples_data_path)
                        data.samples_data = table.to_pylist()
                        log.info(f"[WANDB] Extracted samples_data: {len(data.samples_data)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No samples_data.parquet in {file_path}: {e}")
                    
                    # Read rollouts_metrics.parquet if it exists
                    rollouts_metrics_path = f"{tmpdir}/rollouts_metrics.parquet"
                    try:
                        table = pq.read_table(rollouts_metrics_path)
                        data.rollouts_metrics = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted rollouts_metrics: "
                            f"{len(data.rollouts_metrics)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No rollouts_metrics.parquet in {file_path}: {e}"
                        )

                    # Read golden_answers.parquet if it exists
                    golden_answers_path = f"{tmpdir}/golden_answers.parquet"
                    try:
                        table = pq.read_table(golden_answers_path)
                        data.golden_answers = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted golden_answers: "
                            f"{len(data.golden_answers)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No golden_answers.parquet in {file_path}: {e}"
                        )

                    # Read info_turns.parquet if it exists
                    info_turns_path = f"{tmpdir}/info_turns.parquet"
                    try:
                        table = pq.read_table(info_turns_path)
                        data.info_turns = table.to_pylist()
                        log.info(
                            "[WANDB] Extracted info_turns: "
                            f"{len(data.info_turns)} rows"
                        )
                    except Exception as e:
                        log.debug(
                            f"[WANDB] No info_turns.parquet in {file_path}: {e}"
                        )
                    
                    # Read metrics.parquet if it exists
                    metrics_path = f"{tmpdir}/metrics.parquet"
                    try:
                        table = pq.read_table(metrics_path)
                        data.metrics = table.to_pylist()
                        log.info(f"[WANDB] Extracted metrics: {len(data.metrics)} rows")
                    except Exception as e:
                        log.debug(f"[WANDB] No metrics.parquet in {file_path}: {e}")
                
                elapsed = time.time() - start
                log.info(f"[WANDB] Downloaded {file_path} in {elapsed:.2f}s")
                return (file_path, data)
    except wandb.errors.CommError:
        log.debug(f"[WANDB] File not found: {file_path}")
    except Exception as e:
        log.error(f"[WANDB] Error downloading {file_path}: {e}")
    return (file_path, data)


def _download_rollout_zips_parallel(run: Any, block_indices: list[int]) -> dict[int, RolloutZipData]:
    """Download multiple rollout block zip files in parallel.
    Returns dict mapping index to RolloutZipData."""
    results = {}
    
    if not block_indices:
        return results
    
    log.info(f"[ROLLOUTS] Downloading {len(block_indices)} rollout blocks in parallel: {block_indices}")
    
    # Create file paths for all blocks
    files_to_download = [(idx, f"steps/block_{idx}.zip") for idx in block_indices]
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as executor:
        # Submit all download tasks
        future_to_idx = {
            executor.submit(_download_rollout_zip_sync, run, file_path): idx
            for idx, file_path in files_to_download
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                file_path, data = future.result()
                if data.has_any_data():
                    results[idx] = data
            except Exception as e:
                log.error(f"[ROLLOUTS] Error downloading zip {idx}: {e}")
    
    return results




def _filter_events_by_tails(events: list[dict] | None, missing_tails: set[int]) -> list[dict]:
    """Filter events to only include those with tail_idx in missing_tails."""
    if not events or not missing_tails:
        return []
    return [e for e in events if e.get("tail_idx") in missing_tails]


def _filter_prompts_by_steps(prompts: list[dict] | None, missing_steps: set[int]) -> list[dict]:
    """Filter prompts to only include those with step in missing_steps."""
    if not prompts or not missing_steps:
        return []
    return [p for p in prompts if p.get("step") in missing_steps]


def _filter_rollouts_by_steps(rollouts: list[dict] | None, missing_steps: set[int]) -> list[dict]:
    """Filter rollouts to only include those with step in missing_steps."""
    if not rollouts or not missing_steps:
        return []
    return [g for g in rollouts if g.get("step") in missing_steps]


def _filter_samples_data_by_steps(samples_data: list[dict] | None, missing_steps: set[int]) -> list[dict]:
    """Filter samples_data to only include those with step in missing_steps."""
    if not samples_data or not missing_steps:
        return []
    return [s for s in samples_data if s.get("step") in missing_steps]


def _filter_rollouts_metrics_by_steps(
    metrics: list[dict] | None,
    missing_steps: set[int],
) -> list[dict]:
    """Filter rollouts metrics to only include those with step in missing_steps."""
    if not metrics or not missing_steps:
        return []
    return [m for m in metrics if m.get("step") in missing_steps]


def _filter_golden_answers_by_steps(
    answers: list[dict] | None,
    missing_steps: set[int],
) -> list[dict]:
    """Filter golden answers to only include those with step in missing_steps."""
    if not answers or not missing_steps:
        return []
    return [a for a in answers if a.get("step") in missing_steps]


def _filter_info_turns_by_steps(
    info_turns: list[dict] | None,
    missing_steps: set[int],
) -> list[dict]:
    """Filter info turns to only include those with step in missing_steps."""
    if not info_turns or not missing_steps:
        return []
    return [it for it in info_turns if it.get("step") in missing_steps]


def _filter_metrics_by_steps(metrics: list[dict] | None, missing_steps: set[int]) -> list[dict]:
    """Filter metrics to only include those with step in missing_steps."""
    if not metrics or not missing_steps:
        return []
    return [m for m in metrics if m.get("step") in missing_steps]


def _get_steps_from_rollouts(rollouts: list[dict] | None) -> set[int]:
    """Extract unique steps from a list of rollouts."""
    if not rollouts:
        return set()
    return {g.get("step") for g in rollouts if g.get("step") is not None}


def _get_steps_from_rollouts_metrics(metrics: list[dict] | None) -> set[int]:
    """Extract unique steps from a list of rollouts metrics."""
    if not metrics:
        return set()
    return {m.get("step") for m in metrics if m.get("step") is not None}


def _get_steps_from_golden_answers(answers: list[dict] | None) -> set[int]:
    """Extract unique steps from a list of golden answers."""
    if not answers:
        return set()
    return {a.get("step") for a in answers if a.get("step") is not None}


def _get_steps_from_metrics(metrics: list[dict] | None) -> set[int]:
    """Extract unique steps from a list of step metrics."""
    if not metrics:
        return set()
    return {m.get("step") for m in metrics if m.get("step") is not None}


def _get_tail_indices_from_events(events: list[dict] | None) -> set[int]:
    """Extract unique tail indices from a list of events."""
    if not events:
        return set()
    return {e.get("tail_idx") for e in events if e.get("tail_idx") is not None}


def _insert_event_zip_data(
    con, 
    run_path: str, 
    data: EventZipData, 
    source_name: str,
    missing_tails: set[int] | None = None
) -> tuple[dict, set[int]]:
    """Insert all data from an EventZipData into the database.
    
    Args:
        con: Database connection
        run_path: The run ID
        data: EventZipData containing events and metadata
        source_name: Name for logging (e.g., "block_0", "tail.zip")
        missing_tails: Optional set of tail indices to filter by. If provided,
                      only events with tail_idx in this set will be inserted.
    
    Returns:
        Tuple of (counts dict, set of tail indices that were actually inserted)
    """
    counts = {
        "orchestrator": 0,
        "trainer": 0,
        "inference": 0,
        "gpu": 0,
        "cpu": 0,
        "vllm": 0,
        "prompts_discarded": 0,
        "rollouts_discarded": 0,
        "samples_data_discarded": 0,
        "rollouts_metrics_discarded": 0,
        "golden_answers_discarded": 0,
        "info_turns_discarded": 0,
        "prompts_eval": 0,
        "rollouts_eval": 0,
        "samples_data_eval": 0,
        "rollouts_metrics_eval": 0,
        "golden_answers_eval": 0,
        "info_turns_eval": 0,
    }
    inserted_tails: set[int] = set()
    
    # If missing_tails is provided, filter events; otherwise use all events
    if missing_tails is not None:
        orchestrator = _filter_events_by_tails(data.orchestrator, missing_tails)
        trainer = _filter_events_by_tails(data.trainer, missing_tails)
        inference = _filter_events_by_tails(data.inference, missing_tails)
        gpu = _filter_events_by_tails(data.gpu, missing_tails)
        cpu = _filter_events_by_tails(data.cpu, missing_tails)
        vllm = _filter_events_by_tails(data.vllm, missing_tails)
        prompts_discarded = _filter_events_by_tails(data.prompts_discarded, missing_tails)
        rollouts_discarded = _filter_events_by_tails(data.rollouts_discarded, missing_tails)
        samples_data_discarded = _filter_events_by_tails(data.samples_data_discarded, missing_tails)
        rollouts_metrics_discarded = _filter_events_by_tails(
            data.rollouts_metrics_discarded, missing_tails
        )
        golden_answers_discarded = _filter_events_by_tails(
            data.golden_answers_discarded, missing_tails
        )
        info_turns_discarded = _filter_events_by_tails(
            data.info_turns_discarded, missing_tails
        )
        prompts_eval = _filter_events_by_tails(data.prompts_eval, missing_tails)
        rollouts_eval = _filter_events_by_tails(data.rollouts_eval, missing_tails)
        samples_data_eval = _filter_events_by_tails(data.samples_data_eval, missing_tails)
        rollouts_metrics_eval = _filter_events_by_tails(data.rollouts_metrics_eval, missing_tails)
        golden_answers_eval = _filter_events_by_tails(data.golden_answers_eval, missing_tails)
        info_turns_eval = _filter_events_by_tails(data.info_turns_eval, missing_tails)
    else:
        orchestrator = data.orchestrator
        trainer = data.trainer
        inference = data.inference
        gpu = data.gpu
        cpu = data.cpu
        vllm = data.vllm
        prompts_discarded = data.prompts_discarded
        rollouts_discarded = data.rollouts_discarded
        samples_data_discarded = data.samples_data_discarded
        rollouts_metrics_discarded = data.rollouts_metrics_discarded
        golden_answers_discarded = data.golden_answers_discarded
        info_turns_discarded = data.info_turns_discarded
        prompts_eval = data.prompts_eval
        rollouts_eval = data.rollouts_eval
        samples_data_eval = data.samples_data_eval
        rollouts_metrics_eval = data.rollouts_metrics_eval
        golden_answers_eval = data.golden_answers_eval
        info_turns_eval = data.info_turns_eval
    
    if orchestrator:
        insert_events_orchestrator(con, run_path, orchestrator)
        counts["orchestrator"] = len(orchestrator)
        inserted_tails.update(_get_tail_indices_from_events(orchestrator))
        log.info(f"[EVENTS] Synced {source_name} orchestrator: {counts['orchestrator']} events")
    
    if trainer:
        insert_events_trainer(con, run_path, trainer)
        counts["trainer"] = len(trainer)
        inserted_tails.update(_get_tail_indices_from_events(trainer))
        log.info(f"[EVENTS] Synced {source_name} trainer: {counts['trainer']} events")
    
    if inference:
        insert_events_inference(con, run_path, inference)
        counts["inference"] = len(inference)
        inserted_tails.update(_get_tail_indices_from_events(inference))
        log.info(f"[EVENTS] Synced {source_name} inference: {counts['inference']} events")
    
    if gpu:
        insert_system_metrics_gpu(con, run_path, gpu)
        counts["gpu"] = len(gpu)
        inserted_tails.update(_get_tail_indices_from_events(gpu))
        log.info(f"[EVENTS] Synced {source_name} gpu: {counts['gpu']} metrics")
    
    if cpu:
        insert_system_metrics_cpu(con, run_path, cpu)
        counts["cpu"] = len(cpu)
        inserted_tails.update(_get_tail_indices_from_events(cpu))
        log.info(f"[EVENTS] Synced {source_name} cpu: {counts['cpu']} metrics")
    
    if vllm:
        insert_vllm_metrics(con, run_path, vllm)
        counts["vllm"] = len(vllm)
        inserted_tails.update(_get_tail_indices_from_events(vllm))
        log.info(f"[EVENTS] Synced {source_name} vllm: {counts['vllm']} metrics")
    
    if prompts_discarded:
        insert_prompts_discarded(con, run_path, prompts_discarded)
        counts["prompts_discarded"] = len(prompts_discarded)
        inserted_tails.update(_get_tail_indices_from_events(prompts_discarded))
        log.info(f"[EVENTS] Synced {source_name} prompts_discarded: {counts['prompts_discarded']} prompts")
    
    if rollouts_discarded:
        insert_rollouts_discarded(con, run_path, rollouts_discarded)
        counts["rollouts_discarded"] = len(rollouts_discarded)
        inserted_tails.update(_get_tail_indices_from_events(rollouts_discarded))
        log.info(f"[EVENTS] Synced {source_name} rollouts_discarded: {counts['rollouts_discarded']} rollout turns")
    
    if samples_data_discarded:
        insert_samples_data_discarded(con, run_path, samples_data_discarded)
        counts["samples_data_discarded"] = len(samples_data_discarded)
        inserted_tails.update(_get_tail_indices_from_events(samples_data_discarded))
        log.info(f"[EVENTS] Synced {source_name} samples_data_discarded: {counts['samples_data_discarded']} samples")
    
    if rollouts_metrics_discarded:
        insert_rollouts_metrics_discarded(con, run_path, rollouts_metrics_discarded)
        counts["rollouts_metrics_discarded"] = len(rollouts_metrics_discarded)
        inserted_tails.update(_get_tail_indices_from_events(rollouts_metrics_discarded))
        log.info(
            f"[EVENTS] Synced {source_name} rollouts_metrics_discarded: "
            f"{counts['rollouts_metrics_discarded']} metrics"
        )

    if golden_answers_discarded:
        insert_golden_answers_discarded(con, run_path, golden_answers_discarded)
        counts["golden_answers_discarded"] = len(golden_answers_discarded)
        inserted_tails.update(_get_tail_indices_from_events(golden_answers_discarded))
        log.info(
            f"[EVENTS] Synced {source_name} golden_answers_discarded: "
            f"{counts['golden_answers_discarded']} answers"
        )

    if info_turns_discarded:
        insert_info_turns_discarded(con, run_path, info_turns_discarded)
        counts["info_turns_discarded"] = len(info_turns_discarded)
        inserted_tails.update(_get_tail_indices_from_events(info_turns_discarded))
        log.info(
            f"[EVENTS] Synced {source_name} info_turns_discarded: "
            f"{counts['info_turns_discarded']} info turns"
        )

    if prompts_eval:
        insert_prompts_eval(con, run_path, prompts_eval)
        counts["prompts_eval"] = len(prompts_eval)
        inserted_tails.update(_get_tail_indices_from_events(prompts_eval))
        log.info(f"[EVENTS] Synced {source_name} prompts_eval: {counts['prompts_eval']} prompts")

    if rollouts_eval:
        insert_rollouts_eval(con, run_path, rollouts_eval)
        counts["rollouts_eval"] = len(rollouts_eval)
        inserted_tails.update(_get_tail_indices_from_events(rollouts_eval))
        log.info(f"[EVENTS] Synced {source_name} rollouts_eval: {counts['rollouts_eval']} turns")

    if samples_data_eval:
        insert_samples_data_eval(con, run_path, samples_data_eval)
        counts["samples_data_eval"] = len(samples_data_eval)
        inserted_tails.update(_get_tail_indices_from_events(samples_data_eval))
        log.info(f"[EVENTS] Synced {source_name} samples_data_eval: {counts['samples_data_eval']} samples")

    if rollouts_metrics_eval:
        insert_rollouts_metrics_eval(con, run_path, rollouts_metrics_eval)
        counts["rollouts_metrics_eval"] = len(rollouts_metrics_eval)
        inserted_tails.update(_get_tail_indices_from_events(rollouts_metrics_eval))
        log.info(
            f"[EVENTS] Synced {source_name} rollouts_metrics_eval: "
            f"{counts['rollouts_metrics_eval']} metrics"
        )

    if golden_answers_eval:
        insert_golden_answers_eval(con, run_path, golden_answers_eval)
        counts["golden_answers_eval"] = len(golden_answers_eval)
        inserted_tails.update(_get_tail_indices_from_events(golden_answers_eval))
        log.info(
            f"[EVENTS] Synced {source_name} golden_answers_eval: "
            f"{counts['golden_answers_eval']} answers"
        )

    if info_turns_eval:
        insert_info_turns_eval(con, run_path, info_turns_eval)
        counts["info_turns_eval"] = len(info_turns_eval)
        inserted_tails.update(_get_tail_indices_from_events(info_turns_eval))
        log.info(
            f"[EVENTS] Synced {source_name} info_turns_eval: "
            f"{counts['info_turns_eval']} info turns"
        )
    
    return counts, inserted_tails


TAILS_PER_BLOCK = 360  # Each block contains 360 tails (5 sec/tail * 360 = 30 minutes)


def _calculate_blocks_for_tails(tail_indices: set[int]) -> set[int]:
    """Calculate which block indices contain the given tail indices."""
    if not tail_indices:
        return set()
    return {tail_idx // TAILS_PER_BLOCK for tail_idx in tail_indices}


async def sync_events_background(run_path: str, api_key: str, run: Any, summary: dict):
    """Background sync of events - downloads zip files containing all 6 parquet files.
    
    Simple approach:
    1. Download tail.zip (last 60 seconds)
    2. Check its metadata (min_tail_idx, max_tail_idx)
    3. Check ingested_tails table for what we're missing in that range
    4. Insert missing tails from tail.zip
    5. If we need older tails (before tail.min_tail_idx), download from blocks
    
    Each zip contains:
    - metadata.json (min_tail_idx, max_tail_idx)
    - orchestrator.parquet, trainer.parquet, inference.parquet, gpu.parquet, cpu.parquet, vllm.parquet
    
    All DB inserts are batched in a single transaction to avoid row-group fragmentation.
    """
    log.info(f"[EVENTS] Starting event sync for: {run_path}")
    events_start = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
        totals = {
            "orchestrator": 0,
            "trainer": 0,
            "inference": 0,
            "gpu": 0,
            "cpu": 0,
            "vllm": 0,
            "prompts_discarded": 0,
            "rollouts_discarded": 0,
            "samples_data_discarded": 0,
            "rollouts_metrics_discarded": 0,
            "golden_answers_discarded": 0,
            "info_turns_discarded": 0,
            "prompts_eval": 0,
            "rollouts_eval": 0,
            "samples_data_eval": 0,
            "rollouts_metrics_eval": 0,
            "golden_answers_eval": 0,
            "info_turns_eval": 0,
        }
        
        # Collect all tails to insert at the end in one transaction
        all_inserted_tails: set[int] = set()
        
        # Step 1: Download and process tail.zip first
        log.info(f"[EVENTS] Fetching tail.zip...")
        file_path, tail_data = await loop.run_in_executor(
            None,
            lambda: _download_event_zip_sync(run, "events/tail.zip")
        )
        
        tail_min = tail_data.min_tail_idx
        tail_max = tail_data.max_tail_idx
        log.info(f"[EVENTS] Tail metadata: min_tail_idx={tail_min}, max_tail_idx={tail_max}")
        
        if tail_max is None:
            log.info(f"[EVENTS] No tail metadata available, skipping event sync")
            return totals
        
        # Step 2: Check which tails from tail.zip are missing
        con = connect()
        ingested_in_tail_range = get_ingested_tails(con, run_path, tail_min, tail_max)
        
        tail_range = set(range(tail_min, tail_max + 1))
        missing_in_tail = tail_range - ingested_in_tail_range
        
        log.info(f"[EVENTS] Tail range {tail_min}-{tail_max}: {len(missing_in_tail)} missing tails")
        
        # Also check older tails upfront
        ingested_before_tail = set()
        if tail_min > 0:
            ingested_before_tail = get_ingested_tails(con, run_path, 0, tail_min - 1)
        con.close()
        
        # Prepare list of data sources to process
        data_sources: list[tuple[str, Any, set[int]]] = []  # (name, data, missing_tails)
        
        # Step 3: Queue tail.zip for processing
        if missing_in_tail and tail_data.has_any_data():
            data_sources.append(("tail.zip", tail_data, missing_in_tail))
        
        # Step 4: Check if we need older tails (before tail.min_tail_idx)
        if tail_min > 0:
            older_tails_range = set(range(0, tail_min))
            missing_older_tails = older_tails_range - ingested_before_tail
            
            if missing_older_tails:
                log.info(f"[EVENTS] Need {len(missing_older_tails)} older tails (range 0-{tail_min - 1})")
                
                # Calculate which blocks we need
                blocks_needed = _calculate_blocks_for_tails(missing_older_tails)
                num_finalized_blocks = summary.get("events/num_finalized_blocks", 0)
                current_block_idx = summary.get("events/current_block_idx", 0)
                
                # Separate into finalized blocks vs live block
                finalized_blocks = sorted([b for b in blocks_needed if b < num_finalized_blocks])
                need_live_block = current_block_idx in blocks_needed and current_block_idx >= num_finalized_blocks
                
                log.info(f"[EVENTS] Need {len(finalized_blocks)} finalized blocks, need_live_block={need_live_block}")
                
                # Download finalized blocks
                if finalized_blocks:
                    for i in range(0, len(finalized_blocks), PARALLEL_DOWNLOADS):
                        batch = finalized_blocks[i:i + PARALLEL_DOWNLOADS]
                        log.info(f"[EVENTS] Downloading blocks: {batch}")
                        
                        results = await loop.run_in_executor(
                            None,
                            lambda b=batch: _download_event_zips_parallel(run, b)
                        )
                        
                        for idx in sorted(results.keys()):
                            data = results[idx]
                            block_min = data.min_tail_idx
                            block_max = data.max_tail_idx
                            
                            if block_min is not None and block_max is not None:
                                block_missing = {t for t in missing_older_tails if block_min <= t <= block_max}
                                if block_missing and data.has_any_data():
                                    data_sources.append((f"block_{idx}", data, block_missing))
                
                # Download live block if needed
                if need_live_block:
                    log.info(f"[EVENTS] Fetching block_live.zip...")
                    file_path, live_data = await loop.run_in_executor(
                        None,
                        lambda: _download_event_zip_sync(run, "events/block_live.zip")
                    )
                    
                    if live_data.has_any_data() and live_data.metadata:
                        live_min = live_data.min_tail_idx
                        live_max = live_data.max_tail_idx
                        
                        if live_min is not None and live_max is not None:
                            live_missing = {t for t in missing_older_tails if live_min <= t <= live_max}
                            if live_missing:
                                data_sources.append(("block_live.zip", live_data, live_missing))
        
        # Step 5: Process all data sources in a SINGLE transaction
        if data_sources:
            log.info(f"[EVENTS] Processing {len(data_sources)} data sources in single transaction...")
            con = connect()
            try:
                with transaction(con):
                    for source_name, data, missing_tails in data_sources:
                        counts, inserted = _insert_event_zip_data(
                            con, run_path, data, source_name,
                            missing_tails=missing_tails
                        )
                        for key in totals:
                            totals[key] += counts[key]
                        all_inserted_tails.update(inserted)
                        log.info(f"[EVENTS] Processed {source_name}: {len(inserted)} tails")
                    
                    # Record all ingested tails at once
                    if all_inserted_tails:
                        insert_ingested_tails(con, run_path, all_inserted_tails)
                        log.info(f"[EVENTS] Recorded {len(all_inserted_tails)} total ingested tails")

            finally:
                con.close()
        
        elapsed = time.time() - events_start
        total_events = totals["orchestrator"] + totals["trainer"] + totals["inference"]
        total_metrics = totals["gpu"] + totals["cpu"] + totals["vllm"]
        total_discarded = (
            totals["prompts_discarded"]
            + totals["rollouts_discarded"]
            + totals["samples_data_discarded"]
            + totals["rollouts_metrics_discarded"]
            + totals["golden_answers_discarded"]
        )
        total_eval = (
            totals["prompts_eval"]
            + totals["rollouts_eval"]
            + totals["samples_data_eval"]
            + totals["rollouts_metrics_eval"]
            + totals["golden_answers_eval"]
            + totals["info_turns_eval"]
        )

        # Persist latest observed tail index so startup can compare local progress
        # against last_summary_json (events/current_tail_idx).
        con = connect()
        try:
            update_ingest_state(con, run_path, last_event_zip_idx=tail_max)
        finally:
            con.close()

        log.info(f"[EVENTS] Completed event sync for {run_path} in {elapsed:.2f}s")
        log.info(f"[EVENTS] Events: {totals['orchestrator']} orchestrator, {totals['trainer']} trainer, {totals['inference']} inference")
        log.info(f"[EVENTS] Metrics: {totals['gpu']} gpu, {totals['cpu']} cpu, {totals['vllm']} vllm")
        log.info(
            "[EVENTS] Discarded: "
            f"{totals['prompts_discarded']} prompts, "
            f"{totals['rollouts_discarded']} rollout turns, "
            f"{totals['samples_data_discarded']} samples_data, "
            f"{totals['rollouts_metrics_discarded']} metrics, "
            f"{totals['golden_answers_discarded']} golden answers"
        )
        if total_eval > 0:
            log.info(
                "[EVENTS] Eval: "
                f"{totals['prompts_eval']} prompts, "
                f"{totals['rollouts_eval']} rollout turns, "
                f"{totals['samples_data_eval']} samples_data, "
                f"{totals['rollouts_metrics_eval']} metrics, "
                f"{totals['golden_answers_eval']} golden answers, "
                f"{totals['info_turns_eval']} info turns"
            )
        
        return totals
        
    except Exception as e:
        log.error(f"[EVENTS] Error syncing events for {run_path}: {repr(e)}")
        raise


STEPS_PER_ROLLOUT_BLOCK = 500  # Each rollout block contains up to 500 steps


def _to_int(value: Any) -> int | None:
    """Best-effort conversion of summary values to int."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None
    return None


def _extract_summary_progress(summary: dict[str, Any]) -> dict[str, int | None]:
    """Extract the progress markers we use to validate local ingest completeness."""
    last_training_step = _to_int(summary.get("steps/last_training_step"))
    if last_training_step is None:
        last_training_step = _to_int(summary.get("_step"))

    return {
        "last_training_step": last_training_step,
        "current_tail_idx": _to_int(summary.get("events/current_tail_idx")),
        "events_current_block_idx": _to_int(summary.get("events/current_block_idx")),
        "steps_current_block_idx": _to_int(summary.get("steps/current_block_idx")),
    }


def get_sync_mismatch_reasons(
    last_gen_step: int,
    last_event_zip_idx: int,
    summary_json: str | None,
    last_event_block_idx: int = -1,
    last_rollout_block_idx: int = -1,
    stored_summary_id: str | None = None,
) -> list[str]:
    """Return reasons when local ingest_state is behind the saved summary."""
    if not summary_json:
        return ["missing_last_summary_json"]

    try:
        summary = json.loads(summary_json)
    except (json.JSONDecodeError, TypeError):
        return ["invalid_last_summary_json"]

    # If the summary_id in the saved summary matches the stored summary_id,
    # then we've already synced this exact summary — nothing to do.
    remote_summary_id = summary.get("summary_id")
    if remote_summary_id is not None and stored_summary_id is not None and remote_summary_id == stored_summary_id:
        return []

    progress = _extract_summary_progress(summary)
    reasons: list[str] = []

    remote_last_training_step = progress["last_training_step"]
    if remote_last_training_step is not None and last_gen_step < remote_last_training_step:
        reasons.append(
            f"last_rollout_step:{last_gen_step}<{remote_last_training_step}"
        )

    remote_tail_idx = progress["current_tail_idx"]
    if remote_tail_idx is not None and last_event_zip_idx < remote_tail_idx:
        reasons.append(
            f"last_event_zip_idx:{last_event_zip_idx}<{remote_tail_idx}"
        )

    remote_event_block_idx = progress["events_current_block_idx"]
    if remote_event_block_idx is not None and last_event_block_idx < remote_event_block_idx:
        reasons.append(
            f"last_block_idx:{last_event_block_idx}<{remote_event_block_idx}"
        )

    remote_rollout_block_idx = progress["steps_current_block_idx"]
    if (
        remote_rollout_block_idx is not None
        and last_rollout_block_idx < remote_rollout_block_idx
    ):
        reasons.append(
            f"last_rollout_block_idx:{last_rollout_block_idx}<{remote_rollout_block_idx}"
        )

    return reasons


def run_needs_sync(
    last_gen_step: int,
    last_event_zip_idx: int,
    summary_json: str | None,
    last_event_block_idx: int = -1,
    last_rollout_block_idx: int = -1,
    stored_summary_id: str | None = None,
) -> bool:
    """Determine if a run needs syncing by comparing ingest_state to last_summary_json."""
    reasons = get_sync_mismatch_reasons(
        last_gen_step=last_gen_step,
        last_event_zip_idx=last_event_zip_idx,
        summary_json=summary_json,
        last_event_block_idx=last_event_block_idx,
        last_rollout_block_idx=last_rollout_block_idx,
        stored_summary_id=stored_summary_id,
    )
    return bool(reasons)


def _calculate_blocks_for_steps(steps: set[int]) -> set[int]:
    """Calculate which rollout block indices contain the given steps."""
    if not steps:
        return set()
    return {step // STEPS_PER_ROLLOUT_BLOCK for step in steps}


def _insert_rollout_zip_data(
    con,
    run_path: str,
    data: RolloutZipData,
    source_name: str,
    missing_steps: set[int] | None = None,
    missing_metric_steps: set[int] | None = None,
) -> tuple[int, int, int, int, int, int, set[int], set[int]]:
    """Insert prompts, rollouts, samples_data, rollout metrics, golden answers,
    and step metrics from a RolloutZipData into the database.
    
    Args:
        con: Database connection
        run_path: The run ID
        data: RolloutZipData containing prompts, rollouts, samples_data, rollout metrics,
              golden answers, step metrics, and metadata
        source_name: Name for logging (e.g., "block_0", "last.zip")
        missing_steps: Optional set of steps to filter by for rollouts/metrics/etc.
        missing_metric_steps: Optional set of steps to filter by for step metrics.
    
    Returns:
        Tuple of (prompts_count, rollouts_count, samples_data_count, rollouts_metrics_count,
                 golden_answers_count, step_metrics_count, set of rollout steps inserted,
                 set of metric steps inserted)
    """
    prompts_count = 0
    rollouts_count = 0
    samples_data_count = 0
    rollouts_metrics_count = 0
    golden_answers_count = 0
    info_turns_count = 0
    step_metrics_count = 0
    inserted_steps: set[int] = set()
    inserted_metric_steps: set[int] = set()
    
    # If missing_steps is provided, filter data; otherwise use all data
    if missing_steps is not None:
        prompts = _filter_prompts_by_steps(data.prompts, missing_steps)
        rollouts = _filter_rollouts_by_steps(data.rollouts, missing_steps)
        samples_data_list = _filter_samples_data_by_steps(data.samples_data, missing_steps)
        rollouts_metrics = _filter_rollouts_metrics_by_steps(
            data.rollouts_metrics, missing_steps
        )
        golden_answers = _filter_golden_answers_by_steps(data.golden_answers, missing_steps)
        info_turns = _filter_info_turns_by_steps(data.info_turns, missing_steps)
        if missing_metric_steps is None:
            metrics = _filter_metrics_by_steps(data.metrics, missing_steps)
        else:
            metrics = _filter_metrics_by_steps(data.metrics, missing_metric_steps)
    else:
        prompts = data.prompts
        rollouts = data.rollouts
        samples_data_list = data.samples_data
        rollouts_metrics = data.rollouts_metrics
        golden_answers = data.golden_answers
        info_turns = data.info_turns
        metrics = data.metrics
    
    if prompts:
        insert_prompts(con, run_path, prompts)
        prompts_count = len(prompts)
        log.info(f"[ROLLOUTS] Synced {source_name}: {prompts_count} prompts")
    
    if rollouts:
        insert_rollouts(con, run_path, rollouts)
        rollouts_count = len(rollouts)
        inserted_steps.update(_get_steps_from_rollouts(rollouts))
        log.info(f"[ROLLOUTS] Synced {source_name}: {rollouts_count} rollout turns")
    
    if samples_data_list:
        insert_samples_data(con, run_path, samples_data_list)
        samples_data_count = len(samples_data_list)
        log.info(f"[ROLLOUTS] Synced {source_name}: {samples_data_count} samples data")
    
    if rollouts_metrics:
        insert_rollouts_metrics(con, run_path, rollouts_metrics)
        rollouts_metrics_count = len(rollouts_metrics)
        # Also track steps from metrics (should be same as rollouts)
        inserted_steps.update(_get_steps_from_rollouts_metrics(rollouts_metrics))
        log.info(
            f"[ROLLOUTS] Synced {source_name}: {rollouts_metrics_count} metrics"
        )

    if golden_answers:
        insert_golden_answers(con, run_path, golden_answers)
        golden_answers_count = len(golden_answers)
        inserted_steps.update(_get_steps_from_golden_answers(golden_answers))
        log.info(
            f"[ROLLOUTS] Synced {source_name}: {golden_answers_count} golden answers"
        )

    if info_turns:
        insert_info_turns(con, run_path, info_turns)
        info_turns_count = len(info_turns)
        log.info(
            f"[ROLLOUTS] Synced {source_name}: {info_turns_count} info turns"
        )
    
    if metrics:
        insert_step_metrics(con, run_path, metrics)
        step_metrics_count = len(metrics)
        inserted_metric_steps.update(_get_steps_from_metrics(metrics))
        log.info(f"[ROLLOUTS] Synced {source_name}: {step_metrics_count} step metrics")
    
    return (
        prompts_count,
        rollouts_count,
        samples_data_count,
        rollouts_metrics_count,
        golden_answers_count,
        step_metrics_count,
        inserted_steps,
        inserted_metric_steps,
    )


async def sync_rollouts_blocks(run_path: str, api_key: str, run: Any, summary: dict):
    """Background sync of rollouts - downloads zip files containing rollouts.parquet,
    rollouts_metrics.parquet, and golden_answers.parquet.
    
    Simple approach (matching events pattern):
    1. Download tail.zip (last 5 steps)
    2. Check its metadata (min_step, max_step)
    3. Check ingested_steps table for what we're missing in that range
    4. Insert missing steps from tail.zip
    5. If we need older steps (before tail.min_step), download from block_live.zip and finalized blocks
    
    File structure:
    - rollouts/block_*.zip: Finalized blocks (every 500 steps) with metadata {block_idx, start_step, end_step}
    - rollouts/block_live.zip: Current block being built with metadata {block_idx, min_step, max_step}
    - rollouts/tail.zip: Last 5 training steps with metadata {min_step, max_step}
    
    Each zip contains:
    - metadata.json: Block metadata
    - rollouts.parquet: Main rollout data (step, sample_idx, env, prompt, completion, reward, advantage, tokens_*)
    - rollouts_metrics.parquet: Normalized rollout metrics table (step, sample_idx, env, metric_name, value)
    - golden_answers.parquet: Golden answers table (step, sample_idx, env, key, value)
    
    All DB inserts are batched in a single transaction to avoid row-group fragmentation.
    """
    log.info(f"[ROLLOUTS] Starting rollout sync for: {run_path}")
    gen_start = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
        total_rollouts = 0
        total_samples_data = 0
        total_rollout_metrics = 0
        total_golden_answers = 0
        
        # Collect all steps to insert at the end in one transaction
        all_inserted_steps: set[int] = set()
        
        # Step 1: Download and process tail.zip first
        log.info(f"[ROLLOUTS] Fetching tail.zip...")
        file_path, tail_data = await loop.run_in_executor(
            None,
            lambda: _download_rollout_zip_sync(run, "steps/tail.zip")
        )
        
        tail_min = tail_data.min_step
        tail_max = tail_data.max_step
        log.info(f"[ROLLOUTS] Tail metadata: min_step={tail_min}, max_step={tail_max}")
        
        if tail_max is None:
            log.info(f"[ROLLOUTS] No tail metadata available, skipping rollout sync")
            return 0
        
        # Step 2: Check which steps from tail.zip are missing
        con = connect()
        ingested_in_tail_range = get_ingested_steps(con, run_path, tail_min, tail_max)
        ingested_metrics_in_tail_range = get_ingested_step_metrics(con, run_path, tail_min, tail_max)
        
        tail_range = set(range(tail_min, tail_max + 1))
        missing_in_tail = tail_range - ingested_in_tail_range
        missing_metrics_in_tail = tail_range - ingested_metrics_in_tail_range
        
        log.info(
            f"[ROLLOUTS] Tail range {tail_min}-{tail_max}: "
            f"{len(missing_in_tail)} missing steps, {len(missing_metrics_in_tail)} missing metric steps"
        )
        
        # Also check older steps upfront
        ingested_before_tail = set()
        ingested_metrics_before_tail = set()
        if tail_min > 0:
            ingested_before_tail = get_ingested_steps(con, run_path, 0, tail_min - 1)
            ingested_metrics_before_tail = get_ingested_step_metrics(con, run_path, 0, tail_min - 1)
        con.close()
        
        # Prepare list of data sources to process: (name, data, missing_steps, missing_metric_steps)
        data_sources: list[tuple[str, Any, set[int], set[int]]] = []
        
        # Step 3: Queue tail.zip for processing
        if (missing_in_tail or missing_metrics_in_tail) and tail_data.has_any_data():
            data_sources.append(("tail.zip", tail_data, missing_in_tail, missing_metrics_in_tail))
        
        # Step 4: Check if we need older steps (before tail.min_step)
        if tail_min > 0:
            older_steps_range = set(range(0, tail_min))
            missing_older_steps = older_steps_range - ingested_before_tail
            missing_older_metric_steps = older_steps_range - ingested_metrics_before_tail
            
            if missing_older_steps or missing_older_metric_steps:
                total_missing = len(missing_older_steps | missing_older_metric_steps)
                log.info(f"[ROLLOUTS] Need {total_missing} older steps (range 0-{tail_min - 1})")
                
                # Get remote state from summary
                num_finalized_blocks = summary.get("steps/num_finalized_blocks", 0)
                current_block_idx = summary.get("steps/current_block_idx", 0)
                
                # Calculate which blocks we need
                missing_any_older_steps = missing_older_steps | missing_older_metric_steps
                blocks_needed = _calculate_blocks_for_steps(missing_any_older_steps)
                
                # Separate into finalized blocks vs live block
                finalized_blocks = sorted([b for b in blocks_needed if b < num_finalized_blocks])
                need_live_block = current_block_idx in blocks_needed and current_block_idx >= num_finalized_blocks
                
                log.info(f"[ROLLOUTS] Need {len(finalized_blocks)} finalized blocks, need_live_block={need_live_block}")
                
                # Download finalized blocks and queue them
                if finalized_blocks:
                    for i in range(0, len(finalized_blocks), PARALLEL_DOWNLOADS):
                        batch = finalized_blocks[i:i + PARALLEL_DOWNLOADS]
                        log.info(f"[ROLLOUTS] Downloading blocks: {batch}")
                        
                        results = await loop.run_in_executor(
                            None,
                            lambda b=batch: _download_rollout_zips_parallel(run, b)
                        )
                        
                        for idx in sorted(results.keys()):
                            data = results[idx]
                            block_start = data.start_step
                            block_end = data.end_step
                            
                            log.info(f"[ROLLOUTS] block_{idx} metadata: start_step={block_start}, end_step={block_end}")
                            
                            if block_start is not None and block_end is not None:
                                block_missing = {s for s in missing_older_steps if block_start <= s <= block_end}
                                block_missing_metrics = {
                                    s for s in missing_older_metric_steps if block_start <= s <= block_end
                                }
                                if (block_missing or block_missing_metrics) and data.has_any_data():
                                    data_sources.append(
                                        (f"block_{idx}", data, block_missing, block_missing_metrics)
                                    )
                
                # Download live block if needed
                if need_live_block:
                    log.info(f"[ROLLOUTS] Fetching block_live.zip...")
                    file_path, live_data = await loop.run_in_executor(
                        None,
                        lambda: _download_rollout_zip_sync(run, "steps/block_live.zip")
                    )
                    
                    if live_data.has_any_data() and live_data.metadata:
                        live_min = live_data.min_step
                        live_max = live_data.max_step
                        
                        if live_min is not None and live_max is not None:
                            live_missing = {s for s in missing_older_steps if live_min <= s <= live_max}
                            live_missing_metrics = {
                                s for s in missing_older_metric_steps if live_min <= s <= live_max
                            }
                            if live_missing or live_missing_metrics:
                                data_sources.append(
                                    ("block_live.zip", live_data, live_missing, live_missing_metrics)
                                )
        
        # Step 5: Process all data sources in a SINGLE transaction
        total_prompts = 0
        total_step_metrics = 0
        if data_sources:
            log.info(f"[ROLLOUTS] Processing {len(data_sources)} data sources in single transaction...")
            con = connect()
            try:
                with transaction(con):
                    all_inserted_metric_steps: set[int] = set()
                    for source_name, data, missing_steps, missing_metric_steps in data_sources:
                        (
                            prompt_count,
                            gen_count,
                            samples_count,
                            rollout_metrics_count,
                            golden_answers_count,
                            step_metrics_count,
                            inserted,
                            inserted_metric_steps,
                        ) = _insert_rollout_zip_data(
                            con,
                            run_path,
                            data,
                            source_name,
                            missing_steps=missing_steps,
                            missing_metric_steps=missing_metric_steps,
                        )
                        total_prompts += prompt_count
                        total_rollouts += gen_count
                        total_samples_data += samples_count
                        total_rollout_metrics += rollout_metrics_count
                        total_golden_answers += golden_answers_count
                        total_step_metrics += step_metrics_count
                        all_inserted_steps.update(inserted)
                        all_inserted_metric_steps.update(inserted_metric_steps)
                        log.info(f"[ROLLOUTS] Processed {source_name}: {len(inserted)} steps")
                    
                    # Record all ingested steps at once
                    if all_inserted_steps:
                        insert_ingested_steps(con, run_path, all_inserted_steps)
                        log.info(f"[ROLLOUTS] Recorded {len(all_inserted_steps)} total ingested steps")
                    if all_inserted_metric_steps:
                        insert_ingested_step_metrics(con, run_path, all_inserted_metric_steps)
                        log.info(
                            f"[ROLLOUTS] Recorded {len(all_inserted_metric_steps)} total ingested metric steps"
                        )
                    
                    # Update the last rollout step
                    if all_inserted_steps:
                        max_step = max(all_inserted_steps)
                        update_ingest_state(con, run_path, last_rollout_step=max_step)
            finally:
                con.close()
        
        elapsed = time.time() - gen_start
        log.info(
            f"[ROLLOUTS] Completed rollout sync for {run_path} in {elapsed:.2f}s - "
            f"{total_prompts} prompts, {total_rollouts} rollout turns, "
            f"{total_samples_data} samples_data, {total_rollout_metrics} metrics, "
            f"{total_golden_answers} golden answers, {total_step_metrics} step metrics"
        )
        
        return total_rollouts
        
    except Exception as e:
        log.error(f"[ROLLOUTS] Error syncing rollouts for {run_path}: {repr(e)}")
        raise




async def sync_rollouts_background(run_path: str, api_key: str):
    """Background sync of rollouts and events - downloads in parallel batches without blocking DB."""
    log.info(f"[SYNC] Starting background sync for: {run_path}")
    sync_start = time.time()
    
    # Update sync status
    _sync_status[run_path] = {
        "status": "syncing",
        "started_at": sync_start,
        "rollouts_fetched": 0,
        "events_fetched": 0,
        "gpu_metrics_fetched": 0,
        "cpu_metrics_fetched": 0,
        "vllm_metrics_fetched": 0,
    }
    
    lock = _get_run_ingest_lock(run_path)
    try:
        # Ensure sync doesn't overlap with tracking ingestion for the same run.
        async with lock:
            # Initialize wandb API (not blocking DB)
            log.debug(f"[WANDB] Connecting to wandb API...")
            api = _get_wandb_api(api_key)
            run = _get_fresh_run(api, run_path)
            log.debug(f"[WANDB] Connected to run: {run.name}")

            # Get run summary (not blocking DB)
            log.debug(f"[WANDB] Fetching run summary...")
            summary = _to_serializable(run.summary) if run.summary else {}
            summary_json = json.dumps(summary)
            summary_progress = _extract_summary_progress(summary)

            # Get run config
            log.debug(f"[WANDB] Fetching run config...")
            config = _to_serializable(run.config) if run.config else {}
            config_json = json.dumps(config)
            log.debug(f"[WANDB] Config: {config_json}")
            
            # Save run metadata with config
            save_run_metadata(run, run_path, config_json=config_json)

            # Download immutable code artifacts once during explicit sync.
            _sync_source_artifacts(run, run_path)

            # Save summary and config immediately so UI has access to metadata
            con = connect()
            update_ingest_state(
                con,
                run_path,
                last_summary_json=summary_json,
                last_config_json=config_json,
                last_block_idx=summary_progress["events_current_block_idx"],
                last_rollout_block_idx=summary_progress["steps_current_block_idx"],
            )
            con.close()
            
            # === SYNC ROLLOUTS (block-based) ===
            total_rollouts = 0
            try:
                total_rollouts = await sync_rollouts_blocks(run_path, api_key, run, summary)
                _sync_status[run_path]["rollouts_fetched"] = total_rollouts
            except Exception as e:
                log.error(f"[SYNC] Error syncing rollouts: {repr(e)}")
                # Continue - don't fail the whole sync if rollouts fail
            
            # === SYNC EVENTS (unified - includes all events and metrics) ===
            event_totals = {"orchestrator": 0, "trainer": 0, "inference": 0, "gpu": 0, "cpu": 0, "vllm": 0}
            try:
                event_totals = await sync_events_background(run_path, api_key, run, summary)
                total_events = event_totals["orchestrator"] + event_totals["trainer"] + event_totals["inference"]
                _sync_status[run_path]["events_fetched"] = total_events
                _sync_status[run_path]["gpu_metrics_fetched"] = event_totals["gpu"]
                _sync_status[run_path]["cpu_metrics_fetched"] = event_totals["cpu"]
                _sync_status[run_path]["vllm_metrics_fetched"] = event_totals["vllm"]
            except Exception as e:
                log.error(f"[SYNC] Error syncing events: {repr(e)}")
                # Continue - don't fail the whole sync if events fail
            
            # Save summary_id after successful sync so we can skip unchanged summaries
            fresh_summary_id = summary.get("summary_id")
            if fresh_summary_id is not None:
                con = connect()
                update_ingest_state(con, run_path, summary_id=fresh_summary_id)
                con.close()
            
            elapsed = time.time() - sync_start
            total_events = event_totals["orchestrator"] + event_totals["trainer"] + event_totals["inference"]
            total_metrics = event_totals["gpu"] + event_totals["cpu"] + event_totals["vllm"]
            log.info(f"[SYNC] Completed sync for {run_path} in {elapsed:.2f}s - {total_rollouts} rollouts, {total_events} events, {total_metrics} metrics")
            
            _sync_status[run_path] = {
                "status": "complete",
                "started_at": sync_start,
                "completed_at": time.time(),
                "rollouts_fetched": total_rollouts,
                "events_fetched": total_events,
                "gpu_metrics_fetched": event_totals["gpu"],
                "cpu_metrics_fetched": event_totals["cpu"],
                "vllm_metrics_fetched": event_totals["vllm"],
            }
        
    except Exception as e:
        log.error(f"[SYNC] Error for {run_path}: {repr(e)}")
        _sync_status[run_path] = {
            "status": "error",
            "error": str(e),
            "started_at": sync_start,
        }


async def start_sync(run_path: str, api_key: str) -> dict:
    """Start background sync and return immediately."""
    # Check if already syncing
    if is_syncing(run_path):
        return {"status": "already_syncing"}
    
    # Start background task
    asyncio.create_task(sync_rollouts_background(run_path, api_key))
    
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Evals After Training sync
# ---------------------------------------------------------------------------


def _download_evals_after_training_zip_sync(
    run: Any, file_path: str
) -> tuple[str, dict[str, list[dict] | None]]:
    """Download an evals_after_training zip and extract eval parquet files.

    Each zip may contain:
    - rollouts_eval.parquet
    - samples_data_eval.parquet
    - golden_answers_eval.parquet
    - prompts_eval.parquet
    - rollouts_metrics_eval.parquet
    - info_turns_eval.parquet

    Returns (file_path, dict of table_name -> list of row dicts).
    """
    result: dict[str, list[dict] | None] = {
        "prompts_eval": None,
        "rollouts_eval": None,
        "samples_data_eval": None,
        "rollouts_metrics_eval": None,
        "golden_answers_eval": None,
        "info_turns_eval": None,
    }
    try:
        start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = run.file(file_path).download(root=tmpdir, replace=True)
            if local_file:
                with zipfile.ZipFile(local_file.name, "r") as zf:
                    zf.extractall(tmpdir)

                for table_name in result:
                    parquet_path = f"{tmpdir}/{table_name}.parquet"
                    try:
                        table = pq.read_table(parquet_path)
                        result[table_name] = table.to_pylist()
                        log.info(
                            f"[EVALS-AT] Extracted {table_name} from {file_path}: "
                            f"{len(result[table_name])} rows"
                        )
                    except Exception:
                        pass  # parquet not present in this zip

                elapsed = time.time() - start
                log.info(f"[EVALS-AT] Downloaded {file_path} in {elapsed:.2f}s")
    except Exception as e:
        log.error(f"[EVALS-AT] Error downloading {file_path}: {e}")
    return (file_path, result)


async def sync_evals_after_training_background(run_path: str, api_key: str):
    """Background sync of evals_after_training zip files from W&B.

    Lists all files under evals_after_training/ prefix, skips already-ingested ones,
    downloads new zips in parallel, extracts parquet files, and inserts into eval tables.
    """
    log.info(f"[EVALS-AT] Starting evals-after-training sync for: {run_path}")
    sync_start = time.time()

    _evals_after_training_status[run_path] = {
        "status": "syncing",
        "started_at": sync_start,
        "files_found": 0,
        "files_new": 0,
        "files_ingested": 0,
        "total_rows": 0,
    }

    lock = _get_run_ingest_lock(run_path)
    try:
        async with lock:
            api = _get_wandb_api(api_key)
            run = _get_fresh_run(api, run_path)

            # List all files in evals_after_training/ folder
            all_files = _list_files_with_prefix(run, "evals_after_training/")
            zip_files = [f for f in all_files if f.endswith(".zip")]
            _evals_after_training_status[run_path]["files_found"] = len(zip_files)
            log.info(f"[EVALS-AT] Found {len(zip_files)} zip files for {run_path}")

            if not zip_files:
                _evals_after_training_status[run_path] = {
                    "status": "complete",
                    "started_at": sync_start,
                    "completed_at": time.time(),
                    "files_found": 0,
                    "files_new": 0,
                    "files_ingested": 0,
                    "total_rows": 0,
                    "message": "No evals_after_training files found",
                }
                return

            # Check which files have already been ingested
            con = connect()
            already_ingested = get_ingested_evals_after_training(con, run_path)
            con.close()

            new_files = [f for f in zip_files if f not in already_ingested]
            _evals_after_training_status[run_path]["files_new"] = len(new_files)
            log.info(
                f"[EVALS-AT] {len(new_files)} new files to ingest "
                f"({len(already_ingested)} already ingested) for {run_path}"
            )

            if not new_files:
                _evals_after_training_status[run_path] = {
                    "status": "complete",
                    "started_at": sync_start,
                    "completed_at": time.time(),
                    "files_found": len(zip_files),
                    "files_new": 0,
                    "files_ingested": 0,
                    "total_rows": 0,
                    "message": "All files already ingested",
                }
                return

            # Download all new zips in parallel
            total_rows = 0
            files_ingested = 0

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=PARALLEL_DOWNLOADS
            ) as executor:
                futures = {
                    loop.run_in_executor(
                        executor,
                        _download_evals_after_training_zip_sync,
                        run,
                        file_path,
                    ): file_path
                    for file_path in new_files
                }

                for coro in asyncio.as_completed(futures):
                    file_path_result, data = await coro
                    file_path = file_path_result

                    # Check if any data was extracted
                    has_data = any(
                        v is not None and len(v) > 0 for v in data.values()
                    )
                    if not has_data:
                        log.warning(
                            f"[EVALS-AT] No eval data in {file_path}, marking as ingested"
                        )
                        con = connect()
                        with transaction(con):
                            insert_ingested_evals_after_training(
                                con, run_path, {file_path}
                            )
                        con.close()
                        continue

                    # Insert data into eval tables
                    con = connect()
                    try:
                        with transaction(con):
                            zip_rows = 0
                            if data.get("prompts_eval"):
                                insert_prompts_eval(con, run_path, data["prompts_eval"])
                                zip_rows += len(data["prompts_eval"])
                            if data.get("rollouts_eval"):
                                insert_rollouts_eval(con, run_path, data["rollouts_eval"])
                                zip_rows += len(data["rollouts_eval"])
                            if data.get("samples_data_eval"):
                                insert_samples_data_eval(
                                    con, run_path, data["samples_data_eval"]
                                )
                                zip_rows += len(data["samples_data_eval"])
                            if data.get("rollouts_metrics_eval"):
                                insert_rollouts_metrics_eval(
                                    con, run_path, data["rollouts_metrics_eval"]
                                )
                                zip_rows += len(data["rollouts_metrics_eval"])
                            if data.get("golden_answers_eval"):
                                insert_golden_answers_eval(
                                    con, run_path, data["golden_answers_eval"]
                                )
                                zip_rows += len(data["golden_answers_eval"])
                            if data.get("info_turns_eval"):
                                insert_info_turns_eval(
                                    con, run_path, data["info_turns_eval"]
                                )
                                zip_rows += len(data["info_turns_eval"])

                            # Mark this file as ingested (inside same transaction)
                            insert_ingested_evals_after_training(
                                con, run_path, {file_path}
                            )

                        total_rows += zip_rows
                        files_ingested += 1
                        _evals_after_training_status[run_path][
                            "files_ingested"
                        ] = files_ingested
                        _evals_after_training_status[run_path][
                            "total_rows"
                        ] = total_rows
                        log.info(
                            f"[EVALS-AT] Ingested {file_path}: {zip_rows} rows "
                            f"({files_ingested}/{len(new_files)})"
                        )
                    except Exception as e:
                        log.error(
                            f"[EVALS-AT] Error inserting data from {file_path}: {e}"
                        )
                    finally:
                        con.close()

            elapsed = time.time() - sync_start
            log.info(
                f"[EVALS-AT] Completed for {run_path} in {elapsed:.2f}s - "
                f"{files_ingested} files, {total_rows} total rows"
            )

            _evals_after_training_status[run_path] = {
                "status": "complete",
                "started_at": sync_start,
                "completed_at": time.time(),
                "files_found": len(zip_files),
                "files_new": len(new_files),
                "files_ingested": files_ingested,
                "total_rows": total_rows,
            }

    except Exception as e:
        log.error(f"[EVALS-AT] Error for {run_path}: {repr(e)}")
        _evals_after_training_status[run_path] = {
            "status": "error",
            "error": str(e),
            "started_at": sync_start,
        }


async def start_evals_after_training_sync(run_path: str, api_key: str) -> dict:
    """Start background sync of evals_after_training and return immediately."""
    if is_syncing_evals_after_training(run_path):
        return {"status": "already_syncing"}

    asyncio.create_task(sync_evals_after_training_background(run_path, api_key))
    return {"status": "started"}


async def ingest_rollouts(run_path: str, api_key: str):
    """Ingest rollouts and events for a run (used by tracking loop)."""
    # Safety guard: removed runs must never be tracked/ingested.
    con = connect()
    row = con.execute(
        "SELECT COALESCE(removed, FALSE) FROM runs WHERE run_id = ?",
        [run_path],
    ).fetchone()
    con.close()
    if row and bool(row[0]):
        log.info(f"[INGEST] Skipping ingestion for removed run: {run_path}")
        clear_active_run(run_path)
        return

    lock = _get_run_ingest_lock(run_path)
    if lock.locked():
        # Don't block the tracking loop (it processes runs sequentially).
        # If a sync is running, we'll retry on the next poll.
        log.info(f"[INGEST] Skipping ingestion for {run_path} (another ingest/sync is in progress)")
        return
    
    log.debug(f"[INGEST] Starting ingestion for: {run_path}")
    ingest_start = time.time()

    try:
        async with lock:
            # Initialize wandb API and get summary (not blocking DB)
            api = _get_wandb_api(api_key)
            run = _get_fresh_run(api, run_path)

            summary = _to_serializable(run.summary) if run.summary else {}
            summary_json = json.dumps(summary)
            summary_progress = _extract_summary_progress(summary)

            # Get run config
            config = _to_serializable(run.config) if run.config else {}
            config_json = json.dumps(config)
            log.debug(f"[INGEST] Config: {config_json}")
            
            # Save run metadata with config
            save_run_metadata(run, run_path, config_json=config_json)

            # Save summary and config immediately
            con = connect()
            update_ingest_state(
                con,
                run_path,
                last_summary_json=summary_json,
                last_config_json=config_json,
                last_block_idx=summary_progress["events_current_block_idx"],
                last_rollout_block_idx=summary_progress["steps_current_block_idx"],
            )
            con.close()
            
            # === SYNC ROLLOUTS (block-based) ===
            total_rollouts = 0
            try:
                total_rollouts = await sync_rollouts_blocks(run_path, api_key, run, summary)
            except Exception as e:
                log.error(f"[INGEST] Error syncing rollouts: {repr(e)}")
                # Continue - don't fail the whole ingest if rollouts fail
            
            # === SYNC EVENTS (unified - includes all events and metrics) ===
            event_totals = {"orchestrator": 0, "trainer": 0, "inference": 0, "gpu": 0, "cpu": 0, "vllm": 0}
            try:
                event_totals = await sync_events_background(run_path, api_key, run, summary)
            except Exception as e:
                log.error(f"[INGEST] Error syncing events: {repr(e)}")
                # Continue - don't fail the whole ingest if events fail
            
            # Save summary_id after successful sync so we can skip unchanged summaries
            fresh_summary_id = summary.get("summary_id")
            if fresh_summary_id is not None:
                con = connect()
                update_ingest_state(con, run_path, summary_id=fresh_summary_id)
                con.close()
            
            elapsed = time.time() - ingest_start
            total_events = event_totals["orchestrator"] + event_totals["trainer"] + event_totals["inference"]
            total_metrics = event_totals["gpu"] + event_totals["cpu"] + event_totals["vllm"]
            log.info(f"[INGEST] Completed for {run_path} in {elapsed:.2f}s - {total_rollouts} rollouts, {total_events} events, {total_metrics} metrics")

            # Check if run is still running; if not, stop tracking it
            run_state = getattr(run, "state", None)
            if not _is_run_running(run_state):
                log.info(f"[TRACKING] Run {run_path} is no longer running (state={run_state}), removing from active tracking")
                clear_active_run(run_path)
        
    except Exception as e:
        log.error(f"[INGEST] Error for {run_path}: {repr(e)}")
        raise


async def ingestion_loop():
    """Main ingestion loop that polls all active runs."""
    log.info(f"[LOOP] Starting ingestion loop (polling every {POLL_SECONDS}s)")
    while True:
        if not _compaction_paused:
            try:
                active_runs = get_active_runs()
                if active_runs:
                    log.info(f"[LOOP] Processing {len(active_runs)} active run(s)")
                for run_path, config in list(active_runs.items()):
                    try:
                        await ingest_rollouts(run_path, config["api_key"])
                    except Exception as e:
                        log.error(f"[LOOP] Error in ingestion for {run_path}: {repr(e)}")
            except Exception as e:
                log.error(f"[LOOP] Loop error: {repr(e)}")
        
        await asyncio.sleep(POLL_SECONDS)


def fetch_and_store_run(api_key: str, run_path: str) -> dict:
    """Fetch a single run from W&B and store metadata/config/summary in the database."""
    api = _get_wandb_api(api_key)
    run = _get_fresh_run(api, run_path)

    summary = _to_serializable(run.summary) if run.summary else {}
    summary_json = json.dumps(summary)
    summary_progress = _extract_summary_progress(summary)

    config = _to_serializable(run.config) if run.config else {}
    config_json = json.dumps(config)

    run_data = build_run_metadata(run, run_path)
    run_data["config_json"] = config_json

    con = connect()
    upsert_run(con, run_data)
    update_ingest_state(
        con,
        run_path,
        last_summary_json=summary_json,
        last_config_json=config_json,
        last_block_idx=summary_progress["events_current_block_idx"],
        last_rollout_block_idx=summary_progress["steps_current_block_idx"],
    )
    con.close()

    return {"run_path": run_path, "name": run_data.get("name")}


def _is_run_running(state: str | None) -> bool:
    return str(state).lower() == "running"


def _discover_tagged_runs_for_project(
    api: Any, project_path: str, tag: str
) -> tuple[str, list[dict[str, Any]]]:
    """Discover tagged runs in a single project.

    Accepts an already-initialised wandb.Api to avoid the overhead of
    creating (and login-verifying) a new one per project.
    """
    t0 = time.time()
    discovered: list[dict[str, Any]] = []
    try:
        runs = api.runs(
            project_path,
            filters={"tags": {"$in": [tag]}},
            order="-created_at",
            per_page=100,
            include_sweeps=False,
        )
    except Exception as e:
        log.warning(f"[WANDB] Failed to list runs for {project_path}: {repr(e)} ({time.time() - t0:.2f}s)")
        return project_path, discovered

    t_iter = time.time()
    for run in runs:
        tags = list(getattr(run, "tags", []) or [])
        if tag not in tags:
            continue
        if "telescope-ignore" in tags:
            continue
        if _schema_version_mismatch(tags):
            continue

        run_path = f"{run.entity}/{run.project}/{run.id}"
        run_data = build_run_metadata(run, run_path)
        discovered.append(
            {
                "run_path": run_path,
                "run_data": run_data,
                "created_at": run_data.get("created_at") or "",
                "is_running": _is_run_running(getattr(run, "state", None)),
            }
        )
        # Update progress immediately as runs are discovered (not only when
        # the whole project scan future completes).
        with _discovery_status_lock:
            if _discovery_status.get("status") == "discovering":
                _discovery_status["runs_found"] = int(_discovery_status.get("runs_found", 0)) + 1

    elapsed = time.time() - t0
    log.debug(
        f"[WANDB] project {project_path}: "
        f"{len(discovered)} run(s), "
        f"api.runs={t_iter - t0:.2f}s iter={time.time() - t_iter:.2f}s total={elapsed:.2f}s"
    )
    return project_path, discovered


def fetch_tagged_runs(api_key: str, tag: str) -> list[str]:
    """Fetch tagged runs across all projects and keep only latest 20 as active/visible."""
    global _discovery_status
    t0 = time.time()
    api = _get_wandb_api(api_key)
    entity = api.default_entity
    projects = list(api.projects(entity=entity))
    total_projects = len(projects)
    project_paths = [f"{entity}/{project.name}" for project in projects]
    log.info(f"[WANDB] Api init + project listing took {time.time() - t0:.2f}s")

    with _discovery_status_lock:
        _discovery_status = {
            "status": "discovering",
            "runs_found": 0,
            "projects_scanned": 0,
            "total_projects": total_projects,
        }

    max_workers = min(
        DISCOVERY_PROJECT_SCAN_WORKERS,
        max(1, total_projects),
    )
    log.info(
        f"[WANDB] Starting tagged run discovery for tag '{tag}' "
        f"across {total_projects} project(s) with {max_workers} worker(s)"
    )

    con = connect()
    try:
        # First pass: collect all tagged W&B runs across projects.
        # Store basic metadata incrementally so runs appear in the UI right away.
        all_wandb_runs: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[concurrent.futures.Future, str] = {
                executor.submit(_discover_tagged_runs_for_project, api, project_path, tag): project_path
                for project_path in project_paths
            }

            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                project_path = futures[future]
                try:
                    _, discovered_runs = future.result()
                except Exception as e:
                    log.warning(
                        f"[WANDB] Discovery worker failed for {project_path}: {repr(e)}"
                    )
                    with _discovery_status_lock:
                        _discovery_status["projects_scanned"] = completed
                    continue

                if discovered_runs:
                    _known_projects.add(project_path)

                for discovered in discovered_runs:
                    all_wandb_runs.append(discovered)
                    # Store basic metadata immediately so the run appears in the UI.
                    upsert_run(con, discovered["run_data"])

                with _discovery_status_lock:
                    _discovery_status["projects_scanned"] = completed

        log.info(f"[WANDB] Discovery complete: {len(all_wandb_runs)} tagged runs across {total_projects} projects (known projects: {len(_known_projects)})")

        # Determine top 20 latest runs by created_at (across all projects)
        all_wandb_runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        top_20_run_paths: set[str] = {item["run_path"] for item in all_wandb_runs[:20]}
        discovered_run_paths: list[str] = list(dict.fromkeys([item["run_path"] for item in all_wandb_runs]))

        # As soon as we know the top 20, mark all other discovered tagged runs as removed
        # in one bulk pass so they show up quickly in "Add Run".
        now_iso = datetime.utcnow().isoformat()
        if discovered_run_paths:
            placeholders = ", ".join(["?"] * len(discovered_run_paths))
            con.execute(
                f"""
                UPDATE runs
                SET removed = TRUE, removed_at = ?, updated_at = ?
                WHERE run_id IN ({placeholders})
                """,
                [now_iso, now_iso, *discovered_run_paths],
            )
        if top_20_run_paths:
            top_20_list = list(top_20_run_paths)
            placeholders = ", ".join(["?"] * len(top_20_list))
            con.execute(
                f"""
                UPDATE runs
                SET removed = FALSE, removed_at = NULL, updated_at = ?
                WHERE run_id IN ({placeholders})
                """,
                [now_iso, *top_20_list],
            )

        # Second pass:
        # - latest 20 are visible/syncable
        # - others were already bulk-marked as removed above
        # - keep active tracking only for running runs that are in latest 20
        sync_run_paths: list[str] = []
        for discovered in all_wandb_runs:
            run_path = discovered["run_path"]
            is_top_20 = run_path in top_20_run_paths
            is_running = bool(discovered.get("is_running", False))

            if is_top_20 and is_running:
                set_active_run(run_path, api_key)
            else:
                clear_active_run(run_path)

            if is_top_20:
                sync_run_paths.append(run_path)

        log.info(
            f"[WANDB] Initial bootstrap: stored {len(all_wandb_runs)} run(s), "
            f"marked {len(sync_run_paths)} as active (latest 20) and "
            f"{max(0, len(all_wandb_runs) - len(sync_run_paths))} as removed"
        )

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(sync_run_paths))
    finally:
        con.close()
        with _discovery_status_lock:
            _discovery_status = {"status": "idle"}


_POLL_RUNS_QUERY = """
query PollRuns($project: String!, $entity: String!, $cursor: String, $perPage: Int!, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    name
                    displayName
                    tags
                    state
                    createdAt
                    notes
                    historyLineCount
                    user { name username }
                }
                cursor
            }
            pageInfo { hasNextPage }
        }
    }
}
"""


def _build_run_data_from_node(node: dict, entity: str, project: str) -> dict:
    """Build a run_data dict directly from a GraphQL node (no Run object)."""
    run_id_short = node.get("name", "")
    run_path = f"{entity}/{project}/{run_id_short}"

    run_data: dict[str, Any] = {
        "run_id": run_path,
        "wandb_id": run_id_short,
        "name": node.get("displayName"),
        "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id_short}",
        "entity": entity,
        "project": project,
        "state": node.get("state"),
        "notes": node.get("notes"),
        "updated_at": datetime.utcnow().isoformat(),
    }

    created_at = node.get("createdAt")
    if created_at:
        run_data["created_at"] = str(created_at)

    user_data = node.get("user")
    if isinstance(user_data, dict):
        run_data["user_name"] = user_data.get("username") or user_data.get("name")

    tags = node.get("tags")
    if tags:
        tags_list = list(tags)
        run_data["tags"] = tags_list
        commit, schema_version, table_schema_versions = _extract_commit_and_schema_version(tags_list)
        if commit:
            run_data["trainer_commit"] = commit
        if schema_version:
            run_data["schema_version"] = schema_version
        if table_schema_versions:
            run_data["table_schema_versions"] = table_schema_versions

    run_data["last_history_step"] = node.get("historyLineCount")

    return run_data


def _poll_project_for_new_runs(
    api: Any,
    project_path: str,
    tag: str,
    existing_ids: set[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fast-poll a single project for new tagged runs.

    Executes a direct GraphQL query (per_page=500) to fetch all run IDs
    in ideally one request, completely bypassing the wandb ``Run`` object
    construction (which triggers an expensive per-run introspection query).

    Only runs absent from *existing_ids* get full metadata built.

    Returns ``(project_path, new_runs, state_updates)``.
    """
    from wandb_gql import gql

    t0 = time.time()
    new_runs: list[dict[str, Any]] = []
    state_updates: list[dict[str, Any]] = []

    parts = project_path.split("/", 1)
    if len(parts) != 2:
        log.warning(f"[WANDB] Invalid project_path: {project_path}")
        return project_path, new_runs, state_updates

    entity, project = parts
    filters_json = json.dumps({"tags": {"$in": [tag]}})
    query = gql(_POLL_RUNS_QUERY)
    cursor = None
    has_next = True

    try:
        while has_next:
            variables: dict[str, Any] = {
                "project": project,
                "entity": entity,
                "order": "-created_at",
                "perPage": 500,
                "filters": filters_json,
                "cursor": cursor,
            }
            response = api.client.execute(query, variable_values=variables)

            proj = response.get("project")
            if not proj:
                log.debug(f"[WANDB] Project not found: {project_path}")
                break

            edges = proj.get("runs", {}).get("edges", [])
            page_info = proj.get("runs", {}).get("pageInfo", {})
            has_next = page_info.get("hasNextPage", False)

            for edge in edges:
                node = edge.get("node", {})
                if not node:
                    continue
                cursor = edge.get("cursor")

                run_tags = node.get("tags") or []
                if tag not in run_tags:
                    continue
                if "telescope-ignore" in run_tags:
                    continue
                if _schema_version_mismatch(run_tags):
                    continue

                run_id_short = node.get("name", "")
                run_path = f"{entity}/{project}/{run_id_short}"
                is_running = _is_run_running(node.get("state"))

                if run_path in existing_ids:
                    if is_running:
                        run_data = _build_run_data_from_node(node, entity, project)
                        state_updates.append({"run_path": run_path, "run_data": run_data})
                    continue

                run_data = _build_run_data_from_node(node, entity, project)
                new_runs.append(
                    {
                        "run_path": run_path,
                        "run_data": run_data,
                        "created_at": run_data.get("created_at") or "",
                        "is_running": is_running,
                    }
                )
    except Exception as e:
        log.warning(f"[WANDB] Failed to poll runs for {project_path}: {repr(e)} ({time.time() - t0:.2f}s)")

    elapsed = time.time() - t0
    log.debug(
        f"[WANDB] poll project {project_path}: "
        f"{len(new_runs)} new, {len(state_updates)} state-update(s), "
        f"total={elapsed:.2f}s"
    )
    return project_path, new_runs, state_updates


def poll_known_projects_for_new_runs(api_key: str, tag: str) -> list[str]:
    """Poll only known projects for new tagged runs.

    Uses per_page=500 so that (in most cases) a single GraphQL request
    per project returns all tagged run IDs.  Only runs absent from the
    local DB are fully processed, making this much cheaper than the
    initial full discovery.

    This is the fast-path used by the poll loop.  It never scans all entity
    projects — discovery only happens for explicitly added projects.
    """
    if not _known_projects:
        return []

    # Skip if a discovery is already running (e.g. the initial fetch_tagged_runs)
    with _discovery_status_lock:
        if _discovery_status.get("status") == "discovering":
            log.debug("[WANDB] poll skipped — discovery already in progress")
            return []

    t0 = time.time()
    project_paths = list(_known_projects)
    log.debug(f"[WANDB] poll_known_projects START — {len(project_paths)} project(s): {project_paths}")

    api = _get_wandb_api(api_key)

    # Get what we already have locally
    t_db = time.time()
    con = connect()
    existing_rows = con.execute("SELECT run_id, removed FROM runs").fetchall()
    existing_ids: set[str] = {
        row[0] for row in existing_rows if row and row[0]
    }
    log.debug(f"[WANDB] DB lookup ({len(existing_ids)} existing runs) took {time.time() - t_db:.3f}s")

    # Query each known project for tagged runs (parallel if multiple)
    max_workers = min(DISCOVERY_PROJECT_SCAN_WORKERS, max(1, len(project_paths)))

    all_new: list[dict[str, Any]] = []
    all_state_updates: list[dict[str, Any]] = []

    t_scan = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future, str] = {
            executor.submit(_poll_project_for_new_runs, api, pp, tag, existing_ids): pp
            for pp in project_paths
        }
        for future in concurrent.futures.as_completed(futures):
            project_path = futures[future]
            try:
                _, new_runs, state_updates = future.result()
            except Exception as e:
                log.warning(f"[WANDB] Failed to scan {project_path}: {repr(e)}")
                continue
            all_new.extend(new_runs)
            all_state_updates.extend(state_updates)

    log.debug(f"[WANDB] scan took {time.time() - t_scan:.2f}s — {len(all_new)} new, {len(all_state_updates)} state-update(s)")

    # Insert new runs
    new_run_paths: list[str] = []
    for item in all_new:
        run_path = item["run_path"]
        is_running = item.get("is_running", False)
        upsert_run(con, item["run_data"])
        existing_ids.add(run_path)
        new_run_paths.append(run_path)
        log.debug(f"[WANDB] NEW run: {run_path} (running={is_running})")
        if is_running:
            set_active_run(run_path, api_key)

    # Update state for existing running runs
    for item in all_state_updates:
        upsert_run(con, item["run_data"])

    con.close()

    elapsed = time.time() - t0
    log.debug(
        f"[WANDB] poll_known_projects DONE in {elapsed:.2f}s — "
        f"{len(all_new)} new, {len(all_state_updates)} state-update(s), "
        f"{len(new_run_paths)} run(s) added"
    )
    return new_run_paths


# Number of on-demand sync workers currently running.
_active_sync_workers: int = 0
_next_worker_id: int = 0


def _ensure_sync_workers():
    """Spawn ephemeral sync workers up to SYNC_QUEUE_WORKERS if the queue has items."""
    global _active_sync_workers, _next_worker_id
    needed = min(_sync_queue.qsize(), SYNC_QUEUE_WORKERS) - _active_sync_workers
    for _ in range(needed):
        _next_worker_id += 1
        _active_sync_workers += 1
        asyncio.create_task(_sync_worker(_next_worker_id))


def enqueue_sync(run_paths: list[str], api_key: str, force_sync: bool = False) -> int:
    """Enqueue runs for background sync workers.

    Runs already in the queue or being processed by a worker are skipped
    to prevent duplicate sync rounds (e.g. reconcile loop re-enqueuing runs
    that the initial configure_wandb_and_sync already queued).

    Ephemeral workers are spawned on demand (up to SYNC_QUEUE_WORKERS) and
    exit automatically once the queue is drained.
    """
    enqueued = 0
    for run_path in run_paths:
        if run_path in _sync_queue_pending:
            log.debug(f"[SYNC] Skipping enqueue for {run_path} (already queued/processing)")
            continue
        _sync_queue_pending.add(run_path)
        _sync_queue.put_nowait((run_path, api_key, force_sync))
        enqueued += 1
    if enqueued:
        _ensure_sync_workers()
    return enqueued


async def _sync_worker(worker_id: int):
    """Ephemeral sync worker — processes items until the queue is empty, then exits."""
    global _active_sync_workers
    log.info(f"[SYNC] Worker #{worker_id} started")
    try:
        while True:
            if _compaction_paused:
                break
            try:
                run_path, api_key, force_sync = _sync_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                log.info(f"[SYNC] Worker #{worker_id} picked {run_path} (force_sync={force_sync})")
                if is_syncing(run_path):
                    log.info(f"[SYNC] Skipping {run_path} (already syncing)")
                    continue

                con = connect()
                run_row = con.execute(
                    "SELECT state, COALESCE(removed, FALSE) FROM runs WHERE run_id = ?",
                    [run_path],
                ).fetchone()
                if run_row and bool(run_row[1]):
                    con.close()
                    log.info(f"[SYNC] Skipping {run_path} (run is removed)")
                    clear_active_run(run_path)
                    continue

                if not force_sync:
                    # Startup/local-resume path: only sync if local ingest state says we are behind.
                    istate = get_ingest_state(con, run_path)
                    con.close()

                    run_state = run_row[0] if run_row else None
                    last_gen_step = istate.get("last_rollout_step", -1)
                    last_event_zip_idx = istate.get("last_event_zip_idx", -1)
                    last_event_block_idx = istate.get("last_block_idx", -1)
                    last_rollout_block_idx = istate.get("last_rollout_block_idx", -1)
                    summary_json = istate.get("last_summary_json")
                    stored_summary_id = istate.get("summary_id")
                    mismatch_reasons = get_sync_mismatch_reasons(
                        last_gen_step=last_gen_step,
                        last_event_zip_idx=last_event_zip_idx,
                        summary_json=summary_json,
                        last_event_block_idx=last_event_block_idx,
                        last_rollout_block_idx=last_rollout_block_idx,
                        stored_summary_id=stored_summary_id,
                    )

                    if not mismatch_reasons:
                        log.info(
                            f"[SYNC] Skipping {run_path} (ingest_state matches summary, "
                            f"state={run_state}, last_rollout_step={last_gen_step}, "
                            f"last_event_zip_idx={last_event_zip_idx}, "
                            f"last_block_idx={last_event_block_idx}, "
                            f"last_rollout_block_idx={last_rollout_block_idx})"
                        )
                        continue

                    log.info(
                        f"[SYNC] Syncing {run_path} due to ingest_state mismatch: "
                        f"{', '.join(mismatch_reasons)}"
                    )
                else:
                    con.close()

                await sync_rollouts_background(run_path, api_key)
            except Exception as e:
                log.error(f"[SYNC] Queue worker error for {run_path}: {repr(e)}")
            finally:
                _sync_queue_pending.discard(run_path)
                _sync_queue.task_done()
    finally:
        _active_sync_workers -= 1
        log.info(f"[SYNC] Worker #{worker_id} finished (queue empty)")


async def configure_wandb_and_sync(api_key: str, tag: str = "telescope"):
    """Store W&B key and poll known projects if any exist.

    No full entity discovery — runs are only discovered for explicitly added
    projects (via /add-project).
    """
    log.info(f"[WANDB] API key configured (tag='{tag}')")
    if _known_projects:
        log.info(f"[WANDB] {len(_known_projects)} known project(s) — polling for new runs")
        new_run_paths = await asyncio.to_thread(
            poll_known_projects_for_new_runs, api_key, tag
        )
        if new_run_paths:
            enqueue_sync(new_run_paths, api_key, force_sync=True)


async def discover_and_sync_project(api_key: str, project_path: str, tag: str = "telescope"):
    """Discover telescope-tagged runs in a single project and enqueue them for sync.

    Called when a user adds a project via /add-project.  Sets discovery status
    so the UI can show a brief "Fetching runs" indicator while the search is
    active.  Once discovery finishes, status goes back to idle and the UI
    never shows the card again (sync happens silently).

    Keeps only the latest 20 runs as active/visible — the rest are marked as
    removed but retained in the DB so they can be re-added via "Add Run".
    """
    global _discovery_status
    log.debug(f"[WANDB] Starting discovery for project '{project_path}' (tag='{tag}')")

    with _discovery_status_lock:
        _discovery_status = {
            "status": "discovering",
            "runs_found": 0,
            "projects_scanned": 0,
            "total_projects": 1,
        }

    try:
        api = await asyncio.to_thread(_get_wandb_api, api_key)
        _, discovered = await asyncio.to_thread(
            _discover_tagged_runs_for_project, api, project_path, tag
        )

        if not discovered:
            log.debug(f"[WANDB] No tagged runs found in {project_path}")
            return

        with _discovery_status_lock:
            _discovery_status["runs_found"] = len(discovered)
            _discovery_status["projects_scanned"] = 1

        # Sort by created_at descending — keep latest 20 visible, rest removed.
        discovered.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        top_run_paths: set[str] = {item["run_path"] for item in discovered[:20]}
        all_run_paths = list(dict.fromkeys(item["run_path"] for item in discovered))

        con = connect()
        try:
            for item in discovered:
                upsert_run(con, item["run_data"])

            now_iso = datetime.utcnow().isoformat()
            # Mark all discovered as removed first…
            if all_run_paths:
                placeholders = ", ".join(["?"] * len(all_run_paths))
                con.execute(
                    f"UPDATE runs SET removed = TRUE, removed_at = ?, updated_at = ? "
                    f"WHERE run_id IN ({placeholders})",
                    [now_iso, now_iso, *all_run_paths],
                )
            # …then un-remove the top 20.
            if top_run_paths:
                top_list = list(top_run_paths)
                placeholders = ", ".join(["?"] * len(top_list))
                con.execute(
                    f"UPDATE runs SET removed = FALSE, removed_at = NULL, updated_at = ? "
                    f"WHERE run_id IN ({placeholders})",
                    [now_iso, *top_list],
                )
        finally:
            con.close()

        # Only sync the top 20; track running ones.
        sync_run_paths: list[str] = []
        for item in discovered:
            run_path = item["run_path"]
            is_top = run_path in top_run_paths
            is_running = bool(item.get("is_running", False))
            if is_top and is_running:
                set_active_run(run_path, api_key)
            if is_top:
                sync_run_paths.append(run_path)

        sync_run_paths = list(dict.fromkeys(sync_run_paths))
        log.debug(
            f"[WANDB] Project discovery done: {len(discovered)} found, "
            f"{len(sync_run_paths)} active (latest 20), "
            f"{max(0, len(discovered) - len(sync_run_paths))} removed"
        )
        if sync_run_paths:
            enqueue_sync(sync_run_paths, api_key, force_sync=True)
    except Exception as e:
        log.debug(f"[WANDB] Discovery failed for {project_path}: {repr(e)}")
    finally:
        with _discovery_status_lock:
            _discovery_status = {"status": "idle"}


async def tagged_runs_poll_loop():
    """Poll known projects for new tagged runs.

    Only scans projects explicitly added by the user (or derived from
    existing runs in the DB). Never does a full entity-wide discovery.
    """
    log.info(f"[WANDB] Starting tagged runs poll loop (every {TAGGED_RUNS_POLL_SECONDS}s)")
    poll_count = 0
    while True:
        if _compaction_paused:
            await asyncio.sleep(TAGGED_RUNS_POLL_SECONDS)
            continue
        try:
            api_key = get_wandb_api_key()

            if api_key and _known_projects:
                new_run_paths = await asyncio.to_thread(
                    poll_known_projects_for_new_runs, api_key, TAGGED_RUNS_TAG
                )
                if new_run_paths:
                    log.info(f"[WANDB] Poll found {len(new_run_paths)} new tagged run(s) — syncing")
                    enqueue_sync(new_run_paths, api_key, force_sync=True)
                poll_count += 1
        except Exception as e:
            log.error(f"[WANDB] Tagged runs poll error: {repr(e)}")

        await asyncio.sleep(TAGGED_RUNS_POLL_SECONDS)


async def ingest_state_reconcile_loop():
    """Every minute, ensure running runs are tracked and enqueue mismatched runs."""
    log.info(
        f"[SYNC] Starting ingest state reconcile loop (every {INGEST_STATE_RECONCILE_SECONDS}s)"
    )
    while True:
        if _compaction_paused:
            await asyncio.sleep(INGEST_STATE_RECONCILE_SECONDS)
            continue
        try:
            con = connect()
            api_key = get_wandb_api_key()

            if not api_key:
                con.close()
                await asyncio.sleep(INGEST_STATE_RECONCILE_SECONDS)
                continue

            running_rows = con.execute(
                """
                SELECT run_id
                FROM runs
                WHERE COALESCE(removed, FALSE) = FALSE
                  AND LOWER(COALESCE(state, '')) = 'running'
                """
            ).fetchall()
            tracked_added = 0
            for (run_id,) in running_rows:
                if not is_tracking(run_id):
                    set_active_run(run_id, api_key)
                    tracked_added += 1

            rows = con.execute(
                """
                SELECT r.run_id,
                       COALESCE(i.last_rollout_step, -1) as last_gen_step,
                       COALESCE(i.last_event_zip_idx, -1) as last_event_zip_idx,
                       COALESCE(i.last_block_idx, -1) as last_event_block_idx,
                       COALESCE(i.last_rollout_block_idx, -1) as last_rollout_block_idx,
                       i.last_summary_json,
                       i.summary_id
                FROM runs r
                LEFT JOIN ingest_state i ON r.run_id = i.run_id
                WHERE COALESCE(r.removed, FALSE) = FALSE
                """
            ).fetchall()
            con.close()

            pending_paths: list[str] = []
            for (
                run_id,
                last_gen_step,
                last_event_zip_idx,
                last_event_block_idx,
                last_rollout_block_idx,
                summary_json,
                stored_summary_id,
            ) in rows:
                if run_needs_sync(
                    last_gen_step,
                    last_event_zip_idx,
                    summary_json,
                    last_event_block_idx=last_event_block_idx,
                    last_rollout_block_idx=last_rollout_block_idx,
                    stored_summary_id=stored_summary_id,
                ):
                    pending_paths.append(run_id)

            pending_paths = list(dict.fromkeys(pending_paths))
            enqueued = 0
            if pending_paths:
                enqueued = enqueue_sync(pending_paths, api_key, force_sync=False)

            if tracked_added or enqueued:
                log.info(
                    f"[SYNC] Reconcile pass: added {tracked_added} run(s) to tracking, "
                    f"enqueued {enqueued} run(s) for sync"
                )
        except Exception as e:
            log.error(f"[SYNC] Ingest state reconcile loop error: {repr(e)}")

        await asyncio.sleep(INGEST_STATE_RECONCILE_SECONDS)
