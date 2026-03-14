from __future__ import annotations
import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
import duckdb
import pandas as pd
import zstandard as zstd
from pathlib import Path

# Setup logging
log = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path.home() / ".telescope"


def _get_data_dir() -> Path:
    return Path(os.environ.get("TELESCOPE_DATA_DIR", str(_DEFAULT_DATA_DIR)))


def get_db_path() -> Path:
    return _get_data_dir() / "telescope.duckdb"


DB_PATH = get_db_path()

def decompress_blob(value: bytes | str | None) -> str | None:
    """Decompress a zstd-compressed BLOB back to a UTF-8 string."""
    if value is None:
        return None
    if isinstance(value, (bytes, memoryview)):
        try:
            # Create a new decompressor per call for thread safety.
            # Use stream_reader to handle frames without content size in header
            # (avoids "Destination buffer is too small" errors).
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(value) as reader:
                decompressed = reader.read()
            return decompressed.decode("utf-8")
        except (zstd.ZstdError, UnicodeDecodeError) as e:
            log.warning("Failed to decompress blob (%d bytes): %s", len(value), e)
            return None
    return value


# Run colors - 16 colors that are assigned to runs when created
RUN_COLORS = [
    "#865ed6",
    "#f07fdd",
    "#e67439",
    "#87cec0",
    "#dc4cdc",
    "#ffb83e",
    "#5ac5db",
    "#239487",
    "#fab796",
    "#9bc750",
    "#ad6f51",
    "#c2337a",
    "#a2a9ad",
    "#538ae6",
    "#f1444f",
    "#479a60",
]


@contextmanager
def transaction(con: duckdb.DuckDBPyConnection):
    """Context manager for explicit transactions.
    
    Usage:
        con = connect()
        with transaction(con):
            insert_foo(con, ...)
            insert_bar(con, ...)
        # All inserts are committed together in one row group
        con.close()
    """
    con.execute("BEGIN TRANSACTION")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


_shared_con: duckdb.DuckDBPyConnection | None = None
_con_lock = threading.Lock()


def _get_shared_connection() -> duckdb.DuckDBPyConnection:
    """Return (and lazily create) the single shared DuckDB connection."""
    global _shared_con
    if _shared_con is not None:
        return _shared_con
    with _con_lock:
        if _shared_con is not None:
            return _shared_con
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"[DB] Opening shared connection to: {DB_PATH}")
        con = duckdb.connect(str(DB_PATH))
        con.execute("PRAGMA threads=4;")
        _init_schema(con)
        _shared_con = con
        return _shared_con


def connect() -> duckdb.DuckDBPyConnection:
    """Return a lightweight cursor backed by the shared connection."""
    return _get_shared_connection().cursor()


def _init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they don't exist. Called once at startup."""
    # Orchestrator events table - instant events with timestamp, event_type, step
    con.execute("""
        CREATE TABLE IF NOT EXISTS events_orchestrator (
            run_id TEXT,
            timestamp DOUBLE,
            event_type TEXT,
            step BIGINT,
            node_id BIGINT
        );
    """)
    
    # Trainer events table - duration events with event_type, step, rank, microbatch, start_time, end_time
    con.execute("""
        CREATE TABLE IF NOT EXISTS events_trainer (
            run_id TEXT,
            event_type TEXT,
            step BIGINT,
            rank BIGINT,
            local_rank BIGINT,
            node_id BIGINT,
            gpu_index BIGINT,
            microbatch BIGINT,
            minibatch BIGINT,
            start_time DOUBLE,
            end_time DOUBLE
        );
    """)

    # Inference events table - duration events with event_type, server, start_time, end_time, token counts, sample_id, group_id, vLLM metrics
    con.execute("""
        CREATE TABLE IF NOT EXISTS events_inference (
            run_id TEXT,
            event_type TEXT,
            server BIGINT,
            node_id BIGINT,
            tp_group_id BIGINT,
            tp_size BIGINT,
            start_time DOUBLE,
            end_time DOUBLE,
            prompt_tokens BIGINT,
            rollout_tokens BIGINT,
            sample_id INTEGER,
            group_id INTEGER,
            vllm_request_id TEXT,
            queue_time DOUBLE,
            time_to_first_token DOUBLE,
            prefill_time DOUBLE,
            decode_time DOUBLE,
            inference_time DOUBLE,
            e2e_latency DOUBLE,
            max_tokens INTEGER,
            is_eval BOOLEAN,
            step INTEGER,
            is_canceled BOOLEAN,
            off_policy_steps INTEGER,
            server_lane BIGINT
        );
    """)

    # Prompts table - stores prompt information per group (one row per group)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            run_id TEXT,
            step BIGINT,
            group_id BIGINT,
            env TEXT,
            system_prompt TEXT,
            tokens_system_prompt BIGINT,
            prompt TEXT,
            tokens_prompt BIGINT
        );
    """)
    
    # Rollouts table - stores rollout turns from each step (one row per turn per sample)
    # Supports multi-turn conversations where each sample can have multiple turns
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts (
            run_id TEXT,
            step BIGINT,
            group_id BIGINT,
            sample_idx BIGINT,
            turn_order BIGINT,
            turn_type TEXT,
            content BLOB,
            tokens BIGINT,
            stop_reason TEXT,
            environment_response_time DOUBLE
        );
    """)
    
    # Samples data table - stores sample-level data (one row per sample)
    # Contains reward, advantage, and aggregated sample info
    con.execute("""
        CREATE TABLE IF NOT EXISTS samples_data (
            run_id TEXT,
            step BIGINT,
            group_id BIGINT,
            sample_idx BIGINT,
            reward DOUBLE,
            advantage DOUBLE,
            turns BIGINT,
            total_tokens BIGINT,
            raw_string BLOB,
            compute_reward_time DOUBLE
        );
    """)

    # Rollouts metrics table - normalized metrics with dynamic names per environment
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts_metrics (
            run_id TEXT,
            step BIGINT,
            sample_idx BIGINT,
            env TEXT,
            metric_name TEXT,
            value DOUBLE
        );
    """)
    # Golden answers table - separate from metrics
    con.execute("""
        CREATE TABLE IF NOT EXISTS golden_answers (
            run_id TEXT,
            step BIGINT,
            sample_idx BIGINT,
            env TEXT,
            key TEXT,
            value TEXT
        );
    """)
    
    # Sample tags table - normalized tags with dynamic names per environment
    con.execute("""
        CREATE TABLE IF NOT EXISTS sample_tags (
            run_id TEXT,
            step BIGINT,
            sample_idx BIGINT,
            env TEXT,
            tag_name TEXT,
            tag_value TEXT
        );
    """)

    # System metrics GPU table - stores GPU metrics from trainer
    con.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics_gpu (
            run_id TEXT,
            timestamp DOUBLE,
            node_id BIGINT,
            rank BIGINT,
            local_rank BIGINT,
            gpu_index BIGINT,
            source TEXT,
            metric_name TEXT,
            value DOUBLE
        );
    """)
    
    # System metrics CPU table - stores CPU and memory metrics from trainer
    con.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics_cpu (
            run_id TEXT,
            timestamp DOUBLE,
            node_id BIGINT,
            source TEXT,
            metric_name TEXT,
            value DOUBLE
        );
    """)
    
    # vLLM metrics table - stores metrics from vLLM inference servers
    con.execute("""
        CREATE TABLE IF NOT EXISTS vllm_metrics (
            run_id TEXT,
            timestamp DOUBLE,
            server BIGINT,
            node_id BIGINT,
            tp_group_id BIGINT,
            tp_size BIGINT,
            metric_name TEXT,
            value DOUBLE
        );
    """)
    
    # Step metrics table - stores per-step training metrics (e.g., grad_norm, kl_divergence, entropy)
    con.execute("""
        CREATE TABLE IF NOT EXISTS step_metrics (
            run_id TEXT,
            step BIGINT,
            metric_name TEXT,
            value DOUBLE,
            section TEXT,
            "group" TEXT
        );
    """)

    # Discarded prompts table - stores prompts for discarded rollouts
    con.execute("""
        CREATE TABLE IF NOT EXISTS prompts_discarded (
            run_id TEXT,
            timestamp DOUBLE,
            discard_reason TEXT,
            trainer_step BIGINT,
            inference_step BIGINT,
            group_id BIGINT,
            env TEXT,
            system_prompt TEXT,
            tokens_system_prompt BIGINT,
            prompt TEXT,
            tokens_prompt BIGINT,
            tail_idx BIGINT
        );
    """)
    
    # Discarded rollouts table - stores rollouts that were discarded during training
    # (e.g., due to max_async or zero_advantage) - supports multi-turn
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts_discarded (
            run_id TEXT,
            trainer_step BIGINT,
            inference_step BIGINT,
            group_id BIGINT,
            sample_idx BIGINT,
            turn_order BIGINT,
            turn_type TEXT,
            content BLOB,
            tokens BIGINT,
            stop_reason TEXT,
            tail_idx BIGINT,
            environment_response_time DOUBLE
        );
    """)
    
    # Discarded samples data table - stores sample-level data for discarded samples
    con.execute("""
        CREATE TABLE IF NOT EXISTS samples_data_discarded (
            run_id TEXT,
            timestamp DOUBLE,
            discard_reason TEXT,
            trainer_step BIGINT,
            inference_step BIGINT,
            group_id BIGINT,
            sample_idx BIGINT,
            reward DOUBLE,
            advantage DOUBLE,
            turns BIGINT,
            total_tokens BIGINT,
            raw_string BLOB,
            tail_idx BIGINT,
            compute_reward_time DOUBLE
        );
    """)

    # Discarded rollouts metrics table - normalized metrics for discarded rollouts
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts_metrics_discarded (
            run_id TEXT,
            sample_idx BIGINT,
            env TEXT,
            metric_name TEXT,
            value DOUBLE,
            tail_idx BIGINT
        );
    """)
    # Discarded golden answers table - separate from metrics
    con.execute("""
        CREATE TABLE IF NOT EXISTS golden_answers_discarded (
            run_id TEXT,
            sample_idx BIGINT,
            env TEXT,
            key TEXT,
            value TEXT,
            tail_idx BIGINT
        );
    """)
    
    # Discarded sample tags table - normalized tags for discarded rollouts
    con.execute("""
        CREATE TABLE IF NOT EXISTS sample_tags_discarded (
            run_id TEXT,
            sample_idx BIGINT,
            env TEXT,
            tag_name TEXT,
            tag_value TEXT,
            tail_idx BIGINT
        );
    """)

    # Info turns table - stores per-turn info items (stderr, stdout, summary, etc.)
    # Multiple rows can exist for the same turn (one per info_key)
    con.execute("""
        CREATE TABLE IF NOT EXISTS info_turns (
            run_id TEXT,
            step BIGINT,
            sample_idx BIGINT,
            turn_order BIGINT,
            env TEXT,
            info_key TEXT,
            info_value TEXT,
            info_type TEXT
        );
    """)
    
    # Discarded info turns table - same structure but with tail_idx instead of step
    con.execute("""
        CREATE TABLE IF NOT EXISTS info_turns_discarded (
            run_id TEXT,
            sample_idx BIGINT,
            turn_order BIGINT,
            env TEXT,
            info_key TEXT,
            info_value TEXT,
            info_type TEXT,
            tail_idx BIGINT
        );
    """)
    
    # Eval prompts table - one row per unique eval prompt (inside events zips, keyed by tail_idx)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prompts_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            model_step BIGINT,
            sample_idx BIGINT,
            env TEXT,
            prompt TEXT,
            tokens_prompt BIGINT,
            system_prompt TEXT,
            tokens_system_prompt BIGINT,
            tail_idx BIGINT
        );
    """)
    
    # Eval rollouts table - one row per turn per eval completion (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            model_step BIGINT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            turn_order BIGINT,
            turn_type TEXT,
            content BLOB,
            tokens BIGINT,
            stop_reason TEXT,
            environment_response_time DOUBLE,
            tail_idx BIGINT
        );
    """)
    
    # Eval samples data table - one row per eval completion (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS samples_data_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            model_step BIGINT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            env TEXT,
            turns BIGINT,
            compute_eval_metrics_time DOUBLE,
            tail_idx BIGINT
        );
    """)
    
    # Eval rollouts metrics table - one row per metric per eval completion (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS rollouts_metrics_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            env TEXT,
            metric_name TEXT,
            value DOUBLE,
            tail_idx BIGINT
        );
    """)
    
    # Eval golden answers table - one row per golden answer key/value (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS golden_answers_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            env TEXT,
            key TEXT,
            value TEXT,
            tail_idx BIGINT
        );
    """)
    
    # Eval sample tags table - one row per tag per eval completion (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sample_tags_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            env TEXT,
            tag_name TEXT,
            tag_value TEXT,
            tail_idx BIGINT
        );
    """)

    # Eval info turns table - per-turn text info for eval completions (inside events zips)
    con.execute("""
        CREATE TABLE IF NOT EXISTS info_turns_eval (
            run_id TEXT,
            step BIGINT,
            eval_name TEXT,
            sample_idx BIGINT,
            completion_idx BIGINT,
            turn_order BIGINT,
            env TEXT,
            info_key TEXT,
            info_value TEXT,
            info_type TEXT,
            tail_idx BIGINT
        );
    """)
    
    # Ingest state - tracks what we've already fetched
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingest_state (
            run_id TEXT PRIMARY KEY,
            last_block_idx BIGINT DEFAULT -1,
            last_rollout_step BIGINT DEFAULT -1,
            last_summary_json TEXT,
            last_config_json TEXT,
            last_event_zip_idx BIGINT DEFAULT -1,
            last_rollout_block_idx BIGINT DEFAULT -1,
            summary_id TEXT
        );
    """)
    
    # Runs table - stores W&B run metadata
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            wandb_id TEXT,
            name TEXT,
            url TEXT,
            entity TEXT,
            project TEXT,
            created_at TEXT,
            state TEXT,
            tags TEXT,
            notes TEXT,
            user_name TEXT,
            last_history_step BIGINT,
            config_json TEXT,
            removed BOOLEAN DEFAULT FALSE,
            removed_at TEXT,
            updated_at TEXT,
            color TEXT,
            trainer_commit TEXT,
            schema_version TEXT,
            table_schema_versions_json TEXT
        );
    """)

    # Custom metrics layout (single-row, global config)
    con.execute("""
        CREATE TABLE IF NOT EXISTS custom_metrics_layout (
            id INTEGER PRIMARY KEY,
            layout_json TEXT NOT NULL,
            updated_at TEXT
        );
    """)

    # Custom metrics templates
    con.execute("""
        CREATE TABLE IF NOT EXISTS custom_metrics_templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            layout_json TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT
        );
    """)

    # Ingested tails table - tracks which tail indices have been ingested for each run
    # Each tail represents a 5-second interval of events uploaded by the trainer
    # Stored as a single JSON array per run to avoid row-group bloat from many small inserts
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingested_tails (
            run_id TEXT PRIMARY KEY,
            indices_json TEXT DEFAULT '[]'
        );
    """)
    
    # Ingested steps table - tracks which training steps have been ingested for each run
    # Each step represents a training step with rollouts
    # Stored as a single JSON array per run to avoid row-group bloat from many small inserts
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingested_steps (
            run_id TEXT PRIMARY KEY,
            indices_json TEXT DEFAULT '[]'
        );
    """)

    # Ingested step metrics table - tracks which training steps have step_metrics ingested
    # Stored as a single JSON array per run to avoid row-group bloat from many small inserts
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingested_step_metrics (
            run_id TEXT PRIMARY KEY,
            indices_json TEXT DEFAULT '[]'
        );
    """)

    # Ingested evals_after_training table - tracks which zip file names have been ingested
    # Stored as a single JSON array per run to avoid row-group bloat from many small inserts
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingested_evals_after_training (
            run_id TEXT PRIMARY KEY,
            file_names_json TEXT DEFAULT '[]'
        );
    """)

    # Known projects table - user-added projects to poll for tagged runs
    con.execute("""
        CREATE TABLE IF NOT EXISTS known_projects (
            project TEXT PRIMARY KEY,
            added_at TEXT
        );
    """)



def get_ingest_state(con: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    """Get the current ingest state for a run."""
    result = con.execute(
        """SELECT last_block_idx, last_rollout_step, last_summary_json,
                  last_event_zip_idx, last_rollout_block_idx, last_config_json, summary_id
           FROM ingest_state WHERE run_id = ?""",
        [run_id]
    ).fetchone()

    if result:
        return {
            "last_block_idx": result[0] or -1,
            "last_rollout_step": result[1] if result[1] is not None else -1,
            "last_summary_json": result[2],
            "last_event_zip_idx": result[3] or -1,
            "last_rollout_block_idx": result[4] or -1,
            "last_config_json": result[5],
            "summary_id": result[6],
        }
    return {
        "last_block_idx": -1,
        "last_rollout_step": -1,
        "last_summary_json": None,
        "last_event_zip_idx": -1,
        "last_rollout_block_idx": -1,
        "last_config_json": None,
        "summary_id": None,
    }


def update_ingest_state(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    last_block_idx: int | None = None,
    last_rollout_step: int | None = None,
    last_summary_json: str | None = None,
    last_config_json: str | None = None,
    last_event_zip_idx: int | None = None,
    last_rollout_block_idx: int | None = None,
    summary_id: str | None = None,
):
    """Update the ingest state for a run."""
    log.debug(f"[DB] Updating ingest state for {run_id}")
    # Upsert the state
    con.execute("""
        INSERT INTO ingest_state (run_id, last_block_idx, last_rollout_step, last_summary_json,
                                  last_config_json, last_event_zip_idx, last_rollout_block_idx, summary_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            last_block_idx = COALESCE(EXCLUDED.last_block_idx, ingest_state.last_block_idx),
            last_rollout_step = COALESCE(EXCLUDED.last_rollout_step, ingest_state.last_rollout_step),
            last_summary_json = COALESCE(EXCLUDED.last_summary_json, ingest_state.last_summary_json),
            last_config_json = COALESCE(EXCLUDED.last_config_json, ingest_state.last_config_json),
            last_event_zip_idx = COALESCE(EXCLUDED.last_event_zip_idx, ingest_state.last_event_zip_idx),
            last_rollout_block_idx = COALESCE(EXCLUDED.last_rollout_block_idx, ingest_state.last_rollout_block_idx),
            summary_id = COALESCE(EXCLUDED.summary_id, ingest_state.summary_id)
    """, [run_id, last_block_idx, last_rollout_step, last_summary_json,
          last_config_json, last_event_zip_idx, last_rollout_block_idx, summary_id])


def insert_events_orchestrator(con: duckdb.DuckDBPyConnection, run_id: str, events: list[dict]):
    """Insert orchestrator events (instant events) into the database."""
    if not events:
        return
    
    log.info(f"[DB] Inserting {len(events)} orchestrator events...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": event.get("timestamp"),
            "event_type": event.get("event_type"),
            "step": event.get("step"),
            "node_id": event.get("node_id"),
        }
        for event in events
    ])
    
    # Insert events - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO events_orchestrator (run_id, timestamp, event_type, step, node_id)
        SELECT run_id, timestamp, event_type, step, node_id FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(events)} orchestrator events in {elapsed:.2f}s")


def insert_events_trainer(con: duckdb.DuckDBPyConnection, run_id: str, events: list[dict]):
    """Insert trainer events (duration events) into the database."""
    if not events:
        return
    
    log.info(f"[DB] Inserting {len(events)} trainer events...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "event_type": event.get("event_type"),
            "step": event.get("step"),
            "rank": event.get("rank"),
            "local_rank": event.get("local_rank"),
            "node_id": event.get("node_id"),
            "gpu_index": event.get("gpu_index"),
            "microbatch": event.get("microbatch"),
            "minibatch": event.get("minibatch"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
        }
        for event in events
    ])
    
    # Insert events - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO events_trainer (
            run_id, event_type, step, rank, local_rank, node_id, gpu_index, microbatch, minibatch, start_time, end_time
        )
        SELECT
            run_id, event_type, step, rank, local_rank, node_id, gpu_index, microbatch, minibatch, start_time, end_time
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(events)} trainer events in {elapsed:.2f}s")


def insert_events_inference(con: duckdb.DuckDBPyConnection, run_id: str, events: list[dict]):
    """Insert inference events (duration events) into the database."""
    if not events:
        return
    
    log.info(f"[DB] Inserting {len(events)} inference events...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "event_type": event.get("event_type"),
            "server": event.get("server"),
            "node_id": event.get("node_id"),
            "tp_group_id": event.get("tp_group_id"),
            "tp_size": event.get("tp_size"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "prompt_tokens": event.get("prompt_tokens"),
            "rollout_tokens": event.get("rollout_tokens"),
            "sample_id": event.get("sample_id"),
            "group_id": event.get("group_id"),
            "vllm_request_id": event.get("vllm_request_id"),
            "queue_time": event.get("queue_time"),
            "time_to_first_token": event.get("time_to_first_token"),
            "prefill_time": event.get("prefill_time"),
            "decode_time": event.get("decode_time"),
            "inference_time": event.get("inference_time"),
            "e2e_latency": event.get("e2e_latency"),
            "max_tokens": event.get("max_tokens"),
            "is_eval": event.get("is_eval"),
            "step": event.get("step"),
            "is_canceled": event.get("is_canceled"),
            "off_policy_steps": event.get("off_policy_steps"),
            "server_lane": event.get("server_lane"),
        }
        for event in events
    ])

    # Insert events - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO events_inference (
            run_id, event_type, server, node_id, tp_group_id, tp_size,
            start_time, end_time, prompt_tokens, rollout_tokens, sample_id, group_id,
            vllm_request_id, queue_time, time_to_first_token, prefill_time, decode_time, inference_time,
            e2e_latency, max_tokens, is_eval, step, is_canceled, off_policy_steps, server_lane
        )
        SELECT
            run_id, event_type, server, node_id, tp_group_id, tp_size,
            start_time, end_time, prompt_tokens, rollout_tokens, sample_id, group_id,
            vllm_request_id, queue_time, time_to_first_token, prefill_time, decode_time, inference_time,
            e2e_latency, max_tokens, is_eval, step, is_canceled, off_policy_steps, server_lane
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(events)} inference events in {elapsed:.2f}s")




def insert_prompts(con: duckdb.DuckDBPyConnection, run_id: str, prompts: list[dict]):
    """Insert prompts into the database using fast bulk insert."""
    if not prompts:
        return
    
    log.info(f"[DB] Inserting {len(prompts)} prompts...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": p.get("step"),
            "group_id": p.get("group_id"),
            "env": p.get("env"),
            "system_prompt": p.get("system_prompt"),
            "tokens_system_prompt": p.get("tokens_system_prompt"),
            "prompt": p.get("prompt"),
            "tokens_prompt": p.get("tokens_prompt"),
        }
        for p in prompts
    ])
    
    # Insert prompts - duplicates are avoided by filtering on ingested_steps before calling this
    con.execute("""
        INSERT INTO prompts 
        SELECT * FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(prompts)} prompts in {elapsed:.2f}s")


def insert_rollouts(con: duckdb.DuckDBPyConnection, run_id: str, rollouts: list[dict]):
    """Insert rollouts (turns) into the database using fast bulk insert.
    
    Supports multi-turn conversations where each sample can have multiple turns.
    """
    if not rollouts:
        return
    
    log.info(f"[DB] Inserting {len(rollouts)} rollout turns...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": gen.get("step"),
            "group_id": gen.get("group_id"),
            "sample_idx": gen.get("sample_idx"),
            "turn_order": gen.get("turn_order"),
            "turn_type": gen.get("turn_type"),
            "content": gen.get("content"),
            "tokens": gen.get("tokens"),
            "stop_reason": gen.get("stop_reason"),
            "environment_response_time": gen.get("environment_response_time"),
        }
        for gen in rollouts
    ])
    
    # De-dupe within this batch, then make the insert idempotent (no UNIQUE constraints required).
    df = df.drop_duplicates(subset=["run_id", "step", "group_id", "sample_idx", "turn_order", "turn_type"])
    con.execute("""
        INSERT INTO rollouts
        SELECT d.*
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM rollouts g
            WHERE g.run_id = d.run_id
              AND g.step = d.step
              AND g.group_id = d.group_id
              AND g.sample_idx = d.sample_idx
              AND g.turn_order = d.turn_order
              AND g.turn_type = d.turn_type
        )
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(rollouts)} rollout turns in {elapsed:.2f}s")


def insert_samples_data(con: duckdb.DuckDBPyConnection, run_id: str, samples_data: list[dict]):
    """Insert sample-level data into the database using fast bulk insert."""
    if not samples_data:
        return
    
    log.info(f"[DB] Inserting {len(samples_data)} samples data...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": s.get("step"),
            "group_id": s.get("group_id"),
            "sample_idx": s.get("sample_idx"),
            "reward": s.get("reward"),
            "advantage": s.get("advantage"),
            "turns": s.get("turns"),
            "total_tokens": s.get("total_tokens"),
            "raw_string": s.get("raw_string"),
            "compute_reward_time": s.get("compute_reward_time"),
        }
        for s in samples_data
    ])

    # De-dupe within this batch, then make the insert idempotent (no UNIQUE constraints required).
    df = df.drop_duplicates(subset=["run_id", "step", "group_id", "sample_idx"])
    con.execute("""
        INSERT INTO samples_data
        SELECT d.*
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM samples_data s
            WHERE s.run_id = d.run_id
              AND s.step = d.step
              AND s.group_id = d.group_id
              AND s.sample_idx = d.sample_idx
        )
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(samples_data)} samples data in {elapsed:.2f}s")


def insert_rollouts_metrics(con: duckdb.DuckDBPyConnection, run_id: str, metrics: list[dict]):
    """Insert rollouts metrics into the database, ignoring duplicates using fast bulk insert."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} rollouts metrics...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": m.get("step"),
            "sample_idx": m.get("sample_idx"),
            "env": m.get("env"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
        }
        for m in metrics
    ])
    
    # De-dupe within this batch, then make the insert idempotent (no UNIQUE constraints required).
    df = df.drop_duplicates(subset=["run_id", "step", "sample_idx", "env", "metric_name"])
    con.execute("""
        INSERT INTO rollouts_metrics (run_id, step, sample_idx, env, metric_name, value)
        SELECT d.run_id, d.step, d.sample_idx, d.env, d.metric_name, d.value
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM rollouts_metrics gm
            WHERE gm.run_id = d.run_id
              AND gm.step = d.step
              AND gm.sample_idx = d.sample_idx
              AND gm.env = d.env
              AND gm.metric_name = d.metric_name
        )
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} rollouts metrics in {elapsed:.2f}s")


def insert_golden_answers(con: duckdb.DuckDBPyConnection, run_id: str, answers: list[dict]):
    """Insert golden answers into the database, ignoring duplicates using fast bulk insert."""
    if not answers:
        return
    
    log.info(f"[DB] Inserting {len(answers)} golden answers...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": a.get("step"),
            "sample_idx": a.get("sample_idx"),
            "env": a.get("env"),
            "key": a.get("key"),
            "value": a.get("value"),
        }
        for a in answers
    ])
    
    # De-dupe within this batch, then make the insert idempotent (no UNIQUE constraints required).
    df = df.drop_duplicates(subset=["run_id", "step", "sample_idx", "env", "key"])
    con.execute("""
        INSERT INTO golden_answers (run_id, step, sample_idx, env, key, value)
        SELECT d.run_id, d.step, d.sample_idx, d.env, d.key, d.value
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM golden_answers ga
            WHERE ga.run_id = d.run_id
              AND ga.step = d.step
              AND ga.sample_idx = d.sample_idx
              AND ga.env = d.env
              AND ga.key = d.key
        )
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(answers)} golden answers in {elapsed:.2f}s")


def insert_sample_tags(con: duckdb.DuckDBPyConnection, run_id: str, tags: list[dict]):
    """Insert sample tags into the database, ignoring duplicates using fast bulk insert."""
    if not tags:
        return

    log.info(f"[DB] Inserting {len(tags)} sample tags...")
    start = time.time()

    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": t.get("step"),
            "sample_idx": t.get("sample_idx"),
            "env": t.get("env"),
            "tag_name": t.get("tag_name"),
            "tag_value": t.get("tag_value"),
        }
        for t in tags
    ])

    # De-dupe within this batch, then make the insert idempotent (no UNIQUE constraints required).
    df = df.drop_duplicates(subset=["run_id", "step", "sample_idx", "env", "tag_name"])
    con.execute("""
        INSERT INTO sample_tags (run_id, step, sample_idx, env, tag_name, tag_value)
        SELECT d.run_id, d.step, d.sample_idx, d.env, d.tag_name, d.tag_value
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM sample_tags st
            WHERE st.run_id = d.run_id
              AND st.step = d.step
              AND st.sample_idx = d.sample_idx
              AND st.env = d.env
              AND st.tag_name = d.tag_name
        )
    """)

    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(tags)} sample tags in {elapsed:.2f}s")


def insert_system_metrics_gpu(con: duckdb.DuckDBPyConnection, run_id: str, metrics: list[dict]):
    """Insert GPU system metrics into the database, ignoring duplicates."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} GPU metrics...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": m.get("timestamp"),
            "node_id": m.get("node_id"),
            "rank": m.get("rank"),
            "local_rank": m.get("local_rank"),
            "gpu_index": m.get("gpu_index"),
            "source": m.get("source"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
        }
        for m in metrics
    ])
    
    # Insert metrics - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO system_metrics_gpu (
            run_id, timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
        )
        SELECT run_id, timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} GPU metrics in {elapsed:.2f}s")


def insert_system_metrics_cpu(con: duckdb.DuckDBPyConnection, run_id: str, metrics: list[dict]):
    """Insert CPU system metrics into the database, ignoring duplicates."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} CPU metrics...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": m.get("timestamp"),
            "node_id": m.get("node_id"),
            "source": m.get("source"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
        }
        for m in metrics
    ])
    
    # Insert metrics - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO system_metrics_cpu (run_id, timestamp, node_id, source, metric_name, value)
        SELECT run_id, timestamp, node_id, source, metric_name, value FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} CPU metrics in {elapsed:.2f}s")


def insert_vllm_metrics(con: duckdb.DuckDBPyConnection, run_id: str, metrics: list[dict]):
    """Insert vLLM metrics into the database, ignoring duplicates."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} vLLM metrics...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": m.get("timestamp"),
            "server": m.get("server"),
            "node_id": m.get("node_id"),
            "tp_group_id": m.get("tp_group_id"),
            "tp_size": m.get("tp_size"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
        }
        for m in metrics
    ])
    
    # Insert metrics - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO vllm_metrics (
            run_id, timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
        )
        SELECT run_id, timestamp, server, node_id, tp_group_id, tp_size, metric_name, value FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} vLLM metrics in {elapsed:.2f}s")


def insert_step_metrics(con: duckdb.DuckDBPyConnection, run_id: str, metrics: list[dict]):
    """Insert step metrics into the database."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} step metrics...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": m.get("step"),
            "metric_name": m.get("metric"),
            "value": m.get("value"),
            "section": m.get("section"),
            "group": m.get("group"),
        }
        for m in metrics
    ])

    # Insert metrics - duplicates are avoided by filtering on ingested_step_metrics before calling this
    con.execute("""
        INSERT INTO step_metrics (run_id, step, metric_name, value, section, "group")
        SELECT run_id, step, metric_name, value, section, "group" FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} step metrics in {elapsed:.2f}s")


def insert_prompts_discarded(con: duckdb.DuckDBPyConnection, run_id: str, prompts: list[dict]):
    """Insert discarded prompts into the database."""
    if not prompts:
        return
    
    log.info(f"[DB] Inserting {len(prompts)} discarded prompts...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": p.get("timestamp"),
            "discard_reason": p.get("discard_reason"),
            "trainer_step": p.get("trainer_step"),
            "inference_step": p.get("inference_step"),
            "group_id": p.get("group_id"),
            "env": p.get("env"),
            "system_prompt": p.get("system_prompt"),
            "tokens_system_prompt": p.get("tokens_system_prompt"),
            "prompt": p.get("prompt"),
            "tokens_prompt": p.get("tokens_prompt"),
            "tail_idx": p.get("tail_idx"),
        }
        for p in prompts
    ])
    
    # Insert prompts - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO prompts_discarded 
        SELECT * FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(prompts)} discarded prompts in {elapsed:.2f}s")


def insert_rollouts_discarded(con: duckdb.DuckDBPyConnection, run_id: str, rollouts: list[dict]):
    """Insert discarded rollouts (turns) into the database."""
    if not rollouts:
        return
    
    log.info(f"[DB] Inserting {len(rollouts)} discarded rollout turns...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "trainer_step": gen.get("trainer_step"),
            "inference_step": gen.get("inference_step"),
            "group_id": gen.get("group_id"),
            "sample_idx": gen.get("sample_idx"),
            "turn_order": gen.get("turn_order"),
            "turn_type": gen.get("turn_type"),
            "content": gen.get("content"),
            "tokens": gen.get("tokens"),
            "stop_reason": gen.get("stop_reason"),
            "tail_idx": gen.get("tail_idx"),
            "environment_response_time": gen.get("environment_response_time"),
        }
        for gen in rollouts
    ])
    
    # Insert rollouts - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO rollouts_discarded (
            run_id, trainer_step, inference_step, group_id, sample_idx,
            turn_order, turn_type, content, tokens, stop_reason, tail_idx,
            environment_response_time
        )
        SELECT
            run_id, trainer_step, inference_step, group_id, sample_idx,
            turn_order, turn_type, content, tokens, stop_reason, tail_idx,
            environment_response_time
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(rollouts)} discarded rollout turns in {elapsed:.2f}s")


def insert_samples_data_discarded(con: duckdb.DuckDBPyConnection, run_id: str, samples_data: list[dict]):
    """Insert discarded sample-level data into the database."""
    if not samples_data:
        return
    
    log.info(f"[DB] Inserting {len(samples_data)} discarded samples data...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "timestamp": s.get("timestamp"),
            "discard_reason": s.get("discard_reason"),
            "trainer_step": s.get("trainer_step"),
            "inference_step": s.get("inference_step"),
            "group_id": s.get("group_id"),
            "sample_idx": s.get("sample_idx"),
            "reward": s.get("reward"),
            "advantage": s.get("advantage"),
            "turns": s.get("turns"),
            "total_tokens": s.get("total_tokens"),
            "raw_string": s.get("raw_string"),
            "tail_idx": s.get("tail_idx"),
            "compute_reward_time": s.get("compute_reward_time"),
        }
        for s in samples_data
    ])

    # Insert samples data - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO samples_data_discarded 
        SELECT * FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(samples_data)} discarded samples data in {elapsed:.2f}s")


def insert_rollouts_metrics_discarded(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    metrics: list[dict],
):
    """Insert discarded rollouts metrics into the database."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} discarded rollouts metrics...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "sample_idx": m.get("sample_idx"),
            "env": m.get("env"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
            "tail_idx": m.get("tail_idx"),
        }
        for m in metrics
    ])
    
    # Insert metrics - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO rollouts_metrics_discarded (
            run_id, sample_idx, env, metric_name, value, tail_idx
        )
        SELECT run_id, sample_idx, env, metric_name, value, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} discarded rollouts metrics in {elapsed:.2f}s")


def insert_golden_answers_discarded(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    answers: list[dict],
):
    """Insert discarded golden answers into the database."""
    if not answers:
        return
    
    log.info(f"[DB] Inserting {len(answers)} discarded golden answers...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "sample_idx": a.get("sample_idx"),
            "env": a.get("env"),
            "key": a.get("key"),
            "value": a.get("value"),
            "tail_idx": a.get("tail_idx"),
        }
        for a in answers
    ])
    
    # Insert answers - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO golden_answers_discarded (
            run_id, sample_idx, env, key, value, tail_idx
        )
        SELECT run_id, sample_idx, env, key, value, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(answers)} discarded golden answers in {elapsed:.2f}s")


def insert_sample_tags_discarded(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    tags: list[dict],
):
    """Insert discarded sample tags into the database."""
    if not tags:
        return

    log.info(f"[DB] Inserting {len(tags)} discarded sample tags...")
    start = time.time()

    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "sample_idx": t.get("sample_idx"),
            "env": t.get("env"),
            "tag_name": t.get("tag_name"),
            "tag_value": t.get("tag_value"),
            "tail_idx": t.get("tail_idx"),
        }
        for t in tags
    ])

    # Insert tags - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO sample_tags_discarded (
            run_id, sample_idx, env, tag_name, tag_value, tail_idx
        )
        SELECT run_id, sample_idx, env, tag_name, tag_value, tail_idx
        FROM df
    """)

    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(tags)} discarded sample tags in {elapsed:.2f}s")


def insert_info_turns(con: duckdb.DuckDBPyConnection, run_id: str, info_turns: list[dict]):
    """Insert info turns into the database, ignoring duplicates using fast bulk insert."""
    if not info_turns:
        return
    
    log.info(f"[DB] Inserting {len(info_turns)} info turns...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": it.get("step"),
            "sample_idx": it.get("sample_idx"),
            "turn_order": it.get("turn_order"),
            "env": it.get("env"),
            "info_key": it.get("info_key"),
            "info_value": it.get("info_value"),
            "info_type": it.get("info_type"),
        }
        for it in info_turns
    ])
    
    # De-dupe within this batch, then make the insert idempotent
    df = df.drop_duplicates(subset=["run_id", "step", "sample_idx", "turn_order", "env", "info_key"])
    con.execute("""
        INSERT INTO info_turns
        SELECT d.*
        FROM df d
        WHERE NOT EXISTS (
            SELECT 1
            FROM info_turns it
            WHERE it.run_id = d.run_id
              AND it.step = d.step
              AND it.sample_idx = d.sample_idx
              AND it.turn_order = d.turn_order
              AND it.env = d.env
              AND it.info_key = d.info_key
        )
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(info_turns)} info turns in {elapsed:.2f}s")


def insert_info_turns_discarded(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    info_turns: list[dict],
):
    """Insert discarded info turns into the database."""
    if not info_turns:
        return
    
    log.info(f"[DB] Inserting {len(info_turns)} discarded info turns...")
    start = time.time()
    
    # Create DataFrame for fast bulk insert
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "sample_idx": it.get("sample_idx"),
            "turn_order": it.get("turn_order"),
            "env": it.get("env"),
            "info_key": it.get("info_key"),
            "info_value": it.get("info_value"),
            "info_type": it.get("info_type"),
            "tail_idx": it.get("tail_idx"),
        }
        for it in info_turns
    ])
    
    # Insert - duplicates are avoided by filtering on ingested_tails before calling this
    con.execute("""
        INSERT INTO info_turns_discarded (
            run_id, sample_idx, turn_order, env, info_key, info_value, info_type, tail_idx
        )
        SELECT run_id, sample_idx, turn_order, env, info_key, info_value, info_type, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(info_turns)} discarded info turns in {elapsed:.2f}s")


def insert_prompts_eval(con: duckdb.DuckDBPyConnection, run_id: str, prompts: list[dict]):
    """Insert eval prompts into the database."""
    if not prompts:
        return
    
    log.info(f"[DB] Inserting {len(prompts)} eval prompts...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": p.get("step"),
            "eval_name": p.get("eval_name"),
            "model_step": p.get("model_step"),
            "sample_idx": p.get("sample_idx"),
            "env": p.get("env"),
            "prompt": p.get("prompt"),
            "tokens_prompt": p.get("tokens_prompt"),
            "system_prompt": p.get("system_prompt"),
            "tokens_system_prompt": p.get("tokens_system_prompt"),
            "tail_idx": p.get("tail_idx"),
        }
        for p in prompts
    ])
    
    con.execute("""
        INSERT INTO prompts_eval
        SELECT * FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(prompts)} eval prompts in {elapsed:.2f}s")


def insert_rollouts_eval(con: duckdb.DuckDBPyConnection, run_id: str, rollouts: list[dict]):
    """Insert eval rollouts (turns) into the database."""
    if not rollouts:
        return
    
    log.info(f"[DB] Inserting {len(rollouts)} eval rollout turns...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": r.get("step"),
            "eval_name": r.get("eval_name"),
            "model_step": r.get("model_step"),
            "sample_idx": r.get("sample_idx"),
            "completion_idx": r.get("completion_idx"),
            "turn_order": r.get("turn_order"),
            "turn_type": r.get("turn_type"),
            "content": r.get("content"),
            "tokens": r.get("tokens"),
            "stop_reason": r.get("stop_reason"),
            "environment_response_time": r.get("environment_response_time"),
            "tail_idx": r.get("tail_idx"),
        }
        for r in rollouts
    ])
    
    con.execute("""
        INSERT INTO rollouts_eval (
            run_id, step, eval_name, model_step, sample_idx, completion_idx,
            turn_order, turn_type, content, tokens, stop_reason,
            environment_response_time, tail_idx
        )
        SELECT
            run_id, step, eval_name, model_step, sample_idx, completion_idx,
            turn_order, turn_type, content, tokens, stop_reason,
            environment_response_time, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(rollouts)} eval rollout turns in {elapsed:.2f}s")


def insert_samples_data_eval(con: duckdb.DuckDBPyConnection, run_id: str, samples_data: list[dict]):
    """Insert eval sample-level data into the database."""
    if not samples_data:
        return
    
    log.info(f"[DB] Inserting {len(samples_data)} eval samples data...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": s.get("step"),
            "eval_name": s.get("eval_name"),
            "model_step": s.get("model_step"),
            "sample_idx": s.get("sample_idx"),
            "completion_idx": s.get("completion_idx"),
            "env": s.get("env"),
            "turns": s.get("turns"),
            "compute_eval_metrics_time": s.get("compute_eval_metrics_time"),
            "tail_idx": s.get("tail_idx"),
        }
        for s in samples_data
    ])
    
    con.execute("""
        INSERT INTO samples_data_eval
        SELECT * FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(samples_data)} eval samples data in {elapsed:.2f}s")


def insert_rollouts_metrics_eval(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    metrics: list[dict],
):
    """Insert eval rollouts metrics into the database."""
    if not metrics:
        return
    
    log.info(f"[DB] Inserting {len(metrics)} eval rollouts metrics...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": m.get("step"),
            "eval_name": m.get("eval_name"),
            "sample_idx": m.get("sample_idx"),
            "completion_idx": m.get("completion_idx"),
            "env": m.get("env"),
            "metric_name": m.get("metric_name"),
            "value": m.get("value"),
            "tail_idx": m.get("tail_idx"),
        }
        for m in metrics
    ])
    
    con.execute("""
        INSERT INTO rollouts_metrics_eval (
            run_id, step, eval_name, sample_idx, completion_idx, env, metric_name, value, tail_idx
        )
        SELECT
            run_id, step, eval_name, sample_idx, completion_idx, env, metric_name, value, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(metrics)} eval rollouts metrics in {elapsed:.2f}s")


def insert_golden_answers_eval(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    answers: list[dict],
):
    """Insert eval golden answers into the database."""
    if not answers:
        return
    
    log.info(f"[DB] Inserting {len(answers)} eval golden answers...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": a.get("step"),
            "eval_name": a.get("eval_name"),
            "sample_idx": a.get("sample_idx"),
            "completion_idx": a.get("completion_idx"),
            "env": a.get("env"),
            "key": a.get("key"),
            "value": a.get("value"),
            "tail_idx": a.get("tail_idx"),
        }
        for a in answers
    ])
    
    con.execute("""
        INSERT INTO golden_answers_eval (
            run_id, step, eval_name, sample_idx, completion_idx, env, key, value, tail_idx
        )
        SELECT
            run_id, step, eval_name, sample_idx, completion_idx, env, key, value, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(answers)} eval golden answers in {elapsed:.2f}s")


def insert_sample_tags_eval(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    tags: list[dict],
):
    """Insert eval sample tags into the database."""
    if not tags:
        return

    log.info(f"[DB] Inserting {len(tags)} eval sample tags...")
    start = time.time()

    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": t.get("step"),
            "eval_name": t.get("eval_name"),
            "sample_idx": t.get("sample_idx"),
            "completion_idx": t.get("completion_idx"),
            "env": t.get("env"),
            "tag_name": t.get("tag_name"),
            "tag_value": t.get("tag_value"),
            "tail_idx": t.get("tail_idx"),
        }
        for t in tags
    ])

    con.execute("""
        INSERT INTO sample_tags_eval (
            run_id, step, eval_name, sample_idx, completion_idx, env, tag_name, tag_value, tail_idx
        )
        SELECT
            run_id, step, eval_name, sample_idx, completion_idx, env, tag_name, tag_value, tail_idx
        FROM df
    """)

    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(tags)} eval sample tags in {elapsed:.2f}s")


def insert_info_turns_eval(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    info_turns: list[dict],
):
    """Insert eval info turns into the database."""
    if not info_turns:
        return
    
    log.info(f"[DB] Inserting {len(info_turns)} eval info turns...")
    start = time.time()
    
    df = pd.DataFrame([
        {
            "run_id": run_id,
            "step": it.get("step"),
            "eval_name": it.get("eval_name"),
            "sample_idx": it.get("sample_idx"),
            "completion_idx": it.get("completion_idx"),
            "turn_order": it.get("turn_order"),
            "env": it.get("env"),
            "info_key": it.get("info_key"),
            "info_value": it.get("info_value"),
            "info_type": it.get("info_type"),
            "tail_idx": it.get("tail_idx"),
        }
        for it in info_turns
    ])
    
    con.execute("""
        INSERT INTO info_turns_eval (
            run_id, step, eval_name, sample_idx, completion_idx, turn_order,
            env, info_key, info_value, info_type, tail_idx
        )
        SELECT
            run_id, step, eval_name, sample_idx, completion_idx, turn_order,
            env, info_key, info_value, info_type, tail_idx
        FROM df
    """)
    
    elapsed = time.time() - start
    log.info(f"[DB] Inserted {len(info_turns)} eval info turns in {elapsed:.2f}s")


def _get_next_run_color(con: duckdb.DuckDBPyConnection) -> str:
    """Get the next color for a new run by counting existing runs."""
    result = con.execute("SELECT COUNT(*) FROM runs").fetchone()
    count = result[0] if result else 0
    return RUN_COLORS[count % len(RUN_COLORS)]


def upsert_run(con: duckdb.DuckDBPyConnection, run_data: dict):
    """Insert or update run metadata in the runs table."""
    run_id = run_data.get("run_id")
    if not run_id:
        return
    
    log.debug(f"[DB] Upserting run metadata for {run_id}")
    
    # Convert tags list to JSON string if present
    tags = run_data.get("tags")
    if isinstance(tags, list):
        import json
        tags = json.dumps(tags)
    
    # Check if this run already exists and has a color
    existing = con.execute(
        "SELECT color FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    
    # Only assign a new color if the run doesn't exist yet
    if existing is None:
        color = _get_next_run_color(con)
    else:
        color = existing[0]  # Keep existing color
    
    con.execute("""
        INSERT INTO runs (
            run_id, wandb_id, name, url, entity, project, created_at,
            state, tags, notes, user_name, last_history_step,
            config_json, updated_at, color, trainer_commit, schema_version,
            table_schema_versions_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            wandb_id = COALESCE(EXCLUDED.wandb_id, runs.wandb_id),
            name = COALESCE(EXCLUDED.name, runs.name),
            url = COALESCE(EXCLUDED.url, runs.url),
            entity = COALESCE(EXCLUDED.entity, runs.entity),
            project = COALESCE(EXCLUDED.project, runs.project),
            created_at = COALESCE(EXCLUDED.created_at, runs.created_at),
            state = COALESCE(EXCLUDED.state, runs.state),
            tags = COALESCE(EXCLUDED.tags, runs.tags),
            notes = COALESCE(EXCLUDED.notes, runs.notes),
            user_name = COALESCE(EXCLUDED.user_name, runs.user_name),
            last_history_step = COALESCE(EXCLUDED.last_history_step, runs.last_history_step),
            config_json = COALESCE(EXCLUDED.config_json, runs.config_json),
            updated_at = EXCLUDED.updated_at,
            color = COALESCE(runs.color, EXCLUDED.color),
            trainer_commit = COALESCE(EXCLUDED.trainer_commit, runs.trainer_commit),
            schema_version = COALESCE(EXCLUDED.schema_version, runs.schema_version),
            table_schema_versions_json = COALESCE(EXCLUDED.table_schema_versions_json, runs.table_schema_versions_json)
    """, [
        run_id,
        run_data.get("wandb_id"),
        run_data.get("name"),
        run_data.get("url"),
        run_data.get("entity"),
        run_data.get("project"),
        run_data.get("created_at"),
        run_data.get("state"),
        tags,
        run_data.get("notes"),
        run_data.get("user_name"),
        run_data.get("last_history_step"),
        run_data.get("config_json"),
        run_data.get("updated_at"),
        color,
        run_data.get("trainer_commit"),
        run_data.get("schema_version"),
        json.dumps(run_data["table_schema_versions"]) if run_data.get("table_schema_versions") else None,
    ])


def get_wandb_key_from_netrc() -> str | None:
    """Read the W&B API key from ~/.netrc (written by `wandb login`)."""
    import netrc as netrc_mod

    netrc_path = os.environ.get("NETRC") or os.path.join(Path.home(), ".netrc")
    try:
        nrc = netrc_mod.netrc(netrc_path)
        auth = nrc.authenticators("api.wandb.ai")
        if auth:
            return auth[2]  # (login, account, password) – password is the key
    except (FileNotFoundError, netrc_mod.NetrcParseError):
        pass
    return None


def _key_source_file() -> Path:
    return _get_data_dir() / "wandb_key_source"


def _api_key_file() -> Path:
    return _get_data_dir() / "wandb_key"


def get_wandb_key_source() -> str:
    """Get the configured key source: 'netrc', 'custom', or 'unconfigured'."""
    f = _key_source_file()
    if f.exists():
        return f.read_text().strip() or "unconfigured"
    return "unconfigured"


def set_wandb_key_source(source: str):
    """Set the key source preference ('netrc' or 'custom')."""
    f = _key_source_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source)


def set_wandb_api_key(*, api_key: str):
    """Store a custom W&B API key in ~/.telescope/wandb_key."""
    f = _api_key_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(api_key)
    f.chmod(0o600)
    set_wandb_key_source("custom")


def get_wandb_api_key() -> str | None:
    """Fetch the W&B API key based on the configured source."""
    source = get_wandb_key_source()

    if source == "netrc":
        return get_wandb_key_from_netrc()

    if source == "custom":
        f = _api_key_file()
        if f.exists():
            return f.read_text().strip()

    return None


def delete_wandb_api_key():
    """Remove the stored W&B API key file."""
    f = _api_key_file()
    if f.exists():
        f.unlink()


def get_custom_metrics_layout(con: duckdb.DuckDBPyConnection) -> dict | None:
    """Fetch the global custom metrics layout, if any."""
    row = con.execute(
        "SELECT layout_json FROM custom_metrics_layout WHERE id = 1"
    ).fetchone()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


def set_custom_metrics_layout(con: duckdb.DuckDBPyConnection, layout: dict):
    """Upsert the global custom metrics layout."""
    from datetime import datetime

    updated_at = datetime.utcnow().isoformat()
    layout_json = json.dumps(layout, separators=(",", ":"))
    con.execute("""
        INSERT INTO custom_metrics_layout (id, layout_json, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            layout_json = EXCLUDED.layout_json,
            updated_at = EXCLUDED.updated_at
    """, [layout_json, updated_at])


def list_custom_metrics_templates(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """List all custom metrics templates (id, name, updated_at)."""
    rows = con.execute(
        "SELECT id, name, updated_at FROM custom_metrics_templates ORDER BY name"
    ).fetchall()
    return [{"id": row[0], "name": row[1], "updated_at": row[2]} for row in rows]


def get_custom_metrics_template(con: duckdb.DuckDBPyConnection, template_id: str) -> dict | None:
    """Get a single custom metrics template by ID."""
    row = con.execute(
        "SELECT id, name, layout_json, updated_at FROM custom_metrics_templates WHERE id = ?",
        [template_id],
    ).fetchone()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "layout": json.loads(row[2]), "updated_at": row[3]}


def create_custom_metrics_template(
    con: duckdb.DuckDBPyConnection, template_id: str, name: str, layout: dict
):
    """Create a new custom metrics template."""
    from datetime import datetime

    now = datetime.utcnow().isoformat()
    layout_json = json.dumps(layout, separators=(",", ":"))
    con.execute(
        """
        INSERT INTO custom_metrics_templates (id, name, layout_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [template_id, name, layout_json, now, now],
    )


def update_custom_metrics_template_layout(
    con: duckdb.DuckDBPyConnection, template_id: str, layout: dict
):
    """Update an existing template's layout."""
    from datetime import datetime

    now = datetime.utcnow().isoformat()
    layout_json = json.dumps(layout, separators=(",", ":"))
    con.execute(
        "UPDATE custom_metrics_templates SET layout_json = ?, updated_at = ? WHERE id = ?",
        [layout_json, now, template_id],
    )


def rename_custom_metrics_template(con: duckdb.DuckDBPyConnection, template_id: str, name: str):
    """Rename a custom metrics template."""
    from datetime import datetime

    now = datetime.utcnow().isoformat()
    con.execute(
        "UPDATE custom_metrics_templates SET name = ?, updated_at = ? WHERE id = ?",
        [name, now, template_id],
    )


def delete_custom_metrics_template(con: duckdb.DuckDBPyConnection, template_id: str):
    """Delete a custom metrics template."""
    con.execute("DELETE FROM custom_metrics_templates WHERE id = ?", [template_id])


def insert_known_project(con: duckdb.DuckDBPyConnection, project: str) -> bool:
    """Insert a user-added project. Returns True if newly inserted."""
    from datetime import datetime

    added_at = datetime.utcnow().isoformat() + "Z"
    existing = con.execute(
        "SELECT 1 FROM known_projects WHERE project = ?", [project]
    ).fetchone()
    if existing:
        return False
    con.execute(
        "INSERT INTO known_projects (project, added_at) VALUES (?, ?)",
        [project, added_at],
    )
    return True


def delete_known_project(con: duckdb.DuckDBPyConnection, project: str) -> bool:
    """Delete a user-added project. Returns True if a row was deleted."""
    result = con.execute(
        "DELETE FROM known_projects WHERE project = ? RETURNING project", [project]
    ).fetchone()
    return result is not None


def list_known_projects(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """List all user-added known projects."""
    rows = con.execute(
        "SELECT project, added_at FROM known_projects ORDER BY added_at DESC"
    ).fetchall()
    return [{"project": row[0], "added_at": row[1]} for row in rows]


def set_run_removed(con: duckdb.DuckDBPyConnection, run_id: str, removed: bool):
    """Mark a run as removed or restored without deleting its metadata."""
    from datetime import datetime

    updated_at = datetime.utcnow().isoformat() + "Z"
    removed_at = updated_at if removed else None
    con.execute(
        "UPDATE runs SET removed = ?, removed_at = ?, updated_at = ? WHERE run_id = ?",
        [removed, removed_at, updated_at, run_id],
    )


def update_run_color(con: duckdb.DuckDBPyConnection, run_id: str, color: str):
    """Update the display color for a run in the runs table."""
    from datetime import datetime

    updated_at = datetime.utcnow().isoformat()
    con.execute(
        "UPDATE runs SET color = ?, updated_at = ? WHERE run_id = ?",
        [color, updated_at, run_id],
    )


def get_ingested_tails(con: duckdb.DuckDBPyConnection, run_id: str, min_tail_idx: int | None = None, max_tail_idx: int | None = None) -> set[int]:
    """Get the set of tail indices that have already been ingested for a run.
    
    Optionally filter to a specific range [min_tail_idx, max_tail_idx] inclusive.
    """
    row = con.execute(
        "SELECT indices_json FROM ingested_tails WHERE run_id = ?",
        [run_id]
    ).fetchone()
    
    if not row or not row[0]:
        return set()
    
    all_indices = set(json.loads(row[0]))
    
    # Apply range filter if specified
    if min_tail_idx is not None or max_tail_idx is not None:
        filtered = set()
        for idx in all_indices:
            if min_tail_idx is not None and idx < min_tail_idx:
                continue
            if max_tail_idx is not None and idx > max_tail_idx:
                continue
            filtered.add(idx)
        return filtered
    
    return all_indices


def insert_ingested_tails(con: duckdb.DuckDBPyConnection, run_id: str, tail_indices: set[int]):
    """Record that specific tail indices have been ingested for a run."""
    if not tail_indices:
        return
    
    log.debug(f"[DB] Recording {len(tail_indices)} ingested tails for {run_id}")
    
    # Read existing indices
    row = con.execute(
        "SELECT indices_json FROM ingested_tails WHERE run_id = ?",
        [run_id]
    ).fetchone()
    
    existing = set(json.loads(row[0])) if row and row[0] else set()
    merged = existing | tail_indices
    merged_json = json.dumps(sorted(merged))
    
    # Upsert the merged set
    con.execute("""
        INSERT INTO ingested_tails (run_id, indices_json)
        VALUES (?, ?)
        ON CONFLICT (run_id) DO UPDATE SET indices_json = EXCLUDED.indices_json
    """, [run_id, merged_json])


def get_ingested_steps(con: duckdb.DuckDBPyConnection, run_id: str, min_step: int | None = None, max_step: int | None = None) -> set[int]:
    """Get the set of steps that have already been ingested for a run.
    
    Optionally filter to a specific range [min_step, max_step] inclusive.
    """
    row = con.execute(
        "SELECT indices_json FROM ingested_steps WHERE run_id = ?",
        [run_id]
    ).fetchone()
    
    if not row or not row[0]:
        return set()
    
    all_steps = set(json.loads(row[0]))
    
    # Apply range filter if specified
    if min_step is not None or max_step is not None:
        filtered = set()
        for step in all_steps:
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            filtered.add(step)
        return filtered
    
    return all_steps


def get_ingested_step_metrics(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    min_step: int | None = None,
    max_step: int | None = None,
) -> set[int]:
    """Get the set of steps that have step_metrics ingested for a run.
    
    Optionally filter to a specific range [min_step, max_step] inclusive.
    """
    row = con.execute(
        "SELECT indices_json FROM ingested_step_metrics WHERE run_id = ?",
        [run_id]
    ).fetchone()

    if not row or not row[0]:
        return set()

    all_steps = set(json.loads(row[0]))

    # Apply range filter if specified
    if min_step is not None or max_step is not None:
        filtered = set()
        for step in all_steps:
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            filtered.add(step)
        return filtered

    return all_steps


def insert_ingested_steps(con: duckdb.DuckDBPyConnection, run_id: str, steps: set[int]):
    """Record that specific steps have been ingested for a run."""
    if not steps:
        return
    
    log.debug(f"[DB] Recording {len(steps)} ingested steps for {run_id}")
    
    # Read existing steps
    row = con.execute(
        "SELECT indices_json FROM ingested_steps WHERE run_id = ?",
        [run_id]
    ).fetchone()
    
    existing = set(json.loads(row[0])) if row and row[0] else set()
    merged = existing | steps
    merged_json = json.dumps(sorted(merged))
    
    # Upsert the merged set
    con.execute("""
        INSERT INTO ingested_steps (run_id, indices_json)
        VALUES (?, ?)
        ON CONFLICT (run_id) DO UPDATE SET indices_json = EXCLUDED.indices_json
    """, [run_id, merged_json])


def insert_ingested_step_metrics(con: duckdb.DuckDBPyConnection, run_id: str, steps: set[int]):
    """Record that specific steps have step_metrics ingested for a run."""
    if not steps:
        return

    log.debug(f"[DB] Recording {len(steps)} ingested step_metrics for {run_id}")

    # Read existing steps
    row = con.execute(
        "SELECT indices_json FROM ingested_step_metrics WHERE run_id = ?",
        [run_id]
    ).fetchone()

    existing = set(json.loads(row[0])) if row and row[0] else set()
    merged = existing | steps
    merged_json = json.dumps(sorted(merged))

    # Upsert the merged set
    con.execute("""
        INSERT INTO ingested_step_metrics (run_id, indices_json)
        VALUES (?, ?)
        ON CONFLICT (run_id) DO UPDATE SET indices_json = EXCLUDED.indices_json
    """, [run_id, merged_json])


def get_ingested_evals_after_training(con: duckdb.DuckDBPyConnection, run_id: str) -> set[str]:
    """Get the set of evals_after_training zip file names already ingested for a run."""
    row = con.execute(
        "SELECT file_names_json FROM ingested_evals_after_training WHERE run_id = ?",
        [run_id]
    ).fetchone()

    if not row or not row[0]:
        return set()

    return set(json.loads(row[0]))


def insert_ingested_evals_after_training(
    con: duckdb.DuckDBPyConnection, run_id: str, file_names: set[str]
):
    """Record that specific evals_after_training zip files have been ingested for a run."""
    if not file_names:
        return

    log.debug(f"[DB] Recording {len(file_names)} ingested evals_after_training files for {run_id}")

    # Read existing file names
    row = con.execute(
        "SELECT file_names_json FROM ingested_evals_after_training WHERE run_id = ?",
        [run_id]
    ).fetchone()

    existing = set(json.loads(row[0])) if row and row[0] else set()
    merged = existing | file_names
    merged_json = json.dumps(sorted(merged))

    # Upsert the merged set
    con.execute("""
        INSERT INTO ingested_evals_after_training (run_id, file_names_json)
        VALUES (?, ?)
        ON CONFLICT (run_id) DO UPDATE SET file_names_json = EXCLUDED.file_names_json
    """, [run_id, merged_json])


def get_database_info() -> dict:
    """Get database file size."""
    db_path = str(DB_PATH)
    size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    return {"size_bytes": size_bytes}


def recover_from_failed_compaction():
    """Check for and recover from a crashed compaction on startup."""
    db_path = str(DB_PATH)
    backup_path = db_path + ".pre_compact"
    new_db_path = db_path + ".compacting"
    export_dir = db_path + "_compact_export"

    if os.path.exists(backup_path):
        if not os.path.exists(db_path):
            log.warning("[DB] Found backup from failed compaction, restoring original database...")
            os.rename(backup_path, db_path)
        else:
            log.info("[DB] Found leftover compaction backup, cleaning up...")
            os.remove(backup_path)

    if os.path.exists(new_db_path):
        log.info("[DB] Cleaning up incomplete compacted database...")
        os.remove(new_db_path)
    if os.path.isdir(export_dir):
        log.info("[DB] Cleaning up compaction export directory...")
        shutil.rmtree(export_dir)


def compact_database(progress_callback=None) -> dict:
    """Compact the database by exporting to Parquet and reimporting.

    This triggers DuckDB's compression pipeline on all data, reclaims free
    blocks, and consolidates fragmented row groups.  The original database
    is only removed after the new one is fully written and verified, so a
    crash at any point leaves at least one valid copy on disk.
    """
    db_path = str(DB_PATH)
    export_dir = db_path + "_compact_export"
    new_db_path = db_path + ".compacting"
    backup_path = db_path + ".pre_compact"

    # Clean up leftover files from any previous failed compaction
    for p in [new_db_path, backup_path]:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)

    size_before = os.path.getsize(db_path)

    # Step 1 — Export the current database to Parquet files.
    if progress_callback:
        progress_callback("exporting")
    escaped_export = export_dir.replace("'", "''")
    con = duckdb.connect(db_path)
    try:
        con.execute(f"EXPORT DATABASE '{escaped_export}' (FORMAT PARQUET)")
    finally:
        con.close()

    # Step 2 — Import into a brand-new database file.
    if progress_callback:
        progress_callback("importing")
    con = duckdb.connect(new_db_path)
    try:
        con.execute(f"IMPORT DATABASE '{escaped_export}'")
    finally:
        con.close()

    # Step 3 — Atomic swap: old → backup, new → original path.
    if progress_callback:
        progress_callback("finalizing")
    os.rename(db_path, backup_path)
    os.rename(new_db_path, db_path)

    size_after = os.path.getsize(db_path)

    # Step 4 — Clean up backup and export directory.
    os.remove(backup_path)
    shutil.rmtree(export_dir)

    log.info(
        f"[DB] Compaction complete: {size_before / 1024**3:.2f} GiB → "
        f"{size_after / 1024**3:.2f} GiB "
        f"({(1 - size_after / size_before) * 100:.0f}% reduction)"
    )
    return {"size_before": size_before, "size_after": size_after}
