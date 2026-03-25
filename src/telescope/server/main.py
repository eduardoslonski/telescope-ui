from __future__ import annotations
import asyncio
import json
import logging
import math
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .ingest import (
    ingestion_loop,
    clear_active_run,
    is_tracking,
    is_syncing,
    get_sync_status,
    get_discovery_status,
    start_sync,
    configure_wandb_and_sync,
    discover_and_sync_project,
    enqueue_sync,
    fetch_and_store_run,
    tagged_runs_poll_loop,
    restore_active_runs_from_db,
    run_needs_sync,
    ingest_state_reconcile_loop,
    pause_for_compaction,
    resume_after_compaction,
    start_evals_after_training_sync,
    is_syncing_evals_after_training,
    get_all_known_projects,
    has_known_projects,
    add_project_to_known,
    remove_project_from_known,
    set_discovery_status_discovering,
    _get_wandb_api,
)
from .db import (
    compact_database,
    connect,
    create_custom_metrics_template,
    decompress_blob,
    delete_custom_metrics_template,
    get_custom_metrics_layout,
    get_custom_metrics_template,
    get_database_info,
    get_wandb_api_key,
    get_wandb_key_from_netrc,
    get_wandb_key_source,
    list_custom_metrics_templates,
    recover_from_failed_compaction,
    rename_custom_metrics_template,
    set_custom_metrics_layout,
    set_run_drained,
    set_run_removed,
    set_wandb_api_key,
    set_wandb_key_source,
    delete_wandb_api_key,
    update_custom_metrics_template_layout,
    update_run_color,
    update_run_name,
    update_run_notes,
)
from .run_code import (
    build_code_diff_summary,
    build_code_tree,
    get_run_code_dir,
    get_run_code_run_dir,
    resolve_code_file_path,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI()
MAX_CODE_FILE_BYTES = 2 * 1024 * 1024


# All tables keyed by run_id (used for bulk deletion)
_TABLES_WITH_RUN_ID = [
    "events_orchestrator",
    "events_trainer",
    "events_inference",
    "prompts",
    "rollouts",
    "samples_data",
    "rollouts_metrics",
    "golden_answers",
    "info_turns",
    "system_metrics_gpu",
    "system_metrics_cpu",
    "vllm_metrics",
    "step_metrics",
    "prompts_discarded",
    "rollouts_discarded",
    "samples_data_discarded",
    "rollouts_metrics_discarded",
    "golden_answers_discarded",
    "info_turns_discarded",
    "prompts_eval",
    "rollouts_eval",
    "samples_data_eval",
    "rollouts_metrics_eval",
    "golden_answers_eval",
    "info_turns_eval",
    "logs",
    "ingest_state",
    "ingested_tails",
    "ingested_steps",
    "ingested_step_metrics",
    "ingested_evals_after_training",
]

# Compaction progress (shared between the endpoint and the background task)
_compaction_state: dict = {
    "status": "idle",  # idle | pausing_syncs | exporting | importing | finalizing | done | error
    "error": None,
    "size_before": None,
    "size_after": None,
}


def _delete_local_run_artifacts(run_path: str) -> None:
    run_code_dir = get_run_code_run_dir(run_path)
    if not run_code_dir.exists():
        return
    try:
        shutil.rmtree(run_code_dir)
        log.info(f"[API] Deleted local run artifacts for {run_path}: {run_code_dir}")
    except Exception as e:
        log.warning(f"[API] Failed deleting local run artifacts for {run_path}: {e}")

# CORS for development (Vite dev server on :3000 hitting API on :8005).
# In production the UI is served by this same FastAPI instance (same origin),
# so CORS headers are unused but harmless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static UI serving
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def mount_static_ui() -> None:
    """Mount the pre-built UI if the static directory exists.

    Called from the CLI entry point after the app is created.  In development
    (running uvicorn directly) this is a no-op since the Vite dev server
    serves the UI.
    """
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return

    # Serve static assets (JS/CSS/fonts/images)
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="static-assets")

    # Catch-all: serve index.html for any non-API path (SPA client-side routing)
    @app.get("/{full_path:path}")
    async def _serve_spa(full_path: str):
        # If a static file exists at this path, serve it
        candidate = _STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start = time.time()
    log.debug(f"[API] --> {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    log.debug(f"[API] <-- {request.method} {request.url.path} [{response.status_code}] {elapsed:.1f}ms")
    return response


@app.on_event("startup")
async def _startup():
    from .db import _get_data_dir
    _get_data_dir().mkdir(parents=True, exist_ok=True)
    log.info("[SERVER] Starting up — data dir: %s", _get_data_dir())
    recover_from_failed_compaction()
    asyncio.create_task(ingestion_loop())
    asyncio.create_task(tagged_runs_poll_loop())
    asyncio.create_task(ingest_state_reconcile_loop())

    # Restore state from local DB (no W&B API calls needed)
    con = connect()
    api_key = get_wandb_api_key()
    if api_key:
        # Resume tracking for runs that were running before the server stopped
        restore_active_runs_from_db(api_key)

        # Re-queue runs that need syncing (running or incomplete data)
        rows = con.execute("""
            SELECT r.run_id,
                   COALESCE(i.last_rollout_step, -1) as last_gen_step,
                   COALESCE(i.last_event_zip_idx, -1) as last_event_zip_idx,
                   COALESCE(i.last_block_idx, -1) as last_event_block_idx,
                   COALESCE(i.last_rollout_block_idx, -1) as last_rollout_block_idx,
                   i.last_summary_json
            FROM runs r
            LEFT JOIN ingest_state i ON r.run_id = i.run_id
            WHERE COALESCE(r.removed, FALSE) = FALSE
            ORDER BY r.created_at DESC NULLS LAST
        """).fetchall()
        pending_paths = []
        skipped = 0
        for (
            run_id,
            last_gen_step,
            last_event_zip_idx,
            last_event_block_idx,
            last_rollout_block_idx,
            summary_json,
        ) in rows:
            if run_needs_sync(
                last_gen_step,
                last_event_zip_idx,
                summary_json,
                last_event_block_idx=last_event_block_idx,
                last_rollout_block_idx=last_rollout_block_idx,
            ):
                pending_paths.append(run_id)
            else:
                skipped += 1
        if pending_paths:
            log.info(f"[SERVER] Re-queueing {len(pending_paths)} run(s) for sync (skipped {skipped} already-complete)")
            enqueue_sync(pending_paths, api_key)
        else:
            log.info(f"[SERVER] No runs need syncing ({skipped} already complete)")
    con.close()

    log.info("[SERVER] Startup complete")


@app.get("/health")
def health():
    return {"ok": True}


# Request/Response models
class SyncRequest(BaseModel):
    run_path: str


class StopTrackingRequest(BaseModel):
    run_path: str


class DeleteRunDataRequest(BaseModel):
    run_path: str


class WandbConfigRequest(BaseModel):
    api_key: str | None = None
    use_netrc: bool = False


class AddRunRequest(BaseModel):
    run_path: str


class RemoveRunRequest(BaseModel):
    run_path: str


class DrainRunRequest(BaseModel):
    run_path: str


class AddProjectRequest(BaseModel):
    project: str


class RemoveProjectRequest(BaseModel):
    project: str


class SetRunColorRequest(BaseModel):
    run_path: str
    color: str


class RenameRunRequest(BaseModel):
    run_path: str
    name: str


class UpdateNotesRequest(BaseModel):
    run_path: str
    notes: str


class RolloutsRequest(BaseModel):
    run_path: str
    step: int | None = None


class RolloutsDiscardedRequest(BaseModel):
    run_path: str
    trainer_step: int | None = None


class EvalsRequest(BaseModel):
    run_path: str
    step: int | None = None
    eval_name: str | None = None


class EvalStepMetricsRequest(BaseModel):
    run_path: str
    eval_name: str
    metric_names: list[str] | None = None
    sample_idx: int | None = None
    start_step: int | None = None
    end_step: int | None = None
    env_filters: list[str] | None = None  # Filter by environment names
    limit: int = 10000


class SampleDetailsRequest(BaseModel):
    run_path: str
    group_id: int
    sample_idx: int
    is_eval: bool = False


class SampleStatusKey(BaseModel):
    group_id: int
    sample_idx: int


class SampleStatusesRequest(BaseModel):
    run_path: str
    samples: list[SampleStatusKey]


class InferenceGroupEventsRequest(BaseModel):
    run_path: str
    group_id: int


class TrainerBreakdownEventsRequest(BaseModel):
    run_path: str
    rank: int
    step: int
    parent_event_type: str


class TimelinePaginatedRequest(BaseModel):
    run_path: str
    page: int = 0
    interval_seconds: float = 60.0


class SystemMetricsRequest(BaseModel):
    run_path: str
    metric_names: list[str] | None = None  # Filter by specific metrics
    gpu_index: int | None = None  # Filter by GPU index (for GPU metrics)
    node_id: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    source: str | None = None
    include_associated_system_metrics: bool = False
    include_edge_points: bool = False
    start_time: float | None = None  # Filter after this timestamp
    end_time: float | None = None  # Filter before this timestamp
    limit: int = 10000


class SystemMetricsPaginatedRequest(BaseModel):
    run_path: str
    page: int = 0
    interval_seconds: float = 60.0
    metric_names: list[str] | None = None  # Filter by specific metrics
    anchor_start_time: float | None = None  # Optional absolute anchor for page 0
    align_to_latest: bool = False  # Live mode: use [latest - interval, latest]


class VllmMetricsPaginatedRequest(BaseModel):
    run_path: str
    page: int = 0
    interval_seconds: float = 60.0
    metric_names: list[str] | None = None  # Filter by specific metrics (None = all)
    anchor_start_time: float | None = None  # Optional absolute anchor for page 0
    align_to_latest: bool = False  # Live mode: use [latest - interval, latest]


class StepMetricsRequest(BaseModel):
    run_path: str
    metric_names: list[str] | None = None  # Filter by specific metrics (e.g., reward_sum_mean)
    start_step: int | None = None  # Filter steps after this
    end_step: int | None = None  # Filter steps before this
    tag_filters: dict[str, list[str]] | None = None  # Filter rollout metrics by sample tags (tag_name -> [values])
    env_filters: list[str] | None = None  # Filter rollout metrics by environment names
    limit: int = 10000


class StepMetricsMultiRequest(BaseModel):
    run_paths: list[str]
    metric_names: list[str] | None = None  # Filter by specific metrics (e.g., reward_sum_mean)
    start_step: int | None = None  # Filter steps after this
    end_step: int | None = None  # Filter steps before this
    tag_filters: dict[str, list[str]] | None = None  # Filter rollout metrics by sample tags (tag_name -> [values])
    env_filters: list[str] | None = None  # Filter rollout metrics by environment names
    limit: int = 10000


class StepTimesRequest(BaseModel):
    run_path: str


class InferencePerformanceRequest(BaseModel):
    run_path: str
    bucket_seconds: int = 60


class TrainerPerformanceRequest(BaseModel):
    run_path: str
    bucket_seconds: int = 60


class StepHistogramRequest(BaseModel):
    run_path: str
    step: int
    metric_type: str  # e.g., "reward_sum", "advantage", "length_prompt", "length_completion", "length_sum", or "reward_<name>"


class StepDistributionOverTimeRequest(BaseModel):
    run_path: str
    metric_type: str  # e.g., "reward_sum", "advantage", "length_prompt", etc.
    num_bins: int = 10  # Number of value bins for the heatmap


@app.post("/sync")
async def sync_run(req: SyncRequest):
    """Start background sync of rollouts for a run. Returns immediately."""
    log.info(f"[API] Sync request for run: {req.run_path}")

    api_key = get_wandb_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="No W&B API key configured")

    # Clear drained flag when syncing starts
    con = connect()
    set_run_drained(con, req.run_path, False)
    con.close()

    result = await start_sync(req.run_path, api_key)
    
    if result["status"] == "already_syncing":
        return {"ok": True, "message": "Sync already in progress", "status": "syncing"}
    
    return {"ok": True, "message": "Sync started", "status": "syncing"}


@app.post("/sync-evals-after-training")
async def sync_evals_after_training(req: SyncRequest):
    """Start background sync of evals_after_training zip files. Returns immediately."""
    log.info(f"[API] Sync evals-after-training request for run: {req.run_path}")

    api_key = get_wandb_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="No W&B API key configured")

    result = await start_evals_after_training_sync(req.run_path, api_key)

    if result["status"] == "already_syncing":
        return {"ok": True, "message": "Evals-after-training sync already in progress", "status": "syncing"}

    return {"ok": True, "message": "Evals-after-training sync started", "status": "syncing"}


@app.post("/stop-tracking")
def stop_tracking(req: StopTrackingRequest):
    """Stop tracking a wandb run."""
    log.info(f"[API] Stopping tracking for run: {req.run_path}")
    clear_active_run(req.run_path)
    return {"ok": True, "message": f"Stopped tracking {req.run_path}"}


@app.post("/wandb-config")
async def wandb_config(req: WandbConfigRequest):
    """Store W&B API key. Discovery happens when the user adds a project."""
    if req.use_netrc:
        api_key = (get_wandb_key_from_netrc() or "").strip()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="No W&B API key found in ~/.netrc – run `wandb login` first",
            )
        set_wandb_key_source("netrc")
        delete_wandb_api_key()
    else:
        api_key = (req.api_key or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Missing W&B API key")
        set_wandb_api_key(api_key=api_key)

    asyncio.create_task(configure_wandb_and_sync(api_key, tag="telescope"))

    return {
        "ok": True,
        "message": "Saved W&B config and started syncing tagged runs",
    }


@app.get("/custom-metrics-layout")
def get_custom_layout():
    """Get the global custom metrics dashboard layout."""
    con = connect()
    layout = get_custom_metrics_layout(con)
    con.close()
    return {"layout": layout}


class CustomMetricsLayoutRequest(BaseModel):
    layout: dict


@app.put("/custom-metrics-layout")
def put_custom_layout(req: CustomMetricsLayoutRequest):
    """Save/update the global custom metrics dashboard layout."""
    con = connect()
    set_custom_metrics_layout(con, req.layout)
    con.close()
    return {"ok": True}


@app.get("/custom-metrics-templates")
def list_templates():
    """List all saved custom metrics templates."""
    con = connect()
    templates = list_custom_metrics_templates(con)
    con.close()
    return {"templates": templates}


class CreateTemplateRequest(BaseModel):
    name: str
    layout: dict


@app.post("/custom-metrics-templates")
def create_template(req: CreateTemplateRequest):
    """Create a new custom metrics template."""
    con = connect()
    template_id = uuid.uuid4().hex[:12]
    create_custom_metrics_template(con, template_id, req.name, req.layout)
    con.close()
    return {"id": template_id, "name": req.name}


class UpdateTemplateLayoutRequest(BaseModel):
    layout: dict


@app.put("/custom-metrics-templates/{template_id}")
def update_template(template_id: str, req: UpdateTemplateLayoutRequest):
    """Update a template's layout."""
    con = connect()
    tmpl = get_custom_metrics_template(con, template_id)
    if not tmpl:
        con.close()
        raise HTTPException(status_code=404, detail="Template not found")
    update_custom_metrics_template_layout(con, template_id, req.layout)
    con.close()
    return {"ok": True}


@app.get("/custom-metrics-templates/{template_id}")
def get_template(template_id: str):
    """Get a single template by ID."""
    con = connect()
    tmpl = get_custom_metrics_template(con, template_id)
    con.close()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return tmpl


class RenameTemplateRequest(BaseModel):
    name: str


@app.patch("/custom-metrics-templates/{template_id}")
def rename_template(template_id: str, req: RenameTemplateRequest):
    """Rename a custom metrics template."""
    con = connect()
    tmpl = get_custom_metrics_template(con, template_id)
    if not tmpl:
        con.close()
        raise HTTPException(status_code=404, detail="Template not found")
    rename_custom_metrics_template(con, template_id, req.name)
    con.close()
    return {"ok": True}


@app.delete("/custom-metrics-templates/{template_id}")
def delete_template(template_id: str):
    """Delete a custom metrics template."""
    con = connect()
    delete_custom_metrics_template(con, template_id)
    con.close()
    return {"ok": True}


@app.get("/known-projects")
def get_known_projects():
    """List all known projects (user-added and derived from runs)."""
    projects = get_all_known_projects()
    return {"projects": projects}


@app.post("/add-project")
async def add_project(req: AddProjectRequest):
    """Add a project to the known projects list and discover its tagged runs."""
    project = req.project.strip()
    if not project:
        raise HTTPException(status_code=400, detail="Missing project path")
    parts = project.split("/")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Expected 'entity/project'",
        )
    is_new = add_project_to_known(project)

    # Trigger background discovery for this project so runs appear automatically.
    # Set discovery status synchronously so the next /runs poll sees it
    # before the background task has a chance to start.
    api_key = get_wandb_api_key()
    if api_key and is_new:
        set_discovery_status_discovering()
        asyncio.create_task(discover_and_sync_project(api_key, project))

    return {
        "ok": True,
        "is_new": is_new,
        "message": f"{'Added' if is_new else 'Already known'}: {project}",
    }


@app.post("/remove-project")
def remove_project(req: RemoveProjectRequest):
    """Remove a user-added project from the known projects list."""
    project = req.project.strip()
    if not project:
        raise HTTPException(status_code=400, detail="Missing project path")
    removed = remove_project_from_known(project)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project}' is not a user-added project",
        )
    return {"ok": True, "message": f"Removed project: {project}"}


@app.post("/add-run")
async def add_run(req: AddRunRequest):
    """Add a run back to the list and sync it using the stored API key."""
    run_path = req.run_path.strip()
    if not run_path:
        raise HTTPException(status_code=400, detail="Missing run path")

    api_key = get_wandb_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="W&B API key not configured")

    try:
        await asyncio.to_thread(fetch_and_store_run, api_key, run_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch run metadata: {e}",
        ) from e

    con = connect()
    set_run_removed(con, run_path, False)
    set_run_drained(con, run_path, False)
    con.close()

    await start_sync(run_path, api_key)

    return {"ok": True, "message": f"Added run {run_path} and started sync"}


@app.post("/remove-run")
def remove_run(req: RemoveRunRequest):
    """Remove a run from the list while keeping its metadata row."""
    run_path = req.run_path
    log.info(f"[API] Removing run and data for: {run_path}")

    clear_active_run(run_path)

    con = connect()

    deleted_counts = {}
    for table in _TABLES_WITH_RUN_ID:
        try:
            count_result = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?",
                [run_path],
            ).fetchone()
            count = count_result[0] if count_result else 0
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_path])
            deleted_counts[table] = count
        except Exception as e:
            log.warning(f"[API] Failed to delete from {table}: {e}")
            deleted_counts[table] = 0

    set_run_removed(con, run_path, True)
    con.close()
    _delete_local_run_artifacts(run_path)

    return {
        "ok": True,
        "message": f"Removed run {run_path}",
        "deleted_counts": deleted_counts,
    }


@app.post("/drain-run")
def drain_run(req: DrainRunRequest):
    """Drain a run: delete its data but keep it visible in the sidebar."""
    run_path = req.run_path
    log.info(f"[API] Draining run data for: {run_path}")

    clear_active_run(run_path)

    con = connect()

    deleted_counts = {}
    for table in _TABLES_WITH_RUN_ID:
        try:
            count_result = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?",
                [run_path],
            ).fetchone()
            count = count_result[0] if count_result else 0
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_path])
            deleted_counts[table] = count
        except Exception as e:
            log.warning(f"[API] Failed to delete from {table}: {e}")
            deleted_counts[table] = 0

    set_run_drained(con, run_path, True)
    con.close()
    _delete_local_run_artifacts(run_path)

    return {
        "ok": True,
        "message": f"Drained run {run_path}",
        "deleted_counts": deleted_counts,
    }


@app.post("/set-run-color")
def set_run_color(req: SetRunColorRequest):
    """Set a custom display color for a run."""
    run_path = req.run_path.strip()
    color = req.color.strip()
    if not run_path:
        raise HTTPException(status_code=400, detail="Missing run path")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise HTTPException(
            status_code=400,
            detail="Invalid color format; expected #RRGGBB",
        )

    con = connect()
    row = con.execute(
        "SELECT run_id FROM runs WHERE run_id = ?",
        [run_path],
    ).fetchone()
    if row is None:
        con.close()
        raise HTTPException(status_code=404, detail="Run not found")

    update_run_color(con, run_path, color.lower())
    con.close()
    return {"ok": True, "run_path": run_path, "color": color.lower()}


@app.post("/rename-run")
def rename_run(req: RenameRunRequest):
    """Rename a run locally and on W&B."""
    run_path = req.run_path.strip()
    new_name = req.name.strip()
    if not run_path:
        raise HTTPException(status_code=400, detail="Missing run path")
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    con = connect()
    row = con.execute(
        "SELECT run_id FROM runs WHERE run_id = ?",
        [run_path],
    ).fetchone()
    if row is None:
        con.close()
        raise HTTPException(status_code=404, detail="Run not found")

    # Update locally first
    update_run_name(con, run_path, new_name)
    con.close()

    # Update on W&B in the background
    api_key = get_wandb_api_key()
    if api_key:
        try:
            api = _get_wandb_api(api_key)
            wandb_run = api.run(run_path)
            wandb_run.name = new_name
            wandb_run.update()
            log.info(f"[API] Renamed run {run_path} to '{new_name}' (local + W&B)")
        except Exception as e:
            log.warning(f"[API] Renamed run {run_path} locally but W&B update failed: {e}")
    else:
        log.info(f"[API] Renamed run {run_path} to '{new_name}' (local only, no W&B key)")

    return {"ok": True, "run_path": run_path, "name": new_name}


@app.post("/update-notes")
def update_notes(req: UpdateNotesRequest):
    """Update notes for a run locally and on W&B."""
    run_path = req.run_path.strip()
    notes = req.notes

    if not run_path:
        raise HTTPException(status_code=400, detail="Missing run path")

    con = connect()
    row = con.execute(
        "SELECT run_id FROM runs WHERE run_id = ?",
        [run_path],
    ).fetchone()
    if row is None:
        con.close()
        raise HTTPException(status_code=404, detail="Run not found")

    update_run_notes(con, run_path, notes)
    con.close()

    # Update on W&B in the background
    api_key = get_wandb_api_key()
    if api_key:
        try:
            api = _get_wandb_api(api_key)
            wandb_run = api.run(run_path)
            wandb_run.notes = notes
            wandb_run.update()
            log.info(f"[API] Updated notes for run {run_path} (local + W&B)")
        except Exception as e:
            log.warning(f"[API] Updated notes for run {run_path} locally but W&B update failed: {e}")
    else:
        log.info(f"[API] Updated notes for run {run_path} (local only, no W&B key)")

    return {"ok": True, "run_path": run_path, "notes": notes}


@app.post("/delete-run-data")
def delete_run_data(req: DeleteRunDataRequest):
    """Delete all data for a specific run from the database."""
    run_path = req.run_path
    log.info(f"[API] Deleting all data for run: {run_path}")
    
    # Stop tracking if active
    clear_active_run(run_path)
    
    con = connect()
    
    # Preserve the run's color before deleting so it survives a resync
    existing_color = None
    try:
        result = con.execute("SELECT color FROM runs WHERE run_id = ?", [run_path]).fetchone()
        existing_color = result[0] if result else None
    except Exception:
        pass
    
    tables_to_delete = [*_TABLES_WITH_RUN_ID, "runs"]

    deleted_counts = {}
    for table in tables_to_delete:
        try:
            # Get count before delete
            count_result = con.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", [run_path]).fetchone()
            count = count_result[0] if count_result else 0
            
            # Delete
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_path])
            deleted_counts[table] = count
            log.info(f"[API] Deleted {count} rows from {table}")
        except Exception as e:
            log.warning(f"[API] Failed to delete from {table}: {e}")
            deleted_counts[table] = 0
    
    # Re-insert a minimal runs row with the preserved color so resync keeps it
    if existing_color:
        try:
            con.execute("INSERT INTO runs (run_id, color) VALUES (?, ?)", [run_path, existing_color])
            log.info(f"[API] Preserved color {existing_color} for run {run_path}")
        except Exception as e:
            log.warning(f"[API] Failed to preserve run color: {e}")
    
    total_deleted = sum(deleted_counts.values())
    log.info(f"[API] Total deleted: {total_deleted} rows across {len(tables_to_delete)} tables")
    if total_deleted > 0:
        log.info("[API] Running VACUUM to reclaim space after deletion...")
        con.execute("VACUUM")
    con.close()
    _delete_local_run_artifacts(run_path)
    
    return {
        "ok": True,
        "message": f"Deleted all data for run {run_path}",
        "deleted_counts": deleted_counts,
        "total_deleted": total_deleted,
    }


@app.get("/runs")
def list_runs():
    """List all runs that have been synced/tracked, ordered by created_at (newest first)."""
    log.debug("[API] Listing all runs from runs table")
    con = connect()
    
    # Get runs from the new runs table, ordered by created_at descending
    rows = con.execute("""
        SELECT r.run_id, r.name, r.created_at, r.state, r.entity, r.project, r.url,
               COALESCE(i.last_rollout_step, -1) as last_rollout_step,
               r.color, r.trainer_commit, r.schema_version, r.notes,
               COALESCE(r.drained, FALSE) as drained
        FROM runs r
        LEFT JOIN ingest_state i ON r.run_id = i.run_id
        WHERE COALESCE(r.removed, FALSE) = FALSE
        ORDER BY r.created_at DESC NULLS LAST
    """).fetchall()

    runs = []
    for row in rows:
        run_id = row[0]
        runs.append({
            "run_id": run_id,
            "name": row[1],
            "created_at": row[2],
            "state": row[3],
            "entity": row[4],
            "project": row[5],
            "url": row[6],
            "last_rollout_step": row[7],
            "is_tracking": is_tracking(run_id),
            "is_syncing": is_syncing(run_id) or is_syncing_evals_after_training(run_id),
            "color": row[8],
            "trainer_commit": row[9],
            "schema_version": row[10],
            "notes": row[11],
            "is_drained": bool(row[12]),
        })
    
    log.debug(f"[API] Found {len(runs)} runs in database")
    has_wandb_key = bool((get_wandb_api_key() or "").strip())
    wandb_key_source = get_wandb_key_source()
    has_netrc_wandb_key = bool((get_wandb_key_from_netrc() or "").strip())
    con.close()
    discovery = get_discovery_status()
    return {
        "runs": runs,
        "discovery": discovery,
        "has_wandb_key": has_wandb_key,
        "wandb_key_source": wandb_key_source,
        "has_netrc_wandb_key": has_netrc_wandb_key,
        "has_known_projects": has_known_projects(),
    }


@app.get("/removed-runs")
def list_removed_runs():
    """List all runs that were removed locally."""
    log.debug("[API] Listing removed runs from runs table")
    con = connect()

    rows = con.execute("""
        SELECT r.run_id, r.name, r.created_at, r.state, r.entity, r.project, r.url,
               r.color, r.removed_at, r.config_json, r.trainer_commit, r.schema_version
        FROM runs r
        WHERE COALESCE(r.removed, FALSE) = TRUE
        ORDER BY r.removed_at DESC NULLS LAST, r.created_at DESC NULLS LAST
    """).fetchall()

    runs = [
        {
            "run_id": row[0],
            "name": row[1],
            "created_at": row[2],
            "state": row[3],
            "entity": row[4],
            "project": row[5],
            "url": row[6],
            "color": row[7],
            "removed_at": row[8],
            "config": json.loads(row[9]) if row[9] else None,
            "trainer_commit": row[10],
            "schema_version": row[11],
        }
        for row in rows
    ]

    return {"runs": runs}


class ConfigSearchRequest(BaseModel):
    filters: str  # e.g. "model=PrimeIntellect/Qwen3-1.7B-Wordle-SFT,seq_len=8192"


@app.post("/search-runs-by-config")
def search_runs_by_config(req: ConfigSearchRequest):
    """Search non-removed runs whose config_json matches ALL key=value filters.

    The *filters* string is a comma-separated list of ``key=value`` pairs.
    Keys support dot-notation for nested lookups (e.g. ``optimizer.lr=1e-4``).
    A run matches only if every filter pair is found (AND logic).
    Returns the list of matching run_ids.
    """
    log.info(f"[API] Config search: {req.filters}")

    # Normalise: lowercase + spaces → underscores
    def _norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    # Parse filters -------------------------------------------------------
    pairs: list[tuple[str, str]] = []
    for token in req.filters.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        key = _norm(key)
        value = _norm(value)
        if key:
            pairs.append((key, value))

    if not pairs:
        raise HTTPException(status_code=400, detail="No valid key=value filters provided")

    # Fetch all non-removed runs with config_json -------------------------
    con = connect()
    rows = con.execute("""
        SELECT run_id, config_json
        FROM runs
        WHERE COALESCE(removed, FALSE) = FALSE
          AND config_json IS NOT NULL
    """).fetchall()
    con.close()

    # Helper: resolve a dot-separated key in a nested dict (case-insensitive)
    def _resolve(cfg: dict, dotted_key: str):
        parts = dotted_key.split(".")
        cur = cfg
        for p in parts:
            if isinstance(cur, dict):
                # Case-insensitive + space/underscore-normalised key lookup
                found = False
                for k, v in cur.items():
                    if _norm(k) == p:
                        cur = v
                        found = True
                        break
                if not found:
                    return None
            else:
                return None
        # W&B configs often wrap values as {"value": X}
        if isinstance(cur, dict) and "value" in cur:
            cur = cur["value"]
        return cur

    # Pre-compile value patterns (support regex) ---------------------------
    import re as _re

    compiled: list[tuple[str, _re.Pattern]] = []
    for key, value in pairs:
        try:
            # Treat value as a regex; anchor with ^ and $ for exact match
            # unless the user already includes regex meta-chars
            pattern = _re.compile(value)
        except _re.error:
            # If the value isn't valid regex, escape it for literal match
            pattern = _re.compile(_re.escape(value))
        compiled.append((key, pattern))

    # Filter runs ---------------------------------------------------------
    matching_ids: list[str] = []
    for run_id, config_json_str in rows:
        try:
            cfg = json.loads(config_json_str)
        except Exception:
            continue

        match = True
        for key, pattern in compiled:
            resolved = _resolve(cfg, key)
            if resolved is None:
                match = False
                break
            if not pattern.search(_norm(str(resolved))):
                match = False
                break
        if match:
            matching_ids.append(run_id)

    log.info(f"[API] Config search matched {len(matching_ids)}/{len(rows)} runs")
    return {"run_ids": matching_ids}


@app.post("/rollouts")
def get_rollouts(req: RolloutsRequest):
    """Get rollouts data for a run, including prompts and rollout metrics.
    
    Returns:
    - prompts: One row per prompt group (step, group_id, env, prompt, tokens_prompt)
    - rollouts: One row per turn per sample (step, group_id, sample_idx, turn_order, turn_type, content, tokens)
    - samples_data: One row per sample (step, group_id, sample_idx, reward, advantage, turns, total_tokens)
    - rollout_metrics: One row per metric per sample (step, sample_idx, env, metric_name, value)
    - golden_answers: One row per golden answer key per sample (step, sample_idx, env, key, value)
    """
    log.info(f"[API] Getting rollouts for {req.run_path}, step={req.step}")
    con = connect()
    
    # Determine which step to fetch
    actual_step = req.step
    if actual_step is None:
        max_step_result = con.execute(
            "SELECT MAX(step) FROM rollouts WHERE run_id = ?",
            [req.run_path],
        ).fetchone()
        actual_step = max_step_result[0] if max_step_result and max_step_result[0] is not None else None
    
    if actual_step is None:
        return {
            "prompts": [],
            "rollouts": [],
            "samples_data": [],
            "rollout_metrics": [],
            "golden_answers": [],
            "sample_tags": [],
            "info_turns": [],
            "available_steps": [],
            "total_steps": 0,
            "current_step": None,
            "available_rollout_metric_names": [],
            "available_envs": [],
        }

    # Fetch prompts
    prompt_rows = con.execute(
        """
        SELECT step, group_id, env, system_prompt, tokens_system_prompt, prompt, tokens_prompt
        FROM prompts
        WHERE run_id = ? AND step = ?
        ORDER BY group_id
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    prompts = [
        {
            "step": row[0],
            "group_id": row[1],
            "env": row[2],
            "system_prompt": row[3],
            "tokens_system_prompt": row[4],
            "prompt": row[5],
            "tokens_prompt": row[6],
        }
        for row in prompt_rows
    ]
    
    # Fetch rollout turns for this step (no limit — data is bounded by step)
    rows = con.execute(
        """
        SELECT step, group_id, sample_idx, turn_order, turn_type, content, tokens, stop_reason
        FROM rollouts
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx, turn_order
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    rollouts = [
        {
            "step": row[0],
            "group_id": row[1],
            "sample_idx": row[2],
            "turn_order": row[3],
            "turn_type": row[4],
            "content": decompress_blob(row[5]),
            "tokens": row[6],
            "stop_reason": row[7],
        }
        for row in rows
    ]
    
    # Fetch samples data (reward, advantage per sample)
    samples_rows = con.execute(
        """
        SELECT step, group_id, sample_idx, reward, advantage, turns, total_tokens, raw_string
        FROM samples_data
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    samples_data = [
        {
            "step": row[0],
            "group_id": row[1],
            "sample_idx": row[2],
            "reward": row[3],
            "advantage": row[4],
            "turns": row[5],
            "total_tokens": row[6],
            "raw_string": decompress_blob(row[7]),
        }
        for row in samples_rows
    ]
    
    # Fetch rollout metrics for this step
    metrics_rows = con.execute(
        """
        SELECT step, sample_idx, env, metric_name, value
        FROM rollouts_metrics
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx, metric_name
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    rollout_metrics = [
        {
            "step": row[0],
            "sample_idx": row[1],
            "env": row[2],
            "metric_name": row[3],
            "value": row[4],
        }
        for row in metrics_rows
    ]

    # Fetch golden answers for this step
    golden_rows = con.execute(
        """
        SELECT step, sample_idx, env, key, value
        FROM golden_answers
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx, key
        """,
        [req.run_path, actual_step],
    ).fetchall()

    golden_answers = [
        {
            "step": row[0],
            "sample_idx": row[1],
            "env": row[2],
            "key": row[3],
            "value": row[4],
        }
        for row in golden_rows
    ]

    # Fetch sample tags for this step
    tags_rows = con.execute(
        """
        SELECT step, sample_idx, env, tag_name, tag_value
        FROM sample_tags
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx, tag_name
        """,
        [req.run_path, actual_step],
    ).fetchall()

    sample_tags = [
        {
            "step": row[0],
            "sample_idx": row[1],
            "env": row[2],
            "tag_name": row[3],
            "tag_value": row[4],
        }
        for row in tags_rows
    ]

    # Fetch info turns for this step
    info_rows = con.execute(
        """
        SELECT step, sample_idx, turn_order, env, info_key, info_value, info_type
        FROM info_turns
        WHERE run_id = ? AND step = ?
        ORDER BY sample_idx, turn_order, info_key
        """,
        [req.run_path, actual_step],
    ).fetchall()

    info_turns = [
        {
            "step": row[0],
            "sample_idx": row[1],
            "turn_order": row[2],
            "env": row[3],
            "info_key": row[4],
            "info_value": row[5],
            "info_type": row[6],
        }
        for row in info_rows
    ]
    
    # Get available steps
    steps_result = con.execute(
        """
        SELECT DISTINCT step FROM rollouts WHERE run_id = ? ORDER BY step
        """,
        [req.run_path],
    ).fetchall()
    available_steps = [r[0] for r in steps_result]
    
    # Get available rollout metric names across the entire run
    metric_names_result = con.execute(
        """
        SELECT DISTINCT metric_name FROM rollouts_metrics WHERE run_id = ? ORDER BY metric_name
        """,
        [req.run_path],
    ).fetchall()
    available_rollout_metric_names = [r[0] for r in metric_names_result]
    
    # Get available environments across the entire run (from prompts table)
    envs_result = con.execute(
        """
        SELECT DISTINCT env FROM prompts WHERE run_id = ? AND env IS NOT NULL ORDER BY env
        """,
        [req.run_path],
    ).fetchall()
    available_envs = [r[0] for r in envs_result]
    
    log.info(
        f"[API] Returning {len(prompts)} prompts, {len(rollouts)} rollout turns, "
        f"{len(samples_data)} samples data, {len(rollout_metrics)} metrics, "
        f"{len(golden_answers)} golden answers, {len(sample_tags)} sample tags, "
        f"{len(info_turns)} info turns, {len(available_steps)} steps available"
    )

    return {
        "prompts": prompts,
        "rollouts": rollouts,
        "samples_data": samples_data,
        "rollout_metrics": rollout_metrics,
        "golden_answers": golden_answers,
        "sample_tags": sample_tags,
        "info_turns": info_turns,
        "available_steps": available_steps,
        "total_steps": len(available_steps),
        "current_step": actual_step,
        "available_rollout_metric_names": available_rollout_metric_names,
        "available_envs": available_envs,
    }


@app.post("/rollouts-discarded")
def get_rollouts_discarded(req: RolloutsDiscardedRequest):
    """Get discarded rollouts data for a run, paginated by trainer_step.
    
    Returns:
    - prompts: Discarded prompts (step, group_id, env, prompt, tokens_prompt)
    - rollouts: Discarded rollout turns (step, group_id, sample_idx, turn_order, turn_type, content, tokens)
    - samples_data: Discarded samples data (sample_idx, reward, advantage, turns, total_tokens)
    - rollout_metrics: Discarded rollout metrics (sample_idx, env, metric_name, value)
    - golden_answers: Discarded golden answers (sample_idx, env, key, value)
    """
    log.info(f"[API] Getting discarded rollouts for {req.run_path}, trainer_step={req.trainer_step}")
    con = connect()
    
    # Determine which trainer_step to fetch
    actual_step = req.trainer_step
    if actual_step is None:
        max_step_result = con.execute(
            "SELECT MAX(trainer_step) FROM rollouts_discarded WHERE run_id = ?",
            [req.run_path],
        ).fetchone()
        actual_step = max_step_result[0] if max_step_result and max_step_result[0] is not None else None
    
    if actual_step is None:
        return {
            "prompts": [],
            "rollouts": [],
            "samples_data": [],
            "rollout_metrics": [],
            "golden_answers": [],
            "sample_tags": [],
            "info_turns": [],
            "available_trainer_steps": [],
            "total_trainer_steps": 0,
            "current_trainer_step": None,
            "available_discard_reasons": [],
            "available_envs": [],
        }

    # Fetch discarded prompts
    prompt_rows = con.execute(
        """
        SELECT timestamp, discard_reason, trainer_step, inference_step, group_id, env, 
               system_prompt, tokens_system_prompt, prompt, tokens_prompt
        FROM prompts_discarded
        WHERE run_id = ? AND trainer_step = ?
        ORDER BY group_id
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    prompts = [
        {
            "timestamp": row[0],
            "discard_reason": row[1],
            "trainer_step": row[2],
            "inference_step": row[3],
            "group_id": row[4],
            "env": row[5],
            "system_prompt": row[6],
            "tokens_system_prompt": row[7],
            "prompt": row[8],
            "tokens_prompt": row[9],
        }
        for row in prompt_rows
    ]
    
    # Fetch discarded rollout turns for this step (no limit — data is bounded by step)
    rows = con.execute(
        """
        SELECT trainer_step, inference_step, group_id, sample_idx,
               turn_order, turn_type, content, tokens, stop_reason
        FROM rollouts_discarded
        WHERE run_id = ? AND trainer_step = ?
        ORDER BY sample_idx, turn_order
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    rollouts = [
        {
            "trainer_step": row[0],
            "inference_step": row[1],
            "group_id": row[2],
            "sample_idx": row[3],
            "turn_order": row[4],
            "turn_type": row[5],
            "content": decompress_blob(row[6]),
            "tokens": row[7],
            "stop_reason": row[8],
        }
        for row in rows
    ]
    
    # Fetch discarded samples data (reward, advantage per sample)
    samples_rows = con.execute(
        """
        SELECT timestamp, discard_reason, trainer_step, inference_step, group_id, sample_idx, 
               reward, advantage, turns, total_tokens, raw_string
        FROM samples_data_discarded
        WHERE run_id = ? AND trainer_step = ?
        ORDER BY sample_idx
        """,
        [req.run_path, actual_step],
    ).fetchall()
    
    samples_data = [
        {
            "timestamp": row[0],
            "discard_reason": row[1],
            "raw_string": decompress_blob(row[10]),
            "trainer_step": row[2],
            "inference_step": row[3],
            "group_id": row[4],
            "sample_idx": row[5],
            "reward": row[6],
            "advantage": row[7],
            "turns": row[8],
            "total_tokens": row[9],
        }
        for row in samples_rows
    ]
    
    # Fetch discarded rollout metrics for this trainer_step using sample_idx
    metrics_rows = con.execute(
        """
        SELECT sample_idx, env, metric_name, value, tail_idx
        FROM rollouts_metrics_discarded
        WHERE run_id = ? AND sample_idx IN (
            SELECT DISTINCT sample_idx FROM rollouts_discarded 
            WHERE run_id = ? AND trainer_step = ?
        )
        ORDER BY sample_idx, metric_name
        """,
        [req.run_path, req.run_path, actual_step],
    ).fetchall()
    
    rollout_metrics = [
        {
            "sample_idx": row[0],
            "env": row[1],
            "metric_name": row[2],
            "value": row[3],
            "tail_idx": row[4],
        }
        for row in metrics_rows
    ]

    # Fetch discarded golden answers for this trainer_step using sample_idx
    golden_rows = con.execute(
        """
        SELECT sample_idx, env, key, value, tail_idx
        FROM golden_answers_discarded
        WHERE run_id = ? AND sample_idx IN (
            SELECT DISTINCT sample_idx FROM rollouts_discarded
            WHERE run_id = ? AND trainer_step = ?
        )
        ORDER BY sample_idx, key
        """,
        [req.run_path, req.run_path, actual_step],
    ).fetchall()

    golden_answers = [
        {
            "sample_idx": row[0],
            "env": row[1],
            "key": row[2],
            "value": row[3],
            "tail_idx": row[4],
        }
        for row in golden_rows
    ]

    # Fetch discarded sample tags for this trainer_step using sample_idx
    tags_rows = con.execute(
        """
        SELECT sample_idx, env, tag_name, tag_value, tail_idx
        FROM sample_tags_discarded
        WHERE run_id = ? AND sample_idx IN (
            SELECT DISTINCT sample_idx FROM rollouts_discarded
            WHERE run_id = ? AND trainer_step = ?
        )
        ORDER BY sample_idx, tag_name
        """,
        [req.run_path, req.run_path, actual_step],
    ).fetchall()

    sample_tags = [
        {
            "sample_idx": row[0],
            "env": row[1],
            "tag_name": row[2],
            "tag_value": row[3],
            "tail_idx": row[4],
        }
        for row in tags_rows
    ]

    # Fetch discarded info turns for this trainer_step using sample_idx
    info_rows = con.execute(
        """
        SELECT sample_idx, turn_order, env, info_key, info_value, info_type, tail_idx
        FROM info_turns_discarded
        WHERE run_id = ? AND sample_idx IN (
            SELECT DISTINCT sample_idx FROM rollouts_discarded
            WHERE run_id = ? AND trainer_step = ?
        )
        ORDER BY sample_idx, turn_order, info_key
        """,
        [req.run_path, req.run_path, actual_step],
    ).fetchall()

    info_turns = [
        {
            "sample_idx": row[0],
            "turn_order": row[1],
            "env": row[2],
            "info_key": row[3],
            "info_value": row[4],
            "info_type": row[5],
            "tail_idx": row[6],
        }
        for row in info_rows
    ]

    # Get available trainer_steps
    steps_result = con.execute(
        """
        SELECT DISTINCT trainer_step FROM rollouts_discarded WHERE run_id = ? ORDER BY trainer_step
        """,
        [req.run_path],
    ).fetchall()
    available_trainer_steps = [r[0] for r in steps_result]
    
    # Get available discard reasons for this run
    reasons_result = con.execute(
        """
        SELECT DISTINCT discard_reason FROM samples_data_discarded WHERE run_id = ? ORDER BY discard_reason
        """,
        [req.run_path],
    ).fetchall()
    available_discard_reasons = [r[0] for r in reasons_result]
    
    # Get available environments across the entire run (from prompts_discarded table)
    envs_result = con.execute(
        """
        SELECT DISTINCT env FROM prompts_discarded WHERE run_id = ? AND env IS NOT NULL ORDER BY env
        """,
        [req.run_path],
    ).fetchall()
    available_envs = [r[0] for r in envs_result]
    
    log.info(
        f"[API] Returning {len(prompts)} discarded prompts, {len(rollouts)} discarded rollout turns, "
        f"{len(samples_data)} samples data, {len(rollout_metrics)} metrics, "
        f"{len(golden_answers)} golden answers, {len(sample_tags)} sample tags, "
        f"{len(info_turns)} info turns, "
        f"{len(available_trainer_steps)} trainer_steps available"
    )

    return {
        "prompts": prompts,
        "rollouts": rollouts,
        "samples_data": samples_data,
        "rollout_metrics": rollout_metrics,
        "golden_answers": golden_answers,
        "sample_tags": sample_tags,
        "info_turns": info_turns,
        "available_trainer_steps": available_trainer_steps,
        "total_trainer_steps": len(available_trainer_steps),
        "current_trainer_step": actual_step,
        "available_discard_reasons": available_discard_reasons,
        "available_envs": available_envs,
    }


@app.post("/evals")
def get_evals(req: EvalsRequest):
    """Get eval rollouts data for a run, filtered by step and eval_name."""
    log.info(f"[API] Getting evals for {req.run_path}, step={req.step}, eval_name={req.eval_name}")
    con = connect()

    empty_response = {
        "prompts": [],
        "rollouts": [],
        "samples_data": [],
        "rollout_metrics": [],
        "golden_answers": [],
        "sample_tags": [],
        "info_turns": [],
        "available_steps": [],
        "available_eval_names": [],
        "current_step": None,
        "current_eval_name": None,
        "available_rollout_metric_names": [],
    }

    # Determine which step to fetch
    actual_step = req.step
    if actual_step is None:
        result = con.execute(
            "SELECT MAX(step) FROM prompts_eval WHERE run_id = ?",
            [req.run_path],
        ).fetchone()
        actual_step = result[0] if result and result[0] is not None else None

    if actual_step is None:
        return empty_response

    # Get available eval names for this step
    eval_names_result = con.execute(
        "SELECT DISTINCT eval_name FROM prompts_eval WHERE run_id = ? AND step = ? ORDER BY eval_name",
        [req.run_path, actual_step],
    ).fetchall()
    available_eval_names = [r[0] for r in eval_names_result]

    # Determine eval_name
    actual_eval_name = req.eval_name
    if actual_eval_name is None or actual_eval_name not in available_eval_names:
        actual_eval_name = available_eval_names[0] if available_eval_names else None

    if actual_eval_name is None:
        # Return with available steps but no data
        steps_result = con.execute(
            "SELECT DISTINCT step FROM prompts_eval WHERE run_id = ? ORDER BY step",
            [req.run_path],
        ).fetchall()
        empty_response["available_steps"] = [r[0] for r in steps_result]
        empty_response["current_step"] = actual_step
        return empty_response

    # Fetch eval prompts
    prompt_rows = con.execute(
        """
        SELECT step, eval_name, model_step, sample_idx, env, prompt, tokens_prompt,
               system_prompt, tokens_system_prompt
        FROM prompts_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    prompts = [
        {
            "step": row[0],
            "eval_name": row[1],
            "model_step": row[2],
            "sample_idx": row[3],
            "env": row[4],
            "prompt": row[5],
            "tokens_prompt": row[6],
            "system_prompt": row[7],
            "tokens_system_prompt": row[8],
        }
        for row in prompt_rows
    ]

    # Fetch eval rollout turns
    rows = con.execute(
        """
        SELECT step, eval_name, model_step, sample_idx, completion_idx,
               turn_order, turn_type, content, tokens, stop_reason, environment_response_time
        FROM rollouts_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx, turn_order
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    rollouts = [
        {
            "step": row[0],
            "eval_name": row[1],
            "model_step": row[2],
            "sample_idx": row[3],
            "completion_idx": row[4],
            "turn_order": row[5],
            "turn_type": row[6],
            "content": decompress_blob(row[7]),
            "tokens": row[8],
            "stop_reason": row[9],
            "environment_response_time": row[10],
        }
        for row in rows
    ]

    # Fetch eval samples data
    samples_rows = con.execute(
        """
        SELECT step, eval_name, model_step, sample_idx, completion_idx, env,
               turns, compute_eval_metrics_time
        FROM samples_data_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    samples_data = [
        {
            "step": row[0],
            "eval_name": row[1],
            "model_step": row[2],
            "sample_idx": row[3],
            "completion_idx": row[4],
            "env": row[5],
            "turns": row[6],
            "compute_eval_metrics_time": row[7],
        }
        for row in samples_rows
    ]

    # Fetch eval rollout metrics
    metrics_rows = con.execute(
        """
        SELECT step, eval_name, sample_idx, completion_idx, env, metric_name, value
        FROM rollouts_metrics_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx, metric_name
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    rollout_metrics = [
        {
            "step": row[0],
            "eval_name": row[1],
            "sample_idx": row[2],
            "completion_idx": row[3],
            "env": row[4],
            "metric_name": row[5],
            "value": row[6],
        }
        for row in metrics_rows
    ]

    # Fetch eval golden answers
    golden_rows = con.execute(
        """
        SELECT step, eval_name, sample_idx, completion_idx, env, key, value
        FROM golden_answers_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx, key
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    golden_answers = [
        {
            "step": row[0],
            "eval_name": row[1],
            "sample_idx": row[2],
            "completion_idx": row[3],
            "env": row[4],
            "key": row[5],
            "value": row[6],
        }
        for row in golden_rows
    ]

    # Fetch eval sample tags
    tags_rows = con.execute(
        """
        SELECT step, eval_name, sample_idx, completion_idx, env, tag_name, tag_value
        FROM sample_tags_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx, tag_name
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    sample_tags = [
        {
            "step": row[0],
            "eval_name": row[1],
            "sample_idx": row[2],
            "completion_idx": row[3],
            "env": row[4],
            "tag_name": row[5],
            "tag_value": row[6],
        }
        for row in tags_rows
    ]

    # Fetch eval info turns
    info_rows = con.execute(
        """
        SELECT step, eval_name, sample_idx, completion_idx, turn_order, env,
               info_key, info_value, info_type
        FROM info_turns_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY sample_idx, completion_idx, turn_order, info_key
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()

    info_turns = [
        {
            "step": row[0],
            "eval_name": row[1],
            "sample_idx": row[2],
            "completion_idx": row[3],
            "turn_order": row[4],
            "env": row[5],
            "info_key": row[6],
            "info_value": row[7],
            "info_type": row[8],
        }
        for row in info_rows
    ]

    # Get all available steps (globally)
    steps_result = con.execute(
        "SELECT DISTINCT step FROM prompts_eval WHERE run_id = ? ORDER BY step",
        [req.run_path],
    ).fetchall()
    available_steps = [r[0] for r in steps_result]

    # Get available rollout metric names for this eval
    metric_names_result = con.execute(
        """
        SELECT DISTINCT metric_name FROM rollouts_metrics_eval
        WHERE run_id = ? AND step = ? AND eval_name = ?
        ORDER BY metric_name
        """,
        [req.run_path, actual_step, actual_eval_name],
    ).fetchall()
    available_rollout_metric_names = [r[0] for r in metric_names_result]

    log.info(
        f"[API] Returning eval data: {len(prompts)} prompts, {len(rollouts)} rollout turns, "
        f"{len(samples_data)} samples data, {len(rollout_metrics)} metrics, "
        f"{len(golden_answers)} golden answers, {len(sample_tags)} sample tags, "
        f"{len(info_turns)} info turns "
        f"for eval={actual_eval_name}, step={actual_step}"
    )

    return {
        "prompts": prompts,
        "rollouts": rollouts,
        "samples_data": samples_data,
        "rollout_metrics": rollout_metrics,
        "golden_answers": golden_answers,
        "sample_tags": sample_tags,
        "info_turns": info_turns,
        "available_steps": available_steps,
        "available_eval_names": available_eval_names,
        "current_step": actual_step,
        "current_eval_name": actual_eval_name,
        "available_rollout_metric_names": available_rollout_metric_names,
    }


@app.post("/eval-step-metrics")
def get_eval_step_metrics(req: EvalStepMetricsRequest):
    """Get per-step eval metrics computed from eval tables.

    Computes aggregated stats (mean, std, min, max) per step for:
    - reward_<name>: dynamic metrics from rollouts_metrics_eval
    - length_prompt: prompt token count from prompts_eval
    - length_completion: completion token count from rollouts_eval (model turns)
    - length_sum: total token count from rollouts_eval (all turns)
    - stop_reason_length_pct: % of completions hitting length limit
    - num_samples, num_completions: counts per step
    """
    log.info(f"[API] Getting eval step metrics for {req.run_path}, eval={req.eval_name}, metrics={req.metric_names}, sample_idx={req.sample_idx}")
    con = connect()

    count_result = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT step), MIN(step), MAX(step) FROM rollouts_eval WHERE run_id = ? AND eval_name = ?",
        [req.run_path, req.eval_name],
    ).fetchone()

    total_rollouts = count_result[0]
    total_steps = count_result[1]
    min_step = count_result[2]
    max_step = count_result[3]

    if total_rollouts == 0:
        return {
            "metrics": [],
            "total_returned": 0,
            "available_metrics": [],
            "available_rollout_metric_names": [],
            "min_step": None,
            "max_step": None,
            "total_steps": 0,
        }

    step_filter = ""
    params = [req.run_path, req.eval_name]

    if req.sample_idx is not None:
        step_filter += " AND sample_idx = ?"
        params.append(req.sample_idx)
    if req.start_step is not None:
        step_filter += " AND step >= ?"
        params.append(req.start_step)
    if req.end_step is not None:
        step_filter += " AND step <= ?"
        params.append(req.end_step)

    metric_names_result = con.execute(
        "SELECT DISTINCT metric_name FROM rollouts_metrics_eval WHERE run_id = ? AND eval_name = ? ORDER BY metric_name",
        [req.run_path, req.eval_name],
    ).fetchall()
    available_rollout_metric_names = [r[0] for r in metric_names_result]

    # Build environment filter for eval queries.
    eval_env_filter_sql = ""
    eval_env_filter_params: list = []
    eval_env_filter_rollouts_sql = ""
    eval_env_filter_rollouts_params: list = []
    if req.env_filters:
        placeholders = ", ".join(["?"] * len(req.env_filters))
        # For tables with env column (prompts_eval, samples_data_eval, rollouts_metrics_eval)
        eval_env_filter_sql = f" AND env IN ({placeholders})"
        eval_env_filter_params = list(req.env_filters)
        # For rollouts_eval (no env column), filter via prompts_eval
        eval_env_filter_rollouts_sql = (
            f" AND (step, sample_idx) IN ("
            f"SELECT step, sample_idx FROM prompts_eval "
            f"WHERE run_id = ? AND eval_name = ? AND env IN ({placeholders}))"
        )
        eval_env_filter_rollouts_params = [req.run_path, req.eval_name] + list(req.env_filters)

    metrics_to_include = set(req.metric_names) if req.metric_names else None
    metrics = []

    completion_query = f"""
        SELECT
            step,
            AVG(completion_tokens) as length_completion_mean,
            STDDEV_SAMP(completion_tokens) as length_completion_std,
            MIN(completion_tokens) as length_completion_min,
            MAX(completion_tokens) as length_completion_max
        FROM (
            SELECT step, sample_idx, completion_idx, SUM(tokens) as completion_tokens
            FROM rollouts_eval
            WHERE run_id = ? AND eval_name = ? AND turn_type = 'model' {step_filter} {eval_env_filter_rollouts_sql}
            GROUP BY step, sample_idx, completion_idx
        )
        GROUP BY step
        ORDER BY step ASC
    """
    completion_rows = con.execute(completion_query, params + eval_env_filter_rollouts_params).fetchall()
    completion_columns = ["length_completion_mean", "length_completion_std", "length_completion_min", "length_completion_max"]
    for row in completion_rows:
        step = row[0]
        for i, col_name in enumerate(completion_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({"step": step, "metric_name": col_name, "value": float(value)})

    prompt_query = f"""
        SELECT
            step,
            AVG(tokens_prompt) as length_prompt_mean,
            STDDEV_SAMP(tokens_prompt) as length_prompt_std,
            MIN(tokens_prompt) as length_prompt_min,
            MAX(tokens_prompt) as length_prompt_max
        FROM prompts_eval
        WHERE run_id = ? AND eval_name = ? {step_filter} {eval_env_filter_sql}
        GROUP BY step
        ORDER BY step ASC
    """
    prompt_rows = con.execute(prompt_query, params + eval_env_filter_params).fetchall()
    prompt_columns = ["length_prompt_mean", "length_prompt_std", "length_prompt_min", "length_prompt_max"]
    for row in prompt_rows:
        step = row[0]
        for i, col_name in enumerate(prompt_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({"step": step, "metric_name": col_name, "value": float(value)})

    length_sum_query = f"""
        SELECT
            step,
            AVG(total_tokens) as length_sum_mean,
            STDDEV_SAMP(total_tokens) as length_sum_std,
            MIN(total_tokens) as length_sum_min,
            MAX(total_tokens) as length_sum_max
        FROM (
            SELECT step, sample_idx, completion_idx, SUM(tokens) as total_tokens
            FROM rollouts_eval
            WHERE run_id = ? AND eval_name = ? {step_filter} {eval_env_filter_rollouts_sql}
            GROUP BY step, sample_idx, completion_idx
        )
        GROUP BY step
        ORDER BY step ASC
    """
    length_sum_rows = con.execute(length_sum_query, params + eval_env_filter_rollouts_params).fetchall()
    length_sum_columns = ["length_sum_mean", "length_sum_std", "length_sum_min", "length_sum_max"]
    for row in length_sum_rows:
        step = row[0]
        for i, col_name in enumerate(length_sum_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({"step": step, "metric_name": col_name, "value": float(value)})

    stop_reason_query = f"""
        SELECT
            step,
            COUNT(DISTINCT CASE WHEN stop_reason = 'length' THEN sample_idx * 10000 + completion_idx END) * 100.0 /
                NULLIF(COUNT(DISTINCT sample_idx * 10000 + completion_idx), 0) as stop_reason_length_pct
        FROM rollouts_eval
        WHERE run_id = ? AND eval_name = ? AND turn_type = 'model' {step_filter} {eval_env_filter_rollouts_sql}
        GROUP BY step
        ORDER BY step ASC
    """
    stop_reason_rows = con.execute(stop_reason_query, params + eval_env_filter_rollouts_params).fetchall()
    for row in stop_reason_rows:
        step = row[0]
        if metrics_to_include is None or "stop_reason_length_pct" in metrics_to_include:
            if row[1] is not None:
                metrics.append({"step": step, "metric_name": "stop_reason_length_pct", "value": float(row[1])})

    count_query = f"""
        SELECT
            step,
            COUNT(DISTINCT sample_idx) as num_samples,
            COUNT(DISTINCT sample_idx * 10000 + completion_idx) as num_completions
        FROM rollouts_eval
        WHERE run_id = ? AND eval_name = ? {step_filter} {eval_env_filter_rollouts_sql}
        GROUP BY step
        ORDER BY step ASC
    """
    count_rows = con.execute(count_query, params + eval_env_filter_rollouts_params).fetchall()
    for row in count_rows:
        step = row[0]
        if metrics_to_include is None or "num_samples" in metrics_to_include:
            metrics.append({"step": step, "metric_name": "num_samples", "value": float(row[1])})
        if metrics_to_include is None or "num_completions" in metrics_to_include:
            metrics.append({"step": step, "metric_name": "num_completions", "value": float(row[2])})

    if available_rollout_metric_names:
        rollout_metric_query = f"""
            SELECT
                step,
                metric_name,
                AVG(value) as mean,
                STDDEV_SAMP(value) as std,
                MIN(value) as min_val,
                MAX(value) as max_val
            FROM rollouts_metrics_eval
            WHERE run_id = ? AND eval_name = ? {step_filter} {eval_env_filter_sql}
            GROUP BY step, metric_name
            ORDER BY step ASC, metric_name ASC
        """
        rollout_metric_rows = con.execute(rollout_metric_query, params + eval_env_filter_params).fetchall()
        for row in rollout_metric_rows:
            step = row[0]
            metric_base_name = row[1]
            stat_values = [
                (f"reward_{metric_base_name}_mean", row[2]),
                (f"reward_{metric_base_name}_std", row[3]),
                (f"reward_{metric_base_name}_min", row[4]),
                (f"reward_{metric_base_name}_max", row[5]),
            ]
            for metric_name, value in stat_values:
                if metrics_to_include is None or metric_name in metrics_to_include:
                    if value is not None:
                        metrics.append({"step": step, "metric_name": metric_name, "value": float(value)})

    available_metrics = []
    available_metrics.extend(completion_columns)
    available_metrics.extend(prompt_columns)
    available_metrics.extend(length_sum_columns)
    available_metrics.append("stop_reason_length_pct")
    available_metrics.extend(["num_samples", "num_completions"])
    for name in available_rollout_metric_names:
        available_metrics.extend([
            f"reward_{name}_mean",
            f"reward_{name}_std",
            f"reward_{name}_min",
            f"reward_{name}_max",
        ])

    log.info(
        f"[API] Returning {len(metrics)} eval step metrics from {total_steps} steps, "
        f"{len(available_rollout_metric_names)} rollout metric types"
    )

    return {
        "metrics": metrics,
        "total_returned": len(metrics),
        "available_metrics": available_metrics,
        "available_rollout_metric_names": available_rollout_metric_names,
        "min_step": min_step,
        "max_step": max_step,
        "total_steps": total_steps,
    }


@app.post("/sample-details")
def get_sample_details(req: SampleDetailsRequest):
    """Get a single sample's details from rollouts, discarded rollouts, or evals."""
    log.info(
        f"[API] Getting sample details for {req.run_path}, group_id={req.group_id}, sample_idx={req.sample_idx}, is_eval={req.is_eval}"
    )
    con = connect()

    # For eval samples: the inference event group_id is a global counter, not the
    # eval sample_idx. The step on eval inference events may be NULL. We resolve
    # the mapping by ranking the group_id among ALL eval groups across all steps,
    # then walking through eval steps to find (step, eval_name, sample_idx).
    if req.is_eval:
        # All eval group_ids for this run, sorted
        all_eval_groups = con.execute(
            """
            SELECT DISTINCT group_id
            FROM events_inference
            WHERE run_id = ? AND is_eval = TRUE
            ORDER BY group_id
            """,
            [req.run_path],
        ).fetchall()

        group_ids = [r[0] for r in all_eval_groups]
        try:
            global_eval_idx = group_ids.index(req.group_id)
        except ValueError:
            global_eval_idx = -1

        target_step = None
        target_eval_name = None
        target_sample_idx = None

        if global_eval_idx >= 0:
            # Per-step eval structure: [(step, eval_name, n_samples), ...] sorted
            # by step then eval_name so the ordering matches the group_id sequence.
            eval_steps_info = con.execute(
                """
                SELECT step, eval_name, COUNT(DISTINCT sample_idx) as n_samples
                FROM prompts_eval
                WHERE run_id = ?
                GROUP BY step, eval_name
                ORDER BY step, eval_name
                """,
                [req.run_path],
            ).fetchall()

            offset = 0
            for e_step, e_name, n_samples in eval_steps_info:
                if global_eval_idx < offset + n_samples:
                    target_step = e_step
                    target_eval_name = e_name
                    target_sample_idx = global_eval_idx - offset
                    break
                offset += n_samples

        if target_step is not None and target_eval_name is not None and target_sample_idx is not None:
            step = target_step
            eval_name = target_eval_name
            eval_sample_idx = target_sample_idx

            model_step_row = con.execute(
                """
                SELECT model_step FROM samples_data_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                LIMIT 1
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchone()
            model_step = model_step_row[0] if model_step_row else None

            prompt_rows = con.execute(
                """
                SELECT step, sample_idx, env, system_prompt, tokens_system_prompt, prompt, tokens_prompt
                FROM prompts_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                ORDER BY sample_idx
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            prompts = [
                {
                    "step": row[0],
                    "group_id": row[1],
                    "env": row[2],
                    "system_prompt": row[3],
                    "tokens_system_prompt": row[4],
                    "prompt": row[5],
                    "tokens_prompt": row[6],
                }
                for row in prompt_rows
            ]

            rollout_rows = con.execute(
                """
                SELECT step, sample_idx, completion_idx, turn_order, turn_type, content, tokens, stop_reason
                FROM rollouts_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                ORDER BY completion_idx, turn_order
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            rollouts = [
                {
                    "step": row[0],
                    "group_id": row[1],
                    "sample_idx": row[2],
                    "turn_order": row[3],
                    "turn_type": row[4],
                    "content": decompress_blob(row[5]),
                    "tokens": row[6],
                    "stop_reason": row[7],
                }
                for row in rollout_rows
            ]

            samples_rows = con.execute(
                """
                SELECT step, sample_idx, completion_idx, env, turns
                FROM samples_data_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            samples_data = [
                {
                    "step": row[0],
                    "group_id": row[1],
                    "sample_idx": row[2],
                    "reward": None,
                    "advantage": None,
                    "turns": row[4],
                    "total_tokens": None,
                    "raw_string": None,
                }
                for row in samples_rows
            ]

            metrics_rows = con.execute(
                """
                SELECT step, sample_idx, completion_idx, env, metric_name, value
                FROM rollouts_metrics_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                ORDER BY completion_idx, metric_name
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            rollout_metrics = [
                {
                    "step": row[0],
                    "sample_idx": row[2],
                    "env": row[3],
                    "metric_name": row[4],
                    "value": row[5],
                }
                for row in metrics_rows
            ]

            golden_rows = con.execute(
                """
                SELECT step, sample_idx, completion_idx, env, key, value
                FROM golden_answers_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                ORDER BY completion_idx, key
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            golden_answers = [
                {
                    "step": row[0],
                    "sample_idx": row[2],
                    "env": row[3],
                    "key": row[4],
                    "value": row[5],
                }
                for row in golden_rows
            ]

            info_rows = con.execute(
                """
                SELECT step, sample_idx, completion_idx, turn_order, env, info_key, info_value, info_type
                FROM info_turns_eval
                WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                ORDER BY completion_idx, turn_order, info_key
                """,
                [req.run_path, step, eval_name, eval_sample_idx],
            ).fetchall()

            info_turns = [
                {
                    "step": row[0],
                    "sample_idx": row[2],
                    "turn_order": row[3],
                    "env": row[4],
                    "info_key": row[5],
                    "info_value": row[6],
                    "info_type": row[7],
                }
                for row in info_rows
            ]

            log.info(
                f"[API] Resolved eval sample: group_id={req.group_id} → "
                f"eval_name={eval_name}, sample_idx={eval_sample_idx}, step={step}"
            )

            return {
                "kind": "eval",
                "step": step,
                "eval_name": eval_name,
                "model_step": model_step,
                "group_id": req.group_id,
                "sample_idx": req.sample_idx,
                "prompts": prompts,
                "rollouts": rollouts,
                "samples_data": samples_data,
                "rollout_metrics": rollout_metrics,
                "golden_answers": golden_answers,
                "info_turns": info_turns,
            }

    # Try non-discarded rollouts first
    step_row = con.execute(
        """
        SELECT step
        FROM samples_data
        WHERE run_id = ? AND group_id = ? AND sample_idx = ?
        ORDER BY step DESC
        LIMIT 1
        """,
        [req.run_path, req.group_id, req.sample_idx],
    ).fetchone()

    if step_row is None:
        step_row = con.execute(
            """
            SELECT step
            FROM rollouts
            WHERE run_id = ? AND group_id = ? AND sample_idx = ?
            ORDER BY step DESC
            LIMIT 1
            """,
            [req.run_path, req.group_id, req.sample_idx],
        ).fetchone()

    if step_row is not None:
        step = step_row[0]

        prompt_rows = con.execute(
            """
            SELECT step, group_id, env, system_prompt, tokens_system_prompt, prompt, tokens_prompt
            FROM prompts
            WHERE run_id = ? AND step = ? AND group_id = ?
            ORDER BY group_id
            """,
            [req.run_path, step, req.group_id],
        ).fetchall()

        prompts = [
            {
                "step": row[0],
                "group_id": row[1],
                "env": row[2],
                "system_prompt": row[3],
                "tokens_system_prompt": row[4],
                "prompt": row[5],
                "tokens_prompt": row[6],
            }
            for row in prompt_rows
        ]

        rollout_rows = con.execute(
            """
            SELECT step, group_id, sample_idx, turn_order, turn_type, content, tokens, stop_reason
            FROM rollouts
            WHERE run_id = ? AND step = ? AND group_id = ? AND sample_idx = ?
            ORDER BY turn_order
            """,
            [req.run_path, step, req.group_id, req.sample_idx],
        ).fetchall()

        rollouts = [
            {
                "step": row[0],
                "group_id": row[1],
                "sample_idx": row[2],
                "turn_order": row[3],
                "turn_type": row[4],
                "content": decompress_blob(row[5]),
                "tokens": row[6],
                "stop_reason": row[7],
            }
            for row in rollout_rows
        ]

        samples_rows = con.execute(
            """
            SELECT step, group_id, sample_idx, reward, advantage, turns, total_tokens, raw_string
            FROM samples_data
            WHERE run_id = ? AND step = ? AND group_id = ? AND sample_idx = ?
            """,
            [req.run_path, step, req.group_id, req.sample_idx],
        ).fetchall()

        samples_data = [
            {
                "step": row[0],
                "group_id": row[1],
                "sample_idx": row[2],
                "reward": row[3],
                "advantage": row[4],
                "turns": row[5],
                "total_tokens": row[6],
                "raw_string": decompress_blob(row[7]),
            }
            for row in samples_rows
        ]

        metrics_rows = con.execute(
            """
            SELECT step, sample_idx, env, metric_name, value
            FROM rollouts_metrics
            WHERE run_id = ? AND step = ? AND sample_idx = ?
            ORDER BY metric_name
            """,
            [req.run_path, step, req.sample_idx],
        ).fetchall()

        rollout_metrics = [
            {
                "step": row[0],
                "sample_idx": row[1],
                "env": row[2],
                "metric_name": row[3],
                "value": row[4],
            }
            for row in metrics_rows
        ]

        golden_rows = con.execute(
            """
            SELECT step, sample_idx, env, key, value
            FROM golden_answers
            WHERE run_id = ? AND step = ? AND sample_idx = ?
            ORDER BY key
            """,
            [req.run_path, step, req.sample_idx],
        ).fetchall()

        golden_answers = [
            {
                "step": row[0],
                "sample_idx": row[1],
                "env": row[2],
                "key": row[3],
                "value": row[4],
            }
            for row in golden_rows
        ]

        info_rows = con.execute(
            """
            SELECT step, sample_idx, turn_order, env, info_key, info_value, info_type
            FROM info_turns
            WHERE run_id = ? AND step = ? AND sample_idx = ?
            ORDER BY turn_order, info_key
            """,
            [req.run_path, step, req.sample_idx],
        ).fetchall()

        info_turns = [
            {
                "step": row[0],
                "sample_idx": row[1],
                "turn_order": row[2],
                "env": row[3],
                "info_key": row[4],
                "info_value": row[5],
                "info_type": row[6],
            }
            for row in info_rows
        ]

        return {
            "kind": "rollouts",
            "step": step,
            "group_id": req.group_id,
            "sample_idx": req.sample_idx,
            "prompts": prompts,
            "rollouts": rollouts,
            "samples_data": samples_data,
            "rollout_metrics": rollout_metrics,
            "golden_answers": golden_answers,
            "info_turns": info_turns,
        }

    # Fallback to discarded rollouts
    discarded_row = con.execute(
        """
        SELECT trainer_step, inference_step, discard_reason
        FROM samples_data_discarded
        WHERE run_id = ? AND group_id = ? AND sample_idx = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        [req.run_path, req.group_id, req.sample_idx],
    ).fetchone()

    if discarded_row is None:
        gen_discarded_row = con.execute(
            """
            SELECT trainer_step, inference_step
            FROM rollouts_discarded
            WHERE run_id = ? AND group_id = ? AND sample_idx = ?
            LIMIT 1
            """,
            [req.run_path, req.group_id, req.sample_idx],
        ).fetchone()
        if gen_discarded_row is not None:
            discarded_row = (gen_discarded_row[0], gen_discarded_row[1], "unknown")

    if discarded_row is None:
        return {
            "kind": None,
            "group_id": req.group_id,
            "sample_idx": req.sample_idx,
            "prompts": [],
            "rollouts": [],
            "samples_data": [],
            "rollout_metrics": [],
            "golden_answers": [],
            "sample_tags": [],
            "info_turns": [],
        }

    trainer_step = discarded_row[0]
    inference_step = discarded_row[1]
    discard_reason = discarded_row[2]

    prompt_rows = con.execute(
        """
        SELECT timestamp, discard_reason, trainer_step, inference_step, group_id, env,
               system_prompt, tokens_system_prompt, prompt, tokens_prompt
        FROM prompts_discarded
        WHERE run_id = ? AND trainer_step = ? AND group_id = ?
        ORDER BY timestamp DESC
        """,
        [req.run_path, trainer_step, req.group_id],
    ).fetchall()

    prompts = [
        {
            "timestamp": row[0],
            "discard_reason": row[1],
            "trainer_step": row[2],
            "inference_step": row[3],
            "group_id": row[4],
            "env": row[5],
            "system_prompt": row[6],
            "tokens_system_prompt": row[7],
            "prompt": row[8],
            "tokens_prompt": row[9],
        }
        for row in prompt_rows
    ]

    rollout_rows = con.execute(
        """
        SELECT trainer_step, inference_step, group_id, sample_idx,
               turn_order, turn_type, content, tokens, stop_reason
        FROM rollouts_discarded
        WHERE run_id = ? AND trainer_step = ? AND group_id = ? AND sample_idx = ?
        ORDER BY turn_order
        """,
        [req.run_path, trainer_step, req.group_id, req.sample_idx],
    ).fetchall()

    rollouts = [
        {
            "trainer_step": row[0],
            "inference_step": row[1],
            "group_id": row[2],
            "sample_idx": row[3],
            "turn_order": row[4],
            "turn_type": row[5],
            "content": decompress_blob(row[6]),
            "tokens": row[7],
            "stop_reason": row[8],
        }
        for row in rollout_rows
    ]

    samples_rows = con.execute(
        """
        SELECT timestamp, discard_reason, trainer_step, inference_step, group_id, sample_idx,
               reward, advantage, turns, total_tokens, raw_string
        FROM samples_data_discarded
        WHERE run_id = ? AND trainer_step = ? AND group_id = ? AND sample_idx = ?
        """,
        [req.run_path, trainer_step, req.group_id, req.sample_idx],
    ).fetchall()

    samples_data = [
        {
            "timestamp": row[0],
            "discard_reason": row[1],
            "trainer_step": row[2],
            "inference_step": row[3],
            "group_id": row[4],
            "sample_idx": row[5],
            "reward": row[6],
            "advantage": row[7],
            "turns": row[8],
            "total_tokens": row[9],
            "raw_string": decompress_blob(row[10]),
        }
        for row in samples_rows
    ]

    metrics_rows = con.execute(
        """
        SELECT sample_idx, env, metric_name, value, tail_idx
        FROM rollouts_metrics_discarded
        WHERE run_id = ? AND sample_idx = ?
        ORDER BY metric_name
        """,
        [req.run_path, req.sample_idx],
    ).fetchall()

    rollout_metrics = [
        {
            "sample_idx": row[0],
            "env": row[1],
            "metric_name": row[2],
            "value": row[3],
            "tail_idx": row[4],
        }
        for row in metrics_rows
    ]

    golden_rows = con.execute(
        """
        SELECT sample_idx, env, key, value, tail_idx
        FROM golden_answers_discarded
        WHERE run_id = ? AND sample_idx = ?
        ORDER BY key
        """,
        [req.run_path, req.sample_idx],
    ).fetchall()

    golden_answers = [
        {
            "sample_idx": row[0],
            "env": row[1],
            "key": row[2],
            "value": row[3],
            "tail_idx": row[4],
        }
        for row in golden_rows
    ]

    info_rows = con.execute(
        """
        SELECT sample_idx, turn_order, env, info_key, info_value, info_type, tail_idx
        FROM info_turns_discarded
        WHERE run_id = ? AND sample_idx = ?
        ORDER BY turn_order, info_key
        """,
        [req.run_path, req.sample_idx],
    ).fetchall()

    info_turns = [
        {
            "sample_idx": row[0],
            "turn_order": row[1],
            "env": row[2],
            "info_key": row[3],
            "info_value": row[4],
            "info_type": row[5],
            "tail_idx": row[6],
        }
        for row in info_rows
    ]

    return {
        "kind": "rollouts_discarded",
        "trainer_step": trainer_step,
        "inference_step": inference_step,
        "discard_reason": discard_reason,
        "group_id": req.group_id,
        "sample_idx": req.sample_idx,
        "prompts": prompts,
        "rollouts": rollouts,
        "samples_data": samples_data,
        "rollout_metrics": rollout_metrics,
        "golden_answers": golden_answers,
        "info_turns": info_turns,
    }


@app.post("/sample-statuses")
def get_sample_statuses(req: SampleStatusesRequest):
    """Get kept/discarded status for a list of samples."""
    log.info(
        f"[API] Getting sample statuses for {req.run_path}, samples={len(req.samples)}"
    )
    if not req.samples:
        return {"statuses": []}

    con = connect()
    values_clause = ", ".join(["(?, ?)"] * len(req.samples))
    params: list[object] = []
    for sample in req.samples:
        params.extend([sample.group_id, sample.sample_idx])
    params.extend([req.run_path, req.run_path])

    rows = con.execute(
        f"""
        WITH keys(group_id, sample_idx) AS (VALUES {values_clause})
        SELECT
            k.group_id,
            k.sample_idx,
            CASE
                WHEN g.sample_idx IS NOT NULL THEN 'rollouts'
                WHEN d.sample_idx IS NOT NULL THEN 'rollouts_discarded'
                ELSE NULL
            END AS kind
        FROM keys k
        LEFT JOIN (
            SELECT DISTINCT group_id, sample_idx
            FROM samples_data
            WHERE run_id = ?
        ) g ON g.group_id = k.group_id AND g.sample_idx = k.sample_idx
        LEFT JOIN (
            SELECT DISTINCT group_id, sample_idx
            FROM samples_data_discarded
            WHERE run_id = ?
        ) d ON d.group_id = k.group_id AND d.sample_idx = k.sample_idx
        """,
        params,
    ).fetchall()

    statuses = [
        {"group_id": row[0], "sample_idx": row[1], "kind": row[2]}
        for row in rows
    ]

    return {"statuses": statuses}


@app.post("/events/inference-by-group")
def get_inference_events_by_group(req: InferenceGroupEventsRequest):
    """Get all inference request events for a specific group_id across all time."""
    log.info(f"[API] Getting inference events for group {req.group_id} in {req.run_path}")
    con = connect()

    # Fetch inference events from events_inference (server_lane is now directly on this table).
    rows = con.execute(
        """
        SELECT event_type, server, node_id, tp_group_id, tp_size,
               start_time, end_time, prompt_tokens, rollout_tokens,
               sample_id, group_id,
               vllm_request_id, queue_time, time_to_first_token, prefill_time,
               decode_time, inference_time, e2e_latency, max_tokens,
               server_lane as lane, is_eval, step, is_canceled, off_policy_steps
        FROM events_inference
        WHERE run_id = ? AND group_id = ? AND event_type = 'request' AND (phase IS NULL OR phase != 'start')
        ORDER BY start_time ASC
        """,
        [req.run_path, req.group_id],
    ).fetchall()

    events = [
        {
            "event_type": row[0],
            "server": row[1],
            "node_id": row[2],
            "tp_group_id": row[3],
            "tp_size": row[4],
            "start_time": row[5],
            "end_time": row[6],
            "prompt_tokens": row[7],
            "rollout_tokens": row[8],
            "sample_id": row[9],
            "group_id": row[10],
            "vllm_request_id": row[11],
            "queue_time": row[12],
            "time_to_first_token": row[13],
            "prefill_time": row[14],
            "decode_time": row[15],
            "inference_time": row[16],
            "e2e_latency": row[17],
            "max_tokens": row[18],
            "lane": row[19],
            "is_eval": bool(row[20]) if row[20] is not None else False,
            "step": row[21],
            "is_canceled": bool(row[22]) if row[22] is not None else False,
            "off_policy_steps": row[23],
            "environment_response_time": None,
            "compute_reward_time": None,
        }
        for row in rows
    ]

    # Fetch environment_response_time from rollouts (including discarded) for this group
    # Returns rows where environment_response_time > 0 (environment turns)
    env_time_rows = con.execute(
        """
        SELECT sample_idx, turn_order, environment_response_time FROM (
            SELECT sample_idx, turn_order, environment_response_time
            FROM rollouts
            WHERE run_id = ? AND group_id = ?
              AND environment_response_time IS NOT NULL AND environment_response_time > 0
            UNION ALL
            SELECT sample_idx, turn_order, environment_response_time
            FROM rollouts_discarded
            WHERE run_id = ? AND group_id = ?
              AND environment_response_time IS NOT NULL AND environment_response_time > 0
        ) sub
        ORDER BY sample_idx, turn_order
        """,
        [req.run_path, req.group_id, req.run_path, req.group_id],
    ).fetchall()
    environment_response_times = [
        {"sample_idx": row[0], "turn_order": row[1], "time": row[2]}
        for row in env_time_rows
    ]

    # Fetch compute_reward_time from samples_data (including discarded) for this group
    reward_time_rows = con.execute(
        """
        SELECT sample_idx, compute_reward_time FROM (
            SELECT sample_idx, compute_reward_time
            FROM samples_data
            WHERE run_id = ? AND group_id = ?
              AND compute_reward_time IS NOT NULL AND compute_reward_time > 0
            UNION ALL
            SELECT sample_idx, compute_reward_time
            FROM samples_data_discarded
            WHERE run_id = ? AND group_id = ?
              AND compute_reward_time IS NOT NULL AND compute_reward_time > 0
        ) sub
        ORDER BY sample_idx
        """,
        [req.run_path, req.group_id, req.run_path, req.group_id],
    ).fetchall()
    compute_reward_times = [
        {"sample_idx": row[0], "time": row[1]}
        for row in reward_time_rows
    ]

    # For eval groups, fetch compute_eval_metrics_time from samples_data_eval
    # (samples_data_eval uses (step, eval_name, sample_idx) instead of group_id,
    # so we need to resolve the mapping via rank-based matching)
    if not compute_reward_times and any(e.get("is_eval") for e in events):
        all_eval_groups = con.execute(
            """
            SELECT DISTINCT group_id
            FROM events_inference
            WHERE run_id = ? AND is_eval = TRUE
            ORDER BY group_id
            """,
            [req.run_path],
        ).fetchall()
        group_ids = [r[0] for r in all_eval_groups]
        try:
            global_eval_idx = group_ids.index(req.group_id)
        except ValueError:
            global_eval_idx = -1

        if global_eval_idx >= 0:
            eval_steps_info = con.execute(
                """
                SELECT step, eval_name, COUNT(DISTINCT sample_idx) as n_samples
                FROM prompts_eval
                WHERE run_id = ?
                GROUP BY step, eval_name
                ORDER BY step, eval_name
                """,
                [req.run_path],
            ).fetchall()

            target_step = None
            target_eval_name = None
            target_sample_idx = None
            offset = 0
            for e_step, e_name, n_samples in eval_steps_info:
                if global_eval_idx < offset + n_samples:
                    target_step = e_step
                    target_eval_name = e_name
                    target_sample_idx = global_eval_idx - offset
                    break
                offset += n_samples

            if target_step is not None and target_eval_name is not None and target_sample_idx is not None:
                # compute_eval_metrics_time may only be populated on one completion;
                # assign the value to the last completion (highest sample_id in events)
                eval_time_row = con.execute(
                    """
                    SELECT MAX(compute_eval_metrics_time)
                    FROM samples_data_eval
                    WHERE run_id = ? AND step = ? AND eval_name = ? AND sample_idx = ?
                      AND compute_eval_metrics_time IS NOT NULL AND compute_eval_metrics_time > 0
                    """,
                    [req.run_path, target_step, target_eval_name, target_sample_idx],
                ).fetchone()
                if eval_time_row and eval_time_row[0] is not None:
                    max_sample_id = max(
                        (e["sample_id"] for e in events if e["sample_id"] is not None),
                        default=0,
                    )
                    compute_reward_times = [
                        {"sample_idx": max_sample_id, "time": eval_time_row[0]}
                    ]

    log.info(f"[API] Returning {len(events)} inference events for group {req.group_id}")
    return {
        "events": events,
        "environment_response_times": environment_response_times,
        "compute_reward_times": compute_reward_times,
    }


@app.post("/events/trainer-breakdown")
def get_trainer_breakdown_events(req: TrainerBreakdownEventsRequest):
    """Get parent + child trainer events for one selected event across all pages."""
    parent_event_type = req.parent_event_type.strip()
    if not parent_event_type:
        raise HTTPException(status_code=400, detail="Missing parent_event_type")

    log.info(
        f"[API] Getting trainer breakdown events for {req.run_path}, "
        f"rank={req.rank}, step={req.step}, parent={parent_event_type}"
    )
    con = connect()

    # Escape LIKE wildcards in the parent event type so we only match true children.
    escaped_parent = (
        parent_event_type
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    child_pattern = f"{escaped_parent}/%"

    rows = con.execute(
        """
        SELECT event_type, step, rank, local_rank, node_id, gpu_index, microbatch, minibatch, start_time, end_time
        FROM events_trainer
        WHERE run_id = ?
          AND rank = ?
          AND step = ?
          AND (
              event_type = ?
              OR event_type LIKE ? ESCAPE '\\'
          )
        ORDER BY start_time ASC
        """,
        [req.run_path, req.rank, req.step, parent_event_type, child_pattern],
    ).fetchall()

    events = [
        {
            "event_type": row[0],
            "step": row[1],
            "rank": row[2],
            "local_rank": row[3],
            "node_id": row[4],
            "gpu_index": row[5],
            "microbatch": row[6],
            "minibatch": row[7],
            "start_time": row[8],
            "end_time": row[9],
        }
        for row in rows
    ]

    log.info(f"[API] Returning {len(events)} trainer breakdown events")
    return {"events": events}


@app.get("/run-summary/{run_path:path}")
def get_run_summary(run_path: str):
    """Get summary info for a run."""
    log.info(f"[API] Getting summary for {run_path}")
    con = connect()
    
    result = con.execute(
        """
        SELECT last_summary_json, last_block_idx, last_rollout_step,
               last_event_zip_idx, last_config_json
        FROM ingest_state
        WHERE run_id = ?
        """,
        [run_path],
    ).fetchone()
    
    tracking = is_tracking(run_path)
    syncing = is_syncing(run_path) or is_syncing_evals_after_training(run_path)
    sync_info = get_sync_status(run_path)
    
    # Fetch trainer_commit and schema_version from runs table
    run_row = con.execute(
        "SELECT trainer_commit, schema_version FROM runs WHERE run_id = ?",
        [run_path],
    ).fetchone()
    trainer_commit = run_row[0] if run_row else None
    schema_version = run_row[1] if run_row else None
    
    if not result:
        log.info(f"[API] No data found for {run_path}")
        return {
            "summary": {},
            "config": {},
            "custom_config": None,
            "trainer_commit": trainer_commit,
            "schema_version": schema_version,
            "last_rollout_step": -1,
            "local_rollout_count": 0,
            "local_rollout_steps": 0,
            "local_rollout_metrics_count": 0,
            "available_rollout_metric_names": [],
            "available_envs": [],
            "local_orchestrator_event_count": 0,
            "local_trainer_event_count": 0,
            "local_inference_event_count": 0,
            "local_gpu_metrics_count": 0,
            "local_cpu_metrics_count": 0,
            "local_vllm_metrics_count": 0,
            "local_discarded_rollout_count": 0,
            "local_discarded_rollout_metrics_count": 0,
            "local_discarded_trainer_steps": 0,
            "trainer_info": {},
            "event_info": {},
            "rollout_info": {},
            "step_metrics_info": {},
            "eval_info": {"evals": []},
            "trainer_bucket_info": {"step": None, "groups_done": 0},
            "waiting_buckets": [],
            "data_metric_ranges": {},
            "is_tracking": tracking,
            "is_syncing": syncing,
            "sync_status": sync_info,
        }
    
    summary = json.loads(result[0]) if result[0] else {}
    config = json.loads(result[4]) if result[4] else {}
    custom_config = config.pop("_custom_config", None)

    # Batch all COUNT queries into a single query using scalar subqueries
    # (replaces 12 separate COUNT queries with 1 round-trip)
    rp = run_path  # shorthand for params
    counts = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM rollouts WHERE run_id = ?),
            (SELECT COUNT(*) FROM rollouts_metrics WHERE run_id = ?),
            (SELECT COUNT(*) FROM events_orchestrator WHERE run_id = ?),
            (SELECT COUNT(*) FROM events_trainer WHERE run_id = ?),
            (SELECT COUNT(*) FROM events_inference WHERE run_id = ?),
            (SELECT COUNT(*) FROM system_metrics_gpu WHERE run_id = ?),
            (SELECT COUNT(*) FROM system_metrics_cpu WHERE run_id = ?),
            (SELECT COUNT(*) FROM vllm_metrics WHERE run_id = ?),
            (SELECT COUNT(*) FROM rollouts_discarded WHERE run_id = ?),
            (SELECT COUNT(*) FROM rollouts_metrics_discarded WHERE run_id = ?),
            (SELECT COUNT(DISTINCT trainer_step) FROM rollouts_discarded WHERE run_id = ?),
            (SELECT COUNT(DISTINCT step) FROM rollouts WHERE run_id = ?)
        """,
        [rp] * 12,
    ).fetchone()
    
    rollout_count = counts[0]
    rollout_metrics_count = counts[1]
    orchestrator_event_count = counts[2]
    trainer_event_count = counts[3]
    inference_event_count = counts[4]
    gpu_metrics_count = counts[5]
    cpu_metrics_count = counts[6]
    vllm_metrics_count = counts[7]
    discarded_rollout_count = counts[8]
    discarded_rollout_metrics_count = counts[9]
    discarded_trainer_steps = counts[10]
    rollout_steps = counts[11]
    
    # Get available rollout metric names for this run
    metric_names_result = con.execute(
        "SELECT DISTINCT metric_name FROM rollouts_metrics WHERE run_id = ? ORDER BY metric_name",
        [run_path],
    ).fetchall()
    available_rollout_metric_names = [r[0] for r in metric_names_result]
    
    # Get available environments for this run (from prompts table)
    envs_result = con.execute(
        "SELECT DISTINCT env FROM prompts WHERE run_id = ? AND env IS NOT NULL ORDER BY env", [run_path]
    ).fetchall()
    available_envs = [r[0] for r in envs_result]

    # Get available sample tag names and their values
    tag_names_result = con.execute(
        """
        SELECT tag_name, tag_value, COUNT(*) as cnt
        FROM sample_tags
        WHERE run_id = ?
        GROUP BY tag_name, tag_value
        ORDER BY tag_name, cnt DESC
        """,
        [run_path],
    ).fetchall()
    available_sample_tags: dict[str, list[str]] = {}
    for row in tag_names_result:
        tag_name, tag_value = row[0], row[1]
        if tag_name not in available_sample_tags:
            available_sample_tags[tag_name] = []
        available_sample_tags[tag_name].append(tag_value)
    
    trainer_info = {
        "last_training_step": summary.get("steps/last_training_step"),
    }
    
    # All events and metrics are in a unified events/ folder
    event_info = {
        "last_event_block_idx": result[1] or -1,
        "remote_num_finalized_blocks": summary.get("events/num_finalized_blocks", 0),
        "remote_current_block_idx": summary.get("events/current_block_idx", 0),
        "tail_orchestrator_count": summary.get("events/tail_orchestrator_count", 0),
        "tail_trainer_count": summary.get("events/tail_trainer_count", 0),
        "tail_inference_count": summary.get("events/tail_inference_count", 0),
        "tail_gpu_count": summary.get("events/tail_gpu_count", 0),
        "tail_cpu_count": summary.get("events/tail_cpu_count", 0),
        "tail_vllm_count": summary.get("events/tail_vllm_count", 0),
        "num_gpus": summary.get("events/num_gpus", 0),
        "num_inference_servers": summary.get("events/num_inference_servers", 0),
    }
    
    rollout_info = {
        "remote_num_finalized_blocks": summary.get("steps/num_finalized_blocks", 0),
        "remote_current_block_idx": summary.get("steps/current_block_idx", 0),
        "last_training_step": summary.get("steps/last_training_step", -1),
        "block_live_rollout_count": summary.get("steps/block_live_count", 0),
    }
    
    # Metrics are computed from rollouts data
    # Build custom_metric_sections: { section: { group: [metric_name, ...] } }
    custom_metric_sections: dict[str, dict[str, list[str]]] = {}
    for r in con.execute(
        'SELECT DISTINCT metric_name, section, "group" FROM step_metrics WHERE run_id = ? ORDER BY section, "group", metric_name',
        [run_path],
    ).fetchall():
        section_name = r[1] or "General"
        group_name = r[2] or ""
        custom_metric_sections.setdefault(section_name, {}).setdefault(group_name, []).append(r[0])

    step_metrics_info = {
        "local_steps": rollout_steps,
        "custom_metric_sections": custom_metric_sections,
    }

    # Eval info: discover available eval names and per-eval rollout metric names
    eval_names_rows = con.execute(
        "SELECT DISTINCT eval_name FROM prompts_eval WHERE run_id = ? ORDER BY eval_name",
        [run_path],
    ).fetchall()
    eval_names = [r[0] for r in eval_names_rows]

    evals_list: list[dict] = []
    if eval_names:
        per_eval_metrics = con.execute(
            "SELECT eval_name, metric_name FROM rollouts_metrics_eval WHERE run_id = ? GROUP BY eval_name, metric_name ORDER BY eval_name, metric_name",
            [run_path],
        ).fetchall()
        metrics_by_eval: dict[str, list[str]] = {name: [] for name in eval_names}
        for row in per_eval_metrics:
            if row[0] in metrics_by_eval:
                metrics_by_eval[row[0]].append(row[1])
        for name in eval_names:
            evals_list.append({
                "eval_name": name,
                "available_rollout_metric_names": metrics_by_eval.get(name, []),
            })

    eval_info = {
        "evals": evals_list,
    }

    # Trainer bucket: count rollouts_group_done for the latest step
    trainer_bucket_row = con.execute(
        """
        SELECT step, COUNT(*) as cnt
        FROM events_orchestrator
        WHERE run_id = ? AND event_type = 'rollouts_group_done'
        GROUP BY step
        ORDER BY step DESC
        LIMIT 1
        """,
        [run_path],
    ).fetchone()
    trainer_bucket_info = {
        "step": trainer_bucket_row[0] if trainer_bucket_row else None,
        "groups_done": trainer_bucket_row[1] if trainer_bucket_row else 0,
    }
    
    # Waiting buckets: completed save_batch steps that are ahead of the step
    # currently being trained. If last_training_step is N, trainer is working
    # on N+1, so only steps > N+1 are actually waiting.
    trainer_last_training_step = summary.get("steps/last_training_step")
    if trainer_last_training_step is None:
        trainer_last_training_step = summary.get("_step")
    try:
        trainer_last_training_step = int(trainer_last_training_step)
    except (TypeError, ValueError):
        trainer_last_training_step = -1
    waiting_step_threshold = trainer_last_training_step + 1
    waiting_bucket_rows = con.execute(
        """
        SELECT DISTINCT step
        FROM events_orchestrator
        WHERE run_id = ? AND event_type = 'save_batch' AND step > ?
        ORDER BY step ASC
        """,
        [run_path, waiting_step_threshold],
    ).fetchall()
    waiting_buckets = [row[0] for row in waiting_bucket_rows]
    
    # Compute per-(env, metric_name) min/max from actual data as fallback ranges
    metric_range_rows = con.execute(
        """
        SELECT env, metric_name, MIN(value), MAX(value)
        FROM rollouts_metrics
        WHERE run_id = ?
        GROUP BY env, metric_name
        """,
        [run_path],
    ).fetchall()
    data_metric_ranges: dict[str, dict[str, dict[str, float]]] = {}
    for row in metric_range_rows:
        env = row[0] or ""
        metric_name = row[1]
        if env not in data_metric_ranges:
            data_metric_ranges[env] = {}
        data_metric_ranges[env][metric_name] = {"min": row[2], "max": row[3]}
    
    log.info(
        f"[API] Summary: {rollout_count} rollouts ({rollout_steps} steps), "
        f"{rollout_metrics_count} rollout metrics, {discarded_rollout_count} discarded rollouts "
        f"({discarded_trainer_steps} trainer_steps), {orchestrator_event_count} orchestrator events, "
        f"{trainer_event_count} trainer events, {inference_event_count} inference events, "
        f"{gpu_metrics_count} gpu metrics, {cpu_metrics_count} cpu metrics, "
        f"{vllm_metrics_count} vllm metrics, tracking={tracking}, syncing={syncing}"
    )
    
    return {
        "summary": summary,
        "config": config,
        "custom_config": custom_config,
        "trainer_commit": trainer_commit,
        "schema_version": schema_version,
        "last_rollout_step": result[2] if result[2] is not None else -1,
        "local_rollout_count": rollout_count,
        "local_rollout_steps": rollout_steps,
        "local_rollout_metrics_count": rollout_metrics_count,
        "available_rollout_metric_names": available_rollout_metric_names,
        "available_envs": available_envs,
        "local_orchestrator_event_count": orchestrator_event_count,
        "local_trainer_event_count": trainer_event_count,
        "local_inference_event_count": inference_event_count,
        "local_gpu_metrics_count": gpu_metrics_count,
        "local_cpu_metrics_count": cpu_metrics_count,
        "local_vllm_metrics_count": vllm_metrics_count,
        "local_discarded_rollout_count": discarded_rollout_count,
        "local_discarded_rollout_metrics_count": discarded_rollout_metrics_count,
        "local_discarded_trainer_steps": discarded_trainer_steps,
        "trainer_info": trainer_info,
        "event_info": event_info,
        "rollout_info": rollout_info,
        "step_metrics_info": step_metrics_info,
        "eval_info": eval_info,
        "trainer_bucket_info": trainer_bucket_info,
        "waiting_buckets": waiting_buckets,
        "data_metric_ranges": data_metric_ranges,
        "available_sample_tags": available_sample_tags,
        "is_tracking": tracking,
        "is_syncing": syncing,
        "sync_status": sync_info,
    }


@app.get("/run-code/tree/{run_path:path}")
def get_run_code_tree(run_path: str):
    """Return directory tree for locally downloaded run code."""
    log.info(f"[API] Getting run code tree for {run_path}")
    code_dir = get_run_code_dir(run_path)
    if not code_dir.exists() or not code_dir.is_dir():
        return {
            "run_path": run_path,
            "available": False,
            "tree": [],
            "truncated": False,
            "total_nodes": 0,
        }

    tree, truncated, total_nodes = build_code_tree(code_dir)
    return {
        "run_path": run_path,
        "available": True,
        "tree": tree,
        "truncated": truncated,
        "total_nodes": total_nodes,
    }


@app.get("/run-code/file/{run_path:path}")
def get_run_code_file(run_path: str, file_path: str):
    """Return file content from locally downloaded run code."""
    if not file_path.strip():
        raise HTTPException(status_code=400, detail="Missing file_path")

    code_dir = get_run_code_dir(run_path)
    if not code_dir.exists() or not code_dir.is_dir():
        raise HTTPException(status_code=404, detail="Code directory not found for run")

    try:
        target_path = resolve_code_file_path(run_path, file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_size = target_path.stat().st_size
        with open(target_path, "rb") as f:
            raw = f.read(MAX_CODE_FILE_BYTES + 1)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}") from e

    truncated = len(raw) > MAX_CODE_FILE_BYTES
    if truncated:
        raw = raw[:MAX_CODE_FILE_BYTES]

    is_binary = b"\x00" in raw[:8192]
    content = "" if is_binary else raw.decode("utf-8", errors="replace")

    return {
        "run_path": run_path,
        "file_path": file_path,
        "size_bytes": file_size,
        "truncated": truncated,
        "is_binary": is_binary,
        "content": content,
    }


@app.get("/run-code/diff-summary/{left_run_path:path}")
def get_run_code_diff_summary(left_run_path: str, right_run_path: str):
    """Return aggregate line diff summary between two locally downloaded code snapshots."""
    if not right_run_path.strip():
        raise HTTPException(status_code=400, detail="Missing right_run_path")

    left_code_dir = get_run_code_dir(left_run_path)
    right_code_dir = get_run_code_dir(right_run_path)

    if (
        not left_code_dir.exists()
        or not left_code_dir.is_dir()
        or not right_code_dir.exists()
        or not right_code_dir.is_dir()
    ):
        return {
            "left_run_path": left_run_path,
            "right_run_path": right_run_path,
            "available": False,
            "changed_files": 0,
            "added_lines": 0,
            "removed_lines": 0,
        }

    try:
        summary = build_code_diff_summary(left_code_dir, right_code_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to build code diff summary: {e}"
        ) from e

    return {
        "left_run_path": left_run_path,
        "right_run_path": right_run_path,
        "available": True,
        "changed_files": summary["changed_files"],
        "added_lines": summary["added_lines"],
        "removed_lines": summary["removed_lines"],
    }


@app.post("/events/timeline-paginated")
def get_timeline_paginated(req: TimelinePaginatedRequest):
    """Get paginated timeline events (orchestrator, trainer, and inference) for a specific page/interval.
    
    This endpoint is designed for timeline visualization with proper pagination.
    It calculates the time range based on ALL events (not just a limited set),
    so page 0 always shows the beginning of the run.
    """
    log.info(f"[API] Getting paginated timeline for {req.run_path}, page={req.page}, interval={req.interval_seconds}s")
    con = connect()
    
    # Get the absolute time range from ALL events.
    time_range = con.execute(
        """
        SELECT 
            MIN(min_time) as global_min,
            MAX(max_time) as global_max
        FROM (
            SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
            FROM events_orchestrator WHERE run_id = ?
            UNION ALL
            SELECT MIN(start_time) as min_time, MAX(end_time) as max_time
            FROM events_trainer WHERE run_id = ?
            UNION ALL
            SELECT MIN(start_time) as min_time, MAX(end_time) as max_time
            FROM events_inference WHERE run_id = ?
        )
        """,
        [req.run_path, req.run_path, req.run_path],
    ).fetchone()
    
    global_min_time = time_range[0]
    global_max_time = time_range[1]
    
    if global_min_time is None or global_max_time is None:
        log.info(f"[API] No events found for {req.run_path}")
        return {
            "orchestrator_events": [],
            "trainer_events": [],
            "inference_events": [],
            "total_pages": 1,
            "current_page": 0,
            "interval_start": 0,
            "interval_end": req.interval_seconds,
            "global_min_time": None,
            "global_max_time": None,
        }
    
    # Calculate pagination
    total_duration = global_max_time - global_min_time
    total_pages = max(1, int((total_duration // req.interval_seconds) + 1))
    
    # Calculate the time window for this page
    interval_start = global_min_time + (req.page * req.interval_seconds)
    interval_end = interval_start + req.interval_seconds
    
    # Fetch orchestrator events in this time window
    orchestrator_rows = con.execute(
        """
        SELECT timestamp, event_type, step, node_id
        FROM events_orchestrator
        WHERE run_id = ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp ASC
        """,
        [req.run_path, interval_start, interval_end],
    ).fetchall()
    
    orchestrator_events = [
        {
            "timestamp": row[0],
            "event_type": row[1],
            "step": row[2],
            "node_id": row[3],
        }
        for row in orchestrator_rows
    ]
    
    # Fetch trainer events that overlap with this time window
    # (events that start before interval_end AND end after interval_start)
    trainer_rows = con.execute(
        """
        SELECT event_type, step, rank, local_rank, node_id, gpu_index, microbatch, minibatch, start_time, end_time
        FROM events_trainer
        WHERE run_id = ? AND start_time < ? AND end_time > ?
        ORDER BY start_time ASC
        """,
        [req.run_path, interval_end, interval_start],
    ).fetchall()

    trainer_events = [
        {
            "event_type": row[0],
            "step": row[1],
            "rank": row[2],
            "local_rank": row[3],
            "node_id": row[4],
            "gpu_index": row[5],
            "microbatch": row[6],
            "minibatch": row[7],
            "start_time": row[8],
            "end_time": row[9],
        }
        for row in trainer_rows
    ]
    
    # Fetch inference events from events_inference, joining with samples_data
    # for compute_reward_time, and with rollouts for per-event timing data.
    # server_lane is now directly on events_inference.
    # Uses window functions to compute per-sample turn position so we can
    # map environment_response_time to the correct request event and attach
    # compute_reward_time to the last request event of each sample.
    inference_rows = con.execute(
        """
        WITH ranked_requests AS (
            SELECT event_type, server, node_id, tp_group_id, tp_size, start_time, end_time, prompt_tokens, rollout_tokens,
                   sample_id, group_id,
                   vllm_request_id, queue_time, time_to_first_token, prefill_time,
                   decode_time, inference_time, e2e_latency, max_tokens, is_eval, step, is_canceled, off_policy_steps,
                   server_lane,
                   ROW_NUMBER() OVER (PARTITION BY group_id, sample_id ORDER BY start_time) as turn_pos,
                   COUNT(*) OVER (PARTITION BY group_id, sample_id) as total_turns
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND (phase IS NULL OR phase != 'start')
        ),
        ranked_env AS (
            SELECT group_id, sample_idx,
                   environment_response_time,
                   ROW_NUMBER() OVER (PARTITION BY group_id, sample_idx ORDER BY turn_order) as env_pos
            FROM (
                SELECT group_id, sample_idx, environment_response_time, turn_order
                FROM rollouts WHERE run_id = ?
                  AND environment_response_time IS NOT NULL AND environment_response_time > 0
                UNION ALL
                SELECT group_id, sample_idx, environment_response_time, turn_order
                FROM rollouts_discarded WHERE run_id = ?
                  AND environment_response_time IS NOT NULL AND environment_response_time > 0
            ) sub
        ),
        eval_groups_ranked AS (
            SELECT group_id, ROW_NUMBER() OVER (ORDER BY group_id) - 1 as rnk
            FROM (SELECT DISTINCT group_id FROM events_inference WHERE run_id = ? AND is_eval = TRUE)
        ),
        eval_samples_ranked AS (
            SELECT step, eval_name, sample_idx, ROW_NUMBER() OVER (ORDER BY step, eval_name, sample_idx) - 1 as rnk
            FROM (SELECT DISTINCT step, eval_name, sample_idx FROM prompts_eval WHERE run_id = ?)
        ),
        samples_info AS (
            SELECT group_id, sample_idx, compute_reward_time
            FROM samples_data WHERE run_id = ?
            UNION ALL
            SELECT group_id, sample_idx, compute_reward_time
            FROM samples_data_discarded WHERE run_id = ?
            UNION ALL
            SELECT egr.group_id,
                   esr.sample_idx as sample_idx,
                   MAX(sde.compute_eval_metrics_time) as compute_reward_time
            FROM eval_groups_ranked egr
            JOIN eval_samples_ranked esr ON egr.rnk = esr.rnk
            JOIN samples_data_eval sde ON sde.run_id = ? AND sde.step = esr.step AND sde.eval_name = esr.eval_name AND sde.sample_idx = esr.sample_idx
            WHERE sde.compute_eval_metrics_time IS NOT NULL AND sde.compute_eval_metrics_time > 0
            GROUP BY egr.group_id, esr.sample_idx
        )
        SELECT rr.event_type, rr.server, rr.node_id, rr.tp_group_id, rr.tp_size,
               rr.start_time, rr.end_time, rr.prompt_tokens, rr.rollout_tokens,
               rr.sample_id, rr.group_id,
               rr.vllm_request_id, rr.queue_time, rr.time_to_first_token, rr.prefill_time,
               rr.decode_time, rr.inference_time, rr.e2e_latency, rr.max_tokens,
               rr.server_lane as lane,
               re.environment_response_time,
               CASE WHEN rr.turn_pos = rr.total_turns THEN si.compute_reward_time ELSE NULL END as compute_reward_time,
               rr.is_eval,
               rr.step,
               rr.is_canceled,
               rr.off_policy_steps
        FROM ranked_requests rr
        LEFT JOIN samples_info si ON si.group_id = rr.group_id AND si.sample_idx = rr.sample_id
        LEFT JOIN ranked_env re ON re.group_id = rr.group_id AND re.sample_idx = rr.sample_id AND re.env_pos = rr.turn_pos
        WHERE rr.start_time < ?
          AND (rr.end_time
               + COALESCE(re.environment_response_time, 0)
               + CASE WHEN rr.turn_pos = rr.total_turns THEN COALESCE(si.compute_reward_time, 0) ELSE 0 END
              ) > ?

        UNION ALL

        SELECT event_type, server, node_id, tp_group_id, tp_size, start_time, end_time, prompt_tokens, rollout_tokens,
               sample_id, group_id,
               vllm_request_id, queue_time, time_to_first_token, prefill_time,
               decode_time, inference_time, e2e_latency, max_tokens,
               NULL as lane,
               NULL as environment_response_time,
               NULL as compute_reward_time,
               is_eval,
               step,
               is_canceled,
               off_policy_steps
        FROM events_inference
        WHERE run_id = ? AND event_type != 'request' AND start_time < ? AND end_time > ?

        ORDER BY start_time ASC
        """,
        [req.run_path,
         req.run_path, req.run_path,
         req.run_path, req.run_path,
         req.run_path, req.run_path, req.run_path,
         interval_end, interval_start,
         req.run_path, interval_end, interval_start],
    ).fetchall()

    inference_events = [
        {
            "event_type": row[0],
            "server": row[1],
            "node_id": row[2],
            "tp_group_id": row[3],
            "tp_size": row[4],
            "start_time": row[5],
            "end_time": row[6],
            "prompt_tokens": row[7],
            "rollout_tokens": row[8],
            "sample_id": row[9],
            "group_id": row[10],
            "vllm_request_id": row[11],
            "queue_time": row[12],
            "time_to_first_token": row[13],
            "prefill_time": row[14],
            "decode_time": row[15],
            "inference_time": row[16],
            "e2e_latency": row[17],
            "max_tokens": row[18],
            "lane": row[19],
            "environment_response_time": row[20],
            "compute_reward_time": row[21],
            "is_eval": bool(row[22]) if row[22] is not None else False,
            "step": row[23],
            "is_canceled": bool(row[24]) if row[24] is not None else False,
            "off_policy_steps": row[25],
        }
        for row in inference_rows
    ]
    
    log.info(f"[API] Returning page {req.page}/{total_pages}: {len(orchestrator_events)} orchestrator, {len(trainer_events)} trainer, {len(inference_events)} inference events")
    
    return {
        "orchestrator_events": orchestrator_events,
        "trainer_events": trainer_events,
        "inference_events": inference_events,
        "total_pages": total_pages,
        "current_page": req.page,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "global_min_time": global_min_time,
        "global_max_time": global_max_time,
    }


@app.get("/events/inflight/{run_path:path}")
def get_inflight_generations(run_path: str):
    """Get the latest inflight generation snapshot for a run.

    Returns the in-memory snapshot (from the last tail.zip download).
    This is ephemeral data — not persisted to DB.
    """
    from .ingest import inflight_by_run
    data = inflight_by_run.get(run_path)
    if data is None:
        return {"snapshot_time": None, "running": []}
    return data


@app.post("/system-metrics/gpu")
def get_system_metrics_gpu(req: SystemMetricsRequest):
    """Get GPU system metrics for a run."""
    log.info(
        f"[API] Getting GPU metrics for {req.run_path}, gpu_index={req.gpu_index}, "
        f"node_id={req.node_id}, rank={req.rank}, local_rank={req.local_rank}, source={req.source}, "
        f"include_associated_system_metrics={req.include_associated_system_metrics}, "
        f"include_edge_points={req.include_edge_points}, limit={req.limit}"
    )
    con = connect()

    include_associated_system = (
        req.include_associated_system_metrics
        and req.rank is not None
        and req.gpu_index is None
    )

    if include_associated_system:
        rank_selector_clauses = ["run_id = ?", "rank = ?"]
        rank_selector_params: list[object] = [req.run_path, req.rank]
        if req.node_id is not None:
            rank_selector_clauses.append("node_id = ?")
            rank_selector_params.append(req.node_id)
        if req.local_rank is not None:
            rank_selector_clauses.append("local_rank = ?")
            rank_selector_params.append(req.local_rank)
        rank_selector_where = " AND ".join(rank_selector_clauses)

        trainer_branch_clauses = ["rank = ?"]
        trainer_branch_params: list[object] = [req.rank]
        if req.node_id is not None:
            trainer_branch_clauses.append("node_id = ?")
            trainer_branch_params.append(req.node_id)
        if req.local_rank is not None:
            trainer_branch_clauses.append("local_rank = ?")
            trainer_branch_params.append(req.local_rank)

        system_branch_clauses = [
            "rank = -1",
            f"gpu_index IN (SELECT DISTINCT gpu_index FROM system_metrics_gpu WHERE {rank_selector_where})",
        ]
        system_branch_params: list[object] = list(rank_selector_params)
        if req.node_id is not None:
            system_branch_clauses.append("node_id = ?")
            system_branch_params.append(req.node_id)

        base_where = (
            "run_id = ? AND (("
            + " AND ".join(trainer_branch_clauses)
            + ") OR ("
            + " AND ".join(system_branch_clauses)
            + "))"
        )
        base_params: list[object] = [req.run_path] + trainer_branch_params + system_branch_params
        if req.source:
            base_where += " AND source = ?"
            base_params.append(req.source)
    else:
        # Standard filter behavior.
        base_clauses = ["run_id = ?"]
        base_params: list[object] = [req.run_path]
        if req.gpu_index is not None:
            base_clauses.append("gpu_index = ?")
            base_params.append(req.gpu_index)
        if req.node_id is not None:
            base_clauses.append("node_id = ?")
            base_params.append(req.node_id)
        if req.rank is not None:
            base_clauses.append("rank = ?")
            base_params.append(req.rank)
        if req.local_rank is not None:
            base_clauses.append("local_rank = ?")
            base_params.append(req.local_rank)
        if req.source:
            base_clauses.append("source = ?")
            base_params.append(req.source)
        base_where = " AND ".join(base_clauses)

    metric_filter_sql = ""
    metric_filter_params: list[object] = []
    if req.metric_names:
        placeholders = ", ".join(["?" for _ in req.metric_names])
        metric_filter_sql = f" AND metric_name IN ({placeholders})"
        metric_filter_params = list(req.metric_names)

    if req.include_edge_points and req.start_time is not None and req.end_time is not None:
        # Include one point before and after the interval per series so lines can
        # connect cleanly to the viewport boundaries without dropping at edges.
        query = f"""
            WITH in_window AS (
                SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
                FROM system_metrics_gpu
                WHERE {base_where}{metric_filter_sql} AND timestamp >= ? AND timestamp <= ?
            ),
            before_window AS (
                SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
                FROM (
                    SELECT
                        timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value,
                        ROW_NUMBER() OVER (
                            PARTITION BY node_id, rank, local_rank, gpu_index, source, metric_name
                            ORDER BY timestamp DESC
                        ) AS rn
                    FROM system_metrics_gpu
                    WHERE {base_where}{metric_filter_sql} AND timestamp < ?
                ) ranked_before
                WHERE rn = 1
            ),
            after_window AS (
                SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
                FROM (
                    SELECT
                        timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value,
                        ROW_NUMBER() OVER (
                            PARTITION BY node_id, rank, local_rank, gpu_index, source, metric_name
                            ORDER BY timestamp ASC
                        ) AS rn
                    FROM system_metrics_gpu
                    WHERE {base_where}{metric_filter_sql} AND timestamp > ?
                ) ranked_after
                WHERE rn = 1
            )
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM in_window
            UNION ALL
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM before_window
            UNION ALL
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM after_window
            ORDER BY timestamp ASC
            LIMIT ?
        """
        params = [
            *base_params,
            *metric_filter_params,
            req.start_time,
            req.end_time,
            *base_params,
            *metric_filter_params,
            req.start_time,
            *base_params,
            *metric_filter_params,
            req.end_time,
            req.limit,
        ]
    else:
        query = f"""
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM system_metrics_gpu
            WHERE {base_where}{metric_filter_sql}
        """
        params = list(base_params)
        params.extend(metric_filter_params)

        if req.start_time is not None:
            query += " AND timestamp >= ?"
            params.append(req.start_time)

        if req.end_time is not None:
            query += " AND timestamp <= ?"
            params.append(req.end_time)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(req.limit)
    
    rows = con.execute(query, params).fetchall()
    
    metrics = [
        {
            "timestamp": row[0],
            "node_id": row[1],
            "rank": row[2],
            "local_rank": row[3],
            "gpu_index": row[4],
            "source": row[5],
            "metric_name": row[6],
            "value": row[7],
        }
        for row in rows
    ]
    
    # Get available metric names and GPU indices
    available_metrics = con.execute(
        f"SELECT DISTINCT metric_name FROM system_metrics_gpu WHERE {base_where} ORDER BY metric_name",
        base_params,
    ).fetchall()
    
    available_gpus = con.execute(
        f"SELECT DISTINCT gpu_index FROM system_metrics_gpu WHERE {base_where} ORDER BY gpu_index",
        base_params,
    ).fetchall()
    
    log.info(f"[API] Returning {len(metrics)} GPU metrics")
    
    return {
        "metrics": metrics,
        "total_returned": len(metrics),
        "available_metrics": [r[0] for r in available_metrics],
        "available_gpus": [r[0] for r in available_gpus],
    }


@app.post("/system-metrics/gpu-paginated")
def get_system_metrics_gpu_paginated(req: SystemMetricsPaginatedRequest):
    """Get paginated GPU system metrics for a specific page/interval."""
    log.info(f"[API] Getting paginated GPU metrics for {req.run_path}, page={req.page}, interval={req.interval_seconds}s")
    con = connect()
    
    # Get the absolute time range
    time_range = con.execute(
        """
        SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
        FROM system_metrics_gpu WHERE run_id = ?
        """,
        [req.run_path],
    ).fetchone()
    
    global_min_time = time_range[0]
    global_max_time = time_range[1]
    
    if global_min_time is None or global_max_time is None:
        log.info(f"[API] No GPU metrics found for {req.run_path}")
        return {
            "metrics": [],
            "total_pages": 1,
            "current_page": 0,
            "interval_start": 0,
            "interval_end": req.interval_seconds,
            "global_min_time": None,
            "global_max_time": None,
            "available_metrics": [],
            "available_gpus": [],
        }
    
    # Calculate pagination
    total_duration = global_max_time - global_min_time
    total_pages = max(1, int((total_duration // req.interval_seconds) + 1))
    current_page = req.page

    # Calculate the time window for this page
    if req.align_to_latest:
        interval_end = global_max_time
        interval_start = max(global_min_time, interval_end - req.interval_seconds)
        current_page = max(0, total_pages - 1)
    else:
        interval_start = global_min_time + (req.page * req.interval_seconds)
        interval_end = interval_start + req.interval_seconds
    
    metric_filter_sql = ""
    metric_filter_params = []
    if req.metric_names:
        placeholders = ", ".join(["?" for _ in req.metric_names])
        metric_filter_sql = f" AND metric_name IN ({placeholders})"
        metric_filter_params = req.metric_names

    # Include one point before and after the interval per series so lines can
    # connect cleanly to the viewport boundaries without dropping at edges.
    query = f"""
        WITH in_window AS (
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM system_metrics_gpu
            WHERE run_id = ? AND timestamp >= ? AND timestamp < ?{metric_filter_sql}
        ),
        before_window AS (
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM (
                SELECT
                    timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY node_id, rank, local_rank, gpu_index, source, metric_name
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM system_metrics_gpu
                WHERE run_id = ? AND timestamp < ?{metric_filter_sql}
            ) ranked_before
            WHERE rn = 1
        ),
        after_window AS (
            SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
            FROM (
                SELECT
                    timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY node_id, rank, local_rank, gpu_index, source, metric_name
                        ORDER BY timestamp ASC
                    ) AS rn
                FROM system_metrics_gpu
                WHERE run_id = ? AND timestamp >= ?{metric_filter_sql}
            ) ranked_after
            WHERE rn = 1
        )
        SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
        FROM in_window
        UNION ALL
        SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
        FROM before_window
        UNION ALL
        SELECT timestamp, node_id, rank, local_rank, gpu_index, source, metric_name, value
        FROM after_window
        ORDER BY timestamp ASC
    """
    params = [
        req.run_path, interval_start, interval_end, *metric_filter_params,
        req.run_path, interval_start, *metric_filter_params,
        req.run_path, interval_end, *metric_filter_params,
    ]

    rows = con.execute(query, params).fetchall()
    
    metrics = [
        {
            "timestamp": row[0],
            "node_id": row[1],
            "rank": row[2],
            "local_rank": row[3],
            "gpu_index": row[4],
            "source": row[5],
            "metric_name": row[6],
            "value": row[7],
        }
        for row in rows
    ]
    
    # Get available metric names and GPU indices
    available_metrics = con.execute(
        "SELECT DISTINCT metric_name FROM system_metrics_gpu WHERE run_id = ? ORDER BY metric_name",
        [req.run_path],
    ).fetchall()
    
    available_gpus = con.execute(
        "SELECT DISTINCT gpu_index FROM system_metrics_gpu WHERE run_id = ? ORDER BY gpu_index",
        [req.run_path],
    ).fetchall()
    
    log.info(f"[API] Returning GPU metrics page {req.page}/{total_pages}: {len(metrics)} metrics")
    
    return {
        "metrics": metrics,
        "total_pages": total_pages,
        "current_page": current_page,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "global_min_time": global_min_time,
        "global_max_time": global_max_time,
        "available_metrics": [r[0] for r in available_metrics],
        "available_gpus": [r[0] for r in available_gpus],
    }


@app.post("/system-metrics/cpu-paginated")
def get_system_metrics_cpu_paginated(req: SystemMetricsPaginatedRequest):
    """Get paginated CPU system metrics for a specific page/interval."""
    log.info(f"[API] Getting paginated CPU metrics for {req.run_path}, page={req.page}, interval={req.interval_seconds}s")
    con = connect()
    
    # Get the absolute time range
    time_range = con.execute(
        """
        SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
        FROM system_metrics_cpu WHERE run_id = ?
        """,
        [req.run_path],
    ).fetchone()
    
    global_min_time = time_range[0]
    global_max_time = time_range[1]
    
    if global_min_time is None or global_max_time is None:
        log.info(f"[API] No CPU metrics found for {req.run_path}")
        return {
            "metrics": [],
            "total_pages": 1,
            "current_page": 0,
            "interval_start": 0,
            "interval_end": req.interval_seconds,
            "global_min_time": None,
            "global_max_time": None,
            "available_metrics": [],
        }
    
    # Align pagination to an optional external anchor (e.g. GPU min time on infra page).
    # This keeps page N mapped to the same absolute time window across metric types.
    pagination_anchor = req.anchor_start_time if req.anchor_start_time is not None else global_min_time

    # Calculate pagination
    total_duration = global_max_time - pagination_anchor
    total_pages = max(1, int((total_duration // req.interval_seconds) + 1))
    current_page = req.page

    # Calculate the time window for this page
    if req.align_to_latest:
        interval_end = global_max_time
        interval_start = max(pagination_anchor, interval_end - req.interval_seconds)
        current_page = max(0, total_pages - 1)
    else:
        interval_start = pagination_anchor + (req.page * req.interval_seconds)
        interval_end = interval_start + req.interval_seconds
    
    metric_filter_sql = ""
    metric_filter_params = []
    if req.metric_names:
        placeholders = ", ".join(["?" for _ in req.metric_names])
        metric_filter_sql = f" AND metric_name IN ({placeholders})"
        metric_filter_params = req.metric_names

    # Include one point before and after the interval per series so lines can
    # connect cleanly to the viewport boundaries without dropping at edges.
    query = f"""
        WITH in_window AS (
            SELECT timestamp, node_id, source, metric_name, value
            FROM system_metrics_cpu
            WHERE run_id = ? AND timestamp >= ? AND timestamp < ?{metric_filter_sql}
        ),
        before_window AS (
            SELECT timestamp, node_id, source, metric_name, value
            FROM (
                SELECT
                    timestamp, node_id, source, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY node_id, source, metric_name
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM system_metrics_cpu
                WHERE run_id = ? AND timestamp < ?{metric_filter_sql}
            ) ranked_before
            WHERE rn = 1
        ),
        after_window AS (
            SELECT timestamp, node_id, source, metric_name, value
            FROM (
                SELECT
                    timestamp, node_id, source, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY node_id, source, metric_name
                        ORDER BY timestamp ASC
                    ) AS rn
                FROM system_metrics_cpu
                WHERE run_id = ? AND timestamp >= ?{metric_filter_sql}
            ) ranked_after
            WHERE rn = 1
        )
        SELECT timestamp, node_id, source, metric_name, value
        FROM in_window
        UNION ALL
        SELECT timestamp, node_id, source, metric_name, value
        FROM before_window
        UNION ALL
        SELECT timestamp, node_id, source, metric_name, value
        FROM after_window
        ORDER BY timestamp ASC
    """
    params = [
        req.run_path, interval_start, interval_end, *metric_filter_params,
        req.run_path, interval_start, *metric_filter_params,
        req.run_path, interval_end, *metric_filter_params,
    ]

    rows = con.execute(query, params).fetchall()
    
    metrics = [
        {
            "timestamp": row[0],
            "node_id": row[1],
            "source": row[2],
            "metric_name": row[3],
            "value": row[4],
        }
        for row in rows
    ]
    
    # Get available metric names
    available_metrics = con.execute(
        "SELECT DISTINCT metric_name FROM system_metrics_cpu WHERE run_id = ? ORDER BY metric_name",
        [req.run_path],
    ).fetchall()
    
    log.info(f"[API] Returning CPU metrics page {req.page}/{total_pages}: {len(metrics)} metrics")
    
    return {
        "metrics": metrics,
        "total_pages": total_pages,
        "current_page": current_page,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "global_min_time": global_min_time,
        "global_max_time": global_max_time,
        "available_metrics": [r[0] for r in available_metrics],
    }


@app.post("/vllm-metrics/paginated")
def get_vllm_metrics_paginated(req: VllmMetricsPaginatedRequest):
    """Get paginated vLLM metrics for a specific page/interval."""
    log.info(f"[API] Getting paginated vLLM metrics for {req.run_path}, page={req.page}, interval={req.interval_seconds}s")
    con = connect()
    
    # Get the absolute time range
    time_range = con.execute(
        """
        SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
        FROM vllm_metrics WHERE run_id = ?
        """,
        [req.run_path],
    ).fetchone()
    
    global_min_time = time_range[0]
    global_max_time = time_range[1]
    
    if global_min_time is None or global_max_time is None:
        log.info(f"[API] No vLLM metrics found for {req.run_path}")
        return {
            "metrics": [],
            "total_pages": 1,
            "current_page": 0,
            "interval_start": 0,
            "interval_end": req.interval_seconds,
            "global_min_time": None,
            "global_max_time": None,
            "available_metrics": [],
            "available_servers": [],
        }
    
    # Use anchor_start_time if provided (allows syncing with GPU pagination)
    effective_min_time = req.anchor_start_time if req.anchor_start_time is not None else global_min_time
    
    # Calculate pagination
    total_duration = max(global_max_time - effective_min_time, 0)
    total_pages = max(1, int((total_duration // req.interval_seconds) + 1))
    current_page = req.page
    
    # Calculate the time window for this page
    if req.align_to_latest:
        interval_end = global_max_time
        interval_start = max(effective_min_time, interval_end - req.interval_seconds)
        current_page = max(0, total_pages - 1)
    else:
        interval_start = effective_min_time + (req.page * req.interval_seconds)
        interval_end = interval_start + req.interval_seconds
    
    metric_filter_sql = ""
    metric_filter_params = []
    if req.metric_names:
        placeholders = ", ".join(["?" for _ in req.metric_names])
        metric_filter_sql = f" AND metric_name IN ({placeholders})"
        metric_filter_params = req.metric_names

    # Include one point before and after the interval per series so lines connect at edges
    query = f"""
        WITH in_window AS (
            SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
            FROM vllm_metrics
            WHERE run_id = ? AND timestamp >= ? AND timestamp < ?{metric_filter_sql}
        ),
        before_window AS (
            SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
            FROM (
                SELECT
                    timestamp, server, node_id, tp_group_id, tp_size, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY server, metric_name
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM vllm_metrics
                WHERE run_id = ? AND timestamp < ?{metric_filter_sql}
            ) ranked_before
            WHERE rn = 1
        ),
        after_window AS (
            SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
            FROM (
                SELECT
                    timestamp, server, node_id, tp_group_id, tp_size, metric_name, value,
                    ROW_NUMBER() OVER (
                        PARTITION BY server, metric_name
                        ORDER BY timestamp ASC
                    ) AS rn
                FROM vllm_metrics
                WHERE run_id = ? AND timestamp >= ?{metric_filter_sql}
            ) ranked_after
            WHERE rn = 1
        )
        SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
        FROM in_window
        UNION ALL
        SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
        FROM before_window
        UNION ALL
        SELECT timestamp, server, node_id, tp_group_id, tp_size, metric_name, value
        FROM after_window
        ORDER BY timestamp ASC
    """
    params = [
        req.run_path, interval_start, interval_end, *metric_filter_params,
        req.run_path, interval_start, *metric_filter_params,
        req.run_path, interval_end, *metric_filter_params,
    ]
    
    rows = con.execute(query, params).fetchall()
    
    metrics = [
        {
            "timestamp": row[0],
            "server": row[1],
            "node_id": row[2],
            "tp_group_id": row[3],
            "tp_size": row[4],
            "metric_name": row[5],
            "value": row[6],
        }
        for row in rows
    ]
    
    # Get available metric names and servers
    available_metrics = con.execute(
        "SELECT DISTINCT metric_name FROM vllm_metrics WHERE run_id = ? ORDER BY metric_name",
        [req.run_path],
    ).fetchall()
    
    available_servers = con.execute(
        "SELECT DISTINCT server FROM vllm_metrics WHERE run_id = ? ORDER BY server",
        [req.run_path],
    ).fetchall()
    
    log.info(f"[API] Returning vLLM metrics page {current_page}/{total_pages}: {len(metrics)} metrics")
    
    return {
        "metrics": metrics,
        "total_pages": total_pages,
        "current_page": current_page,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "global_min_time": global_min_time,
        "global_max_time": global_max_time,
        "available_metrics": [r[0] for r in available_metrics],
        "available_servers": [r[0] for r in available_servers],
    }


@app.post("/step-metrics")
def get_step_metrics(req: StepMetricsRequest):
    """Get per-step training metrics computed from prompts, samples_data and rollout metrics data.
    
    Computes aggregated stats (mean, std, min, max) for each metric per step:
    - reward_sum (total reward from samples_data.reward)
    - reward_<name> (dynamic metrics from rollouts_metrics table)
    - advantage (from samples_data.advantage)
    - tokens_prompt (prompt token count from prompts table)
    - tokens (completion token count from rollouts table, only from final turns)
    """
    log.info(f"[API] Getting step metrics for {req.run_path}, metrics={req.metric_names}, limit={req.limit}")
    con = connect()
    
    # Check if we have any rollouts
    count_result = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT step), MIN(step), MAX(step) FROM rollouts WHERE run_id = ?",
        [req.run_path],
    ).fetchone()
    
    total_rollouts = count_result[0]
    total_steps = count_result[1]
    min_step = count_result[2]
    max_step = count_result[3]
    
    if total_rollouts == 0:
        return {
            "metrics": [],
            "total_returned": 0,
            "available_metrics": [],
            "available_rollout_metric_names": [],
            "available_custom_metrics": [],
            "custom_metric_sections": {},
            "min_step": None,
            "max_step": None,
            "total_steps": 0,
        }
    
    # Build step filter
    step_filter = ""
    params = [req.run_path]
    
    if req.start_step is not None:
        step_filter += " AND step >= ?"
        params.append(req.start_step)
    
    if req.end_step is not None:
        step_filter += " AND step <= ?"
        params.append(req.end_step)
    
    # Get all available rollout metric names from the rollouts_metrics table
    metric_names_result = con.execute(
        "SELECT DISTINCT metric_name FROM rollouts_metrics WHERE run_id = ? ORDER BY metric_name",
        [req.run_path],
    ).fetchall()
    available_rollout_metric_names = [r[0] for r in metric_names_result]

    # Get available sample tag names and their values
    tag_names_result = con.execute(
        """
        SELECT tag_name, tag_value, COUNT(*) as cnt
        FROM sample_tags
        WHERE run_id = ?
        GROUP BY tag_name, tag_value
        ORDER BY tag_name, cnt DESC
        """,
        [req.run_path],
    ).fetchall()
    available_sample_tags: dict[str, list[str]] = {}
    for row in tag_names_result:
        tag_name, tag_value = row[0], row[1]
        if tag_name not in available_sample_tags:
            available_sample_tags[tag_name] = []
        available_sample_tags[tag_name].append(tag_value)

    # Build tag filter SQL for rollout metrics queries.
    # _build_tag_filter returns (sql_fragment, params) where the sql_fragment
    # uses the given prefix for step/sample_idx columns.
    def _build_tag_filter(prefix: str = "") -> tuple[str, list]:
        if not req.tag_filters:
            return "", []
        step_col = f"{prefix}step" if prefix else "step"
        sidx_col = f"{prefix}sample_idx" if prefix else "sample_idx"
        conditions = []
        params_out: list = []
        for tag_name, tag_values in req.tag_filters.items():
            if tag_values:
                placeholders = ", ".join(["?"] * len(tag_values))
                conditions.append(
                    f"SELECT DISTINCT step, sample_idx FROM sample_tags "
                    f"WHERE run_id = ? AND tag_name = ? AND tag_value IN ({placeholders})"
                )
                params_out.extend([req.run_path, tag_name] + tag_values)
        if not conditions:
            return "", []
        subquery = conditions[0]
        for cond in conditions[1:]:
            subquery = f"({subquery}) INTERSECT ({cond})"
        return f" AND ({step_col}, {sidx_col}) IN ({subquery})", params_out

    tag_filter_sql, tag_filter_params = _build_tag_filter()
    tag_filter_sql_gm, tag_filter_params_gm = _build_tag_filter("gm.")

    # Group-level tag filter for prompts table (group_id instead of sample_idx)
    def _build_tag_filter_group(prefix: str = "") -> tuple[str, list]:
        if not req.tag_filters:
            return "", []
        step_col = f"{prefix}step" if prefix else "step"
        gid_col = f"{prefix}group_id" if prefix else "group_id"
        conditions = []
        params_out: list = []
        for tag_name, tag_values in req.tag_filters.items():
            if tag_values:
                placeholders = ", ".join(["?"] * len(tag_values))
                conditions.append(
                    f"SELECT DISTINCT step, sample_idx FROM sample_tags "
                    f"WHERE run_id = ? AND tag_name = ? AND tag_value IN ({placeholders})"
                )
                params_out.extend([req.run_path, tag_name] + tag_values)
        if not conditions:
            return "", []
        subquery = conditions[0]
        for cond in conditions[1:]:
            subquery = f"({subquery}) INTERSECT ({cond})"
        group_subquery = (
            f"SELECT DISTINCT sd.step, sd.group_id "
            f"FROM samples_data sd "
            f"WHERE sd.run_id = ? AND (sd.step, sd.sample_idx) IN ({subquery})"
        )
        return f" AND ({step_col}, {gid_col}) IN ({group_subquery})", [req.run_path] + params_out

    tag_filter_sql_group, tag_filter_params_group = _build_tag_filter_group()

    # Build environment filter SQL.
    # For tables with group_id (samples_data, rollouts): filter via prompts table.
    # For tables with env column (rollouts_metrics, prompts): filter directly.
    def _build_env_filter(prefix: str = "") -> tuple[str, list]:
        """Env filter for tables with (step, group_id) — samples_data, rollouts."""
        if not req.env_filters:
            return "", []
        step_col = f"{prefix}step" if prefix else "step"
        gid_col = f"{prefix}group_id" if prefix else "group_id"
        placeholders = ", ".join(["?"] * len(req.env_filters))
        subquery = f"SELECT step, group_id FROM prompts WHERE run_id = ? AND env IN ({placeholders})"
        return f" AND ({step_col}, {gid_col}) IN ({subquery})", [req.run_path] + list(req.env_filters)

    def _build_env_filter_direct(prefix: str = "") -> tuple[str, list]:
        """Env filter for tables with env column — rollouts_metrics, prompts."""
        if not req.env_filters:
            return "", []
        env_col = f"{prefix}env" if prefix else "env"
        placeholders = ", ".join(["?"] * len(req.env_filters))
        return f" AND {env_col} IN ({placeholders})", list(req.env_filters)

    env_filter_sql, env_filter_params = _build_env_filter()
    env_filter_sql_group, env_filter_params_group = _build_env_filter_direct()  # for prompts
    env_filter_sql_rm, env_filter_params_rm = _build_env_filter_direct()  # for rollouts_metrics
    env_filter_sql_gm, env_filter_params_gm = _build_env_filter_direct("gm.")  # for rollouts_metrics Gini (prefixed)

    # Compute base metrics from samples_data table (reward_sum, advantage)
    base_query = f"""
        SELECT
            step,
            -- Reward sum (total reward) - from samples_data
            AVG(reward) as reward_sum_mean,
            STDDEV_SAMP(reward) as reward_sum_std,
            MIN(reward) as reward_sum_min,
            MAX(reward) as reward_sum_max,
            -- Advantage - from samples_data
            AVG(advantage) as advantage_mean,
            STDDEV_SAMP(advantage) as advantage_std,
            MIN(advantage) as advantage_min,
            MAX(advantage) as advantage_max
        FROM samples_data
        WHERE run_id = ? AND reward IS NOT NULL {step_filter} {tag_filter_sql} {env_filter_sql}
        GROUP BY step
        ORDER BY step ASC
    """

    base_rows = con.execute(base_query, params + tag_filter_params + env_filter_params).fetchall()

    # Compute completion tokens from rollouts table (only model turns)
    # First sum tokens per sample where turn_type = 'model', then aggregate per step
    completion_query = f"""
        SELECT
            step,
            AVG(completion_tokens) as length_completion_mean,
            STDDEV_SAMP(completion_tokens) as length_completion_std,
            MIN(completion_tokens) as length_completion_min,
            MAX(completion_tokens) as length_completion_max
        FROM (
            SELECT step, sample_idx, SUM(tokens) as completion_tokens
            FROM rollouts
            WHERE run_id = ? AND turn_type = 'model' {step_filter} {tag_filter_sql} {env_filter_sql}
            GROUP BY step, sample_idx
        )
        GROUP BY step
        ORDER BY step ASC
    """

    completion_rows = con.execute(completion_query, params + tag_filter_params + env_filter_params).fetchall()
    completion_by_step = {row[0]: row[1:] for row in completion_rows}

    # Compute prompt token metrics from prompts table
    prompt_query = f"""
        SELECT
            step,
            AVG(tokens_prompt) as length_prompt_mean,
            STDDEV_SAMP(tokens_prompt) as length_prompt_std,
            MIN(tokens_prompt) as length_prompt_min,
            MAX(tokens_prompt) as length_prompt_max
        FROM prompts
        WHERE run_id = ? {step_filter} {tag_filter_sql_group} {env_filter_sql_group}
        GROUP BY step
        ORDER BY step ASC
    """

    prompt_rows = con.execute(prompt_query, params + tag_filter_params_group + env_filter_params_group).fetchall()
    prompt_metrics_by_step = {row[0]: row[1:] for row in prompt_rows}
    
    # Compute total token stats from samples_data (total_tokens includes everything)
    length_sum_query = f"""
        SELECT
            step,
            AVG(total_tokens) as length_sum_mean,
            STDDEV_SAMP(total_tokens) as length_sum_std,
            MIN(total_tokens) as length_sum_min,
            MAX(total_tokens) as length_sum_max
        FROM samples_data
        WHERE run_id = ? AND reward IS NOT NULL {step_filter} {tag_filter_sql} {env_filter_sql}
        GROUP BY step
        ORDER BY step ASC
    """

    length_sum_rows = con.execute(length_sum_query, params + tag_filter_params + env_filter_params).fetchall()
    length_sum_by_step = {row[0]: row[1:] for row in length_sum_rows}

    # Compute stop_reason = 'length' percentage per step
    # Count samples where at least one model rollout has stop_reason = 'length'
    stop_reason_length_query = f"""
        SELECT
            step,
            COUNT(DISTINCT CASE WHEN stop_reason = 'length' THEN sample_idx END) * 100.0 /
                NULLIF(COUNT(DISTINCT sample_idx), 0) as stop_reason_length_pct
        FROM rollouts
        WHERE run_id = ? AND turn_type = 'model' {step_filter} {tag_filter_sql} {env_filter_sql}
        GROUP BY step
        ORDER BY step ASC
    """

    stop_reason_length_rows = con.execute(stop_reason_length_query, params + tag_filter_params + env_filter_params).fetchall()
    stop_reason_length_by_step = {row[0]: row[1] for row in stop_reason_length_rows}

    # Compute within-group completion-length heterogeneity metrics, averaged per step.
    # Each group contributes equally:
    # - CV: std/mean completion tokens within group
    # - Max/Median ratio: straggler indicator within group
    # - Gini: inequality of completion tokens within group
    group_homogeneity_query = f"""
        WITH completion_per_sample AS (
            SELECT
                step,
                group_id,
                sample_idx,
                SUM(tokens) as completion_tokens
            FROM rollouts
            WHERE run_id = ? AND turn_type = 'model' {step_filter} {tag_filter_sql} {env_filter_sql}
            GROUP BY step, group_id, sample_idx
        ),
        group_stats AS (
            SELECT
                step,
                group_id,
                AVG(completion_tokens) as mean_tokens,
                STDDEV_SAMP(completion_tokens) as std_tokens,
                MAX(completion_tokens) as max_tokens,
                MEDIAN(completion_tokens) as median_tokens,
                COUNT(*) as n
            FROM completion_per_sample
            GROUP BY step, group_id
        ),
        group_ranked AS (
            SELECT
                step,
                group_id,
                completion_tokens,
                ROW_NUMBER() OVER (PARTITION BY step, group_id ORDER BY completion_tokens) as rnk,
                COUNT(*) OVER (PARTITION BY step, group_id) as n,
                SUM(completion_tokens) OVER (PARTITION BY step, group_id) as sum_tokens
            FROM completion_per_sample
        ),
        group_gini AS (
            SELECT
                step,
                group_id,
                CASE
                    WHEN MAX(sum_tokens) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * completion_tokens)) / (MAX(n) * MAX(sum_tokens))
                        - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM group_ranked
            GROUP BY step, group_id
        ),
        group_metrics AS (
            SELECT
                gs.step,
                gs.group_id,
                CASE
                    WHEN gs.mean_tokens > 0 AND gs.n > 1 THEN COALESCE(gs.std_tokens, 0) / gs.mean_tokens
                    ELSE 0
                END as cv,
                CASE
                    WHEN gs.median_tokens > 0 THEN gs.max_tokens / gs.median_tokens
                    WHEN gs.max_tokens = 0 THEN 1
                    ELSE NULL
                END as max_median_ratio,
                gg.gini
            FROM group_stats gs
            LEFT JOIN group_gini gg
                ON gs.step = gg.step AND gs.group_id = gg.group_id
        )
        SELECT
            step,
            AVG(cv) as group_length_cv_mean,
            AVG(max_median_ratio) as group_length_max_median_ratio_mean,
            AVG(gini) as group_length_gini_mean
        FROM group_metrics
        GROUP BY step
        ORDER BY step ASC
    """

    group_homogeneity_rows = con.execute(group_homogeneity_query, params + tag_filter_params + env_filter_params).fetchall()
    group_homogeneity_by_step = {
        row[0]: {
            "group_length_cv_mean": row[1],
            "group_length_max_median_ratio_mean": row[2],
            "group_length_gini_mean": row[3],
        }
        for row in group_homogeneity_rows
    }
    
    # Compute Gini coefficient of rewards within each group, averaged per step
    # Gini measures reward sparsity: 1 = only one sample got reward, 0 = all equal
    # Uses rank-based formula: G = (2*Σ(rank_i * val_i))/(n*Σ(val_i)) - (n+1)/n
    # Only considers non-discarded samples (from samples_data table)
    gini_query = f"""
        WITH ranked AS (
            SELECT
                step, group_id, reward,
                ROW_NUMBER() OVER (PARTITION BY step, group_id ORDER BY reward) as rnk,
                COUNT(*) OVER (PARTITION BY step, group_id) as n,
                SUM(reward) OVER (PARTITION BY step, group_id) as sum_reward
            FROM samples_data
            WHERE run_id = ? AND reward IS NOT NULL {step_filter} {tag_filter_sql} {env_filter_sql}
        ),
        gini_per_group AS (
            SELECT
                step, group_id,
                CASE
                    WHEN MAX(sum_reward) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * reward)) / (MAX(n) * MAX(sum_reward)) - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM ranked
            GROUP BY step, group_id
        )
        SELECT
            step,
            AVG(gini) as reward_gini_mean
        FROM gini_per_group
        GROUP BY step
        ORDER BY step ASC
    """

    gini_rows = con.execute(gini_query, params + tag_filter_params + env_filter_params).fetchall()
    gini_by_step = {row[0]: row[1] for row in gini_rows}

    # Compute off-policy steps stats from events_inference table
    # off_policy_steps = number of weight updates while rollout was in-flight
    # Join with prompts to get the training step (events_inference.step may be NULL)
    off_policy_query = f"""
        SELECT
            p.step,
            AVG(ei.off_policy_steps) as off_policy_steps_mean,
            STDDEV_SAMP(ei.off_policy_steps) as off_policy_steps_std
        FROM events_inference ei
        JOIN prompts p ON p.run_id = ei.run_id AND p.group_id = ei.group_id
        WHERE ei.run_id = ? AND ei.event_type = 'request' AND ei.off_policy_steps IS NOT NULL
            {step_filter.replace("step", "p.step")}
        GROUP BY p.step
        ORDER BY p.step ASC
    """
    off_policy_rows = con.execute(off_policy_query, params).fetchall()
    off_policy_by_step = {row[0]: (row[1], row[2]) for row in off_policy_rows}

    # Base column names for samples_data-based metrics
    base_columns = [
        "step",
        "reward_sum_mean", "reward_sum_std", "reward_sum_min", "reward_sum_max",
        "advantage_mean", "advantage_std", "advantage_min", "advantage_max",
    ]
    
    # Completion tokens column names (from rollouts table, turn_type='model')
    completion_columns = [
        "length_completion_mean", "length_completion_std", "length_completion_min", "length_completion_max",
    ]
    
    # Prompt-based column names
    prompt_columns = [
        "length_prompt_mean", "length_prompt_std", "length_prompt_min", "length_prompt_max",
    ]
    
    # Length sum (prompt + completion) column names
    length_sum_columns = [
        "length_sum_mean", "length_sum_std", "length_sum_min", "length_sum_max",
    ]
    
    # Filter to requested metrics if specified
    metrics_to_include = set(req.metric_names) if req.metric_names else None
    
    # Flatten the base aggregated results into individual metric records
    metrics = []
    for row in base_rows:
        step = row[0]
        for i, col_name in enumerate(base_columns[1:], start=1):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": col_name,
                        "value": float(value),
                    })
        
        # Add completion token metrics for this step (from rollouts table, turn_type='model')
        if step in completion_by_step:
            completion_vals = completion_by_step[step]
            for i, col_name in enumerate(completion_columns):
                if metrics_to_include is None or col_name in metrics_to_include:
                    value = completion_vals[i]
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": col_name,
                            "value": float(value),
                        })
        
        # Add prompt metrics for this step
        if step in prompt_metrics_by_step:
            prompt_vals = prompt_metrics_by_step[step]
            for i, col_name in enumerate(prompt_columns):
                if metrics_to_include is None or col_name in metrics_to_include:
                    value = prompt_vals[i]
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": col_name,
                            "value": float(value),
                        })
        
        # Add length_sum metrics for this step (prompt + completion tokens)
        if step in length_sum_by_step:
            length_sum_vals = length_sum_by_step[step]
            for i, col_name in enumerate(length_sum_columns):
                if metrics_to_include is None or col_name in metrics_to_include:
                    value = length_sum_vals[i]
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": col_name,
                            "value": float(value),
                        })
        
        # Add stop_reason_length_pct metric for this step
        if step in stop_reason_length_by_step:
            if metrics_to_include is None or "stop_reason_length_pct" in metrics_to_include:
                value = stop_reason_length_by_step[step]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": "stop_reason_length_pct",
                        "value": float(value),
                    })

        # Add rollout-group homogeneity metrics for this step
        if step in group_homogeneity_by_step:
            for metric_name, value in group_homogeneity_by_step[step].items():
                if metrics_to_include is None or metric_name in metrics_to_include:
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": metric_name,
                            "value": float(value),
                        })
        
        # Add Gini coefficient metric for this step
        if step in gini_by_step:
            if metrics_to_include is None or "reward_gini_mean" in metrics_to_include:
                value = gini_by_step[step]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": "reward_gini_mean",
                        "value": float(value),
                    })

    # Add off-policy steps metrics (from events_inference, independent of samples_data steps)
    for step, (off_policy_mean, off_policy_std) in off_policy_by_step.items():
        if metrics_to_include is None or "off_policy_steps_mean" in metrics_to_include:
            if off_policy_mean is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "off_policy_steps_mean",
                    "value": float(off_policy_mean),
                })
        if metrics_to_include is None or "off_policy_steps_std" in metrics_to_include:
            if off_policy_std is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "off_policy_steps_std",
                    "value": float(off_policy_std),
                })

    # Compute metrics from rollouts_metrics table for each metric_name
    if available_rollout_metric_names:
        # Build step filter for rollout metrics
        metric_step_filter = ""
        metric_params = [req.run_path]
        
        if req.start_step is not None:
            metric_step_filter += " AND step >= ?"
            metric_params.append(req.start_step)
        
        if req.end_step is not None:
            metric_step_filter += " AND step <= ?"
            metric_params.append(req.end_step)
        
        metrics_query = f"""
            SELECT
                step,
                metric_name,
                AVG(value) as mean,
                STDDEV_SAMP(value) as std,
                MIN(value) as min,
                MAX(value) as max
            FROM rollouts_metrics
            WHERE run_id = ? {metric_step_filter} {tag_filter_sql} {env_filter_sql_rm}
            GROUP BY step, metric_name
            ORDER BY step ASC, metric_name ASC
        """

        metrics_rows = con.execute(
            metrics_query, metric_params + tag_filter_params + env_filter_params_rm
        ).fetchall()
        
        for row in metrics_rows:
            step = row[0]
            metric_base_name = row[1]
            mean_val = row[2]
            std_val = row[3]
            min_val = row[4]
            max_val = row[5]
            
            # Create metric entries for each stat
            stat_values = [
                (f"reward_{metric_base_name}_mean", mean_val),
                (f"reward_{metric_base_name}_std", std_val),
                (f"reward_{metric_base_name}_min", min_val),
                (f"reward_{metric_base_name}_max", max_val),
            ]
            
            for metric_name, value in stat_values:
                if metrics_to_include is None or metric_name in metrics_to_include:
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": metric_name,
                            "value": float(value),
                        })
    
    # Compute Gini coefficient for each rollout metric (sparsity within groups)
    # Join rollouts_metrics with samples_data to get group_id
    if available_rollout_metric_names:
        gen_metric_gini_query = f"""
            WITH ranked AS (
                SELECT 
                    gm.step, gm.metric_name, sd.group_id, gm.value,
                    ROW_NUMBER() OVER (PARTITION BY gm.step, gm.metric_name, sd.group_id ORDER BY gm.value) as rnk,
                    COUNT(*) OVER (PARTITION BY gm.step, gm.metric_name, sd.group_id) as n,
                    SUM(gm.value) OVER (PARTITION BY gm.step, gm.metric_name, sd.group_id) as sum_value
                FROM rollouts_metrics gm
                JOIN samples_data sd ON sd.run_id = gm.run_id AND sd.step = gm.step AND sd.sample_idx = gm.sample_idx
                WHERE gm.run_id = ? AND gm.value IS NOT NULL {metric_step_filter} {tag_filter_sql_gm} {env_filter_sql_gm}
            ),
            gini_per_group AS (
                SELECT
                    step, metric_name, group_id,
                    CASE
                        WHEN MAX(sum_value) > 0 AND MAX(n) > 1 THEN
                            (2.0 * SUM(rnk * value)) / (MAX(n) * MAX(sum_value)) - (MAX(n) + 1.0) / MAX(n)
                        ELSE 0
                    END as gini
                FROM ranked
                GROUP BY step, metric_name, group_id
            )
            SELECT
                step, metric_name, AVG(gini) as gini_mean
            FROM gini_per_group
            GROUP BY step, metric_name
            ORDER BY step ASC, metric_name ASC
        """

        gen_gini_rows = con.execute(
            gen_metric_gini_query, metric_params + tag_filter_params_gm + env_filter_params_gm
        ).fetchall()
        for row in gen_gini_rows:
            step, base_name, gini_val = row[0], row[1], row[2]
            gini_metric_name = f"reward_{base_name}_gini_mean"
            if metrics_to_include is None or gini_metric_name in metrics_to_include:
                if gini_val is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": gini_metric_name,
                        "value": float(gini_val),
                    })
    
    # Build available metrics list
    available_metrics = list(base_columns[1:])  # Base metrics from rollouts
    available_metrics.extend(prompt_columns)  # Add prompt metrics
    available_metrics.append("stop_reason_length_pct")  # Add stop reason length percentage
    available_metrics.extend([
        "group_length_cv_mean",
        "group_length_max_median_ratio_mean",
        "group_length_gini_mean",
    ])
    available_metrics.append("reward_gini_mean")  # Add reward Gini coefficient (sparsity)
    available_metrics.extend(["off_policy_steps_mean", "off_policy_steps_std"])
    for metric_name in available_rollout_metric_names:
        available_metrics.extend([
            f"reward_{metric_name}_mean",
            f"reward_{metric_name}_std",
            f"reward_{metric_name}_min",
            f"reward_{metric_name}_max",
            f"reward_{metric_name}_gini_mean",
        ])
    
    # Query the step_metrics table for custom metrics (grad_norm, kl_divergence, entropy, etc.)
    custom_metrics_query = f"""
        SELECT 
            step,
            metric_name,
            value
        FROM step_metrics
        WHERE run_id = ? {step_filter}
        ORDER BY step ASC, metric_name ASC
    """
    
    custom_rows = con.execute(custom_metrics_query, params).fetchall()
    
    # Get available custom metric names with their sections/groups from step_metrics table
    available_custom_metrics_result = con.execute(
        'SELECT DISTINCT metric_name, section, "group" FROM step_metrics WHERE run_id = ? ORDER BY section, "group", metric_name',
        [req.run_path],
    ).fetchall()
    available_custom_metrics = [r[0] for r in available_custom_metrics_result]
    # Build section -> group -> [metric_names] mapping
    custom_metric_sections: dict[str, dict[str, list[str]]] = {}
    for r in available_custom_metrics_result:
        section_name = r[1] or "General"
        group_name = r[2] or ""
        custom_metric_sections.setdefault(section_name, {}).setdefault(group_name, []).append(r[0])
    
    for row in custom_rows:
        step = row[0]
        metric_name = row[1]
        value = row[2]
        
        if metrics_to_include is None or metric_name in metrics_to_include:
            if value is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(value),
                })
    
    # Add custom metrics to available metrics list
    available_metrics.extend(available_custom_metrics)
    
    # Compute discarded rollout metrics from rollouts_discarded table
    # Use trainer_step as the step for discarded metrics
    discarded_step_filter = ""
    discarded_params = [req.run_path]
    
    if req.start_step is not None:
        discarded_step_filter += " AND trainer_step >= ?"
        discarded_params.append(req.start_step)
    
    if req.end_step is not None:
        discarded_step_filter += " AND trainer_step <= ?"
        discarded_params.append(req.end_step)
    
    # Compute count and percentage metrics for discarded samples
    discarded_count_query = f"""
        SELECT 
            trainer_step as step,
            COUNT(*) as discarded_count,
            SUM(CASE WHEN discard_reason = 'zero_advantage' THEN 1 ELSE 0 END) as zero_advantage_count,
            SUM(CASE WHEN discard_reason = 'max_async' THEN 1 ELSE 0 END) as max_async_count,
            SUM(CASE WHEN discard_reason = 'zero_advantage' AND reward = 0 THEN 1 ELSE 0 END) as zero_advantage_all_zero_count,
            SUM(CASE WHEN discard_reason = 'zero_advantage' AND reward > 0 THEN 1 ELSE 0 END) as zero_advantage_all_positive_count,
            AVG(CASE WHEN discard_reason = 'zero_advantage' THEN reward ELSE NULL END) as zero_advantage_mean_reward
        FROM samples_data_discarded
        WHERE run_id = ? AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_count_rows = con.execute(discarded_count_query, discarded_params).fetchall()
    
    for row in discarded_count_rows:
        step = row[0]
        total_count = row[1]
        zero_advantage_count = row[2]
        max_async_count = row[3]
        zero_advantage_all_zero_count = row[4]
        zero_advantage_all_positive_count = row[5]
        zero_advantage_mean_reward = row[6]
        
        # Discarded count
        if metrics_to_include is None or "discarded_count" in metrics_to_include:
            metrics.append({
                "step": step,
                "metric_name": "discarded_count",
                "value": float(total_count),
            })
        
        # Zero advantage percentage
        if metrics_to_include is None or "discarded_zero_advantage_pct" in metrics_to_include:
            if total_count > 0:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_zero_advantage_pct",
                    "value": float(zero_advantage_count / total_count * 100),
                })
        
        # Zero advantage (all 0) percentage - percentage of zero advantage due to all rewards being 0
        if metrics_to_include is None or "discarded_zero_advantage_all_zero_pct" in metrics_to_include:
            if zero_advantage_count > 0:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_zero_advantage_all_zero_pct",
                    "value": float(zero_advantage_all_zero_count / zero_advantage_count * 100),
                })
        
        # Zero advantage (all > 0) percentage - percentage of zero advantage due to all rewards being > 0
        if metrics_to_include is None or "discarded_zero_advantage_all_positive_pct" in metrics_to_include:
            if zero_advantage_count > 0:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_zero_advantage_all_positive_pct",
                    "value": float(zero_advantage_all_positive_count / zero_advantage_count * 100),
                })
        
        # Zero advantage mean reward
        if metrics_to_include is None or "discarded_zero_advantage_mean_reward" in metrics_to_include:
            if zero_advantage_mean_reward is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_zero_advantage_mean_reward",
                    "value": float(zero_advantage_mean_reward),
                })
        
        # Max async percentage
        if metrics_to_include is None or "discarded_max_async_pct" in metrics_to_include:
            if total_count > 0:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_max_async_pct",
                    "value": float(max_async_count / total_count * 100),
                })
    
    # Compute completion token stats for discarded samples (from rollouts_discarded, turn_type='model')
    # First sum tokens per sample where turn_type = 'model', then aggregate per step
    discarded_completion_query = f"""
        SELECT 
            trainer_step as step,
            AVG(completion_tokens) as discarded_length_completion_mean,
            STDDEV_SAMP(completion_tokens) as discarded_length_completion_std,
            MIN(completion_tokens) as discarded_length_completion_min,
            MAX(completion_tokens) as discarded_length_completion_max
        FROM (
            SELECT trainer_step, sample_idx, SUM(tokens) as completion_tokens
            FROM rollouts_discarded
            WHERE run_id = ? AND turn_type = 'model' {discarded_step_filter}
            GROUP BY trainer_step, sample_idx
        )
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_completion_rows = con.execute(discarded_completion_query, discarded_params).fetchall()
    
    discarded_completion_columns = [
        "discarded_length_completion_mean", "discarded_length_completion_std", 
        "discarded_length_completion_min", "discarded_length_completion_max"
    ]
    
    for row in discarded_completion_rows:
        step = row[0]
        for i, col_name in enumerate(discarded_completion_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": col_name,
                        "value": float(value),
                    })
    
    # Compute prompt token stats for discarded rollouts (from prompts_discarded table)
    discarded_prompt_query = f"""
        SELECT 
            trainer_step as step,
            AVG(tokens_prompt) as discarded_length_prompt_mean,
            STDDEV_SAMP(tokens_prompt) as discarded_length_prompt_std,
            MIN(tokens_prompt) as discarded_length_prompt_min,
            MAX(tokens_prompt) as discarded_length_prompt_max
        FROM prompts_discarded
        WHERE run_id = ? AND tokens_prompt IS NOT NULL {discarded_step_filter}
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_prompt_rows = con.execute(discarded_prompt_query, discarded_params).fetchall()
    discarded_prompt_metrics_by_step = {row[0]: row[1:] for row in discarded_prompt_rows}
    
    discarded_prompt_columns = [
        "discarded_length_prompt_mean", "discarded_length_prompt_std", 
        "discarded_length_prompt_min", "discarded_length_prompt_max"
    ]
    
    for step, prompt_vals in discarded_prompt_metrics_by_step.items():
        for i, col_name in enumerate(discarded_prompt_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = prompt_vals[i]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": col_name,
                        "value": float(value),
                    })
    
    # Compute total token stats from samples_data_discarded (total_tokens includes everything)
    discarded_sum_query = f"""
        SELECT 
            trainer_step as step,
            AVG(total_tokens) as discarded_length_sum_mean,
            STDDEV_SAMP(total_tokens) as discarded_length_sum_std,
            MIN(total_tokens) as discarded_length_sum_min,
            MAX(total_tokens) as discarded_length_sum_max
        FROM samples_data_discarded
        WHERE run_id = ? AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_sum_rows = con.execute(discarded_sum_query, discarded_params).fetchall()
    
    discarded_sum_columns = [
        "discarded_length_sum_mean", "discarded_length_sum_std", 
        "discarded_length_sum_min", "discarded_length_sum_max"
    ]
    
    for row in discarded_sum_rows:
        step = row[0]
        for i, col_name in enumerate(discarded_sum_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": col_name,
                        "value": float(value),
                    })
    
    # Compute discarded stop_reason = 'length' percentage per step
    # Count samples where at least one model rollout has stop_reason = 'length'
    discarded_stop_reason_length_query = f"""
        SELECT 
            trainer_step as step,
            COUNT(DISTINCT CASE WHEN stop_reason = 'length' THEN sample_idx END) * 100.0 / 
                NULLIF(COUNT(DISTINCT sample_idx), 0) as discarded_stop_reason_length_pct
        FROM rollouts_discarded
        WHERE run_id = ? AND turn_type = 'model' {discarded_step_filter}
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_stop_reason_length_rows = con.execute(discarded_stop_reason_length_query, discarded_params).fetchall()
    
    for row in discarded_stop_reason_length_rows:
        step = row[0]
        if metrics_to_include is None or "discarded_stop_reason_length_pct" in metrics_to_include:
            value = row[1]
            if value is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_stop_reason_length_pct",
                    "value": float(value),
                })

    # Compute discarded within-group completion-length heterogeneity metrics.
    # Per group: max/median ratio + gini; then average across groups per step.
    discarded_group_homogeneity_query = f"""
        WITH completion_per_sample AS (
            SELECT
                trainer_step as step,
                group_id,
                sample_idx,
                SUM(tokens) as completion_tokens
            FROM rollouts_discarded
            WHERE run_id = ? AND turn_type = 'model' {discarded_step_filter}
            GROUP BY trainer_step, group_id, sample_idx
        ),
        group_stats AS (
            SELECT
                step,
                group_id,
                MAX(completion_tokens) as max_tokens,
                MEDIAN(completion_tokens) as median_tokens
            FROM completion_per_sample
            GROUP BY step, group_id
        ),
        group_ranked AS (
            SELECT
                step,
                group_id,
                completion_tokens,
                ROW_NUMBER() OVER (PARTITION BY step, group_id ORDER BY completion_tokens) as rnk,
                COUNT(*) OVER (PARTITION BY step, group_id) as n,
                SUM(completion_tokens) OVER (PARTITION BY step, group_id) as sum_tokens
            FROM completion_per_sample
        ),
        group_gini AS (
            SELECT
                step,
                group_id,
                CASE
                    WHEN MAX(sum_tokens) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * completion_tokens)) / (MAX(n) * MAX(sum_tokens))
                        - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM group_ranked
            GROUP BY step, group_id
        ),
        group_metrics AS (
            SELECT
                gs.step,
                gs.group_id,
                CASE
                    WHEN gs.median_tokens > 0 THEN gs.max_tokens / gs.median_tokens
                    WHEN gs.max_tokens = 0 THEN 1
                    ELSE NULL
                END as max_median_ratio,
                gg.gini
            FROM group_stats gs
            LEFT JOIN group_gini gg
                ON gs.step = gg.step AND gs.group_id = gg.group_id
        )
        SELECT
            step,
            AVG(max_median_ratio) as discarded_group_length_max_median_ratio_mean,
            AVG(gini) as discarded_group_length_gini_mean
        FROM group_metrics
        GROUP BY step
        ORDER BY step ASC
    """

    discarded_group_homogeneity_rows = con.execute(
        discarded_group_homogeneity_query,
        discarded_params,
    ).fetchall()

    for row in discarded_group_homogeneity_rows:
        step = row[0]
        max_median_ratio = row[1]
        gini = row[2]
        if (
            metrics_to_include is None
            or "discarded_group_length_max_median_ratio_mean" in metrics_to_include
        ):
            if max_median_ratio is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_group_length_max_median_ratio_mean",
                    "value": float(max_median_ratio),
                })
        if metrics_to_include is None or "discarded_group_length_gini_mean" in metrics_to_include:
            if gini is not None:
                metrics.append({
                    "step": step,
                    "metric_name": "discarded_group_length_gini_mean",
                    "value": float(gini),
                })
    
    # Compute discarded reward_sum and advantage metrics from samples_data_discarded
    discarded_base_query = f"""
        SELECT 
            trainer_step as step,
            AVG(reward) as discarded_reward_sum_mean,
            STDDEV_SAMP(reward) as discarded_reward_sum_std,
            MIN(reward) as discarded_reward_sum_min,
            MAX(reward) as discarded_reward_sum_max,
            AVG(advantage) as discarded_advantage_mean,
            STDDEV_SAMP(advantage) as discarded_advantage_std,
            MIN(advantage) as discarded_advantage_min,
            MAX(advantage) as discarded_advantage_max
        FROM samples_data_discarded
        WHERE run_id = ? AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY trainer_step
        ORDER BY trainer_step ASC
    """
    
    discarded_base_rows = con.execute(discarded_base_query, discarded_params).fetchall()
    
    discarded_base_columns = [
        "discarded_reward_sum_mean", "discarded_reward_sum_std", "discarded_reward_sum_min", "discarded_reward_sum_max",
        "discarded_advantage_mean", "discarded_advantage_std", "discarded_advantage_min", "discarded_advantage_max",
    ]
    
    for row in discarded_base_rows:
        step = row[0]
        for i, col_name in enumerate(discarded_base_columns):
            if metrics_to_include is None or col_name in metrics_to_include:
                value = row[i + 1]
                if value is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": col_name,
                        "value": float(value),
                    })
    
    # Compute discarded rollout metrics per reward name from rollouts_metrics_discarded
    # Need to join with rollouts_discarded to get trainer_step
    discarded_gen_metrics_query = f"""
        SELECT 
            gd.trainer_step as step,
            gmd.metric_name,
            AVG(gmd.value) as mean,
            STDDEV_SAMP(gmd.value) as std,
            MIN(gmd.value) as min_val,
            MAX(gmd.value) as max_val
        FROM rollouts_metrics_discarded gmd
        JOIN (
            SELECT DISTINCT run_id, trainer_step, sample_idx
            FROM rollouts_discarded
            WHERE run_id = ? {discarded_step_filter}
        ) gd ON gmd.run_id = gd.run_id AND gmd.sample_idx = gd.sample_idx
        WHERE gmd.run_id = ?
        GROUP BY gd.trainer_step, gmd.metric_name
        ORDER BY gd.trainer_step ASC, gmd.metric_name ASC
    """
    
    discarded_gen_params = [req.run_path] + discarded_params[1:] + [req.run_path]
    
    try:
        discarded_gen_metrics_rows = con.execute(discarded_gen_metrics_query, discarded_gen_params).fetchall()
        
        discarded_gen_metric_names = set()
        for row in discarded_gen_metrics_rows:
            step = row[0]
            metric_base_name = row[1]
            mean_val = row[2]
            std_val = row[3]
            min_val = row[4]
            max_val = row[5]
            
            discarded_gen_metric_names.add(metric_base_name)
            
            stat_values = [
                (f"discarded_reward_{metric_base_name}_mean", mean_val),
                (f"discarded_reward_{metric_base_name}_std", std_val),
                (f"discarded_reward_{metric_base_name}_min", min_val),
                (f"discarded_reward_{metric_base_name}_max", max_val),
            ]
            
            for metric_name, value in stat_values:
                if metrics_to_include is None or metric_name in metrics_to_include:
                    if value is not None:
                        metrics.append({
                            "step": step,
                            "metric_name": metric_name,
                            "value": float(value),
                        })
    except Exception as e:
        log.warning(f"[API] Could not compute discarded rollout metrics: {e}")
        discarded_gen_metric_names = set()
    
    # Compute discarded off-policy steps (join events_inference with prompts_discarded)
    try:
        discarded_off_policy_query = f"""
            SELECT
                pd.trainer_step,
                AVG(ei.off_policy_steps) as off_policy_steps_mean,
                STDDEV_SAMP(ei.off_policy_steps) as off_policy_steps_std
            FROM events_inference ei
            JOIN prompts_discarded pd ON pd.run_id = ei.run_id AND pd.group_id = ei.group_id
            WHERE ei.run_id = ? AND ei.event_type = 'request'
                AND ei.off_policy_steps IS NOT NULL {discarded_step_filter}
            GROUP BY pd.trainer_step
            ORDER BY pd.trainer_step ASC
        """
        discarded_off_policy_rows = con.execute(discarded_off_policy_query, discarded_params).fetchall()
        for row in discarded_off_policy_rows:
            step = row[0]
            if metrics_to_include is None or "discarded_off_policy_steps_mean" in metrics_to_include:
                if row[1] is not None:
                    metrics.append({"step": step, "metric_name": "discarded_off_policy_steps_mean", "value": float(row[1])})
            if metrics_to_include is None or "discarded_off_policy_steps_std" in metrics_to_include:
                if row[2] is not None:
                    metrics.append({"step": step, "metric_name": "discarded_off_policy_steps_std", "value": float(row[2])})
    except Exception as e:
        log.warning(f"[API] Could not compute discarded off-policy steps: {e}")

    # Add discarded metrics to available metrics list
    discarded_metrics_list = [
        "discarded_count", "discarded_zero_advantage_pct", "discarded_max_async_pct",
        "discarded_stop_reason_length_pct",
        "discarded_group_length_max_median_ratio_mean", "discarded_group_length_gini_mean",
        "discarded_reward_sum_mean", "discarded_reward_sum_std", "discarded_reward_sum_min", "discarded_reward_sum_max",
        "discarded_advantage_mean", "discarded_advantage_std", "discarded_advantage_min", "discarded_advantage_max",
        "discarded_length_prompt_mean", "discarded_length_prompt_std", "discarded_length_prompt_min", "discarded_length_prompt_max",
        "discarded_length_completion_mean", "discarded_length_completion_std", "discarded_length_completion_min", "discarded_length_completion_max",
        "discarded_length_sum_mean", "discarded_length_sum_std", "discarded_length_sum_min", "discarded_length_sum_max",
        "discarded_off_policy_steps_mean", "discarded_off_policy_steps_std",
    ]
    for gen_name in discarded_gen_metric_names:
        discarded_metrics_list.extend([
            f"discarded_reward_{gen_name}_mean",
            f"discarded_reward_{gen_name}_std",
            f"discarded_reward_{gen_name}_min",
            f"discarded_reward_{gen_name}_max",
        ])
    available_metrics.extend(discarded_metrics_list)

    # Compute canceled count per step from events_inference
    # Canceled samples are attributed to the step of the weight_broadcast that follows them.
    # i.e. canceled samples between broadcast(step N-1) and broadcast(step N) → step N
    try:
        canceled_count_query = f"""
            WITH broadcasts AS (
                SELECT
                    step,
                    MIN(start_time) as broadcast_time,
                    LAG(MIN(start_time)) OVER (ORDER BY step) as prev_broadcast_time
                FROM events_inference
                WHERE run_id = ? AND event_type = 'weight_broadcast' AND step IS NOT NULL
                GROUP BY step
            ),
            canceled AS (
                SELECT DISTINCT sample_id, end_time
                FROM events_inference
                WHERE run_id = ? AND event_type = 'request' AND is_canceled = true
            )
            SELECT
                b.step,
                COUNT(DISTINCT c.sample_id) as canceled_count
            FROM broadcasts b
            LEFT JOIN canceled c
                ON c.end_time > COALESCE(b.prev_broadcast_time, -1e18)
                AND c.end_time <= b.broadcast_time
            WHERE 1=1 {step_filter}
            GROUP BY b.step
            ORDER BY b.step ASC
        """
        canceled_count_rows = con.execute(
            canceled_count_query, [req.run_path, req.run_path] + params[1:]
        ).fetchall()

        for row in canceled_count_rows:
            step = row[0]
            if metrics_to_include is None or "canceled_count" in metrics_to_include:
                metrics.append({
                    "step": step,
                    "metric_name": "canceled_count",
                    "value": float(row[1]),
                })

        available_metrics.append("canceled_count")
    except Exception as e:
        log.warning(f"[API] Could not compute canceled count metric: {e}")

    # =========================================================================
    # Compute timing metrics from events_trainer table (single batched query)
    # =========================================================================
    # Event types are clean operation names (e.g. "forward", "backward").
    # The microbatch index is stored in a separate "microbatch" column.
    
    timing_operations = [
        "forward",
        "backward",
        "loss_computation",
        "compute_kl",
        "compute_entropy",
        "data_to_device",
        "prepare_tensors",
    ]
    ops_in_placeholders = ",".join(["?" for _ in timing_operations])
    
    # Build timing step filter
    timing_step_filter = ""
    timing_params = [req.run_path]
    
    if req.start_step is not None:
        timing_step_filter += " AND step >= ?"
        timing_params.append(req.start_step)
    
    if req.end_step is not None:
        timing_step_filter += " AND step <= ?"
        timing_params.append(req.end_step)
    
    # Build params for the timing operations query (run_path + operations + step filters)
    timing_ops_params = [req.run_path] + timing_operations
    if req.start_step is not None:
        timing_ops_params.append(req.start_step)
    if req.end_step is not None:
        timing_ops_params.append(req.end_step)
    
    # Single query for ALL timing operations
    timing_rows = con.execute(f"""
        SELECT
            step,
            event_type as operation,
            SUM(end_time - start_time) as total_time,
            AVG(end_time - start_time) as microbatch_mean,
            STDDEV_SAMP(end_time - start_time) as microbatch_std,
            MIN(end_time - start_time) as microbatch_min,
            MAX(end_time - start_time) as microbatch_max,
            COUNT(*) as microbatch_count
        FROM events_trainer
        WHERE run_id = ?
          AND rank = 0
          AND event_type IN ({ops_in_placeholders})
          {timing_step_filter}
        GROUP BY step, event_type
        ORDER BY step ASC
    """, timing_ops_params).fetchall()
    
    for row in timing_rows:
        step = row[0]
        operation = row[1]
        total_time = row[2]
        microbatch_mean = row[3]
        microbatch_std = row[4]
        microbatch_min = row[5]
        microbatch_max = row[6]
        microbatch_count = row[7]
        
        # Full step total time
        metric_name = f"timing_{operation}_total"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if total_time is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(total_time),
                })
        
        # Microbatch count (only emit once from forward operation to avoid duplicates)
        if operation == "forward":
            metric_name = "timing_microbatch_count"
            if metrics_to_include is None or metric_name in metrics_to_include:
                if microbatch_count is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": metric_name,
                        "value": float(microbatch_count),
                    })
        
        # Microbatch mean
        metric_name = f"timing_{operation}_microbatch_mean"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if microbatch_mean is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(microbatch_mean),
                })
        
        # Microbatch std
        metric_name = f"timing_{operation}_microbatch_std"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if microbatch_std is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(microbatch_std),
                })
        
        # Microbatch min
        metric_name = f"timing_{operation}_microbatch_min"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if microbatch_min is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(microbatch_min),
                })
        
        # Microbatch max
        metric_name = f"timing_{operation}_microbatch_max"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if microbatch_max is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(microbatch_max),
                })
    
    # Compute full step timing metrics
    # timing_step_active: Time within step (first event start to last event end)
    # timing_step_total: Wall-clock time including wait (last event of prev step to last event of this step)
    
    step_boundaries_query = f"""
        SELECT 
            step,
            MIN(start_time) as step_start,
            MAX(end_time) as step_end
        FROM events_trainer
        WHERE run_id = ? AND rank = 0 {timing_step_filter}
        GROUP BY step
        ORDER BY step ASC
    """
    
    step_boundaries_rows = con.execute(step_boundaries_query, timing_params).fetchall()
    
    # Build maps for step boundaries
    step_start_map = {}
    step_end_map = {}
    sorted_steps = []
    
    for row in step_boundaries_rows:
        step = row[0]
        step_start = row[1]
        step_end = row[2]
        step_start_map[step] = step_start
        step_end_map[step] = step_end
        sorted_steps.append(step)
        
        # timing_step_active: first event to last event within step
        active_time = step_end - step_start if step_start is not None and step_end is not None else None
        metric_name = "timing_step_active"
        if metrics_to_include is None or metric_name in metrics_to_include:
            if active_time is not None:
                metrics.append({
                    "step": step,
                    "metric_name": metric_name,
                    "value": float(active_time),
                })
    
    # timing_step_total: last event of previous step to last event of this step
    # For first step, we'll use training_loop_start as baseline
    for i, step in enumerate(sorted_steps):
        if i == 0:
            # First step - will compute after we get training_loop_start
            continue
        else:
            prev_step = sorted_steps[i - 1]
            prev_end = step_end_map.get(prev_step)
            curr_end = step_end_map.get(step)
            
            if prev_end is not None and curr_end is not None:
                total_time = curr_end - prev_end
                metric_name = "timing_step_total"
                if metrics_to_include is None or metric_name in metrics_to_include:
                    metrics.append({
                        "step": step,
                        "metric_name": metric_name,
                        "value": float(total_time),
                    })
    
    # Compute "waiting for data" metric
    # This is the idle time between steps: time from end of step N-1 to start of step N
    # For the first step, we use the training_loop_start orchestrator event as baseline
    waiting_query = f"""
        WITH step_boundaries AS (
            SELECT 
                step,
                MIN(start_time) as step_start,
                MAX(end_time) as step_end
            FROM events_trainer
            WHERE run_id = ? AND rank = 0 {timing_step_filter}
            GROUP BY step
        ),
        step_with_prev AS (
            SELECT 
                step,
                step_start,
                step_end,
                LAG(step_end) OVER (ORDER BY step) as prev_step_end
            FROM step_boundaries
        ),
        training_start AS (
            SELECT MIN(timestamp) as first_event_time
            FROM events_orchestrator
            WHERE run_id = ? AND event_type = 'training_loop_start'
        )
        SELECT 
            s.step,
            CASE 
                WHEN s.prev_step_end IS NULL THEN 
                    s.step_start - COALESCE(t.first_event_time, s.step_start)
                ELSE 
                    s.step_start - s.prev_step_end
            END as wait_time
        FROM step_with_prev s
        CROSS JOIN training_start t
        ORDER BY s.step ASC
    """
    
    try:
        # Build params for waiting query (needs run_path twice)
        waiting_params = [req.run_path, req.run_path]
        
        waiting_rows = con.execute(waiting_query, waiting_params).fetchall()
        
        for row in waiting_rows:
            step = row[0]
            wait_time = row[1]
            
            # Apply step filter manually since the CTE doesn't filter well
            if req.start_step is not None and step < req.start_step:
                continue
            if req.end_step is not None and step > req.end_step:
                continue
            
            metric_name = "timing_waiting_for_data"
            if metrics_to_include is None or metric_name in metrics_to_include:
                if wait_time is not None and wait_time >= 0:
                    metrics.append({
                        "step": step,
                        "metric_name": metric_name,
                        "value": float(wait_time),
                    })
        
        # For the first step, compute timing_step_total using training_loop_start
        if sorted_steps:
            first_step = sorted_steps[0]
            first_step_end = step_end_map.get(first_step)
            
            # Get training_loop_start time
            training_start_row = con.execute(
                "SELECT MIN(timestamp) FROM events_orchestrator WHERE run_id = ? AND event_type = 'training_loop_start'",
                [req.run_path]
            ).fetchone()
            
            if training_start_row and training_start_row[0] is not None and first_step_end is not None:
                training_start = training_start_row[0]
                total_time = first_step_end - training_start
                metric_name = "timing_step_total"
                if metrics_to_include is None or metric_name in metrics_to_include:
                    if total_time >= 0:
                        metrics.append({
                            "step": first_step,
                            "metric_name": metric_name,
                            "value": float(total_time),
                        })
    except Exception as e:
        log.warning(f"[API] Could not compute waiting_for_data metric: {e}")
    
    # Compute batch completion time from orchestrator save_batch events
    # Measure wall-clock interval between successive save_batch events
    # Always compute save_batch_step_map (needed for inference timing percentages)
    save_batch_step_map: dict[int, float] = {}
    inference_timing_metrics = [
        "timing_avg_inference_time", "timing_avg_compute_reward_time",
        "timing_generation_normal_pct", "timing_generation_discarded_pct",
        "timing_generation_canceled_pct", "timing_generation_all_pct",
        "timing_compute_reward_normal_pct", "timing_compute_reward_discarded_pct",
        "timing_compute_reward_canceled_pct", "timing_compute_reward_all_pct",
        "timing_idle_pct",
    ]
    need_save_batch = metrics_to_include is None or "timing_save_batch_total" in metrics_to_include or any(
        m in metrics_to_include for m in inference_timing_metrics
    )
    if need_save_batch:
        try:
            save_batch_query = f"""
                WITH batch_events AS (
                    SELECT
                        step,
                        MIN(timestamp) as batch_time
                    FROM events_orchestrator
                    WHERE run_id = ? AND event_type = 'save_batch' AND step IS NOT NULL
                    GROUP BY step
                ),
                ordered AS (
                    SELECT
                        step,
                        batch_time,
                        LAG(batch_time) OVER (ORDER BY step) as prev_batch_time
                    FROM batch_events
                ),
                training_start AS (
                    SELECT MIN(timestamp) as first_event_time
                    FROM events_orchestrator
                    WHERE run_id = ? AND event_type = 'training_loop_start'
                )
                SELECT
                    o.step,
                    CASE
                        WHEN o.prev_batch_time IS NULL THEN
                            o.batch_time - COALESCE(t.first_event_time, o.batch_time)
                        ELSE
                            o.batch_time - o.prev_batch_time
                    END as batch_duration
                FROM ordered o
                CROSS JOIN training_start t
                ORDER BY o.step ASC
            """

            save_batch_rows = con.execute(
                save_batch_query,
                [req.run_path, req.run_path],
            ).fetchall()

            for row in save_batch_rows:
                step = row[0]
                batch_duration = row[1]

                if step is None:
                    continue
                if req.start_step is not None and step < req.start_step:
                    continue
                if req.end_step is not None and step > req.end_step:
                    continue

                if batch_duration is not None and batch_duration >= 0:
                    save_batch_step_map[step] = float(batch_duration)
                    if metrics_to_include is None or "timing_save_batch_total" in metrics_to_include:
                        metrics.append({
                            "step": step,
                            "metric_name": "timing_save_batch_total",
                            "value": float(batch_duration),
                        })
        except Exception as e:
            log.warning(f"[API] Could not compute save_batch timing metric: {e}")
    
    # Compute trainer weight_broadcast timing (single event per step, not microbatched)
    try:
        weight_broadcast_trainer_query = f"""
            SELECT 
                step,
                SUM(end_time - start_time) as total_time
            FROM events_trainer
            WHERE run_id = ? 
              AND rank = 0
              AND event_type = 'weight_broadcast'
              {timing_step_filter}
            GROUP BY step
            ORDER BY step ASC
        """
        weight_broadcast_trainer_rows = con.execute(weight_broadcast_trainer_query, timing_params).fetchall()
        
        for row in weight_broadcast_trainer_rows:
            step = row[0]
            total_time = row[1]
            
            metric_name = "timing_weight_sync_trainer_total"
            if metrics_to_include is None or metric_name in metrics_to_include:
                if total_time is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": metric_name,
                        "value": float(total_time),
                    })
    except Exception as e:
        log.warning(f"[API] Could not compute trainer weight_broadcast metric: {e}")
    
    # Compute inference weight_broadcast timing (from events_inference table)
    # Uses the stored step column to group weight_broadcast events per step
    try:
        weight_broadcast_inference_query = """
            SELECT
                step,
                (MAX(end_time) - MIN(start_time)) as wall_clock_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'weight_broadcast' AND step IS NOT NULL
            GROUP BY step
            ORDER BY step ASC
        """
        weight_broadcast_inference_rows = con.execute(
            weight_broadcast_inference_query, 
            [req.run_path]
        ).fetchall()
        
        for row in weight_broadcast_inference_rows:
            step = row[0]
            wall_clock_time = row[1]
            
            # Apply step filter
            if req.start_step is not None and step < req.start_step:
                continue
            if req.end_step is not None and step > req.end_step:
                continue
            
            metric_name = "timing_weight_sync_inference_total"
            if metrics_to_include is None or metric_name in metrics_to_include:
                if wall_clock_time is not None:
                    metrics.append({
                        "step": step,
                        "metric_name": metric_name,
                        "value": float(wall_clock_time),
                    })
    except Exception as e:
        log.warning(f"[API] Could not compute inference weight_broadcast metric: {e}")

    # =========================================================================
    # Compute inference timing metrics (generation/reward/idle breakdown)
    # Uses save_batch timestamps as step windows. Events are attributed to
    # the step whose window contains the event's end_time. Generation time
    # is the actual lane occupation (end_time - start_time), clipped to the
    # step window. Compute reward time is attributed to the step where the
    # last turn of the sample ends.
    # =========================================================================
    any_inference_timing_needed = metrics_to_include is None or any(
        m in metrics_to_include for m in inference_timing_metrics
    )
    if any_inference_timing_needed:
        try:
            # Fetch config for total_lanes computation
            config_row = con.execute(
                "SELECT last_config_json FROM ingest_state WHERE run_id = ?",
                [req.run_path],
            ).fetchone()
            run_config = json.loads(config_row[0]) if config_row and config_row[0] else {}

            num_servers = run_config.get("num_inference_servers") or max(
                1, run_config.get("inference_num_workers", 1) // max(1, run_config.get("inference_tensor_parallel_size", 1))
            )
            max_concurrent = run_config.get("max_concurrent_prompts") or (
                run_config.get("max_concurrent_prompts_per_server", 12) * num_servers
            )
            cfg_group_size = run_config.get("group_size", 1)
            lanes_per_server = max(1, int((max_concurrent / num_servers) * cfg_group_size))
            total_lanes = lanes_per_server * num_servers

            # --- Single query: generation + compute_reward sums per step ---
            # Step windows from save_batch; events attributed by end_time;
            # generation = clipped (end_time - start_time);
            # compute_reward attributed to step of last turn per sample.
            timing_rows = con.execute(f"""
                WITH batch_events AS (
                    SELECT step, MIN(timestamp) as batch_time
                    FROM events_orchestrator
                    WHERE run_id = ? AND event_type = 'save_batch' AND step IS NOT NULL
                    GROUP BY step
                ),
                training_start AS (
                    SELECT MIN(timestamp) as ts FROM events_orchestrator
                    WHERE run_id = ? AND event_type = 'training_loop_start'
                ),
                step_windows AS (
                    SELECT step,
                           COALESCE(LAG(batch_time) OVER (ORDER BY step), t.ts) as step_start,
                           batch_time as step_end
                    FROM batch_events CROSS JOIN training_start t
                ),
                request_events AS (
                    SELECT sample_id, group_id, start_time, end_time,
                           inference_time, is_canceled
                    FROM events_inference
                    WHERE run_id = ? AND event_type = 'request'
                      AND (is_eval = false OR is_eval IS NULL)
                ),
                -- last event per sample (for compute_reward attribution)
                last_event_per_sample AS (
                    SELECT group_id, sample_id, MAX(end_time) as last_end
                    FROM request_events
                    GROUP BY group_id, sample_id
                ),
                -- sample classification + compute_reward_time
                sample_info AS (
                    SELECT group_id, sample_idx, compute_reward_time, 'normal' as status
                    FROM samples_data WHERE run_id = ?
                    UNION ALL
                    SELECT group_id, sample_idx, compute_reward_time, 'discarded' as status
                    FROM samples_data_discarded WHERE run_id = ?
                ),
                -- Generation: attribute each event to a step, clip duration
                gen_attributed AS (
                    SELECT
                        sw.step,
                        GREATEST(0, LEAST(re.end_time, sw.step_end) - GREATEST(re.start_time, sw.step_start)) as clipped_gen,
                        re.inference_time,
                        CASE
                            WHEN COALESCE(re.is_canceled, false) THEN 'canceled'
                            WHEN si.status IS NOT NULL THEN si.status
                            ELSE 'normal'
                        END as status
                    FROM request_events re
                    JOIN step_windows sw
                        ON re.end_time > sw.step_start AND re.end_time <= sw.step_end
                    LEFT JOIN sample_info si
                        ON si.group_id = re.group_id AND si.sample_idx = re.sample_id
                ),
                -- Compute reward: attribute to step of last event per sample
                cr_attributed AS (
                    SELECT
                        sw.step,
                        COALESCE(si.compute_reward_time, 0) as cr_time,
                        COALESCE(si.status, 'normal') as status
                    FROM last_event_per_sample le
                    JOIN step_windows sw
                        ON le.last_end > sw.step_start AND le.last_end <= sw.step_end
                    LEFT JOIN sample_info si
                        ON si.group_id = le.group_id AND si.sample_idx = le.sample_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM request_events re2
                        WHERE re2.group_id = le.group_id AND re2.sample_id = le.sample_id
                          AND COALESCE(re2.is_canceled, false) = true
                    )
                ),
                gen_agg AS (
                    SELECT
                        step,
                        SUM(clipped_gen) as gen_all,
                        SUM(CASE WHEN status = 'normal' THEN clipped_gen ELSE 0 END) as gen_normal,
                        SUM(CASE WHEN status = 'discarded' THEN clipped_gen ELSE 0 END) as gen_discarded,
                        SUM(CASE WHEN status = 'canceled' THEN clipped_gen ELSE 0 END) as gen_canceled,
                        AVG(inference_time) as avg_inference_time
                    FROM gen_attributed
                    GROUP BY step
                ),
                cr_agg AS (
                    SELECT
                        step,
                        SUM(cr_time) as cr_all,
                        SUM(CASE WHEN status = 'normal' THEN cr_time ELSE 0 END) as cr_normal,
                        SUM(CASE WHEN status = 'discarded' THEN cr_time ELSE 0 END) as cr_discarded,
                        AVG(CASE WHEN cr_time > 0 THEN cr_time ELSE NULL END) as avg_cr_time
                    FROM cr_attributed
                    GROUP BY step
                )
                SELECT
                    sw.step,
                    sw.step_end - sw.step_start as step_duration,
                    g.gen_all, g.gen_normal, g.gen_discarded, g.gen_canceled,
                    g.avg_inference_time,
                    c.cr_all, c.cr_normal, c.cr_discarded,
                    c.avg_cr_time
                FROM step_windows sw
                LEFT JOIN gen_agg g ON g.step = sw.step
                LEFT JOIN cr_agg c ON c.step = sw.step
                WHERE (g.step IS NOT NULL OR c.step IS NOT NULL)
                ORDER BY sw.step ASC
            """, [req.run_path, req.run_path, req.run_path, req.run_path, req.run_path]).fetchall()

            for row in timing_rows:
                step = row[0]
                step_duration = row[1]

                if req.start_step is not None and step < req.start_step:
                    continue
                if req.end_step is not None and step > req.end_step:
                    continue
                if step_duration is None or step_duration <= 0:
                    continue

                available_lane_seconds = step_duration * total_lanes

                g_all = float(row[2] or 0)
                g_normal = float(row[3] or 0)
                g_discarded = float(row[4] or 0)
                g_canceled = float(row[5] or 0)
                avg_inf = row[6]

                cr_all = float(row[7] or 0)
                cr_normal = float(row[8] or 0)
                cr_discarded = float(row[9] or 0)
                avg_cr = row[10]

                # Avg metrics
                if metrics_to_include is None or "timing_avg_inference_time" in metrics_to_include:
                    if avg_inf is not None:
                        metrics.append({"step": step, "metric_name": "timing_avg_inference_time", "value": float(avg_inf)})
                if metrics_to_include is None or "timing_avg_compute_reward_time" in metrics_to_include:
                    if avg_cr is not None:
                        metrics.append({"step": step, "metric_name": "timing_avg_compute_reward_time", "value": float(avg_cr)})

                # Percentage metrics
                pct = {
                    "timing_generation_normal_pct": g_normal / available_lane_seconds * 100,
                    "timing_generation_discarded_pct": g_discarded / available_lane_seconds * 100,
                    "timing_generation_canceled_pct": g_canceled / available_lane_seconds * 100,
                    "timing_generation_all_pct": g_all / available_lane_seconds * 100,
                    "timing_compute_reward_normal_pct": cr_normal / available_lane_seconds * 100,
                    "timing_compute_reward_discarded_pct": cr_discarded / available_lane_seconds * 100,
                    "timing_compute_reward_canceled_pct": 0.0,
                    "timing_compute_reward_all_pct": cr_all / available_lane_seconds * 100,
                }
                pct["timing_idle_pct"] = max(0.0, 100.0 - pct["timing_generation_all_pct"] - pct["timing_compute_reward_all_pct"])

                for metric_name, value in pct.items():
                    if metrics_to_include is None or metric_name in metrics_to_include:
                        metrics.append({"step": step, "metric_name": metric_name, "value": float(value)})

        except Exception as e:
            log.warning(f"[API] Could not compute inference timing metrics: {e}")

    # Add timing metrics to available metrics list
    timing_metrics_list = ["timing_step_total", "timing_step_active", "timing_microbatch_count"]
    for operation in timing_operations:
        timing_metrics_list.extend([
            f"timing_{operation}_total",
            f"timing_{operation}_microbatch_mean",
            f"timing_{operation}_microbatch_std",
            f"timing_{operation}_microbatch_min",
            f"timing_{operation}_microbatch_max",
        ])
    timing_metrics_list.append("timing_waiting_for_data")
    timing_metrics_list.append("timing_save_batch_total")
    timing_metrics_list.append("timing_weight_sync_trainer_total")
    timing_metrics_list.append("timing_weight_sync_inference_total")
    timing_metrics_list.extend([
        "timing_avg_inference_time", "timing_avg_compute_reward_time",
        "timing_generation_normal_pct", "timing_generation_discarded_pct",
        "timing_generation_canceled_pct", "timing_generation_all_pct",
        "timing_compute_reward_normal_pct", "timing_compute_reward_discarded_pct",
        "timing_compute_reward_canceled_pct", "timing_compute_reward_all_pct",
        "timing_idle_pct",
    ])
    available_metrics.extend(timing_metrics_list)
    
    log.info(
        f"[API] Returning {len(metrics)} step metrics from {len(base_rows)} steps, "
        f"{len(available_rollout_metric_names)} rollout metric types, "
        f"{len(available_custom_metrics)} custom metrics"
    )
    
    return {
        "metrics": metrics,
        "total_returned": len(metrics),
        "available_metrics": available_metrics,
        "available_rollout_metric_names": available_rollout_metric_names,
        "available_sample_tags": available_sample_tags,
        "available_custom_metrics": available_custom_metrics,
        "custom_metric_sections": custom_metric_sections,
        "min_step": min_step,
        "max_step": max_step,
        "total_steps": total_steps,
    }


@app.post("/step-metrics/multi")
def get_step_metrics_multi(req: StepMetricsMultiRequest):
    """Get per-step training metrics for multiple runs in one batched request.

    Optimized: uses batch SQL queries (IN clause + GROUP BY run_id) so the
    total number of queries is constant (~18) regardless of how many runs
    are requested, instead of ~25 queries per run.
    """
    run_paths = [run_path for run_path in req.run_paths if run_path]
    if not run_paths:
        return {"runs": []}

    log.info(
        "[API] Getting step metrics (batched) for %s runs, metrics=%s, limit=%s",
        len(run_paths), req.metric_names, req.limit,
    )

    con = connect()
    metrics_to_include = set(req.metric_names) if req.metric_names else None

    # Shared IN-clause placeholder
    in_ph = ", ".join(["?" for _ in run_paths])

    # Step filters (shared across all runs)
    step_filter = ""
    step_params: list = []
    if req.start_step is not None:
        step_filter += " AND step >= ?"
        step_params.append(req.start_step)
    if req.end_step is not None:
        step_filter += " AND step <= ?"
        step_params.append(req.end_step)

    # Discarded tables use trainer_step instead of step
    discarded_step_filter = ""
    discarded_step_params: list = []
    if req.start_step is not None:
        discarded_step_filter += " AND trainer_step >= ?"
        discarded_step_params.append(req.start_step)
    if req.end_step is not None:
        discarded_step_filter += " AND trainer_step <= ?"
        discarded_step_params.append(req.end_step)

    timing_step_filter = step_filter  # alias – same column name

    # Column-name constants (same as single-run endpoint)
    base_stat_cols = [
        "reward_sum_mean", "reward_sum_std", "reward_sum_min", "reward_sum_max",
        "advantage_mean", "advantage_std", "advantage_min", "advantage_max",
    ]
    completion_columns = [
        "length_completion_mean", "length_completion_std",
        "length_completion_min", "length_completion_max",
    ]
    prompt_columns = [
        "length_prompt_mean", "length_prompt_std",
        "length_prompt_min", "length_prompt_max",
    ]
    length_sum_columns = [
        "length_sum_mean", "length_sum_std",
        "length_sum_min", "length_sum_max",
    ]

    # Per-run accumulators
    run_metrics: dict[str, list] = {rp: [] for rp in run_paths}
    run_gen_metric_names: dict[str, list[str]] = {rp: [] for rp in run_paths}
    run_custom_metrics: dict[str, list[str]] = {rp: [] for rp in run_paths}
    run_discarded_gen_names: dict[str, set[str]] = {rp: set() for rp in run_paths}
    run_meta: dict[str, dict] = {
        rp: {"min_step": None, "max_step": None, "total_steps": 0, "has_data": False}
        for rp in run_paths
    }

    def _add(rid: str, step: int, name: str, value):
        """Append a metric entry if it passes the include filter."""
        if metrics_to_include is None or name in metrics_to_include:
            if value is not None:
                run_metrics[rid].append(
                    {"step": step, "metric_name": name, "value": float(value)}
                )

    # =====================================================================
    # 1. Count check per run
    # =====================================================================
    for row in con.execute(f"""
        SELECT run_id, COUNT(*), COUNT(DISTINCT step), MIN(step), MAX(step)
        FROM rollouts WHERE run_id IN ({in_ph})
        GROUP BY run_id
    """, run_paths).fetchall():
        rid = row[0]
        if row[1] > 0:
            run_meta[rid] = {
                "min_step": row[3], "max_step": row[4],
                "total_steps": row[2], "has_data": True,
            }

    active_runs = [rp for rp in run_paths if run_meta[rp]["has_data"]]

    # If no run has data, return empty results for every run
    if not active_runs:
        return {"runs": [
            {
                "run_path": rp, "metrics": [], "total_returned": 0,
                "available_metrics": [], "available_rollout_metric_names": [],
                "available_custom_metrics": [], "custom_metric_sections": {},
                "min_step": None, "max_step": None, "total_steps": 0,
            }
            for rp in run_paths
        ]}

    a_in_ph = ", ".join(["?" for _ in active_runs])
    a_params = active_runs + step_params             # active_runs + step filter
    a_disc_params = active_runs + discarded_step_params
    a_timing_params = active_runs + step_params

    # Build tag filter for multi-run rollout metrics queries
    def _build_multi_tag_filter(prefix: str = "") -> tuple[str, list]:
        if not req.tag_filters:
            return "", []
        rid_col = f"{prefix}run_id" if prefix else "run_id"
        step_col = f"{prefix}step" if prefix else "step"
        sidx_col = f"{prefix}sample_idx" if prefix else "sample_idx"
        conditions = []
        params_out: list = []
        for tag_name, tag_values in req.tag_filters.items():
            if tag_values:
                placeholders = ", ".join(["?"] * len(tag_values))
                conditions.append(
                    f"SELECT DISTINCT run_id, step, sample_idx FROM sample_tags "
                    f"WHERE tag_name = ? AND tag_value IN ({placeholders})"
                )
                params_out.extend([tag_name] + tag_values)
        if not conditions:
            return "", []
        subquery = conditions[0]
        for cond in conditions[1:]:
            subquery = f"({subquery}) INTERSECT ({cond})"
        return f" AND ({rid_col}, {step_col}, {sidx_col}) IN ({subquery})", params_out

    multi_tag_sql, multi_tag_params = _build_multi_tag_filter()
    multi_tag_sql_gm, multi_tag_params_gm = _build_multi_tag_filter("gm.")

    # Group-level tag filter for prompts table (which has group_id but not sample_idx).
    # Joins through samples_data to map sample_idx tags to group_id.
    def _build_multi_tag_filter_group(prefix: str = "") -> tuple[str, list]:
        if not req.tag_filters:
            return "", []
        rid_col = f"{prefix}run_id" if prefix else "run_id"
        step_col = f"{prefix}step" if prefix else "step"
        gid_col = f"{prefix}group_id" if prefix else "group_id"
        conditions = []
        params_out: list = []
        for tag_name, tag_values in req.tag_filters.items():
            if tag_values:
                placeholders = ", ".join(["?"] * len(tag_values))
                conditions.append(
                    f"SELECT DISTINCT run_id, step, sample_idx FROM sample_tags "
                    f"WHERE tag_name = ? AND tag_value IN ({placeholders})"
                )
                params_out.extend([tag_name] + tag_values)
        if not conditions:
            return "", []
        subquery = conditions[0]
        for cond in conditions[1:]:
            subquery = f"({subquery}) INTERSECT ({cond})"
        # Map sample_idx back to group_id via samples_data
        group_subquery = (
            f"SELECT DISTINCT sd.run_id, sd.step, sd.group_id "
            f"FROM samples_data sd "
            f"WHERE (sd.run_id, sd.step, sd.sample_idx) IN ({subquery})"
        )
        return f" AND ({rid_col}, {step_col}, {gid_col}) IN ({group_subquery})", params_out

    multi_tag_sql_group, multi_tag_params_group = _build_multi_tag_filter_group()

    # Build environment filter for multi-run queries.
    def _build_multi_env_filter(prefix: str = "") -> tuple[str, list]:
        """Env filter for tables with (run_id, step, group_id) — samples_data, rollouts."""
        if not req.env_filters:
            return "", []
        rid_col = f"{prefix}run_id" if prefix else "run_id"
        step_col = f"{prefix}step" if prefix else "step"
        gid_col = f"{prefix}group_id" if prefix else "group_id"
        placeholders = ", ".join(["?"] * len(req.env_filters))
        subquery = f"SELECT run_id, step, group_id FROM prompts WHERE env IN ({placeholders})"
        return f" AND ({rid_col}, {step_col}, {gid_col}) IN ({subquery})", list(req.env_filters)

    def _build_multi_env_filter_direct(prefix: str = "") -> tuple[str, list]:
        """Env filter for tables with env column — rollouts_metrics, prompts."""
        if not req.env_filters:
            return "", []
        env_col = f"{prefix}env" if prefix else "env"
        placeholders = ", ".join(["?"] * len(req.env_filters))
        return f" AND {env_col} IN ({placeholders})", list(req.env_filters)

    multi_env_sql, multi_env_params = _build_multi_env_filter()
    multi_env_sql_group, multi_env_params_group = _build_multi_env_filter_direct()  # prompts
    multi_env_sql_rm, multi_env_params_rm = _build_multi_env_filter_direct()  # rollouts_metrics
    multi_env_sql_gm, multi_env_params_gm = _build_multi_env_filter_direct("gm.")  # rollouts_metrics Gini

    # =====================================================================
    # 2. Available rollout metric names per run
    # =====================================================================
    for row in con.execute(f"""
        SELECT DISTINCT run_id, metric_name
        FROM rollouts_metrics WHERE run_id IN ({a_in_ph})
        ORDER BY run_id, metric_name
    """, active_runs).fetchall():
        run_gen_metric_names[row[0]].append(row[1])

    # =====================================================================
    # 3. Base metrics from samples_data (reward_sum, advantage)
    # =====================================================================
    base_rows = con.execute(f"""
        SELECT
            run_id, step,
            AVG(reward), STDDEV_SAMP(reward), MIN(reward), MAX(reward),
            AVG(advantage), STDDEV_SAMP(advantage), MIN(advantage), MAX(advantage)
        FROM samples_data
        WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {step_filter} {multi_tag_sql} {multi_env_sql}
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall()

    # Track (run_id, step) pairs from base rows for joining other tables
    base_keys: set[tuple[str, int]] = set()
    for row in base_rows:
        rid, step = row[0], row[1]
        base_keys.add((rid, step))
        for i, col in enumerate(base_stat_cols):
            _add(rid, step, col, row[i + 2])

    # =====================================================================
    # 4. Completion tokens (from rollouts, turn_type='model')
    # =====================================================================
    completion_map: dict[tuple[str, int], tuple] = {}
    for row in con.execute(f"""
        SELECT
            run_id, step,
            AVG(completion_tokens), STDDEV_SAMP(completion_tokens),
            MIN(completion_tokens), MAX(completion_tokens)
        FROM (
            SELECT run_id, step, sample_idx, SUM(tokens) as completion_tokens
            FROM rollouts
            WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {step_filter} {multi_tag_sql} {multi_env_sql}
            GROUP BY run_id, step, sample_idx
        )
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall():
        completion_map[(row[0], row[1])] = row[2:]

    # =====================================================================
    # 5. Prompt tokens (from prompts)
    # =====================================================================
    prompt_map: dict[tuple[str, int], tuple] = {}
    for row in con.execute(f"""
        SELECT
            run_id, step,
            AVG(tokens_prompt), STDDEV_SAMP(tokens_prompt),
            MIN(tokens_prompt), MAX(tokens_prompt)
        FROM prompts
        WHERE run_id IN ({a_in_ph}) {step_filter} {multi_tag_sql_group} {multi_env_sql_group}
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params_group + multi_env_params_group).fetchall():
        prompt_map[(row[0], row[1])] = row[2:]

    # =====================================================================
    # 6. Length sum (total_tokens from samples_data)
    # =====================================================================
    length_sum_map: dict[tuple[str, int], tuple] = {}
    for row in con.execute(f"""
        SELECT
            run_id, step,
            AVG(total_tokens), STDDEV_SAMP(total_tokens),
            MIN(total_tokens), MAX(total_tokens)
        FROM samples_data
        WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {step_filter} {multi_tag_sql} {multi_env_sql}
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall():
        length_sum_map[(row[0], row[1])] = row[2:]

    # =====================================================================
    # 7. Stop reason 'length' percentage
    # =====================================================================
    stop_reason_map: dict[tuple[str, int], float] = {}
    for row in con.execute(f"""
        SELECT
            run_id, step,
            COUNT(DISTINCT CASE WHEN stop_reason = 'length' THEN sample_idx END) * 100.0 /
                NULLIF(COUNT(DISTINCT sample_idx), 0)
        FROM rollouts
        WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {step_filter} {multi_tag_sql} {multi_env_sql}
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall():
        stop_reason_map[(row[0], row[1])] = row[2]

    # =====================================================================
    # 7a. Group completion-length heterogeneity (CV, Max/Median, Gini)
    # =====================================================================
    group_homogeneity_map: dict[tuple[str, int], tuple[float | None, float | None, float | None]] = {}
    for row in con.execute(f"""
        WITH completion_per_sample AS (
            SELECT
                run_id,
                step,
                group_id,
                sample_idx,
                SUM(tokens) as completion_tokens
            FROM rollouts
            WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {step_filter} {multi_tag_sql} {multi_env_sql}
            GROUP BY run_id, step, group_id, sample_idx
        ),
        group_stats AS (
            SELECT
                run_id,
                step,
                group_id,
                AVG(completion_tokens) as mean_tokens,
                STDDEV_SAMP(completion_tokens) as std_tokens,
                MAX(completion_tokens) as max_tokens,
                MEDIAN(completion_tokens) as median_tokens,
                COUNT(*) as n
            FROM completion_per_sample
            GROUP BY run_id, step, group_id
        ),
        group_ranked AS (
            SELECT
                run_id,
                step,
                group_id,
                completion_tokens,
                ROW_NUMBER() OVER (PARTITION BY run_id, step, group_id ORDER BY completion_tokens) as rnk,
                COUNT(*) OVER (PARTITION BY run_id, step, group_id) as n,
                SUM(completion_tokens) OVER (PARTITION BY run_id, step, group_id) as sum_tokens
            FROM completion_per_sample
        ),
        group_gini AS (
            SELECT
                run_id,
                step,
                group_id,
                CASE
                    WHEN MAX(sum_tokens) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * completion_tokens)) / (MAX(n) * MAX(sum_tokens))
                        - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM group_ranked
            GROUP BY run_id, step, group_id
        ),
        group_metrics AS (
            SELECT
                gs.run_id,
                gs.step,
                gs.group_id,
                CASE
                    WHEN gs.mean_tokens > 0 AND gs.n > 1 THEN COALESCE(gs.std_tokens, 0) / gs.mean_tokens
                    ELSE 0
                END as cv,
                CASE
                    WHEN gs.median_tokens > 0 THEN gs.max_tokens / gs.median_tokens
                    WHEN gs.max_tokens = 0 THEN 1
                    ELSE NULL
                END as max_median_ratio,
                gg.gini
            FROM group_stats gs
            LEFT JOIN group_gini gg
                ON gs.run_id = gg.run_id AND gs.step = gg.step AND gs.group_id = gg.group_id
        )
        SELECT
            run_id,
            step,
            AVG(cv) as group_length_cv_mean,
            AVG(max_median_ratio) as group_length_max_median_ratio_mean,
            AVG(gini) as group_length_gini_mean
        FROM group_metrics
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall():
        group_homogeneity_map[(row[0], row[1])] = (row[2], row[3], row[4])

    # =====================================================================
    # 7b. Gini coefficient of rewards within groups (reward sparsity)
    # =====================================================================
    gini_map: dict[tuple[str, int], float] = {}
    for row in con.execute(f"""
        WITH ranked AS (
            SELECT
                run_id, step, group_id, reward,
                ROW_NUMBER() OVER (PARTITION BY run_id, step, group_id ORDER BY reward) as rnk,
                COUNT(*) OVER (PARTITION BY run_id, step, group_id) as n,
                SUM(reward) OVER (PARTITION BY run_id, step, group_id) as sum_reward
            FROM samples_data
            WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {step_filter} {multi_tag_sql} {multi_env_sql}
        ),
        gini_per_group AS (
            SELECT
                run_id, step, group_id,
                CASE
                    WHEN MAX(sum_reward) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * reward)) / (MAX(n) * MAX(sum_reward)) - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM ranked
            GROUP BY run_id, step, group_id
        )
        SELECT
            run_id, step, AVG(gini)
        FROM gini_per_group
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_params + multi_tag_params + multi_env_params).fetchall():
        gini_map[(row[0], row[1])] = row[2]

    # =====================================================================
    # 7c. Off-policy steps (from events_inference)
    # =====================================================================
    off_policy_map: dict[tuple[str, int], tuple[float | None, float | None]] = {}
    for row in con.execute(f"""
        SELECT
            p.run_id, p.step,
            AVG(ei.off_policy_steps),
            STDDEV_SAMP(ei.off_policy_steps)
        FROM events_inference ei
        JOIN prompts p ON p.run_id = ei.run_id AND p.group_id = ei.group_id
        WHERE ei.run_id IN ({a_in_ph}) AND ei.event_type = 'request'
            AND ei.off_policy_steps IS NOT NULL {step_filter.replace("step", "p.step")}
        GROUP BY p.run_id, p.step
        ORDER BY p.run_id, p.step ASC
    """, a_params).fetchall():
        off_policy_map[(row[0], row[1])] = (row[2], row[3])

    # Emit off-policy metrics (independent of base_keys since they come from events_inference)
    for (rid, step), (op_mean, op_std) in off_policy_map.items():
        _add(rid, step, "off_policy_steps_mean", op_mean)
        _add(rid, step, "off_policy_steps_std", op_std)

    # Emit joined metrics for each base-row step
    for key in base_keys:
        rid, step = key
        if key in completion_map:
            for i, col in enumerate(completion_columns):
                _add(rid, step, col, completion_map[key][i])
        if key in prompt_map:
            for i, col in enumerate(prompt_columns):
                _add(rid, step, col, prompt_map[key][i])
        if key in length_sum_map:
            for i, col in enumerate(length_sum_columns):
                _add(rid, step, col, length_sum_map[key][i])
        if key in stop_reason_map:
            _add(rid, step, "stop_reason_length_pct", stop_reason_map[key])
        if key in group_homogeneity_map:
            group_cv, group_max_median_ratio, group_gini = group_homogeneity_map[key]
            _add(rid, step, "group_length_cv_mean", group_cv)
            _add(rid, step, "group_length_max_median_ratio_mean", group_max_median_ratio)
            _add(rid, step, "group_length_gini_mean", group_gini)
        if key in gini_map:
            _add(rid, step, "reward_gini_mean", gini_map[key])

    # =====================================================================
    # 8. Rollout metrics (rollouts_metrics)
    # =====================================================================
    for row in con.execute(f"""
        SELECT
            run_id, step, metric_name,
            AVG(value), STDDEV_SAMP(value), MIN(value), MAX(value)
        FROM rollouts_metrics
        WHERE run_id IN ({a_in_ph}) {step_filter} {multi_tag_sql} {multi_env_sql_rm}
        GROUP BY run_id, step, metric_name
        ORDER BY run_id, step ASC, metric_name ASC
    """, a_params + multi_tag_params + multi_env_params_rm).fetchall():
        rid, step, base_name = row[0], row[1], row[2]
        for suffix, val in [("_mean", row[3]), ("_std", row[4]),
                            ("_min", row[5]), ("_max", row[6])]:
            _add(rid, step, f"reward_{base_name}{suffix}", val)

    # =====================================================================
    # 8b. Gini coefficient for each rollout metric (sparsity within groups)
    # =====================================================================
    for row in con.execute(f"""
        WITH ranked AS (
            SELECT
                gm.run_id, gm.step, gm.metric_name, sd.group_id, gm.value,
                ROW_NUMBER() OVER (PARTITION BY gm.run_id, gm.step, gm.metric_name, sd.group_id ORDER BY gm.value) as rnk,
                COUNT(*) OVER (PARTITION BY gm.run_id, gm.step, gm.metric_name, sd.group_id) as n,
                SUM(gm.value) OVER (PARTITION BY gm.run_id, gm.step, gm.metric_name, sd.group_id) as sum_value
            FROM rollouts_metrics gm
            JOIN samples_data sd ON sd.run_id = gm.run_id AND sd.step = gm.step AND sd.sample_idx = gm.sample_idx
            WHERE gm.run_id IN ({a_in_ph}) AND gm.value IS NOT NULL {step_filter} {multi_tag_sql_gm} {multi_env_sql_gm}
        ),
        gini_per_group AS (
            SELECT
                run_id, step, metric_name, group_id,
                CASE
                    WHEN MAX(sum_value) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * value)) / (MAX(n) * MAX(sum_value)) - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM ranked
            GROUP BY run_id, step, metric_name, group_id
        )
        SELECT
            run_id, step, metric_name, AVG(gini)
        FROM gini_per_group
        GROUP BY run_id, step, metric_name
        ORDER BY run_id, step ASC, metric_name ASC
    """, a_params + multi_tag_params_gm + multi_env_params_gm).fetchall():
        _add(row[0], row[1], f"reward_{row[2]}_gini_mean", row[3])

    # =====================================================================
    # 9. General metrics from step_metrics
    # =====================================================================
    for row in con.execute(f"""
        SELECT run_id, step, metric_name, value
        FROM step_metrics
        WHERE run_id IN ({a_in_ph}) {step_filter}
        ORDER BY run_id, step ASC, metric_name ASC
    """, a_params).fetchall():
        _add(row[0], row[1], row[2], row[3])

    # =====================================================================
    # 10. Available custom metric names + sections/groups per run
    # =====================================================================
    run_custom_metric_sections: dict[str, dict[str, dict[str, list[str]]]] = {rp: {} for rp in run_paths}
    for row in con.execute(f"""
        SELECT DISTINCT run_id, metric_name, section, "group"
        FROM step_metrics WHERE run_id IN ({a_in_ph})
        ORDER BY run_id, section, "group", metric_name
    """, active_runs).fetchall():
        run_custom_metrics[row[0]].append(row[1])
        section_name = row[2] or "General"
        group_name = row[3] or ""
        run_custom_metric_sections[row[0]].setdefault(section_name, {}).setdefault(group_name, []).append(row[1])

    # =====================================================================
    # 11. Discarded: count & percentage metrics
    # =====================================================================
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            COUNT(*),
            SUM(CASE WHEN discard_reason = 'zero_advantage' THEN 1 ELSE 0 END),
            SUM(CASE WHEN discard_reason = 'max_async' THEN 1 ELSE 0 END),
            SUM(CASE WHEN discard_reason = 'zero_advantage' AND reward = 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN discard_reason = 'zero_advantage' AND reward > 0 THEN 1 ELSE 0 END),
            AVG(CASE WHEN discard_reason = 'zero_advantage' THEN reward ELSE NULL END)
        FROM samples_data_discarded
        WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        rid, step = row[0], row[1]
        total_count, zero_adv, max_async = row[2], row[3], row[4]
        zero_adv_zero, zero_adv_pos, zero_adv_mean = row[5], row[6], row[7]
        _add(rid, step, "discarded_count", total_count)
        if total_count and total_count > 0:
            _add(rid, step, "discarded_zero_advantage_pct", zero_adv / total_count * 100)
            _add(rid, step, "discarded_max_async_pct", max_async / total_count * 100)
        if zero_adv and zero_adv > 0:
            _add(rid, step, "discarded_zero_advantage_all_zero_pct",
                 zero_adv_zero / zero_adv * 100)
            _add(rid, step, "discarded_zero_advantage_all_positive_pct",
                 zero_adv_pos / zero_adv * 100)
        _add(rid, step, "discarded_zero_advantage_mean_reward", zero_adv_mean)

    # =====================================================================
    # 12. Discarded completion tokens
    # =====================================================================
    discarded_compl_cols = [
        "discarded_length_completion_mean", "discarded_length_completion_std",
        "discarded_length_completion_min", "discarded_length_completion_max",
    ]
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            AVG(completion_tokens), STDDEV_SAMP(completion_tokens),
            MIN(completion_tokens), MAX(completion_tokens)
        FROM (
            SELECT run_id, trainer_step, sample_idx, SUM(tokens) as completion_tokens
            FROM rollouts_discarded
            WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {discarded_step_filter}
            GROUP BY run_id, trainer_step, sample_idx
        )
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        rid, step = row[0], row[1]
        for i, col in enumerate(discarded_compl_cols):
            _add(rid, step, col, row[i + 2])

    # =====================================================================
    # 13. Discarded prompt tokens
    # =====================================================================
    discarded_prompt_cols = [
        "discarded_length_prompt_mean", "discarded_length_prompt_std",
        "discarded_length_prompt_min", "discarded_length_prompt_max",
    ]
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            AVG(tokens_prompt), STDDEV_SAMP(tokens_prompt),
            MIN(tokens_prompt), MAX(tokens_prompt)
        FROM prompts_discarded
        WHERE run_id IN ({a_in_ph}) AND tokens_prompt IS NOT NULL {discarded_step_filter}
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        rid, step = row[0], row[1]
        for i, col in enumerate(discarded_prompt_cols):
            _add(rid, step, col, row[i + 2])

    # =====================================================================
    # 14. Discarded total tokens (length_sum)
    # =====================================================================
    discarded_sum_cols = [
        "discarded_length_sum_mean", "discarded_length_sum_std",
        "discarded_length_sum_min", "discarded_length_sum_max",
    ]
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            AVG(total_tokens), STDDEV_SAMP(total_tokens),
            MIN(total_tokens), MAX(total_tokens)
        FROM samples_data_discarded
        WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        rid, step = row[0], row[1]
        for i, col in enumerate(discarded_sum_cols):
            _add(rid, step, col, row[i + 2])

    # =====================================================================
    # 15. Discarded stop reason 'length' percentage
    # =====================================================================
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            COUNT(DISTINCT CASE WHEN stop_reason = 'length' THEN sample_idx END) * 100.0 /
                NULLIF(COUNT(DISTINCT sample_idx), 0)
        FROM rollouts_discarded
        WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {discarded_step_filter}
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        _add(row[0], row[1], "discarded_stop_reason_length_pct", row[2])

    # =====================================================================
    # 15a. Discarded group completion-length heterogeneity (Max/Median, Gini)
    # =====================================================================
    for row in con.execute(f"""
        WITH completion_per_sample AS (
            SELECT
                run_id,
                trainer_step as step,
                group_id,
                sample_idx,
                SUM(tokens) as completion_tokens
            FROM rollouts_discarded
            WHERE run_id IN ({a_in_ph}) AND turn_type = 'model' {discarded_step_filter}
            GROUP BY run_id, trainer_step, group_id, sample_idx
        ),
        group_stats AS (
            SELECT
                run_id,
                step,
                group_id,
                MAX(completion_tokens) as max_tokens,
                MEDIAN(completion_tokens) as median_tokens
            FROM completion_per_sample
            GROUP BY run_id, step, group_id
        ),
        group_ranked AS (
            SELECT
                run_id,
                step,
                group_id,
                completion_tokens,
                ROW_NUMBER() OVER (PARTITION BY run_id, step, group_id ORDER BY completion_tokens) as rnk,
                COUNT(*) OVER (PARTITION BY run_id, step, group_id) as n,
                SUM(completion_tokens) OVER (PARTITION BY run_id, step, group_id) as sum_tokens
            FROM completion_per_sample
        ),
        group_gini AS (
            SELECT
                run_id,
                step,
                group_id,
                CASE
                    WHEN MAX(sum_tokens) > 0 AND MAX(n) > 1 THEN
                        (2.0 * SUM(rnk * completion_tokens)) / (MAX(n) * MAX(sum_tokens))
                        - (MAX(n) + 1.0) / MAX(n)
                    ELSE 0
                END as gini
            FROM group_ranked
            GROUP BY run_id, step, group_id
        ),
        group_metrics AS (
            SELECT
                gs.run_id,
                gs.step,
                gs.group_id,
                CASE
                    WHEN gs.median_tokens > 0 THEN gs.max_tokens / gs.median_tokens
                    WHEN gs.max_tokens = 0 THEN 1
                    ELSE NULL
                END as max_median_ratio,
                gg.gini
            FROM group_stats gs
            LEFT JOIN group_gini gg
                ON gs.run_id = gg.run_id AND gs.step = gg.step AND gs.group_id = gg.group_id
        )
        SELECT
            run_id,
            step,
            AVG(max_median_ratio) as discarded_group_length_max_median_ratio_mean,
            AVG(gini) as discarded_group_length_gini_mean
        FROM group_metrics
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_disc_params).fetchall():
        _add(row[0], row[1], "discarded_group_length_max_median_ratio_mean", row[2])
        _add(row[0], row[1], "discarded_group_length_gini_mean", row[3])

    # =====================================================================
    # 16. Discarded base metrics (reward_sum, advantage)
    # =====================================================================
    discarded_base_cols = [
        "discarded_reward_sum_mean", "discarded_reward_sum_std",
        "discarded_reward_sum_min", "discarded_reward_sum_max",
        "discarded_advantage_mean", "discarded_advantage_std",
        "discarded_advantage_min", "discarded_advantage_max",
    ]
    for row in con.execute(f"""
        SELECT
            run_id, trainer_step,
            AVG(reward), STDDEV_SAMP(reward), MIN(reward), MAX(reward),
            AVG(advantage), STDDEV_SAMP(advantage), MIN(advantage), MAX(advantage)
        FROM samples_data_discarded
        WHERE run_id IN ({a_in_ph}) AND reward IS NOT NULL {discarded_step_filter}
        GROUP BY run_id, trainer_step
        ORDER BY run_id, trainer_step ASC
    """, a_disc_params).fetchall():
        rid, step = row[0], row[1]
        for i, col in enumerate(discarded_base_cols):
            _add(rid, step, col, row[i + 2])

    # =====================================================================
    # 17. Discarded rollout metrics (per reward name)
    # =====================================================================
    try:
        for row in con.execute(f"""
            SELECT
                gd.run_id, gd.trainer_step, gmd.metric_name,
                AVG(gmd.value), STDDEV_SAMP(gmd.value),
                MIN(gmd.value), MAX(gmd.value)
            FROM rollouts_metrics_discarded gmd
            JOIN (
                SELECT DISTINCT run_id, trainer_step, sample_idx
                FROM rollouts_discarded
                WHERE run_id IN ({a_in_ph}) {discarded_step_filter}
            ) gd ON gmd.run_id = gd.run_id AND gmd.sample_idx = gd.sample_idx
            WHERE gmd.run_id IN ({a_in_ph})
            GROUP BY gd.run_id, gd.trainer_step, gmd.metric_name
            ORDER BY gd.run_id, gd.trainer_step ASC, gmd.metric_name ASC
        """, active_runs + discarded_step_params + active_runs).fetchall():
            rid, step, base_name = row[0], row[1], row[2]
            run_discarded_gen_names[rid].add(base_name)
            for suffix, val in [("_mean", row[3]), ("_std", row[4]),
                                ("_min", row[5]), ("_max", row[6])]:
                _add(rid, step, f"discarded_reward_{base_name}{suffix}", val)
    except Exception as e:
        log.warning(f"[API] Could not compute discarded rollout metrics: {e}")

    # =====================================================================
    # 17a. Discarded off-policy steps (join events_inference with prompts_discarded)
    # =====================================================================
    try:
        for row in con.execute(f"""
            SELECT
                pd.run_id, pd.trainer_step,
                AVG(ei.off_policy_steps),
                STDDEV_SAMP(ei.off_policy_steps)
            FROM events_inference ei
            JOIN prompts_discarded pd ON pd.run_id = ei.run_id AND pd.group_id = ei.group_id
            WHERE ei.run_id IN ({a_in_ph}) AND ei.event_type = 'request'
                AND ei.off_policy_steps IS NOT NULL {discarded_step_filter.replace("trainer_step", "pd.trainer_step")}
            GROUP BY pd.run_id, pd.trainer_step
            ORDER BY pd.run_id, pd.trainer_step ASC
        """, active_runs + discarded_step_params).fetchall():
            _add(row[0], row[1], "discarded_off_policy_steps_mean", row[2])
            _add(row[0], row[1], "discarded_off_policy_steps_std", row[3])
    except Exception as e:
        log.warning(f"[API] Could not compute discarded off-policy steps: {e}")

    # =====================================================================
    # 17b. Canceled count per step (from events_inference)
    # =====================================================================
    try:
        for row in con.execute(f"""
            WITH broadcasts AS (
                SELECT
                    run_id,
                    step,
                    MIN(start_time) as broadcast_time,
                    LAG(MIN(start_time)) OVER (PARTITION BY run_id ORDER BY step) as prev_broadcast_time
                FROM events_inference
                WHERE run_id IN ({a_in_ph}) AND event_type = 'weight_broadcast' AND step IS NOT NULL
                GROUP BY run_id, step
            ),
            canceled AS (
                SELECT DISTINCT run_id, sample_id, end_time
                FROM events_inference
                WHERE run_id IN ({a_in_ph}) AND event_type = 'request' AND is_canceled = true
            )
            SELECT
                b.run_id,
                b.step,
                COUNT(DISTINCT c.sample_id) as canceled_count
            FROM broadcasts b
            LEFT JOIN canceled c
                ON c.run_id = b.run_id
                AND c.end_time > COALESCE(b.prev_broadcast_time, -1e18)
                AND c.end_time <= b.broadcast_time
            WHERE 1=1 {step_filter}
            GROUP BY b.run_id, b.step
            ORDER BY b.run_id, b.step ASC
        """, active_runs + active_runs + step_params).fetchall():
            _add(row[0], row[1], "canceled_count", row[2])
    except Exception as e:
        log.warning(f"[API] Could not compute canceled count metric: {e}")

    # =====================================================================
    # 18. Timing: all operations in one query (replaces 8 per-run queries)
    # =====================================================================
    timing_operations = [
        "forward", "backward", "loss_computation", "compute_kl",
        "compute_entropy", "data_to_device", "prepare_tensors",
    ]
    ops_in_ph = ",".join(["?" for _ in timing_operations])
    a_timing_ops_params = active_runs + timing_operations + step_params

    for row in con.execute(f"""
        SELECT
            run_id, step,
            event_type as operation,
            SUM(end_time - start_time),
            AVG(end_time - start_time),
            STDDEV_SAMP(end_time - start_time),
            MIN(end_time - start_time),
            MAX(end_time - start_time),
            COUNT(*)
        FROM events_trainer
        WHERE run_id IN ({a_in_ph})
          AND rank = 0
          AND event_type IN ({ops_in_ph})
          {timing_step_filter}
        GROUP BY run_id, step, event_type
        ORDER BY run_id, step ASC
    """, a_timing_ops_params).fetchall():
        rid, step, operation = row[0], row[1], row[2]
        total_time, mb_mean, mb_std, mb_min, mb_max, mb_count = row[3:]
        _add(rid, step, f"timing_{operation}_total", total_time)
        if operation == "forward":
            _add(rid, step, "timing_microbatch_count", mb_count)
        _add(rid, step, f"timing_{operation}_microbatch_mean", mb_mean)
        _add(rid, step, f"timing_{operation}_microbatch_std", mb_std)
        _add(rid, step, f"timing_{operation}_microbatch_min", mb_min)
        _add(rid, step, f"timing_{operation}_microbatch_max", mb_max)

    # =====================================================================
    # 19. Step boundaries → timing_step_active & timing_step_total
    # =====================================================================
    # Collect per-run step boundaries for step_total computation
    run_sorted_steps: dict[str, list[int]] = {rp: [] for rp in active_runs}
    run_step_end: dict[str, dict[int, float]] = {rp: {} for rp in active_runs}

    for row in con.execute(f"""
        SELECT run_id, step, MIN(start_time), MAX(end_time)
        FROM events_trainer
        WHERE run_id IN ({a_in_ph}) AND rank = 0 {timing_step_filter}
        GROUP BY run_id, step
        ORDER BY run_id, step ASC
    """, a_timing_params).fetchall():
        rid, step, s_start, s_end = row
        run_sorted_steps[rid].append(step)
        run_step_end[rid][step] = s_end
        if s_start is not None and s_end is not None:
            _add(rid, step, "timing_step_active", s_end - s_start)

    # timing_step_total: gap between consecutive step ends
    for rid in active_runs:
        steps = run_sorted_steps[rid]
        for i in range(1, len(steps)):
            prev_end = run_step_end[rid].get(steps[i - 1])
            curr_end = run_step_end[rid].get(steps[i])
            if prev_end is not None and curr_end is not None:
                _add(rid, steps[i], "timing_step_total", curr_end - prev_end)

    # =====================================================================
    # 20. Waiting for data + first-step timing_step_total
    # =====================================================================
    try:
        waiting_rows = con.execute(f"""
            WITH step_boundaries AS (
                SELECT run_id, step,
                       MIN(start_time) as step_start,
                       MAX(end_time) as step_end
                FROM events_trainer
                WHERE run_id IN ({a_in_ph}) AND rank = 0 {timing_step_filter}
                GROUP BY run_id, step
            ),
            step_with_prev AS (
                SELECT run_id, step, step_start, step_end,
                       LAG(step_end) OVER (PARTITION BY run_id ORDER BY step) as prev_step_end
                FROM step_boundaries
            ),
            training_start AS (
                SELECT run_id, MIN(timestamp) as first_event_time
                FROM events_orchestrator
                WHERE run_id IN ({a_in_ph}) AND event_type = 'training_loop_start'
                GROUP BY run_id
            )
            SELECT
                s.run_id, s.step,
                CASE
                    WHEN s.prev_step_end IS NULL THEN
                        s.step_start - COALESCE(t.first_event_time, s.step_start)
                    ELSE
                        s.step_start - s.prev_step_end
                END as wait_time
            FROM step_with_prev s
            LEFT JOIN training_start t ON s.run_id = t.run_id
            ORDER BY s.run_id, s.step ASC
        """, a_timing_params + active_runs).fetchall()

        for row in waiting_rows:
            rid, step, wait_time = row
            if req.start_step is not None and step < req.start_step:
                continue
            if req.end_step is not None and step > req.end_step:
                continue
            if wait_time is not None and wait_time >= 0:
                _add(rid, step, "timing_waiting_for_data", wait_time)

        # First-step timing_step_total using training_loop_start baseline
        ts_rows = con.execute(f"""
            SELECT run_id, MIN(timestamp)
            FROM events_orchestrator
            WHERE run_id IN ({a_in_ph}) AND event_type = 'training_loop_start'
            GROUP BY run_id
        """, active_runs).fetchall()
        training_start_map = {r[0]: r[1] for r in ts_rows if r[1] is not None}

        for rid in active_runs:
            steps = run_sorted_steps[rid]
            if steps and rid in training_start_map:
                first_step = steps[0]
                first_end = run_step_end[rid].get(first_step)
                if first_end is not None:
                    total_time = first_end - training_start_map[rid]
                    if total_time >= 0:
                        _add(rid, first_step, "timing_step_total", total_time)
    except Exception as e:
        log.warning(f"[API] Could not compute waiting_for_data metric: {e}")

    # =====================================================================
    # 21. Save-batch timing
    # =====================================================================
    # save_batch_multi_map: {run_id: {step: batch_duration}} for percentage calculations
    # Always compute the map (needed by inference timing percentages)
    save_batch_multi_map: dict[str, dict[int, float]] = defaultdict(dict)
    multi_inference_timing_metrics = [
        "timing_avg_inference_time", "timing_avg_compute_reward_time",
        "timing_generation_normal_pct", "timing_generation_discarded_pct",
        "timing_generation_canceled_pct", "timing_generation_all_pct",
        "timing_compute_reward_normal_pct", "timing_compute_reward_discarded_pct",
        "timing_compute_reward_canceled_pct", "timing_compute_reward_all_pct",
        "timing_idle_pct",
    ]
    need_save_batch_multi = metrics_to_include is None or "timing_save_batch_total" in metrics_to_include or any(
        m in metrics_to_include for m in multi_inference_timing_metrics
    )
    if need_save_batch_multi:
        try:
            for row in con.execute(f"""
                WITH batch_events AS (
                    SELECT run_id, step, MIN(timestamp) as batch_time
                    FROM events_orchestrator
                    WHERE run_id IN ({a_in_ph}) AND event_type = 'save_batch' AND step IS NOT NULL
                    GROUP BY run_id, step
                ),
                ordered AS (
                    SELECT run_id, step, batch_time,
                           LAG(batch_time) OVER (PARTITION BY run_id ORDER BY step) as prev_batch_time
                    FROM batch_events
                ),
                training_start AS (
                    SELECT run_id, MIN(timestamp) as first_event_time
                    FROM events_orchestrator
                    WHERE run_id IN ({a_in_ph}) AND event_type = 'training_loop_start'
                    GROUP BY run_id
                )
                SELECT
                    o.run_id, o.step,
                    CASE
                        WHEN o.prev_batch_time IS NULL THEN
                            o.batch_time - COALESCE(t.first_event_time, o.batch_time)
                        ELSE
                            o.batch_time - o.prev_batch_time
                    END as batch_duration
                FROM ordered o
                LEFT JOIN training_start t ON o.run_id = t.run_id
                ORDER BY o.run_id, o.step ASC
            """, active_runs + active_runs).fetchall():
                rid, step, batch_duration = row
                if step is None:
                    continue
                if req.start_step is not None and step < req.start_step:
                    continue
                if req.end_step is not None and step > req.end_step:
                    continue
                if batch_duration is not None and batch_duration >= 0:
                    save_batch_multi_map[rid][step] = float(batch_duration)
                    if metrics_to_include is None or "timing_save_batch_total" in metrics_to_include:
                        _add(rid, step, "timing_save_batch_total", batch_duration)
        except Exception as e:
            log.warning(f"[API] Could not compute save_batch timing metric: {e}")

    # =====================================================================
    # 22. Trainer weight_broadcast timing
    # =====================================================================
    try:
        for row in con.execute(f"""
            SELECT run_id, step, SUM(end_time - start_time)
            FROM events_trainer
            WHERE run_id IN ({a_in_ph}) AND rank = 0 AND event_type = 'weight_broadcast'
                  {timing_step_filter}
            GROUP BY run_id, step
            ORDER BY run_id, step ASC
        """, a_timing_params).fetchall():
            _add(row[0], row[1], "timing_weight_sync_trainer_total", row[2])
    except Exception as e:
        log.warning(f"[API] Could not compute trainer weight_broadcast metric: {e}")

    # =====================================================================
    # 23. Inference weight_broadcast timing
    # =====================================================================
    try:
        for row in con.execute(f"""
            SELECT run_id, step, (MAX(end_time) - MIN(start_time)) as wall_clock_time
            FROM events_inference
            WHERE run_id IN ({a_in_ph}) AND event_type = 'weight_broadcast' AND step IS NOT NULL
            GROUP BY run_id, step
            ORDER BY run_id, step ASC
        """, active_runs).fetchall():
            rid, step, wc_time = row
            if req.start_step is not None and step < req.start_step:
                continue
            if req.end_step is not None and step > req.end_step:
                continue
            _add(rid, step, "timing_weight_sync_inference_total", wc_time)
    except Exception as e:
        log.warning(f"[API] Could not compute inference weight_broadcast metric: {e}")

    # =====================================================================
    # 24. Inference timing metrics (generation/reward/idle breakdown)
    # Uses save_batch timestamps as step windows. Events are attributed to
    # the step whose window contains the event's end_time. Generation time
    # is the actual lane occupation (end_time - start_time), clipped to the
    # step window. Compute reward time is attributed to the step where the
    # last turn of the sample ends.
    # =====================================================================
    any_inference_timing_needed = metrics_to_include is None or any(
        m in metrics_to_include for m in multi_inference_timing_metrics
    )
    if any_inference_timing_needed:
        try:
            # Fetch configs for all runs to compute total_lanes per run
            run_total_lanes: dict[str, int] = {}
            config_rows = con.execute(f"""
                SELECT run_id, last_config_json FROM ingest_state
                WHERE run_id IN ({a_in_ph})
            """, active_runs).fetchall()
            for crow in config_rows:
                rid = crow[0]
                rcfg = json.loads(crow[1]) if crow[1] else {}
                ns = rcfg.get("num_inference_servers") or max(
                    1, rcfg.get("inference_num_workers", 1) // max(1, rcfg.get("inference_tensor_parallel_size", 1))
                )
                mc = rcfg.get("max_concurrent_prompts") or (
                    rcfg.get("max_concurrent_prompts_per_server", 12) * ns
                )
                gs = rcfg.get("group_size", 1)
                lps = max(1, int((mc / ns) * gs))
                run_total_lanes[rid] = lps * ns

            # --- Single query per run: generation + compute_reward sums per step ---
            # Step windows from save_batch; events attributed by end_time;
            # generation = clipped (end_time - start_time);
            # compute_reward attributed to step of last turn per sample.
            for run_id in active_runs:
                tl = run_total_lanes.get(run_id, 1)
                timing_rows = con.execute("""
                    WITH batch_events AS (
                        SELECT step, MIN(timestamp) as batch_time
                        FROM events_orchestrator
                        WHERE run_id = ? AND event_type = 'save_batch' AND step IS NOT NULL
                        GROUP BY step
                    ),
                    training_start AS (
                        SELECT MIN(timestamp) as ts FROM events_orchestrator
                        WHERE run_id = ? AND event_type = 'training_loop_start'
                    ),
                    step_windows AS (
                        SELECT step,
                               COALESCE(LAG(batch_time) OVER (ORDER BY step), t.ts) as step_start,
                               batch_time as step_end
                        FROM batch_events CROSS JOIN training_start t
                    ),
                    request_events AS (
                        SELECT sample_id, group_id, start_time, end_time,
                               inference_time, is_canceled
                        FROM events_inference
                        WHERE run_id = ? AND event_type = 'request'
                          AND (is_eval = false OR is_eval IS NULL)
                    ),
                    last_event_per_sample AS (
                        SELECT group_id, sample_id, MAX(end_time) as last_end
                        FROM request_events
                        GROUP BY group_id, sample_id
                    ),
                    sample_info AS (
                        SELECT group_id, sample_idx, compute_reward_time, 'normal' as status
                        FROM samples_data WHERE run_id = ?
                        UNION ALL
                        SELECT group_id, sample_idx, compute_reward_time, 'discarded' as status
                        FROM samples_data_discarded WHERE run_id = ?
                    ),
                    gen_attributed AS (
                        SELECT
                            sw.step,
                            GREATEST(0, LEAST(re.end_time, sw.step_end) - GREATEST(re.start_time, sw.step_start)) as clipped_gen,
                            re.inference_time,
                            CASE
                                WHEN COALESCE(re.is_canceled, false) THEN 'canceled'
                                WHEN si.status IS NOT NULL THEN si.status
                                ELSE 'normal'
                            END as status
                        FROM request_events re
                        JOIN step_windows sw
                            ON re.end_time > sw.step_start AND re.end_time <= sw.step_end
                        LEFT JOIN sample_info si
                            ON si.group_id = re.group_id AND si.sample_idx = re.sample_id
                    ),
                    cr_attributed AS (
                        SELECT
                            sw.step,
                            COALESCE(si.compute_reward_time, 0) as cr_time,
                            COALESCE(si.status, 'normal') as status
                        FROM last_event_per_sample le
                        JOIN step_windows sw
                            ON le.last_end > sw.step_start AND le.last_end <= sw.step_end
                        LEFT JOIN sample_info si
                            ON si.group_id = le.group_id AND si.sample_idx = le.sample_id
                        WHERE NOT EXISTS (
                            SELECT 1 FROM request_events re2
                            WHERE re2.group_id = le.group_id AND re2.sample_id = le.sample_id
                              AND COALESCE(re2.is_canceled, false) = true
                        )
                    ),
                    gen_agg AS (
                        SELECT
                            step,
                            SUM(clipped_gen) as gen_all,
                            SUM(CASE WHEN status = 'normal' THEN clipped_gen ELSE 0 END) as gen_normal,
                            SUM(CASE WHEN status = 'discarded' THEN clipped_gen ELSE 0 END) as gen_discarded,
                            SUM(CASE WHEN status = 'canceled' THEN clipped_gen ELSE 0 END) as gen_canceled,
                            AVG(inference_time) as avg_inference_time
                        FROM gen_attributed
                        GROUP BY step
                    ),
                    cr_agg AS (
                        SELECT
                            step,
                            SUM(cr_time) as cr_all,
                            SUM(CASE WHEN status = 'normal' THEN cr_time ELSE 0 END) as cr_normal,
                            SUM(CASE WHEN status = 'discarded' THEN cr_time ELSE 0 END) as cr_discarded,
                            AVG(CASE WHEN cr_time > 0 THEN cr_time ELSE NULL END) as avg_cr_time
                        FROM cr_attributed
                        GROUP BY step
                    )
                    SELECT
                        sw.step,
                        sw.step_end - sw.step_start as step_duration,
                        g.gen_all, g.gen_normal, g.gen_discarded, g.gen_canceled,
                        g.avg_inference_time,
                        c.cr_all, c.cr_normal, c.cr_discarded,
                        c.avg_cr_time
                    FROM step_windows sw
                    LEFT JOIN gen_agg g ON g.step = sw.step
                    LEFT JOIN cr_agg c ON c.step = sw.step
                    WHERE (g.step IS NOT NULL OR c.step IS NOT NULL)
                    ORDER BY sw.step ASC
                """, [run_id, run_id, run_id, run_id, run_id]).fetchall()

                for row in timing_rows:
                    step = row[0]
                    step_duration = row[1]

                    if req.start_step is not None and step < req.start_step:
                        continue
                    if req.end_step is not None and step > req.end_step:
                        continue
                    if step_duration is None or step_duration <= 0:
                        continue

                    available_lane_seconds = step_duration * tl

                    g_all = float(row[2] or 0)
                    g_normal = float(row[3] or 0)
                    g_discarded = float(row[4] or 0)
                    g_canceled = float(row[5] or 0)
                    avg_inf = row[6]

                    cr_all = float(row[7] or 0)
                    cr_normal = float(row[8] or 0)
                    cr_discarded = float(row[9] or 0)
                    avg_cr = row[10]

                    # Avg metrics
                    if metrics_to_include is None or "timing_avg_inference_time" in metrics_to_include:
                        if avg_inf is not None:
                            _add(run_id, step, "timing_avg_inference_time", float(avg_inf))
                    if metrics_to_include is None or "timing_avg_compute_reward_time" in metrics_to_include:
                        if avg_cr is not None:
                            _add(run_id, step, "timing_avg_compute_reward_time", float(avg_cr))

                    # Percentage metrics
                    pct = {
                        "timing_generation_normal_pct": g_normal / available_lane_seconds * 100,
                        "timing_generation_discarded_pct": g_discarded / available_lane_seconds * 100,
                        "timing_generation_canceled_pct": g_canceled / available_lane_seconds * 100,
                        "timing_generation_all_pct": g_all / available_lane_seconds * 100,
                        "timing_compute_reward_normal_pct": cr_normal / available_lane_seconds * 100,
                        "timing_compute_reward_discarded_pct": cr_discarded / available_lane_seconds * 100,
                        "timing_compute_reward_canceled_pct": 0.0,
                        "timing_compute_reward_all_pct": cr_all / available_lane_seconds * 100,
                    }
                    pct["timing_idle_pct"] = max(0.0, 100.0 - pct["timing_generation_all_pct"] - pct["timing_compute_reward_all_pct"])

                    for mname, val in pct.items():
                        if metrics_to_include is None or mname in metrics_to_include:
                            _add(run_id, step, mname, val)

        except Exception as e:
            log.warning(f"[API] Could not compute inference timing metrics (multi): {e}")

    # =====================================================================
    # Build final response per run
    # =====================================================================
    runs_out = []
    for rp in run_paths:
        meta = run_meta[rp]

        # -- available_metrics (static + dynamic parts per run) --
        avail: list[str] = list(base_stat_cols)
        avail.extend(completion_columns)
        avail.extend(prompt_columns)
        avail.extend(length_sum_columns)
        avail.append("stop_reason_length_pct")
        avail.extend([
            "group_length_cv_mean",
            "group_length_max_median_ratio_mean",
            "group_length_gini_mean",
        ])
        avail.append("reward_gini_mean")
        avail.extend(["off_policy_steps_mean", "off_policy_steps_std"])
        for gn in run_gen_metric_names[rp]:
            avail.extend([f"reward_{gn}_mean", f"reward_{gn}_std",
                          f"reward_{gn}_min", f"reward_{gn}_max",
                          f"reward_{gn}_gini_mean"])
        avail.extend(run_custom_metrics[rp])

        # Discarded metrics
        discarded_static = [
            "discarded_count", "discarded_zero_advantage_pct", "discarded_max_async_pct",
            "discarded_stop_reason_length_pct",
            "discarded_group_length_max_median_ratio_mean", "discarded_group_length_gini_mean",
            "discarded_reward_sum_mean", "discarded_reward_sum_std",
            "discarded_reward_sum_min", "discarded_reward_sum_max",
            "discarded_advantage_mean", "discarded_advantage_std",
            "discarded_advantage_min", "discarded_advantage_max",
            "discarded_length_prompt_mean", "discarded_length_prompt_std",
            "discarded_length_prompt_min", "discarded_length_prompt_max",
            "discarded_length_completion_mean", "discarded_length_completion_std",
            "discarded_length_completion_min", "discarded_length_completion_max",
            "discarded_length_sum_mean", "discarded_length_sum_std",
            "discarded_length_sum_min", "discarded_length_sum_max",
            "discarded_off_policy_steps_mean", "discarded_off_policy_steps_std",
        ]
        for dgn in run_discarded_gen_names[rp]:
            discarded_static.extend([
                f"discarded_reward_{dgn}_mean", f"discarded_reward_{dgn}_std",
                f"discarded_reward_{dgn}_min", f"discarded_reward_{dgn}_max",
            ])
        avail.extend(discarded_static)

        # Canceled metrics
        avail.append("canceled_count")

        # Timing metrics
        timing_list = ["timing_step_total", "timing_step_active", "timing_microbatch_count"]
        for op in timing_operations:
            timing_list.extend([
                f"timing_{op}_total", f"timing_{op}_microbatch_mean",
                f"timing_{op}_microbatch_std", f"timing_{op}_microbatch_min",
                f"timing_{op}_microbatch_max",
            ])
        timing_list.extend([
            "timing_waiting_for_data", "timing_save_batch_total",
            "timing_weight_sync_trainer_total", "timing_weight_sync_inference_total",
            "timing_avg_inference_time", "timing_avg_compute_reward_time",
            "timing_generation_normal_pct", "timing_generation_discarded_pct",
            "timing_generation_canceled_pct", "timing_generation_all_pct",
            "timing_compute_reward_normal_pct", "timing_compute_reward_discarded_pct",
            "timing_compute_reward_canceled_pct", "timing_compute_reward_all_pct",
            "timing_idle_pct",
        ])
        avail.extend(timing_list)

        metrics_list = run_metrics[rp]
        runs_out.append({
            "run_path": rp,
            "metrics": metrics_list,
            "total_returned": len(metrics_list),
            "available_metrics": avail,
            "available_rollout_metric_names": run_gen_metric_names[rp],
            "available_custom_metrics": run_custom_metrics[rp],
            "custom_metric_sections": run_custom_metric_sections[rp],
            "min_step": meta["min_step"],
            "max_step": meta["max_step"],
            "total_steps": meta["total_steps"],
        })

    log.info(
        "[API] Batched step-metrics: %s runs, %s total metric points",
        len(run_paths),
        sum(len(run_metrics[rp]) for rp in run_paths),
    )

    return {"runs": runs_out}


@app.post("/step-times")
def get_step_times(req: StepTimesRequest):
    """Get the last trainer event end time for each step."""
    log.info(f"[API] Getting step times for {req.run_path}")
    con = connect()
    
    rows = con.execute(
        """
        SELECT step, MAX(end_time) as last_time
        FROM events_trainer
        WHERE run_id = ? AND step >= 0
        GROUP BY step
        ORDER BY step ASC
        """,
        [req.run_path],
    ).fetchall()
    
    if not rows:
        return {"step_times": [], "first_step_time": None}
    
    step_times = [
        {"step": row[0], "time": row[1]}
        for row in rows
        if row[1] is not None
    ]
    
    first_step_time = step_times[0]["time"] if step_times else None
    
    return {
        "step_times": step_times,
        "first_step_time": first_step_time,
    }


@app.post("/inference-performance")
def get_inference_performance(req: InferencePerformanceRequest):
    """Get inference performance metrics bucketed by time intervals."""
    log.info(f"[API] Getting inference performance for {req.run_path}")
    con = connect()
    bucket = req.bucket_seconds

    # Inference calls per bucket (all inference requests by end_time)
    inference_calls = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Requests done per bucket (non-canceled completed requests)
    requests_done = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Rollouts group done kept per bucket (groups that appear in samples_data)
    rollouts_group_done_kept = con.execute(
        """
        WITH group_completion AS (
            SELECT group_id, MAX(end_time) as completion_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
              AND (is_canceled = false OR is_canceled IS NULL)
            GROUP BY group_id
        )
        SELECT FLOOR(gc.completion_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM group_completion gc
        WHERE EXISTS (
            SELECT 1 FROM samples_data sd
            WHERE sd.run_id = ? AND sd.group_id = gc.group_id
        )
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [req.run_path, bucket, bucket, req.run_path],
    ).fetchall()

    # Rollouts group done discarded per bucket (groups that appear in samples_data_discarded)
    rollouts_group_done_discarded = con.execute(
        """
        WITH group_completion AS (
            SELECT group_id, MAX(end_time) as completion_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
              AND (is_canceled = false OR is_canceled IS NULL)
            GROUP BY group_id
        )
        SELECT FLOOR(gc.completion_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM group_completion gc
        WHERE EXISTS (
            SELECT 1 FROM samples_data_discarded sdd
            WHERE sdd.run_id = ? AND sdd.group_id = gc.group_id
        )
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [req.run_path, bucket, bucket, req.run_path],
    ).fetchall()

    # Rollouts group done per bucket (kept + discarded combined)
    rollouts_group_done = con.execute(
        """
        WITH group_completion AS (
            SELECT group_id, MAX(end_time) as completion_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
              AND (is_canceled = false OR is_canceled IS NULL)
            GROUP BY group_id
        ),
        kept_and_discarded AS (
            SELECT gc.group_id, gc.completion_time
            FROM group_completion gc
            WHERE EXISTS (
                SELECT 1 FROM samples_data sd
                WHERE sd.run_id = ? AND sd.group_id = gc.group_id
            )
            UNION ALL
            SELECT gc.group_id, gc.completion_time
            FROM group_completion gc
            WHERE EXISTS (
                SELECT 1 FROM samples_data_discarded sdd
                WHERE sdd.run_id = ? AND sdd.group_id = gc.group_id
            )
        )
        SELECT FLOOR(completion_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM kept_and_discarded
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [req.run_path, req.run_path, req.run_path, bucket, bucket],
    ).fetchall()

    # Rollouts group done canceled per bucket (groups with any canceled request)
    rollouts_group_done_canceled = con.execute(
        """
        WITH canceled_groups AS (
            SELECT group_id, MAX(end_time) as completion_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
              AND is_canceled = true
            GROUP BY group_id
        )
        SELECT FLOOR(completion_time / ?) * ? as bucket_time, COUNT(*) as cnt
        FROM canceled_groups
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [req.run_path, bucket, bucket],
    ).fetchall()

    # Average prefill time per bucket (non-canceled requests with prefill_time)
    avg_time_prefill = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(prefill_time) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND prefill_time IS NOT NULL AND prefill_time > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average decode time per bucket (non-canceled requests with decode_time)
    avg_time_decode = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(decode_time) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND decode_time IS NOT NULL AND decode_time > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average compute reward time per bucket
    # compute_reward_time lives in samples_data / samples_data_discarded,
    # keyed by group_id. We join with group completion time from events_inference
    # to bucket it by time.
    avg_time_compute_reward = con.execute(
        """
        WITH group_completion AS (
            SELECT group_id, MAX(end_time) as completion_time
            FROM events_inference
            WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
              AND (is_canceled = false OR is_canceled IS NULL)
            GROUP BY group_id
        ),
        sample_cr AS (
            SELECT sd.group_id, sd.compute_reward_time
            FROM samples_data sd
            WHERE sd.run_id = ? AND sd.compute_reward_time IS NOT NULL AND sd.compute_reward_time > 0
            UNION ALL
            SELECT sdd.group_id, sdd.compute_reward_time
            FROM samples_data_discarded sdd
            WHERE sdd.run_id = ? AND sdd.compute_reward_time IS NOT NULL AND sdd.compute_reward_time > 0
        ),
        group_avg_cr AS (
            SELECT group_id, AVG(compute_reward_time) as avg_cr
            FROM sample_cr
            GROUP BY group_id
        )
        SELECT FLOOR(gc.completion_time / ?) * ? as bucket_time, AVG(gacr.avg_cr) as avg_val
        FROM group_avg_cr gacr
        JOIN group_completion gc ON gc.group_id = gacr.group_id
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [req.run_path, req.run_path, req.run_path, bucket, bucket],
    ).fetchall()

    # Average queue time per bucket
    avg_time_queue = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(queue_time) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND queue_time IS NOT NULL AND queue_time > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average time to first token per bucket
    avg_time_ttft = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(time_to_first_token) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND time_to_first_token IS NOT NULL AND time_to_first_token > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average inference time per bucket
    avg_time_inference = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(inference_time) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND inference_time IS NOT NULL AND inference_time > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average e2e latency per bucket
    avg_time_e2e = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(e2e_latency) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND e2e_latency IS NOT NULL AND e2e_latency > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # Average full generation duration per bucket (end_time - start_time)
    avg_time_generation = con.execute(
        """
        SELECT FLOOR(end_time / ?) * ? as bucket_time, AVG(end_time - start_time) as avg_val
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request' AND end_time IS NOT NULL AND start_time IS NOT NULL
            AND (is_canceled = false OR is_canceled IS NULL)
            AND (end_time - start_time) > 0
        GROUP BY bucket_time
        ORDER BY bucket_time ASC
        """,
        [bucket, bucket, req.run_path],
    ).fetchall()

    # --- Inference utilization (idle vs working) ---
    # Total number of lanes = distinct (server, server_lane) combinations
    num_lanes_row = con.execute(
        """
        SELECT COUNT(DISTINCT (COALESCE(server, 0) * 100000 + COALESCE(server_lane, 0)))
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request'
            AND start_time IS NOT NULL AND end_time IS NOT NULL
        """,
        [req.run_path],
    ).fetchone()
    num_lanes = num_lanes_row[0] if num_lanes_row and num_lanes_row[0] else 0

    # Fetch all request events with start/end times for overlap computation
    inference_events = con.execute(
        """
        SELECT start_time, end_time
        FROM events_inference
        WHERE run_id = ? AND event_type = 'request'
            AND start_time IS NOT NULL AND end_time IS NOT NULL
        ORDER BY start_time ASC
        """,
        [req.run_path],
    ).fetchall()

    # Include in-flight generations as synthetic events (start_time -> snapshot_time)
    from .ingest import inflight_by_run
    inflight_data = inflight_by_run.get(req.run_path)
    if inflight_data and inflight_data.get("snapshot_time") and inflight_data.get("running"):
        snap_time = inflight_data["snapshot_time"]
        inference_events = list(inference_events) + [
            (g["start_time"], snap_time)
            for g in inflight_data["running"]
            if g.get("start_time") is not None
        ]

    # Fetch weight_broadcast events for generating vs weight sync breakdown
    weight_broadcast_events = con.execute(
        """
        SELECT start_time, end_time
        FROM events_inference
        WHERE run_id = ? AND event_type = 'weight_broadcast'
            AND start_time IS NOT NULL AND end_time IS NOT NULL
        ORDER BY start_time ASC
        """,
        [req.run_path],
    ).fetchall()

    # Step times (for vertical lines)
    step_rows = con.execute(
        """
        SELECT step, MAX(end_time) as last_time
        FROM events_trainer
        WHERE run_id = ? AND step >= 0
        GROUP BY step
        ORDER BY step ASC
        """,
        [req.run_path],
    ).fetchall()

    # Last event time (for detecting unfinished intervals)
    last_time_row = con.execute(
        """
        SELECT MAX(end_time) FROM events_inference
        WHERE run_id = ? AND end_time IS NOT NULL
        """,
        [req.run_path],
    ).fetchone()
    last_time = last_time_row[0] if last_time_row and last_time_row[0] is not None else None

    # Extend last_time to include inflight snapshot time
    if inflight_data and inflight_data.get("snapshot_time") and inflight_data.get("running"):
        snap_time = inflight_data["snapshot_time"]
        if last_time is None or snap_time > last_time:
            last_time = snap_time

    con.close()

    # Compute utilization buckets
    utilization_buckets = []
    if num_lanes > 0 and inference_events:
        capacity = num_lanes * bucket  # total lane-seconds per bucket
        request_overlap: dict[float, float] = defaultdict(float)
        wb_overlap: dict[float, float] = defaultdict(float)

        for start_time, end_time in inference_events:
            first_b = math.floor(start_time / bucket) * bucket
            last_b = math.floor(end_time / bucket) * bucket
            b = first_b
            while b <= last_b:
                b_end = b + bucket
                overlap = min(end_time, b_end) - max(start_time, b)
                if overlap > 0:
                    request_overlap[b] += overlap
                b += bucket

        for start_time, end_time in weight_broadcast_events:
            first_b = math.floor(start_time / bucket) * bucket
            last_b = math.floor(end_time / bucket) * bucket
            b = first_b
            while b <= last_b:
                b_end = b + bucket
                overlap = min(end_time, b_end) - max(start_time, b)
                if overlap > 0:
                    wb_overlap[b] += overlap
                b += bucket

        all_bucket_keys = sorted(set(request_overlap.keys()) | set(wb_overlap.keys()))
        for b_start in all_bucket_keys:
            raw_generating_pct = min(100.0, request_overlap.get(b_start, 0) / capacity * 100)
            # Weight sync blocks all lanes at once, so measure against wall-clock time only
            wb_pct = min(100.0, wb_overlap.get(b_start, 0) / bucket * 100)
            # Discount weight sync from generating so they don't double-count
            generating_pct = max(0.0, raw_generating_pct - wb_pct)
            total_busy = min(100.0, generating_pct + wb_pct)
            idle_pct = max(0.0, 100.0 - total_busy)
            utilization_buckets.append({
                "time": b_start,
                "working": round(total_busy, 4),
                "generating": round(generating_pct, 4),
                "weight_broadcast": round(wb_pct, 4),
                "idle": round(idle_pct, 4),
            })

    # Find earliest time across all data for relative time calculation
    all_bucket_rows = [
        inference_calls, requests_done, rollouts_group_done,
        rollouts_group_done_kept, rollouts_group_done_discarded, rollouts_group_done_canceled,
        avg_time_prefill, avg_time_decode, avg_time_compute_reward,
        avg_time_queue, avg_time_ttft, avg_time_inference, avg_time_e2e, avg_time_generation,
    ]
    first_time = None
    for rows in all_bucket_rows:
        if rows:
            t = rows[0][0]
            if t is not None and (first_time is None or t < first_time):
                first_time = t
    if step_rows:
        t = step_rows[0][1]
        if t is not None and (first_time is None or t < first_time):
            first_time = t

    def format_buckets(rows):
        return [{"time": row[0], "count": row[1]} for row in rows if row[0] is not None]

    def format_avg_buckets(rows):
        return [{"time": row[0], "value": row[1]} for row in rows if row[0] is not None and row[1] is not None]

    return {
        "inference_calls": format_buckets(inference_calls),
        "requests_done": format_buckets(requests_done),
        "rollouts_group_done": format_buckets(rollouts_group_done),
        "rollouts_group_done_kept": format_buckets(rollouts_group_done_kept),
        "rollouts_group_done_discarded": format_buckets(rollouts_group_done_discarded),
        "rollouts_group_done_canceled": format_buckets(rollouts_group_done_canceled),
        "avg_time_prefill": format_avg_buckets(avg_time_prefill),
        "avg_time_decode": format_avg_buckets(avg_time_decode),
        "avg_time_compute_reward": format_avg_buckets(avg_time_compute_reward),
        "avg_time_queue": format_avg_buckets(avg_time_queue),
        "avg_time_ttft": format_avg_buckets(avg_time_ttft),
        "avg_time_inference": format_avg_buckets(avg_time_inference),
        "avg_time_e2e": format_avg_buckets(avg_time_e2e),
        "avg_time_generation": format_avg_buckets(avg_time_generation),
        "utilization_buckets": utilization_buckets,
        "num_lanes": num_lanes,
        "step_times": [
            {"step": row[0], "time": row[1]}
            for row in step_rows
            if row[1] is not None
        ],
        "first_time": first_time,
        "last_time": last_time,
    }


@app.post("/trainer-performance")
def get_trainer_performance(req: TrainerPerformanceRequest):
    """Get trainer performance metrics bucketed by time intervals.

    For each time bucket, computes the percentage of time spent on each
    trainer event type (forward, backward, optimizer, etc.), averaged
    across all ranks.  Also returns derived metrics: idle, working, and
    working_except_weight_sync.
    """
    log.info(f"[API] Getting trainer performance for {req.run_path}")
    con = connect()
    bucket = req.bucket_seconds

    # Fetch all top-level trainer events (exclude sub-events like loss/shift)
    events = con.execute(
        """
        SELECT event_type, rank, start_time, end_time
        FROM events_trainer
        WHERE run_id = ?
            AND start_time IS NOT NULL AND end_time IS NOT NULL
            AND event_type NOT LIKE '%/%'
        ORDER BY start_time ASC
        """,
        [req.run_path],
    ).fetchall()

    # Step times (for vertical lines / first_time)
    step_rows = con.execute(
        """
        SELECT step, MAX(end_time) as last_time
        FROM events_trainer
        WHERE run_id = ? AND step >= 0
        GROUP BY step
        ORDER BY step ASC
        """,
        [req.run_path],
    ).fetchall()

    con.close()

    if not events:
        return {"buckets": [], "first_time": None, "last_time": None, "step_times": [], "event_types": []}

    # Collect unique ranks and event types
    all_ranks = set()
    all_event_types = set()
    for evt_type, rank, _start, _end in events:
        all_ranks.add(rank)
        all_event_types.add(evt_type)
    num_ranks = len(all_ranks)

    # Accumulate overlap seconds: bucket_data[bucket_start][(rank, event_type)] = seconds
    bucket_data: dict[float, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))

    for evt_type, rank, start_time, end_time in events:
        first_b = math.floor(start_time / bucket) * bucket
        last_b = math.floor(end_time / bucket) * bucket

        b = first_b
        while b <= last_b:
            b_end = b + bucket
            overlap = min(end_time, b_end) - max(start_time, b)
            if overlap > 0:
                bucket_data[b][(rank, evt_type)] += overlap
            b += bucket

    # Build result: average across ranks, convert to percentages
    result_buckets = []
    for b_start in sorted(bucket_data.keys()):
        entry: dict = {"time": b_start}

        type_totals: dict[str, float] = defaultdict(float)
        for (rank, evt_type), seconds in bucket_data[b_start].items():
            type_totals[evt_type] += seconds / num_ranks

        total_busy = 0.0
        weight_sync_pct = 0.0
        for evt_type, avg_seconds in type_totals.items():
            pct = avg_seconds / bucket * 100
            entry[evt_type] = round(pct, 4)
            total_busy += pct
            if evt_type == "weight_broadcast":
                weight_sync_pct = pct

        entry["working"] = round(min(100, total_busy), 4)
        entry["idle"] = round(max(0, 100 - total_busy), 4)
        entry["working_except_weight_sync"] = round(max(0, total_busy - weight_sync_pct), 4)
        entry["training"] = round(max(0, total_busy - weight_sync_pct), 4)

        result_buckets.append(entry)

    # First time
    first_time = sorted(bucket_data.keys())[0] if bucket_data else None
    if step_rows:
        t = step_rows[0][1]
        if t is not None and (first_time is None or t < first_time):
            first_time = t

    # Last event time (for detecting unfinished intervals)
    last_time = max(end_time for _, _, _, end_time in events) if events else None

    return {
        "buckets": result_buckets,
        "first_time": first_time,
        "last_time": last_time,
        "step_times": [
            {"step": row[0], "time": row[1]}
            for row in step_rows
            if row[1] is not None
        ],
        "event_types": sorted(all_event_types),
    }


@app.post("/step-histogram")
def get_step_histogram(req: StepHistogramRequest):
    """Get raw values for histogram visualization at a specific step.
    
    Returns all individual values for a metric type at a given step,
    suitable for client-side histogram computation.
    
    Supported metric_types:
    - reward_sum: Total reward from samples_data.reward
    - advantage: Advantage values from samples_data.advantage
    - length_prompt: Prompt token counts (from prompts table)
    - length_completion: Completion token counts (sum of tokens from rollouts where turn_type='model')
    - length_sum: Total token counts from samples_data.total_tokens
    - reward_<name>: Dynamic rollout metric values (e.g., reward_accuracy)
    """
    log.info(f"[API] Getting histogram data for {req.run_path}, step={req.step}, metric={req.metric_type}")
    con = connect()
    
    values = []
    
    # Check if this is an eval metric type:
    #   eval/{eval_name}/{base_metric_type}
    #   eval/{eval_name}/s/{sample_idx}/{base_metric_type}
    is_eval = req.metric_type.startswith("eval/")
    eval_name = None
    eval_sample_idx = None
    if is_eval:
        sample_match = re.match(r'^eval/([^/]+)/s/(\d+)/(.+)$', req.metric_type)
        if sample_match:
            eval_name = sample_match.group(1)
            eval_sample_idx = int(sample_match.group(2))
            metric_type = sample_match.group(3)
        else:
            parts = req.metric_type.split("/", 2)
            if len(parts) == 3:
                eval_name = parts[1]
                metric_type = parts[2]
            else:
                is_eval = False
    
    if is_eval and eval_name:
        sample_filter = " AND sample_idx = ?" if eval_sample_idx is not None else ""
        base_params = [req.run_path, eval_name, req.step] + ([eval_sample_idx] if eval_sample_idx is not None else [])

        if metric_type == "length_prompt":
            rows = con.execute(
                f"SELECT tokens_prompt FROM prompts_eval WHERE run_id = ? AND eval_name = ? AND step = ? {sample_filter} AND tokens_prompt IS NOT NULL ORDER BY sample_idx",
                base_params,
            ).fetchall()
            values = [row[0] for row in rows]
        elif metric_type == "length_completion":
            rows = con.execute(
                f"SELECT SUM(tokens) as ct FROM rollouts_eval WHERE run_id = ? AND eval_name = ? AND step = ? {sample_filter} AND turn_type = 'model' AND tokens IS NOT NULL GROUP BY sample_idx, completion_idx ORDER BY sample_idx, completion_idx",
                base_params,
            ).fetchall()
            values = [row[0] for row in rows]
        elif metric_type == "length_sum":
            rows = con.execute(
                f"SELECT SUM(tokens) as ct FROM rollouts_eval WHERE run_id = ? AND eval_name = ? AND step = ? {sample_filter} AND tokens IS NOT NULL GROUP BY sample_idx, completion_idx ORDER BY sample_idx, completion_idx",
                base_params,
            ).fetchall()
            values = [row[0] for row in rows]
        elif metric_type.startswith("reward_"):
            metric_name = metric_type[7:]
            rows = con.execute(
                f"SELECT value FROM rollouts_metrics_eval WHERE run_id = ? AND eval_name = ? AND step = ? {sample_filter} AND metric_name = ? AND value IS NOT NULL ORDER BY sample_idx, completion_idx",
                base_params + [metric_name],
            ).fetchall()
            values = [row[0] for row in rows]
        else:
            log.warning(f"[API] Unknown eval metric_type: {metric_type}")

        log.info(f"[API] Returning {len(values)} eval histogram values for {req.metric_type} at step {req.step}")
        return {"values": values, "step": req.step, "metric_type": req.metric_type, "count": len(values)}

    # Check if this is a discarded metric type
    is_discarded = req.metric_type.startswith("discarded_")
    metric_type = req.metric_type[len("discarded_"):] if is_discarded else req.metric_type
    
    # Choose tables and step column based on discarded flag
    samples_table = "samples_data_discarded" if is_discarded else "samples_data"
    prompts_table = "prompts_discarded" if is_discarded else "prompts"
    rollouts_table = "rollouts_discarded" if is_discarded else "rollouts"
    gen_metrics_table = "rollouts_metrics_discarded" if is_discarded else "rollouts_metrics"
    step_col = "trainer_step" if is_discarded else "step"
    
    if metric_type == "reward_sum":
        rows = con.execute(
            f"""
            SELECT reward FROM {samples_table}
            WHERE run_id = ? AND {step_col} = ? AND reward IS NOT NULL
            ORDER BY sample_idx
            """,
            [req.run_path, req.step],
        ).fetchall()
        values = [row[0] for row in rows]
        
    elif metric_type == "advantage":
        rows = con.execute(
            f"""
            SELECT advantage FROM {samples_table}
            WHERE run_id = ? AND {step_col} = ? AND advantage IS NOT NULL
            ORDER BY sample_idx
            """,
            [req.run_path, req.step],
        ).fetchall()
        values = [row[0] for row in rows]
        
    elif metric_type == "length_prompt":
        rows = con.execute(
            f"""
            SELECT tokens_prompt FROM {prompts_table}
            WHERE run_id = ? AND {step_col} = ? AND tokens_prompt IS NOT NULL
            ORDER BY group_id
            """,
            [req.run_path, req.step],
        ).fetchall()
        values = [row[0] for row in rows]
        
    elif metric_type == "length_completion":
        rows = con.execute(
            f"""
            SELECT SUM(tokens) as completion_tokens FROM {rollouts_table}
            WHERE run_id = ? AND {step_col} = ? AND turn_type = 'model' AND tokens IS NOT NULL
            GROUP BY sample_idx
            ORDER BY sample_idx
            """,
            [req.run_path, req.step],
        ).fetchall()
        values = [row[0] for row in rows]
    
    elif metric_type == "length_sum":
        rows = con.execute(
            f"""
            SELECT total_tokens FROM {samples_table}
            WHERE run_id = ? AND {step_col} = ? AND total_tokens IS NOT NULL
            ORDER BY sample_idx
            """,
            [req.run_path, req.step],
        ).fetchall()
        values = [row[0] for row in rows]
        
    elif metric_type.startswith("reward_"):
        metric_name = metric_type[7:]  # Remove "reward_" prefix
        if is_discarded:
            # For discarded rollout metrics, join with rollouts_discarded to get trainer_step
            rows = con.execute(
                f"""
                SELECT gmd.value FROM {gen_metrics_table} gmd
                WHERE gmd.run_id = ? AND gmd.sample_idx IN (
                    SELECT DISTINCT sample_idx FROM {rollouts_table}
                    WHERE run_id = ? AND {step_col} = ?
                ) AND gmd.metric_name = ? AND gmd.value IS NOT NULL
                ORDER BY gmd.sample_idx
                """,
                [req.run_path, req.run_path, req.step, metric_name],
            ).fetchall()
        else:
            rows = con.execute(
                f"""
                SELECT value FROM {gen_metrics_table}
                WHERE run_id = ? AND step = ? AND metric_name = ? AND value IS NOT NULL
                ORDER BY sample_idx
                """,
                [req.run_path, req.step, metric_name],
            ).fetchall()
        values = [row[0] for row in rows]
    else:
        log.warning(f"[API] Unknown metric_type: {req.metric_type}")
    
    log.info(f"[API] Returning {len(values)} histogram values for {req.metric_type} at step {req.step}")
    
    return {
        "values": values,
        "step": req.step,
        "metric_type": req.metric_type,
        "count": len(values),
    }


@app.post("/step-distribution-over-time")
def get_step_distribution_over_time(req: StepDistributionOverTimeRequest):
    """Get distribution data over all steps for heatmap visualization.
    
    Returns a 2D histogram where:
    - X axis: steps
    - Y axis: value bins
    - Values: counts in each bin
    
    This is useful for visualizing how distributions change over training.
    """
    log.info(f"[API] Getting distribution over time for {req.run_path}, metric={req.metric_type}")
    con = connect()
    
    # Check if this is an eval metric type:
    #   eval/{eval_name}/{base_metric_type}
    #   eval/{eval_name}/s/{sample_idx}/{base_metric_type}
    is_eval = req.metric_type.startswith("eval/")
    eval_name = None
    eval_sample_idx = None
    if is_eval:
        sample_match = re.match(r'^eval/([^/]+)/s/(\d+)/(.+)$', req.metric_type)
        if sample_match:
            eval_name = sample_match.group(1)
            eval_sample_idx = int(sample_match.group(2))
            eval_metric_type = sample_match.group(3)
        else:
            parts = req.metric_type.split("/", 2)
            if len(parts) == 3:
                eval_name = parts[1]
                eval_metric_type = parts[2]
            else:
                is_eval = False

    if is_eval and eval_name:
        sample_filter = " AND sample_idx = ?" if eval_sample_idx is not None else ""
        base_params = [req.run_path, eval_name] + ([eval_sample_idx] if eval_sample_idx is not None else [])

        if eval_metric_type == "length_prompt":
            all_values_query = f"SELECT step, tokens_prompt as value FROM prompts_eval WHERE run_id = ? AND eval_name = ? {sample_filter} AND tokens_prompt IS NOT NULL ORDER BY step, sample_idx"
            params = base_params
        elif eval_metric_type == "length_completion":
            all_values_query = f"SELECT step, SUM(tokens) as value FROM rollouts_eval WHERE run_id = ? AND eval_name = ? {sample_filter} AND turn_type = 'model' AND tokens IS NOT NULL GROUP BY step, sample_idx, completion_idx ORDER BY step, sample_idx, completion_idx"
            params = base_params
        elif eval_metric_type == "length_sum":
            all_values_query = f"SELECT step, SUM(tokens) as value FROM rollouts_eval WHERE run_id = ? AND eval_name = ? {sample_filter} AND tokens IS NOT NULL GROUP BY step, sample_idx, completion_idx ORDER BY step, sample_idx, completion_idx"
            params = base_params
        elif eval_metric_type.startswith("reward_"):
            metric_name = eval_metric_type[7:]
            all_values_query = f"SELECT step, value FROM rollouts_metrics_eval WHERE run_id = ? AND eval_name = ? {sample_filter} AND metric_name = ? AND value IS NOT NULL ORDER BY step, sample_idx, completion_idx"
            params = base_params + [metric_name]
        else:
            log.warning(f"[API] Unknown eval distribution metric_type: {eval_metric_type}")
            return {"steps": [], "bin_edges": [], "counts": [], "global_min": None, "global_max": None}

        # Fall through to shared binning logic below (same as non-eval path)
        rows = con.execute(all_values_query, params).fetchall()
        if not rows:
            return {"steps": [], "bin_edges": [], "counts": [], "global_min": None, "global_max": None}

        from collections import defaultdict
        step_values = defaultdict(list)
        all_values = []
        for row in rows:
            step_values[row[0]].append(row[1])
            all_values.append(row[1])

        global_min = min(all_values)
        global_max = max(all_values)
        if global_min == global_max:
            global_min -= 0.5
            global_max += 0.5

        num_bins = req.num_bins
        bin_width = (global_max - global_min) / num_bins
        bin_edges = [global_min + i * bin_width for i in range(num_bins + 1)]

        sorted_steps = sorted(step_values.keys())
        steps = sorted_steps
        counts = []
        for step in sorted_steps:
            vals = step_values[step]
            step_counts = [0] * num_bins
            for v in vals:
                bin_idx = int((v - global_min) / bin_width)
                if bin_idx >= num_bins:
                    bin_idx = num_bins - 1
                if bin_idx < 0:
                    bin_idx = 0
                step_counts[bin_idx] += 1
            counts.append(step_counts)

        return {"steps": steps, "bin_edges": bin_edges, "counts": counts, "global_min": global_min, "global_max": global_max}

    # Check if this is a discarded metric type
    is_discarded = req.metric_type.startswith("discarded_")
    metric_type = req.metric_type[len("discarded_"):] if is_discarded else req.metric_type
    
    # Choose tables and step column based on discarded flag
    samples_table = "samples_data_discarded" if is_discarded else "samples_data"
    prompts_table = "prompts_discarded" if is_discarded else "prompts"
    rollouts_table = "rollouts_discarded" if is_discarded else "rollouts"
    gen_metrics_table = "rollouts_metrics_discarded" if is_discarded else "rollouts_metrics"
    step_col = "trainer_step" if is_discarded else "step"
    
    # Build query based on metric type
    if metric_type == "reward_sum":
        all_values_query = f"""
            SELECT {step_col} as step, reward as value FROM {samples_table}
            WHERE run_id = ? AND reward IS NOT NULL
            ORDER BY {step_col}, sample_idx
        """
        params = [req.run_path]
    elif metric_type == "advantage":
        all_values_query = f"""
            SELECT {step_col} as step, advantage as value FROM {samples_table}
            WHERE run_id = ? AND advantage IS NOT NULL
            ORDER BY {step_col}, sample_idx
        """
        params = [req.run_path]
    elif metric_type == "length_prompt":
        all_values_query = f"""
            SELECT {step_col} as step, tokens_prompt as value FROM {prompts_table}
            WHERE run_id = ? AND tokens_prompt IS NOT NULL
            ORDER BY {step_col}, group_id
        """
        params = [req.run_path]
    elif metric_type == "length_completion":
        all_values_query = f"""
            SELECT {step_col} as step, SUM(tokens) as value FROM {rollouts_table}
            WHERE run_id = ? AND turn_type = 'model' AND tokens IS NOT NULL
            GROUP BY {step_col}, sample_idx
            ORDER BY {step_col}, sample_idx
        """
        params = [req.run_path]
    elif metric_type == "length_sum":
        all_values_query = f"""
            SELECT {step_col} as step, total_tokens as value FROM {samples_table}
            WHERE run_id = ? AND total_tokens IS NOT NULL
            ORDER BY {step_col}, sample_idx
        """
        params = [req.run_path]
    elif metric_type == "off_policy_steps":
        all_values_query = f"""
            SELECT p.{step_col} as step, ei.off_policy_steps as value
            FROM events_inference ei
            JOIN {prompts_table} p ON p.run_id = ei.run_id AND p.group_id = ei.group_id
            WHERE ei.run_id = ? AND ei.event_type = 'request' AND ei.off_policy_steps IS NOT NULL
            ORDER BY p.{step_col}
        """
        params = [req.run_path]
    elif metric_type.startswith("reward_"):
        metric_name = metric_type[7:]
        if is_discarded:
            # For discarded rollout metrics, join with rollouts_discarded for trainer_step
            all_values_query = f"""
                SELECT gd.trainer_step as step, gmd.value FROM {gen_metrics_table} gmd
                JOIN (
                    SELECT DISTINCT run_id, trainer_step, sample_idx
                    FROM {rollouts_table}
                    WHERE run_id = ?
                ) gd ON gmd.run_id = gd.run_id AND gmd.sample_idx = gd.sample_idx
                WHERE gmd.run_id = ? AND gmd.metric_name = ? AND gmd.value IS NOT NULL
                ORDER BY gd.trainer_step, gmd.sample_idx
            """
            params = [req.run_path, req.run_path, metric_name]
        else:
            all_values_query = f"""
                SELECT step, value FROM {gen_metrics_table}
                WHERE run_id = ? AND metric_name = ? AND value IS NOT NULL
                ORDER BY step, sample_idx
            """
            params = [req.run_path, metric_name]
    else:
        log.warning(f"[API] Unknown metric_type: {req.metric_type}")
        return {
            "steps": [],
            "bin_edges": [],
            "counts": [],
            "global_min": None,
            "global_max": None,
        }
    
    rows = con.execute(all_values_query, params).fetchall()
    
    if not rows:
        return {
            "steps": [],
            "bin_edges": [],
            "counts": [],
            "global_min": None,
            "global_max": None,
        }
    
    # Group values by step
    from collections import defaultdict
    step_values = defaultdict(list)
    all_values = []
    
    for row in rows:
        step = row[0]
        value = row[1]
        step_values[step].append(value)
        all_values.append(value)
    
    # Calculate global min/max
    global_min = min(all_values)
    global_max = max(all_values)
    
    # Handle case where all values are the same
    if global_min == global_max:
        global_min -= 0.5
        global_max += 0.5
    
    # Create bin edges
    num_bins = req.num_bins
    bin_width = (global_max - global_min) / num_bins
    bin_edges = [global_min + i * bin_width for i in range(num_bins + 1)]
    
    # Sort steps
    sorted_steps = sorted(step_values.keys())
    
    # For each step, compute histogram counts
    # counts[step_idx][bin_idx] = count
    counts = []
    
    for step in sorted_steps:
        values = step_values[step]
        bin_counts = [0] * num_bins
        
        for value in values:
            bin_idx = int((value - global_min) / bin_width)
            # Handle edge case for max value
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            bin_counts[bin_idx] += 1
        
        counts.append(bin_counts)
    
    log.info(f"[API] Returning distribution over {len(sorted_steps)} steps, {num_bins} bins")
    
    return {
        "steps": sorted_steps,
        "bin_edges": bin_edges,
        "counts": counts,  # 2D array: [step_idx][bin_idx]
        "global_min": global_min,
        "global_max": global_max,
    }



# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

class LogsRequest(BaseModel):
    run_path: str
    page: int = 0
    page_size: int = 500
    components: list[str] | None = None
    levels: list[str] | None = None
    sources: list[str] | None = None
    search: str | None = None
    time_start: float | None = None
    time_end: float | None = None


@app.post("/api/logs")
def get_logs(req: LogsRequest):
    """Get paginated and filtered log records for a run."""
    con = connect()
    try:
        conditions = ["run_id = ?"]
        params: list = [req.run_path]

        if req.components:
            placeholders = ", ".join(["?"] * len(req.components))
            conditions.append(f"component IN ({placeholders})")
            params.extend(req.components)

        if req.levels:
            placeholders = ", ".join(["?"] * len(req.levels))
            conditions.append(f"level IN ({placeholders})")
            params.extend(req.levels)

        if req.sources:
            placeholders = ", ".join(["?"] * len(req.sources))
            conditions.append(f"source IN ({placeholders})")
            params.extend(req.sources)

        if req.search:
            conditions.append("message ILIKE ?")
            params.append(f"%{req.search}%")

        if req.time_start is not None:
            conditions.append("timestamp >= ?")
            params.append(req.time_start)

        if req.time_end is not None:
            conditions.append("timestamp <= ?")
            params.append(req.time_end)

        where = " AND ".join(conditions)

        total = con.execute(f"SELECT COUNT(*) FROM logs WHERE {where}", params).fetchone()[0]

        offset = req.page * req.page_size
        rows = con.execute(
            f"""SELECT timestamp, level, component, source, message
                FROM logs WHERE {where}
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?""",
            params + [req.page_size, offset]
        ).fetchall()

        return {
            "logs": [
                {"timestamp": r[0], "level": r[1], "component": r[2], "source": r[3], "message": r[4]}
                for r in rows
            ],
            "total": total,
            "page": req.page,
            "page_size": req.page_size,
            "total_pages": max(1, (total + req.page_size - 1) // req.page_size),
        }
    finally:
        con.close()


@app.get("/api/logs/summary/{run_path:path}")
def get_logs_summary(run_path: str):
    """Get available components, levels, sources for filter dropdowns."""
    con = connect()
    try:
        components = [r[0] for r in con.execute(
            "SELECT DISTINCT component FROM logs WHERE run_id = ? ORDER BY component", [run_path]
        ).fetchall()]
        levels = [r[0] for r in con.execute(
            "SELECT DISTINCT level FROM logs WHERE run_id = ? ORDER BY level", [run_path]
        ).fetchall()]
        sources = [r[0] for r in con.execute(
            "SELECT DISTINCT source FROM logs WHERE run_id = ? ORDER BY source", [run_path]
        ).fetchall()]
        total_count = con.execute(
            "SELECT COUNT(*) FROM logs WHERE run_id = ?", [run_path]
        ).fetchone()[0]
        return {
            "components": components,
            "levels": levels,
            "sources": sources,
            "total_count": total_count,
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Database compaction
# ---------------------------------------------------------------------------

@app.get("/database-info")
def database_info():
    """Return the current database file size."""
    return get_database_info()


@app.get("/compact-database/status")
def compact_database_status():
    """Return the current compaction progress."""
    return dict(_compaction_state)


@app.post("/compact-database")
async def start_compact_database():
    """Start a background database compaction (export → reimport)."""
    if _compaction_state["status"] not in ("idle", "done", "error"):
        raise HTTPException(status_code=409, detail="Compaction already in progress")

    _compaction_state.update(
        status="pausing_syncs",
        error=None,
        size_before=None,
        size_after=None,
    )
    asyncio.create_task(_run_compaction())
    return {"ok": True}


async def _run_compaction():
    """Execute the full compaction pipeline in the background."""
    try:
        # 1 — Pause all background loops and drain active syncs
        _compaction_state["status"] = "pausing_syncs"
        await pause_for_compaction()

        # 2 — Record size before
        info = get_database_info()
        _compaction_state["size_before"] = info["size_bytes"]

        # 3 — Run the (blocking) compaction in a thread
        def _progress(step: str):
            _compaction_state["status"] = step

        result = await asyncio.to_thread(compact_database, _progress)

        _compaction_state["status"] = "done"
        _compaction_state["size_before"] = result["size_before"]
        _compaction_state["size_after"] = result["size_after"]

    except Exception as e:
        _compaction_state["status"] = "error"
        _compaction_state["error"] = str(e)
        log.error(f"[COMPACT] Compaction failed: {e}", exc_info=True)

    finally:
        resume_after_compaction()
