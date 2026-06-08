import logging

from app.worker import celery_app

logger = logging.getLogger(__name__)

# Legacy `process_document` and `generate_3d_model` Celery tasks were
# removed; the LangGraph `run_pipeline` task below is the canonical path.

@celery_app.task(bind=True)
def run_pipeline(self, file_path: str, file_type: str) -> dict:
    """Run the LangGraph 3D generation pipeline end-to-end (streaming)."""
    from app.pipeline.state import Pipeline3DState

    # Use the Celery task ID as the LangGraph thread_id so this run's
    # checkpoints are isolated and can be resumed by ID if the worker crashes.
    thread_id = str(self.request.id)
    self.update_state(state="PROCESSING", meta={
        "status": "Starting LangGraph pipeline",
        "thread_id": thread_id,
    })
    logger.info("run_pipeline: %s (%s) thread_id=%s", file_path, file_type, thread_id)

    initial_state: Pipeline3DState = {
        "file_path": file_path,
        "file_type": file_type,
        "raw_text": "",
        "parsed_content": {},
        "spec": None,
        "spec_valid": False,
        "spec_retry_count": 0,
        "mesh_output": None,
        "mesh_valid": False,
        "mesh_retry_count": 0,
        "model_info": None,
        "errors": [],
    }

    # Stream node-level events so Celery state reflects live progress.
    # Frontend polling /task/{id} now sees the current node name instead
    # of just "PROCESSING" for 20 minutes.
    from app.pipeline.graph import run_pipeline_streaming

    def _on_node_event(node_name: str, state_update: dict):
        # Truncate errors list to last 5 to keep Celery meta small
        recent_errors = (state_update.get("errors") or [])[-5:]
        self.update_state(state="PROCESSING", meta={
            "status": f"Running {node_name}",
            "current_node": node_name,
            # current_step removed — node_name above is the authoritative current node
            "recent_errors": recent_errors,
            "thread_id": thread_id,
        })

    try:
        final_state = run_pipeline_streaming(
            initial_state, thread_id, on_event=_on_node_event,
        )
        model_info = final_state.get("model_info") or {}
        self.update_state(
            state="COMPLETED",
            meta={
                "status": "Pipeline complete",
                "model_info": model_info,
                "thread_id": thread_id,
                "errors": final_state.get("errors", []),
            },
        )
        return model_info
    except Exception as e:
        error_msg = f"Pipeline failed: {e}"
        logger.error(error_msg)
        self.update_state(state="FAILED", meta={
            "status": "Pipeline failed",
            "error": str(e),
            "thread_id": thread_id,
        })
        raise


@celery_app.task(bind=True)
def resume_pipeline(self, thread_id: str) -> dict:
    """Resume a previously checkpointed LangGraph pipeline run.

    Use case: the original worker crashed (OOM, kill, deploy) mid-mesh-gen.
    The state up to the last completed node was saved to the SqliteSaver
    checkpointer. Call this with the original thread_id to pick up from there
    instead of restarting from the parse stage.
    """
    from app.pipeline.graph import get_run_state, resume_run

    snapshot = get_run_state(thread_id)
    if snapshot is None:
        msg = f"No checkpoint found for thread_id={thread_id}"
        logger.error(msg)
        self.update_state(state="FAILED", meta={"status": msg, "thread_id": thread_id})
        raise ValueError(msg)

    logger.info("resume_pipeline: thread_id=%s resuming from %s",
                thread_id, snapshot.get("model_info", {}).get("model_id", "?"))
    self.update_state(state="PROCESSING", meta={
        "status": "Resuming from last checkpoint",
        "thread_id": thread_id,
        "resumed_from": snapshot.get("model_info", {}).get("model_id"),
    })

    def _on_node_event(node_name: str, state_update: dict):
        recent_errors = (state_update.get("errors") or [])[-5:]
        self.update_state(state="PROCESSING", meta={
            "status": f"Running {node_name}",
            "current_node": node_name,
            # current_step removed — node_name above is the authoritative current node
            "recent_errors": recent_errors,
            "thread_id": thread_id,
            "resumed": True,
        })

    try:
        final_state = resume_run(thread_id, on_event=_on_node_event)
        model_info = final_state.get("model_info") or {}
        self.update_state(state="COMPLETED", meta={
            "status": "Pipeline complete (resumed)",
            "model_info": model_info,
            "thread_id": thread_id,
            "errors": final_state.get("errors", []),
        })
        return model_info
    except Exception as e:
        logger.error("Resume failed: %s", e)
        self.update_state(state="FAILED", meta={
            "status": "Pipeline resume failed",
            "error": str(e),
            "thread_id": thread_id,
        })
        raise
