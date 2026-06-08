import logging
import time

from app.worker import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def run_pipeline(self, file_path: str, file_type: str, texture: bool = True) -> dict:
    """Run the LangGraph 3D generation pipeline end-to-end (streaming)."""
    from app.pipeline.state import Pipeline3DState

    thread_id = str(self.request.id)
    delivery = self.request.delivery_info or {}
    _meta_base = {
        "task_id": thread_id,
        "worker": self.request.hostname,
        "queue": delivery.get("routing_key", "document_processing"),
    }
    self.update_state(state="PROCESSING", meta={
        **_meta_base,
        "status": "Starting LangGraph pipeline",
        "current_node": None,
        "node_history": [],
        "thread_id": thread_id,
        "ts": time.time(),
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
        "texture_enabled": texture,
        "model_info": None,
        "errors": [],
    }

    from app.pipeline.graph import run_pipeline_streaming

    node_history: list[str] = []

    def _on_node_event(node_name: str, state_update: dict):
        # Collapse subgraph prefixes: "spec_extraction:extract_spec_llm" -> keep full name
        # but cap history at 20 to keep meta small
        node_history.append(node_name)
        recent_errors = (state_update.get("errors") or [])[-5:]
        self.update_state(state="PROCESSING", meta={
            **_meta_base,
            "status": f"Running {node_name}",
            "current_node": node_name,
            "node_history": node_history[-20:],
            "recent_errors": recent_errors,
            "thread_id": thread_id,
            "ts": time.time(),
        })

    try:
        final_state = run_pipeline_streaming(
            initial_state, thread_id, on_event=_on_node_event,
        )
        model_info = final_state.get("model_info") or {}
        self.update_state(
            state="COMPLETED",
            meta={
                **_meta_base,
                "status": "Pipeline complete",
                "current_node": "store_result",
                "node_history": node_history[-20:],
                "model_info": model_info,
                "thread_id": thread_id,
                "errors": final_state.get("errors", []),
                "ts": time.time(),
            },
        )
        return model_info
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        self.update_state(state="FAILED", meta={
            **_meta_base,
            "status": "Pipeline failed",
            "error": str(e),
            "node_history": node_history[-20:],
            "thread_id": thread_id,
            "ts": time.time(),
        })
        raise


@celery_app.task(bind=True)
def resume_pipeline(self, thread_id: str) -> dict:
    """Resume a previously checkpointed LangGraph pipeline run."""
    from app.pipeline.graph import get_run_state, resume_run

    delivery = self.request.delivery_info or {}
    _meta_base = {
        "task_id": str(self.request.id),
        "worker": self.request.hostname,
        "queue": delivery.get("routing_key", "document_processing"),
    }

    snapshot = get_run_state(thread_id)
    if snapshot is None:
        msg = f"No checkpoint found for thread_id={thread_id}"
        logger.error(msg)
        self.update_state(state="FAILED", meta={**_meta_base, "status": msg, "thread_id": thread_id})
        raise ValueError(msg)

    logger.info("resume_pipeline: thread_id=%s", thread_id)
    self.update_state(state="PROCESSING", meta={
        **_meta_base,
        "status": "Resuming from last checkpoint",
        "current_node": None,
        "node_history": [],
        "thread_id": thread_id,
        "ts": time.time(),
    })

    node_history: list[str] = []

    def _on_node_event(node_name: str, state_update: dict):
        node_history.append(node_name)
        recent_errors = (state_update.get("errors") or [])[-5:]
        self.update_state(state="PROCESSING", meta={
            **_meta_base,
            "status": f"Running {node_name}",
            "current_node": node_name,
            "node_history": node_history[-20:],
            "recent_errors": recent_errors,
            "thread_id": thread_id,
            "resumed": True,
            "ts": time.time(),
        })

    try:
        final_state = resume_run(thread_id, on_event=_on_node_event)
        model_info = final_state.get("model_info") or {}
        self.update_state(state="COMPLETED", meta={
            **_meta_base,
            "status": "Pipeline complete (resumed)",
            "current_node": "store_result",
            "node_history": node_history[-20:],
            "model_info": model_info,
            "thread_id": thread_id,
            "errors": final_state.get("errors", []),
            "ts": time.time(),
        })
        return model_info
    except Exception as e:
        logger.error("Resume failed: %s", e)
        self.update_state(state="FAILED", meta={
            **_meta_base,
            "status": "Pipeline resume failed",
            "error": str(e),
            "node_history": node_history[-20:],
            "thread_id": thread_id,
            "ts": time.time(),
        })
        raise
