"""LangGraph 3D-generation pipeline.

Architecture:
    parse_document
         |
    validate_parsed_document   <-- best-practice gap filled: warn on empty parse
         |
    spec_extraction (subgraph) <-- extract_spec_llm <-> validate_spec <-> fallback
         |
    mesh_generation (subgraph) <-- generate_mesh <-> validate_mesh
         |
    store_result
         |
        END

Features wired:
    * StateGraph + TypedDict state                         (graph topology)
    * Reducer on errors (Annotated[List, operator.add])    (cross-node accumulation)
    * Conditional edges for retry / fallback routing       (branching)
    * Compiled subgraphs for spec and mesh                 (modularity)
    * SqliteSaver checkpointer (persistent, resumable)     (durability)
    * Optional interrupt_after for human-in-the-loop pause (HITL)
    * Sync `pipeline.invoke` / `pipeline.stream`           (block / progress)
    * Async `pipeline.ainvoke` / `pipeline.astream`        (non-blocking)
    * Helper `run_pipeline_streaming(state, thread_id, on_event)` for Celery use

Backward compat:
    `pipeline` is still the module-level compiled graph (with checkpointer +
    validate_parse + subgraphs). Existing callers that do `pipeline.invoke(state)`
    must now pass a `config={"configurable": {"thread_id": ...}}`. The helper
    `run_pipeline(initial_state, thread_id)` wraps this for convenience.
"""

import logging
import os
import sqlite3

from langgraph.graph import END, StateGraph

from app.pipeline.nodes import (
    build_fallback_spec_node,
    extract_spec_llm_node,
    generate_mesh_node,
    parse_document_node,
    store_result_node,
    validate_mesh_node,
    validate_parsed_document_node,
    validate_spec_node,
)
from app.pipeline.state import Pipeline3DState

logger = logging.getLogger(__name__)

MAX_SPEC_RETRIES = 3
MAX_MESH_RETRIES = 2

# ----------------------------------------------------------------------
# Checkpointer (SqliteSaver) — persistent, allows resuming a run after a
# Celery worker crash, OOM, or planned interrupt. The DB lives next to the
# generated assets so it survives container restarts when that volume is
# mounted. Override the path with PIPELINE_CHECKPOINT_DB env var.
# ----------------------------------------------------------------------
_DEFAULT_CHECKPOINT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'generated', 'pipeline_checkpoints.db',
))
CHECKPOINT_DB_PATH = os.environ.get("PIPELINE_CHECKPOINT_DB", _DEFAULT_CHECKPOINT_PATH)

_checkpointer = None


def _get_checkpointer():
    """Return a module-level singleton SqliteSaver, or None on failure.

    Uses a single connection with `check_same_thread=False` so Celery
    worker threads share it. SQLite locking + WAL handles concurrent writes.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        _checkpointer = SqliteSaver(conn)
        logger.info("Pipeline checkpointer ready at %s", CHECKPOINT_DB_PATH)
    except Exception as e:
        logger.warning("Pipeline checkpointer disabled (%s). Runs will not be resumable.", e)
        _checkpointer = None
    return _checkpointer


# ----------------------------------------------------------------------
# Routers (conditional-edge decision functions)
# ----------------------------------------------------------------------

def _route_after_validate_spec(state: Pipeline3DState) -> str:
    if state.get("spec_valid"):
        return "generate_mesh"
    if state.get("spec_retry_count", 0) >= MAX_SPEC_RETRIES:
        return "build_fallback_spec"
    return "extract_spec_llm"


def _route_after_validate_mesh(state: Pipeline3DState) -> str:
    if state.get("mesh_valid"):
        return "store_result"
    if state.get("mesh_retry_count", 0) >= MAX_MESH_RETRIES:
        return "store_result"
    return "generate_mesh"


# ----------------------------------------------------------------------
# Subgraphs
# ----------------------------------------------------------------------

def _build_spec_extraction_subgraph():
    """LLM spec extraction with bounded retry and hand-crafted fallback.

    Compiled WITHOUT its own checkpointer; the parent graph's checkpointer
    transparently persists this subgraph's internal node transitions.
    """
    g = StateGraph(Pipeline3DState)
    g.add_node("extract_spec_llm", extract_spec_llm_node)
    g.add_node("validate_spec", validate_spec_node)
    g.add_node("build_fallback_spec", build_fallback_spec_node)
    g.set_entry_point("extract_spec_llm")
    g.add_edge("extract_spec_llm", "validate_spec")
    g.add_conditional_edges(
        "validate_spec",
        _route_after_validate_spec,
        {
            "generate_mesh": END,
            "build_fallback_spec": "build_fallback_spec",
            "extract_spec_llm": "extract_spec_llm",
        },
    )
    g.add_edge("build_fallback_spec", END)
    return g.compile()


def _build_mesh_generation_subgraph():
    """Mesh generation with retry-on-validation-failure.

    Mesh gen takes ~20 min; checkpointer (set on the parent graph) persists
    after each internal step so a crash mid-generation can resume from the
    last completed retry instead of restarting from scratch.
    """
    g = StateGraph(Pipeline3DState)
    g.add_node("generate_mesh", generate_mesh_node)
    g.add_node("validate_mesh", validate_mesh_node)
    g.set_entry_point("generate_mesh")
    g.add_edge("generate_mesh", "validate_mesh")
    g.add_conditional_edges(
        "validate_mesh",
        _route_after_validate_mesh,
        {
            "store_result": END,
            "generate_mesh": "generate_mesh",
        },
    )
    return g.compile()


# ----------------------------------------------------------------------
# Main pipeline builder
# ----------------------------------------------------------------------

def build_pipeline(*, interrupt_after=None, with_checkpointer: bool = True):
    """Build and compile the main pipeline graph."""
    spec_subgraph = _build_spec_extraction_subgraph()
    mesh_subgraph = _build_mesh_generation_subgraph()

    graph = StateGraph(Pipeline3DState)
    graph.add_node("parse_document", parse_document_node)
    graph.add_node("validate_parsed_document", validate_parsed_document_node)
    graph.add_node("spec_extraction", spec_subgraph)
    graph.add_node("mesh_generation", mesh_subgraph)
    graph.add_node("store_result", store_result_node)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "validate_parsed_document")
    graph.add_edge("validate_parsed_document", "spec_extraction")
    graph.add_edge("spec_extraction", "mesh_generation")
    graph.add_edge("mesh_generation", "store_result")
    graph.add_edge("store_result", END)

    compile_kwargs = {}
    if with_checkpointer:
        ckpt = _get_checkpointer()
        if ckpt is not None:
            compile_kwargs["checkpointer"] = ckpt
    if interrupt_after:
        compile_kwargs["interrupt_after"] = list(interrupt_after)
    return graph.compile(**compile_kwargs)


# Module-level compiled pipeline (checkpointer attached, no interrupts).
pipeline = build_pipeline()


# ----------------------------------------------------------------------
# Helpers for callers (tasks.py / API routes)
# ----------------------------------------------------------------------

def make_thread_config(thread_id: str) -> dict:
    """Build the `config` dict LangGraph expects when a checkpointer is
    attached. Pass the Celery task ID as `thread_id`."""
    return {"configurable": {"thread_id": str(thread_id)}}


def run_pipeline(initial_state: Pipeline3DState, thread_id: str) -> dict:
    """Synchronous, blocking run. Returns the final state dict."""
    return pipeline.invoke(initial_state, config=make_thread_config(thread_id))


def run_pipeline_streaming(initial_state, thread_id: str, on_event=None) -> dict:
    """Streaming run. Calls `on_event(node_name, state_update)` after each
    node, then returns the final state from the checkpointer.

    Pass `initial_state=None` to RESUME a previously checkpointed run with
    the same thread_id.
    """
    config = make_thread_config(thread_id)
    last_state: dict = {}
    try:
        for event in pipeline.stream(initial_state, config=config, subgraphs=True):
            if isinstance(event, tuple) and len(event) == 2:
                parent_path, update_dict = event
                prefix = ":".join(str(p) for p in parent_path) + ":" if parent_path else ""
            else:
                update_dict = event
                prefix = ""
            if not isinstance(update_dict, dict):
                continue
            for node_name, state_update in update_dict.items():
                if node_name in ("__start__", "__end__"):
                    continue
                full_name = f"{prefix}{node_name}"
                if on_event is not None and isinstance(state_update, dict):
                    try:
                        on_event(full_name, state_update)
                    except Exception:
                        logger.exception("on_event callback raised; continuing")
                if isinstance(state_update, dict):
                    last_state.update(state_update)
    except Exception:
        logger.exception("Streaming pipeline run failed")
        raise
    try:
        snapshot = pipeline.get_state(config)
        if snapshot is not None and snapshot.values:
            return dict(snapshot.values)
    except Exception:
        logger.warning("Could not fetch final state from checkpointer; using accumulated state")
    return last_state


async def arun_pipeline(initial_state: Pipeline3DState, thread_id: str) -> dict:
    """Async equivalent of `run_pipeline`."""
    return await pipeline.ainvoke(initial_state, config=make_thread_config(thread_id))


async def arun_pipeline_streaming(initial_state, thread_id: str, on_event=None) -> dict:
    """Async streaming run via `astream`."""
    config = make_thread_config(thread_id)
    last_state: dict = {}
    try:
        async for event in pipeline.astream(initial_state, config=config, subgraphs=True):
            if isinstance(event, tuple) and len(event) == 2:
                parent_path, update_dict = event
                prefix = ":".join(str(p) for p in parent_path) + ":" if parent_path else ""
            else:
                update_dict = event
                prefix = ""
            if not isinstance(update_dict, dict):
                continue
            for node_name, state_update in update_dict.items():
                if node_name in ("__start__", "__end__"):
                    continue
                full_name = f"{prefix}{node_name}"
                if on_event is not None and isinstance(state_update, dict):
                    try:
                        res = on_event(full_name, state_update)
                        if hasattr(res, "__await__"):
                            await res
                    except Exception:
                        logger.exception("on_event callback raised; continuing")
                if isinstance(state_update, dict):
                    last_state.update(state_update)
    except Exception:
        logger.exception("Async streaming pipeline run failed")
        raise
    try:
        snapshot = await pipeline.aget_state(config)
        if snapshot is not None and snapshot.values:
            return dict(snapshot.values)
    except Exception:
        logger.warning("Could not fetch final state (async); using accumulated state")
    return last_state


def get_run_state(thread_id: str):
    """Return the latest checkpointed state for a given thread_id, or None."""
    try:
        snapshot = pipeline.get_state(make_thread_config(thread_id))
        return dict(snapshot.values) if snapshot and snapshot.values else None
    except Exception as e:
        logger.warning("get_run_state(%s) failed: %s", thread_id, e)
        return None


def resume_run(thread_id: str, on_event=None) -> dict:
    """Resume a paused / interrupted run from its last checkpoint."""
    return run_pipeline_streaming(None, thread_id, on_event=on_event)


def build_pipeline_with_interrupts(interrupt_after):
    """Build a separately-compiled pipeline that pauses after the given
    top-level nodes. Use for HITL workflows."""
    return build_pipeline(interrupt_after=interrupt_after)
