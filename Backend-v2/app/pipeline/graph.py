import logging

from langgraph.graph import END, StateGraph

from app.pipeline.nodes import (
    build_fallback_spec_node,
    extract_spec_llm_node,
    generate_mesh_node,
    parse_document_node,
    store_result_node,
    validate_mesh_node,
    validate_spec_node,
)
from app.pipeline.state import Pipeline3DState

logger = logging.getLogger(__name__)

MAX_SPEC_RETRIES = 3
MAX_MESH_RETRIES = 2


def _route_after_validate_spec(state: Pipeline3DState) -> str:
    if state["spec_valid"]:
        return "generate_mesh"
    if state.get("spec_retry_count", 0) >= MAX_SPEC_RETRIES:
        return "build_fallback_spec"
    return "extract_spec_llm"


def _route_after_validate_mesh(state: Pipeline3DState) -> str:
    if state["mesh_valid"]:
        return "store_result"
    if state.get("mesh_retry_count", 0) >= MAX_MESH_RETRIES:
        # Store whatever we have rather than looping forever
        return "store_result"
    return "generate_mesh"


def build_pipeline() -> StateGraph:
    graph = StateGraph(Pipeline3DState)

    graph.add_node("parse_document", parse_document_node)
    graph.add_node("extract_spec_llm", extract_spec_llm_node)
    graph.add_node("validate_spec", validate_spec_node)
    graph.add_node("build_fallback_spec", build_fallback_spec_node)
    graph.add_node("generate_mesh", generate_mesh_node)
    graph.add_node("validate_mesh", validate_mesh_node)
    graph.add_node("store_result", store_result_node)

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "extract_spec_llm")
    graph.add_edge("extract_spec_llm", "validate_spec")
    graph.add_conditional_edges(
        "validate_spec",
        _route_after_validate_spec,
        {
            "generate_mesh": "generate_mesh",
            "build_fallback_spec": "build_fallback_spec",
            "extract_spec_llm": "extract_spec_llm",
        },
    )
    graph.add_edge("build_fallback_spec", "generate_mesh")
    graph.add_edge("generate_mesh", "validate_mesh")
    graph.add_conditional_edges(
        "validate_mesh",
        _route_after_validate_mesh,
        {
            "store_result": "store_result",
            "generate_mesh": "generate_mesh",
        },
    )
    graph.add_edge("store_result", END)

    return graph.compile()


# Module-level compiled graph — imported by tasks and routes
pipeline = build_pipeline()
