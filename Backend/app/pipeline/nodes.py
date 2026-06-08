import logging
from datetime import datetime

from app.pipeline.state import Pipeline3DState

logger = logging.getLogger(__name__)


def parse_document_node(state: Pipeline3DState) -> dict:
    from app.services.document_parser import document_parser

    logger.info("Pipeline: parse_document | %s", state["file_path"])
    parsed = document_parser.parse_document(state["file_path"], state["file_type"])
    return {
        "raw_text": parsed.get("content", ""),
        "parsed_content": parsed,
        "current_step": "extract_spec_llm",
        "errors": [],
    }


def extract_spec_llm_node(state: Pipeline3DState) -> dict:
    from app.services.llm_service import get_llm_service
    from app.services.prompt_engineering import get_prompt_engineer

    retry = state.get("spec_retry_count", 0)
    logger.info("Pipeline: extract_spec_llm (attempt %d)", retry + 1)

    llm = get_llm_service()
    pe = get_prompt_engineer()

    prompt = pe.create_extraction_prompt(state["raw_text"], "document_analysis")
    try:
        response = llm.generate_response(prompt, max_tokens=1024, temperature=0.7)
        json_data = llm.extract_json_from_text(response)
        return {
            "spec": json_data,
            "spec_valid": False,
            "current_step": "validate_spec",
            "errors": [],
        }
    except Exception as e:
        logger.warning("LLM extract failed: %s", e)
        return {
            "spec": None,
            "spec_valid": False,
            "current_step": "validate_spec",
            "errors": [f"LLM extract error: {e}"],
        }


def validate_spec_node(state: Pipeline3DState) -> dict:
    from app.models.spec_models import ObjectSpec

    spec = state.get("spec")
    retry = state.get("spec_retry_count", 0)

    if not spec:
        return {
            "spec_valid": False,
            "spec_retry_count": retry + 1,
            "errors": ["LLM returned no JSON"],
        }

    # DocumentAnalysis wraps objects in a list — unwrap first object
    obj_data = spec
    if "objects" in spec and spec["objects"]:
        obj_data = spec["objects"][0]

    try:
        ObjectSpec(**obj_data)
        logger.info("Pipeline: spec valid")
        return {
            "spec": obj_data,
            "spec_valid": True,
            "errors": [],
        }
    except Exception as e:
        logger.warning("Spec validation failed: %s", e)
        return {
            "spec_valid": False,
            "spec_retry_count": retry + 1,
            "errors": [f"Spec invalid: {e}"],
        }


def build_fallback_spec_node(state: Pipeline3DState) -> dict:
    logger.info("Pipeline: building fallback spec")
    raw_text = state.get("raw_text", "")
    title = "3D Object"
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line) > 5:
            title = line[:60]
            break

    return {
        "spec": {
            "name": title,
            "shape": "CUSTOM",
            "dimensions": {"length": 100, "width": 100, "height": 100, "unit": "mm"},
            "material": {"type": "Plastic", "color": "Matte Black"},
        },
        "spec_valid": True,
        "errors": ["Fallback spec used after LLM retry exhaustion"],
        "current_step": "generate_mesh",
    }


def generate_mesh_node(state: Pipeline3DState) -> dict:
    from app.services.hunyuan3d_service import get_hunyuan3d

    spec = state.get("spec") or {}
    retry = state.get("mesh_retry_count", 0)
    logger.info("Pipeline: generate_mesh (attempt %d)", retry + 1)

    name = spec.get("name", "3D Object")
    description = spec.get("description", "")
    mat = spec.get("material", {})
    material_str = ""
    if isinstance(mat, dict):
        material_str = f"{mat.get('type', '')} {mat.get('color', '')}".strip()

    text_prompt = f"A detailed 3D model of {name.lower()}"
    if description:
        text_prompt += f", {description.lower()}"
    if material_str:
        text_prompt += f", made of {material_str.lower()}"
    text_prompt += ", high quality, professional 3D rendering"

    try:
        service = get_hunyuan3d()
        result = service.text_to_3d(text_prompt)
        return {
            "mesh_output": result,
            "mesh_valid": False,
            "current_step": "validate_mesh",
            "errors": [],
        }
    except Exception as e:
        logger.warning("Mesh generation failed: %s", e)
        return {
            "mesh_output": None,
            "mesh_valid": False,
            "current_step": "validate_mesh",
            "errors": [f"Mesh generation error: {e}"],
        }


def validate_mesh_node(state: Pipeline3DState) -> dict:
    mesh_output = state.get("mesh_output")
    retry = state.get("mesh_retry_count", 0)

    if not mesh_output:
        return {
            "mesh_valid": False,
            "mesh_retry_count": retry + 1,
            "errors": ["Mesh output is None"],
        }

    # hunyuan3d _export_mesh returns uid + preview_url + download_url + format
    if "uid" in mesh_output and "preview_url" in mesh_output:
        logger.info("Pipeline: mesh valid (uid=%s)", mesh_output["uid"])
        return {"mesh_valid": True, "errors": []}

    logger.warning("Mesh output missing expected keys: %s", list(mesh_output.keys()))
    return {
        "mesh_valid": False,
        "mesh_retry_count": retry + 1,
        "errors": [f"Mesh output missing expected keys: {list(mesh_output.keys())}"],
    }


def store_result_node(state: Pipeline3DState) -> dict:
    from app.services import gallery_db

    spec = state.get("spec") or {}
    mesh_output = state.get("mesh_output") or {}
    uid = mesh_output.get("uid")

    model_info = {
        "model_id": f"model_{int(datetime.now().timestamp())}",
        "title": spec.get("name", "Generated Model"),
        "description": spec.get("description", ""),
        "preview_url": mesh_output.get("preview_url"),
        "download_url": mesh_output.get("download_url"),
        "format": mesh_output.get("format", "glb"),
        "uid": uid,
        "generation_time": datetime.now().isoformat(),
        "pipeline": "langgraph",
        "errors": state.get("errors", []),
    }

    if uid:
        try:
            gallery_db.insert(
                uid=uid,
                prompt=spec.get("description", ""),
                source="pipeline",
                preview_url=mesh_output.get("preview_url", ""),
                download_url=mesh_output.get("download_url", ""),
            )
        except Exception:
            logger.exception("gallery_db: failed to save pipeline result %s", uid)

    logger.info("Pipeline: complete | model_id=%s", model_info["model_id"])
    return {"model_info": model_info, "current_step": "done"}
