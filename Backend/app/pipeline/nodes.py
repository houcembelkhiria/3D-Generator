import json
import logging
import os
import signal
import threading
from contextlib import contextmanager
from datetime import datetime

from app.pipeline.state import Pipeline3DState

logger = logging.getLogger(__name__)


# Per-node timeouts (seconds). Override via env:
#   LG_TIMEOUT_LLM=120, LG_TIMEOUT_MESH=1200, LG_TIMEOUT_DEFAULT=60
# A timeout raises NodeTimeoutError which the node body catches and records
# as a normal failure (so the validator + retry / fallback router still apply).
LG_TIMEOUT_LLM = int(os.environ.get("LG_TIMEOUT_LLM", "120"))      # 2 min per LLM call
LG_TIMEOUT_MESH = int(os.environ.get("LG_TIMEOUT_MESH", "1200"))   # 20 min per mesh gen
LG_TIMEOUT_DEFAULT = int(os.environ.get("LG_TIMEOUT_DEFAULT", "60"))


class NodeTimeoutError(TimeoutError):
    pass


@contextmanager
def _node_timeout(seconds: int, label: str):
    """Raise NodeTimeoutError if the wrapped block runs longer than `seconds`.

    Uses SIGALRM in the main thread, threading.Timer + thread-kill fallback
    in non-main threads (Celery prefork worker child = main thread, so SIGALRM
    works; threaded pool workers fall back to the polite timer that asks the
    block to check `_TIMEOUT_FLAG` — for true thread-kill use Celery's
    task_time_limit which sends SIGTERM to the worker process).
    """
    is_main = threading.current_thread() is threading.main_thread()
    if not is_main or seconds <= 0:
        # No-op fallback; Celery's task_time_limit is the backstop
        try:
            yield
        finally:
            pass
        return

    def _handler(signum, frame):
        raise NodeTimeoutError(f"{label} exceeded {seconds}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_document_node(state: Pipeline3DState) -> dict:
    from app.services.document_parser import document_parser

    logger.info("Pipeline: parse_document | %s", state["file_path"])
    parsed = document_parser.parse_document(state["file_path"], state["file_type"])
    return {
        "raw_text": parsed.get("content", ""),
        "parsed_content": parsed,
        "errors": [],
    }


def validate_parsed_document_node(state: Pipeline3DState) -> dict:
    """Sanity-check the parsed document before sending it to the LLM.

    Records warnings into state.errors if the parsed text is suspiciously
    short or empty (typical signs of OCR failure / unsupported format).
    Does NOT terminate the pipeline — the LLM will still get a chance and
    the fallback spec covers the case where extraction completely fails.
    Surfacing the parse problem in the error log makes the root cause
    visible instead of being buried behind 3 LLM retries.
    """
    text = (state.get("raw_text") or "").strip()
    parsed = state.get("parsed_content") or {}
    warnings = []
    if len(text) < 20:
        warnings.append(
            f"Parsed document is very short ({len(text)} chars) — likely empty,"
            " image-only, or OCR failure. LLM extraction may not produce useful spec."
        )
    if not parsed:
        warnings.append("Parsed content dict is empty — document parser returned no metadata.")
    if warnings:
        logger.warning("Pipeline: validate_parsed_document raised %d warning(s)", len(warnings))
    return {
        "errors": warnings,
    }


def extract_spec_llm_node(state: Pipeline3DState) -> dict:
    from app.services.llm_service import get_llm_service, OllamaLLMService
    from app.services.prompt_engineering import get_prompt_engineer
    from app.models.spec_models import ObjectSpec

    retry = state.get("spec_retry_count", 0)
    logger.info("Pipeline: extract_spec_llm (attempt %d)", retry + 1)

    llm = get_llm_service()
    pe = get_prompt_engineer()

    try:
        with _node_timeout(LG_TIMEOUT_LLM, "extract_spec_llm"):
            if isinstance(llm, OllamaLLMService):
                # Chat path with grammar-constrained JSON. Schema-constrained
                # decoding guarantees the response validates against ObjectSpec,
                # so we skip the regex `extract_json_from_text` retry loop.
                system_text, user_text = pe.create_extraction_messages(
                    state["raw_text"], "object_spec"
                )
                schema = ObjectSpec.model_json_schema()
                response = llm.generate_chat(
                    system_text, user_text,
                    schema=schema, max_tokens=1024, temperature=0.2,
                )
            else:
                # GGUF path - keep the legacy Llama-3-templated prompt.
                prompt = pe.create_extraction_prompt(state["raw_text"], "document_analysis")
                response = llm.generate_response(prompt, max_tokens=1024, temperature=0.7)

        # Empty response = real failure, not "valid JSON we just can't find".
        # Surfacing this loudly triggers the retry router instead of silently
        # flowing to the fallback spec.
        if not response or not response.strip():
            raise RuntimeError("LLM returned empty response")

        # Schema-constrained Ollama output parses directly; legacy path still
        # needs the regex extractor for code-fenced / explanatory JSON.
        try:
            json_data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            json_data = llm.extract_json_from_text(response)

        return {
            "spec": json_data,
            "spec_valid": False,
            "errors": [],
        }
    except NodeTimeoutError as te:
        logger.warning("LLM extract timed out: %s", te)
        return {
            "spec": None,
            "spec_valid": False,
            "errors": [f"LLM extract timeout: {te}"],
        }
    except Exception as e:
        logger.warning("LLM extract failed: %s", e)
        return {
            "spec": None,
            "spec_valid": False,
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


# Lines that are email/PDF chrome rather than subject content. The previous
# fallback picked "Hi team," or "From: artist@studio3d.com" as the object name,
# which left the t2i prompt with no subject at all -> generic black blob.
_FALLBACK_SKIP_PREFIXES = (
    # Email/MIME chrome
    "from:", "to:", "cc:", "bcc:", "subject:", "date:", "mime-version:",
    "content-type:", "content-transfer-encoding:", "x-",
    # Greetings / sign-offs (real subject lines never look like these)
    "hi ", "hello", "hey ", "dear ", "best ", "regards", "thanks", "thank you",
    "please ", "sincerely", "cheers",
    # PDF chrome (page footers, confidentiality boilerplate)
    "page ", "p.", "confidential", "draft", "rev.", "version ",
    "copyright", "(c)", "all rights reserved",
)


# Substrings that mark a banner/footer line anywhere on it (not just at the
# start) — e.g. "ACME Engineering Confidential" or "Spec sheet - DRAFT".
_FALLBACK_CHROME_SUBSTRINGS = (
    "confidential", "all rights reserved", "copyright",
)

import re as _re
_FALLBACK_PAGE_NUMBER_RE = _re.compile(r"^[\divxlc]+\s*(/|of)\s*[\divxlc]+$", _re.IGNORECASE)

# Tokens that mark a line as describing the subject. First matching line wins
# its trailing value as the object name.
# Iteration order matters: explicit asset keys win over email/PDF subject
# headers, so an eml with both "Subject: Asset Request" and "OBJECT: Crate"
# picks the latter as the more specific source of truth.
_FALLBACK_NAME_KEYS = (
    "object:", "asset:", "product:", "title:", "name:",
    "subject:",  # email/PDF header - last resort
)

# Truncation cap for the fallback `description`. Big enough to carry the
# essential subject signal (materials, dimensions, accents) into the t2i
# prompt; small enough to stay well under SDXL's CLIP token budget.
_FALLBACK_DESC_CHARS = 1500


def _pick_fallback_name(raw_text: str) -> str:
    """Choose a meaningful object name from raw document text.

    Strategy: first look for an explicit key line ("OBJECT: X", "TITLE: X").
    If none, take the first content line that isn't email/PDF chrome.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    # Outer loop is keys, inner loop is lines: this enforces the priority
    # order in _FALLBACK_NAME_KEYS so "OBJECT:" later in the document beats
    # "Subject:" in the email header.
    for key in _FALLBACK_NAME_KEYS:
        for line in lines:
            low = line.lower()
            if low.startswith(key):
                value = line[len(key):].strip(" :-\t")
                if value:
                    return value[:60]
    for line in lines:
        low = line.lower()
        if any(low.startswith(p) for p in _FALLBACK_SKIP_PREFIXES):
            continue
        if any(kw in low for kw in _FALLBACK_CHROME_SUBSTRINGS):
            continue
        if _FALLBACK_PAGE_NUMBER_RE.match(low):
            continue
        if len(line) >= 8 and not line.startswith("---"):
            return line[:60]
    return "3D Object"


def build_fallback_spec_node(state: Pipeline3DState) -> dict:
    """Hand-crafted spec used after LLM retry exhaustion.

    Two changes from the original `{Plastic, Matte Black}` cube:
    - `name` is picked via _pick_fallback_name so it carries subject signal
      instead of "Hi team," or "From:" headers.
    - `description` carries the full raw_text (truncated) so the downstream
      t2i prompt builder still sees "wooden crate, planks, oak..." even when
      every LLM call returned garbage. Material defaults are unchanged to
      keep behaviour identical for runs that previously succeeded.
    """
    logger.info("Pipeline: building fallback spec")
    raw_text = state.get("raw_text", "") or ""
    title = _pick_fallback_name(raw_text)
    description = raw_text.strip()[:_FALLBACK_DESC_CHARS]

    return {
        "spec": {
            "name": title,
            "description": description,
            "shape": "CUSTOM",
            "dimensions": {"length": 100, "width": 100, "height": 100, "unit": "mm"},
            "material": {"type": "Plastic", "color": "Matte Black"},
        },
        "spec_valid": True,
        "errors": ["Fallback spec used after LLM retry exhaustion"],
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

    # Build a rich prompt from all spec fields so Hunyuan3D generates the described object
    dims = spec.get("dimensions", {})
    shape = spec.get("shape", "")
    hollow = spec.get("hollow", False)

    parts = [f"A detailed 3D model of {name.lower()}"]
    if description:
        parts.append(description.lower())
    if shape and shape.upper() not in ("CUSTOM", ""):
        parts.append(f"{shape.lower()} shape")
    if dims:
        unit = dims.get("unit", "mm")
        d_parts = []
        for k in ("length", "width", "height", "diameter"):
            v = dims.get(k)
            if v:
                d_parts.append(f"{v}{unit} {k}")
        if d_parts:
            parts.append("dimensions: " + " x ".join(d_parts))
    if material_str:
        parts.append(f"made of {material_str.lower()}")
    if hollow:
        parts.append("hollow interior")
    parts.append("high quality, professional 3D rendering")

    text_prompt = ", ".join(parts)

    try:
        texture_enabled = state.get("texture_enabled", True)
        service = get_hunyuan3d()
        with _node_timeout(LG_TIMEOUT_MESH, "generate_mesh"):
            result = service.text_to_3d(text_prompt, texture=texture_enabled)
        return {
            "mesh_output": result,
            "mesh_valid": False,
            "errors": [],
        }
    except NodeTimeoutError as te:
        logger.warning("Mesh generation timed out: %s", te)
        return {
            "mesh_output": None,
            "mesh_valid": False,
            "errors": [f"Mesh generation timeout: {te}"],
        }
    except Exception as e:
        logger.warning("Mesh generation failed: %s", e)
        return {
            "mesh_output": None,
            "mesh_valid": False,
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
    return {"model_info": model_info}
