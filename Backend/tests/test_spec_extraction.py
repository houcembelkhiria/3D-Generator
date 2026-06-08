"""Regression tests for the LLM spec-extraction path.

Covers the four failure modes that produced the "wooden crate -> black blob"
incident:

1. Code-completion / embedding models slipping past Ollama autopick.
2. Empty LLM response silently passing through to the generic fallback.
3. Fallback spec discarding raw_text and shipping a useless subject to t2i.
4. Llama-3-only prompt template not portable to other Ollama model families.

These tests are pure-Python and do NOT touch Ollama, the network, llama-cpp,
SDXL, or Hunyuan3D — they monkey-patch the LLM service so the suite runs in
a couple of seconds with no GPU / model weights required.
"""
from __future__ import annotations

import pytest

from app.services.llm_service import (
    OllamaLLMService,
    _autopick_ollama_model,
    _is_chat_capable,
)
from app.pipeline.nodes import (
    build_fallback_spec_node,
    extract_spec_llm_node,
    _pick_fallback_name,
)


# ---------------------------------------------------------------------------
# Phase 1: Ollama autopick filter
# ---------------------------------------------------------------------------

class TestAutopick:
    def test_rejects_coder_models(self):
        assert _is_chat_capable("qwen-coder:latest") is False
        assert _is_chat_capable("codellama:7b") is False
        assert _is_chat_capable("starcoder2:3b") is False

    def test_rejects_embedding_models(self):
        assert _is_chat_capable("nomic-embed-text:latest") is False
        assert _is_chat_capable("mxbai-embed-large") is False

    def test_accepts_instruction_tuned_models(self):
        assert _is_chat_capable("qwen2.5:3b-instruct") is True
        assert _is_chat_capable("llama3.2:3b") is True
        assert _is_chat_capable("phi3:mini") is True
        assert _is_chat_capable("gemma2:2b") is True

    def test_autopick_skips_lone_coder_model(self):
        """The bug condition: only qwen-coder installed -> return None so the
        pipeline logs a real error instead of silently picking it."""
        assert _autopick_ollama_model(["qwen-coder:latest"]) is None

    def test_autopick_prefers_default_when_installed(self):
        models = ["qwen-coder:latest", OllamaLLMService.DEFAULT_MODEL, "llama3.2:3b"]
        assert _autopick_ollama_model(models) == OllamaLLMService.DEFAULT_MODEL

    def test_autopick_falls_back_to_priority_list(self):
        models = ["qwen-coder:latest", "llama3.2:3b"]
        assert _autopick_ollama_model(models) == "llama3.2:3b"

    def test_autopick_skips_coder_even_within_preferred_family(self):
        models = ["qwen-coder:latest", "qwen3:4b-instruct"]
        assert _autopick_ollama_model(models) == "qwen3:4b-instruct"


# ---------------------------------------------------------------------------
# Phase 3: empty LLM response is treated as a real failure
# ---------------------------------------------------------------------------

class _StubGGUFService:
    """Stand-in for `LLMService` (GGUF path) — exercises the legacy code path."""
    def __init__(self, response: str):
        self._response = response

    def generate_response(self, prompt, max_tokens=1024, temperature=0.7, **_):
        return self._response

    def extract_json_from_text(self, text):
        import json
        try:
            return json.loads(text)
        except Exception:
            return None


class TestExtractEmptyResponse:
    def test_empty_response_records_error(self, monkeypatch):
        stub = _StubGGUFService("")
        monkeypatch.setattr(
            "app.services.llm_service.get_llm_service", lambda: stub
        )
        state = {"raw_text": "OBJECT: Wooden Crate\nDimensions: 600x400x350mm"}
        result = extract_spec_llm_node(state)

        assert result["spec"] is None
        assert result["spec_valid"] is False
        assert result["errors"], "expected an error to be recorded"
        assert "empty" in result["errors"][0].lower()

    def test_whitespace_only_response_records_error(self, monkeypatch):
        stub = _StubGGUFService("   \n\n  \t")
        monkeypatch.setattr(
            "app.services.llm_service.get_llm_service", lambda: stub
        )
        state = {"raw_text": "OBJECT: Crate"}
        result = extract_spec_llm_node(state)
        assert result["spec"] is None
        assert "empty" in result["errors"][0].lower()


# ---------------------------------------------------------------------------
# Phase 4: smart fallback name + raw_text-preserving description
# ---------------------------------------------------------------------------

WOODEN_CRATE_EML = """\
From: artist@studio3d.com
To: pipeline@3dgenerator.local
Subject: Asset Request - Wooden Storage Crate
Date: Mon, 01 Jun 2024 10:30:00 +0000

Hi team,

Please generate a 3D model for the following asset:

OBJECT: Wooden Storage Crate
DESCRIPTION: A classic wooden shipping crate with visible planks, metal corner brackets, and rope handles.

DIMENSIONS:
- Length: 600 mm
- Width: 400 mm
- Height: 350 mm

MATERIAL:
- Primary: Wood (Oak, weathered finish)
"""

SCIFI_PDF_TEXT = """\
Page 1 of 4
ACME Engineering Confidential

Product Brief: SciFi Helmet
Overall height: 280 mm
Width: 220 mm
Material: Carbon Fiber
"""


class TestFallbackNamePicker:
    """The exact failure that produced the black blob: fallback used
    'Hi team,' as the object name. These tests pin the new behaviour."""

    def test_email_prefers_OBJECT_key_over_subject(self):
        assert _pick_fallback_name(WOODEN_CRATE_EML) == "Wooden Storage Crate"

    def test_email_skips_greeting_and_headers(self):
        name = _pick_fallback_name(WOODEN_CRATE_EML)
        assert name != "Hi team,"
        assert not name.lower().startswith("from:")
        assert not name.lower().startswith("subject:")

    def test_pdf_skips_page_footer_and_confidentiality(self):
        # PDF section heading "Product Brief: SciFi Helmet" is acceptable —
        # what matters is that the page footer and confidentiality banner are
        # skipped. The LLM happy-path returns just "SciFi Helmet"; the
        # fallback's job is only to avoid the "Hi team," / page-footer disaster.
        name = _pick_fallback_name(SCIFI_PDF_TEXT)
        assert "helmet" in name.lower()
        assert "page" not in name.lower()
        assert "confidential" not in name.lower()

    def test_pdf_skips_inline_confidential_banner(self):
        text = "Page 1\nACME Confidential\n\nVending Machine Cabinet\nSize: 1.8m"
        assert _pick_fallback_name(text) == "Vending Machine Cabinet"

    def test_unknown_format_falls_back_to_3d_object(self):
        assert _pick_fallback_name("") == "3D Object"
        assert _pick_fallback_name("hi\nhello\nthanks") == "3D Object"

    def test_subject_used_when_no_explicit_object_key(self):
        text = "Subject: Build me a robot\n\nHi, please build a robot."
        assert _pick_fallback_name(text) == "Build me a robot"


class TestFallbackSpecBuilder:
    """The load-bearing safety net: even when LLM is broken, the t2i prompt
    must still see real subject content (this is what failed in the original
    incident — description was empty, so SDXL only saw 'plastic matte black')."""

    def test_description_carries_subject_keywords(self):
        state = {"raw_text": WOODEN_CRATE_EML}
        spec = build_fallback_spec_node(state)["spec"]
        desc = spec["description"].lower()
        assert "wooden" in desc
        assert "crate" in desc
        assert "oak" in desc

    def test_pdf_description_carries_subject_keywords(self):
        state = {"raw_text": SCIFI_PDF_TEXT}
        spec = build_fallback_spec_node(state)["spec"]
        desc = spec["description"].lower()
        assert "scifi helmet" in desc or "helmet" in desc
        assert "carbon" in desc

    def test_description_is_truncated_for_long_docs(self):
        long_doc = "OBJECT: Massive Asset\n" + ("filler line\n" * 1000)
        spec = build_fallback_spec_node({"raw_text": long_doc})["spec"]
        # Cap is 1500 chars — small enough to fit any t2i prompt budget.
        assert len(spec["description"]) <= 1500

    def test_fallback_preserves_pipeline_compatibility(self):
        """All keys that downstream nodes (generate_mesh_node) read must
        still be present — adding `description` must NOT break shape/dims/material."""
        spec = build_fallback_spec_node({"raw_text": WOODEN_CRATE_EML})["spec"]
        assert spec["shape"] == "CUSTOM"
        assert spec["dimensions"]["unit"] == "mm"
        assert spec["material"]["type"] == "Plastic"
        assert spec["name"] == "Wooden Storage Crate"

    def test_handles_missing_raw_text(self):
        # Real pipeline state can have raw_text=None on parse failure.
        spec = build_fallback_spec_node({"raw_text": None})["spec"]
        assert spec["name"] == "3D Object"
        assert spec["description"] == ""


# ---------------------------------------------------------------------------
# Phase 2: PromptEngineer message split + PDF truncation
# ---------------------------------------------------------------------------

class TestPromptEngineering:
    def test_create_extraction_messages_returns_tuple(self):
        from app.services.prompt_engineering import get_prompt_engineer
        pe = get_prompt_engineer()
        system, user = pe.create_extraction_messages("OBJECT: Crate")
        assert isinstance(system, str) and isinstance(user, str)
        assert "TASK:" in system
        assert "DOCUMENT TEXT:" in user
        # Crucially: NO Llama-3 chat tokens — those break non-Llama models.
        assert "<|begin_of_text|>" not in system
        assert "<|begin_of_text|>" not in user

    def test_long_pdf_text_is_truncated(self):
        from app.services.prompt_engineering import get_prompt_engineer, PromptEngineer
        pe = get_prompt_engineer()
        long_doc = "X" * (PromptEngineer.MAX_DOCUMENT_CHARS + 5000)
        system, user = pe.create_extraction_messages(long_doc)
        # truncated to MAX_DOCUMENT_CHARS plus a short ellipsis marker
        assert len(user) < PromptEngineer.MAX_DOCUMENT_CHARS + 200
        assert "truncated" in user.lower()

    def test_legacy_prompt_path_still_works(self):
        """The GGUF path still uses the Llama-3 template — must not regress."""
        from app.services.prompt_engineering import get_prompt_engineer
        pe = get_prompt_engineer()
        prompt = pe.create_extraction_prompt("OBJECT: Crate")
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
