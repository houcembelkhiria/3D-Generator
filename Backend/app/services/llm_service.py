"""
LLM Service for integrating Llama-3-8B-Instruct model.
Handles model loading, inference, and prompt management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.hunyuan3d_config import Hunyuan3DSettings

try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("llama-cpp-python not installed. LLM functionality will be disabled.")

logger = logging.getLogger(__name__)


class LLMSession:
    """Represents a single LLM inference session"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize LLM session.
        
        Args:
            model_path: Path to the GGUF model file
            **kwargs: Additional parameters for Llama model
        """
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM functionality not available. Please install llama-cpp-python.")
        
        self.model_path = model_path
        self.model = None
        self.default_params = {
            "n_ctx": 4096,  # Context window
            "n_threads": 8,  # CPU threads
            "n_gpu_layers": 0,  # GPU layers (0 for CPU only)
            **kwargs
        }
        
    def load_model(self):
        """Load the LLM model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading LLM model from: {self.model_path}")
        try:
            self.model = Llama(
                model_path=self.model_path,
                **self.default_params
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default generation parameters
        gen_params = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["</s>", "User:", "Assistant:"],
            **kwargs
        }
        
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            response = self.model(
                prompt=prompt,
                **gen_params
            )
            
            generated_text = response["choices"][0]["text"]
            logger.debug(f"Generated {len(generated_text)} characters")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise


class LLMService:
    """Main service for LLM operations"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LLM service.
        
        Args:
            model_path: Path to the GGUF model file (optional, can be set later)
        """
        _settings = Hunyuan3DSettings()
        self.model_path = model_path or _settings.llm_model_path
        self.session: Optional[LLMSession] = None
    
    def initialize_session(self, **kwargs) -> LLMSession:
        """
        Initialize and return an LLM session.
        
        Args:
            **kwargs: Additional parameters for the session
            
        Returns:
            Initialized LLMSession
        """
        if self.session is None:
            self.session = LLMSession(self.model_path, **kwargs)
            self.session.load_model()
        
        return self.session
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated response text
        """
        session = self.initialize_session()
        return session.generate(prompt, **kwargs)
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from generated text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON object or None if no valid JSON found
        """
        # Look for JSON between code blocks or standalone
        import re
        
        # Pattern to match JSON in code blocks
        json_patterns = [
            r'```(?:json)?\s*({.*?})\s*```',  # JSON in code blocks
            r'({[^{]*(?:\{[^{]*\}[^{]*)*})',  # Standalone JSON objects
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean the match
                    json_str = match.strip()
                    if json_str.startswith('```'):
                        json_str = json_str[3:]
                    if json_str.endswith('```'):
                        json_str = json_str[:-3]
                    json_str = json_str.strip()
                    
                    # Parse JSON
                    parsed = json.loads(json_str)
                    return parsed
                    
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def format_system_prompt(self, task_description: str, schema_example: Optional[Dict] = None) -> str:
        """
        Format a system prompt for the LLM.
        
        Args:
            task_description: Description of the task
            schema_example: Example of expected JSON schema
            
        Returns:
            Formatted system prompt
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant specialized in extracting structured information from technical documents for 3D model generation.

{task_description}

Respond in JSON format only. Do not include any explanations or markdown formatting outside the JSON structure.

<|eot_id|><|start_header_id|>user<|end_header_id|>

"""
        
        if schema_example:
            prompt += f"Here's the expected JSON structure:\n{json.dumps(schema_example, indent=2)}\n\n"
        
        prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        return prompt




class OllamaLLMService:
    """LLM service backed by a local Ollama instance (no GGUF required).

    Uses `/api/chat` with role messages (see `generate_chat`) so Ollama applies
    the per-model template server-side - portable across Qwen, Llama, Mistral,
    Phi, Gemma. The legacy `/api/generate` raw-prompt path is preserved for
    callers that still pass a Llama-3-templated string.
    """

    # Default model is read from centralised config (Hunyuan3DSettings.ollama_default_model).
    # With Ollama's grammar-constrained decoding (`format=<schema>`), a 3-4B
    # SLM matches a 7-8B general LLM for spec extraction at ~half the RAM.
    DEFAULT_MODEL = Hunyuan3DSettings().ollama_default_model

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info("OllamaLLMService: model=%s endpoint=%s", model, self.base_url)

    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **_) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    def generate_chat(
        self,
        system: str,
        user: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> str:
        """Call `/api/chat` with role messages. If `schema` is a JSON schema
        dict, Ollama applies grammar-constrained decoding so the output is
        guaranteed to validate against it (Ollama >= 0.5). Falls back to
        `format="json"` on older Ollama versions that reject schema objects.
        """
        import urllib.request, urllib.error
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }
        if schema is not None:
            body["format"] = schema

        def _post(payload_body: Dict[str, Any]) -> str:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload_body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            return (data.get("message") or {}).get("content", "")

        try:
            return _post(body)
        except urllib.error.HTTPError as e:
            if schema is not None and e.code in (400, 422):
                logger.warning(
                    "Ollama rejected schema-format (%s); retrying with format=json", e
                )
                body["format"] = "json"
                return _post(body)
            raise

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        return _extract_json(text)

    def format_system_prompt(self, task_description: str, schema_example: Optional[Dict] = None) -> str:
        return _format_system_prompt(task_description, schema_example)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re
    json_patterns = [
        r'''```(?:json)?\s*({.*?})\s*```''',
        r'''({[^{]*(?:\{[^{]*\}[^{]*)*})''',
    ]
    for pattern in json_patterns:
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    return None


def _format_system_prompt(task_description: str, schema_example: Optional[Dict] = None) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant specialized in extracting structured information from technical documents for 3D model generation.

{task_description}

Respond in JSON format only. Do not include any explanations or markdown formatting outside the JSON structure.

<|eot_id|><|start_header_id|>user<|end_header_id|>

"""
    if schema_example:
        prompt += f"Here's the expected JSON structure:\n{json.dumps(schema_example, indent=2)}\n\n"
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt

# Global LLM service instance
llm_service = LLMService()


# Substrings that mark a model as unsuitable for chat-style JSON extraction.
# Code-completion / embedding / non-instruction-tuned base models silently
# return empty strings (or pure code) when handed a chat-templated prompt,
# which used to bypass our retry loop and trigger the generic-black-blob
# fallback. Filter them out at autopick; users can override via OLLAMA_MODEL.
_OLLAMA_REJECT_KEYWORDS = (
    # Code-completion families (return code/empty for chat prompts)
    "coder", "code-", "codellama", "starcoder", "deepseek-coder", "wizardcoder",
    # Embedding models (no generation head)
    "embed", "embedding",
    # Vision-only and safety classifiers (no general chat)
    "vision-only", "guard",
)


def _is_chat_capable(model_name: str) -> bool:
    n = model_name.lower()
    return not any(kw in n for kw in _OLLAMA_REJECT_KEYWORDS)


def _autopick_ollama_model(models: list[str]) -> Optional[str]:
    """Pick the best Ollama-installed model for chat-style JSON extraction.

    Order of preference:
    1. The configured `OllamaLLMService.DEFAULT_MODEL` if installed.
    2. Instruction-tuned small models (qwen2.5/qwen3 3-4B, llama-3.2 3B, phi-3,
       gemma-2/3) in the `preferred` priority list, filtered through
       `_is_chat_capable` to drop coder/embed variants.
    3. None — caller logs a warning and leaves `llm_service` as the bare GGUF
       service so the pipeline raises an honest "no model" error rather than
       picking the wrong tool.
    """
    if OllamaLLMService.DEFAULT_MODEL in models:
        return OllamaLLMService.DEFAULT_MODEL
    preferred = ["qwen2.5", "qwen3", "qwen", "llama3.2", "llama3", "llama",
                 "mistral", "phi", "gemma"]
    for pref in preferred:
        for m in models:
            if pref in m.lower() and _is_chat_capable(m):
                return m
    return None


def get_llm_service() -> "LLMService | OllamaLLMService":
    """Get the global LLM service instance.

    Priority:
    1. OLLAMA_MODEL env var → use that Ollama model explicitly (no filter).
    2. GGUF file exists     → use local llama-cpp.
    3. Ollama reachable     → auto-pick a chat-capable instruction-tuned model.
    4. Nothing usable       → return bare LLMService (pipeline will surface
                              an explicit error instead of silently using a
                              code-completion model).
    """
    global llm_service

    _settings = Hunyuan3DSettings()
    # Only use Ollama when OLLAMA_MODEL is explicitly set in the environment.
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        if not isinstance(llm_service, OllamaLLMService) or llm_service.model != ollama_model:
            llm_service = OllamaLLMService(model=ollama_model)
        return llm_service

    model_path = _settings.llm_model_path
    if os.path.exists(model_path):
        return llm_service  # existing LLMService with GGUF

    # GGUF missing — probe Ollama
    if not isinstance(llm_service, OllamaLLMService):
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
                tags = json.loads(r.read())
            models = [m["name"] for m in tags.get("models", [])]
            chosen = _autopick_ollama_model(models)
            if chosen:
                logger.info("GGUF not found - using Ollama model: %s", chosen)
                llm_service = OllamaLLMService(model=chosen)
            else:
                logger.warning(
                    "No chat-capable Ollama model installed (found %s). "
                    "Pull one with e.g. `ollama pull %s` or set OLLAMA_MODEL.",
                    models or "no models", OllamaLLMService.DEFAULT_MODEL,
                )
        except Exception as e:
            logger.warning("Ollama not reachable (%s); LLM will use fallback spec", e)

    return llm_service



# Example usage and testing
if __name__ == "__main__":
    # Example configuration — reads from Hunyuan3DSettings (LLAMA_MODEL_PATH env var)
    model_path = Hunyuan3DSettings().llm_model_path
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please download a Llama-3 model in GGUF format and update the path.")
        exit(1)
    
    # Initialize service
    service = LLMService(model_path)
    
    # Test prompt
    test_prompt = service.format_system_prompt(
        "Extract technical specifications from the document.",
        {
            "dimensions": {"length": 100, "width": 50, "height": 30, "unit": "mm"},
            "material": "ABS Plastic",
            "color": "Black"
        }
    )
    
    test_prompt += "Document content: A rectangular box with dimensions 200x100x50mm made of ABS plastic in black color."
    
    print("Testing LLM service...")
    try:
        response = service.generate_response(test_prompt, max_tokens=512)
        print(f"Response: {response}")
        
        # Try to extract JSON
        json_result = service.extract_json_from_text(response)
        if json_result:
            print(f"Extracted JSON: {json.dumps(json_result, indent=2)}")
        else:
            print("No JSON found in response")
            
    except Exception as e:
        print(f"Error: {e}")