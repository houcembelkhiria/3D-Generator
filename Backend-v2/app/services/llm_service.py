"""
LLM Service for integrating Llama-3-8B-Instruct model.
Handles model loading, inference, and prompt management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

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
        self.model_path = model_path
        self.session: Optional[LLMSession] = None
        
        # Default model path (can be overridden)
        if self.model_path is None:
            self.model_path = os.getenv("LLAMA_MODEL_PATH", "./models/llama-3-8b-instruct.Q4_K_M.gguf")
    
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


# Global LLM service instance
llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Get the global LLM service instance"""
    return llm_service


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    model_path = os.getenv("LLAMA_MODEL_PATH", "./models/llama-3-8b-instruct.Q4_K_M.gguf")
    
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