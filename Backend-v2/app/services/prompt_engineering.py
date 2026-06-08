"""
Prompt engineering service for structured JSON output.
Manages system prompts and ensures consistent JSON schema enforcement.
"""

import json
from typing import Dict, Any, Optional, List
from app.models.spec_models import ObjectSpec, DocumentAnalysis


class PromptEngineer:
    """Service for crafting effective prompts that enforce JSON schema compliance"""
    
    def __init__(self):
        self.system_instructions = self._get_base_instructions()
    
    def _get_base_instructions(self) -> str:
        """Get the base system instructions for JSON schema enforcement"""
        return """You are an expert technical document analyzer specializing in extracting precise 3D model specifications.

IMPORTANT RULES:
1. ALWAYS respond with valid JSON only
2. NEVER include markdown formatting, explanations, or text outside JSON
3. Use the exact field names and structure provided
4. If information is missing, use null or omit optional fields
5. Be precise with measurements and technical specifications
6. Focus on extractable, quantifiable information
7. For dimensions, always include the unit of measurement
8. Estimate reasonably when exact values aren't provided
9. Include confidence scores when relevant
10. Flag any ambiguities or uncertainties in the warnings field"""
    
    def create_extraction_prompt(self, document_text: str, target_schema: str = "object_spec") -> str:
        """
        Create a prompt for extracting structured information from document text.
        
        Args:
            document_text: The cleaned document text to analyze
            target_schema: Target schema type ("object_spec" or "document_analysis")
            
        Returns:
            Formatted prompt string ready for LLM
        """
        if target_schema == "object_spec":
            schema_example = ObjectSpec.model_config["json_schema_extra"]["example"]
            task_description = "Extract precise 3D object specifications from the technical document. Focus on dimensions, materials, and measurable properties."
        else:  # document_analysis
            schema_example = DocumentAnalysis.model_config["json_schema_extra"]["example"]
            task_description = "Perform comprehensive analysis of the technical document. Extract all 3D object specifications and any tabular data present."
        
        # Build the prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_instructions}

TASK: {task_description}

RESPONSE FORMAT:
{json.dumps(schema_example, indent=2)}

<|eot_id|><|start_header_id|>user<|end_header_id|>

DOCUMENT TEXT:
{document_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def create_table_extraction_prompt(self, table_text: str) -> str:
        """
        Create a prompt specifically for extracting tabular data.
        
        Args:
            table_text: Text containing table data
            
        Returns:
            Formatted prompt for table extraction
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_instructions}

TASK: Extract structured tabular data from the provided text. Identify headers and organize data into rows.

RESPONSE FORMAT:
{{
  "headers": ["Column1", "Column2", "Column3"],
  "rows": [
    ["Row1-Col1", "Row1-Col2", "Row1-Col3"],
    ["Row2-Col1", "Row2-Col2", "Row2-Col3"]
  ],
  "caption": "Optional table title or description"
}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

TABLE TEXT:
{table_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def create_refinement_prompt(self, initial_response: str, feedback: str) -> str:
        """
        Create a prompt for refining/revising a previous response.
        
        Args:
            initial_response: Previous LLM response
            feedback: Feedback or correction instructions
            
        Returns:
            Refinement prompt
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_instructions}

TASK: Revise and improve the previous response based on the feedback provided. Maintain the same JSON structure but correct any errors or incorporate the feedback.

<|eot_id|><|start_header_id|>user<|end_header_id|>

PREVIOUS RESPONSE:
{initial_response}

FEEDBACK TO ADDRESS:
{feedback}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def create_validation_prompt(self, json_data: str, validation_rules: List[str]) -> str:
        """
        Create a prompt for validating JSON data against specific rules.
        
        Args:
            json_data: JSON data to validate
            validation_rules: List of validation rules to check
            
        Returns:
            Validation prompt
        """
        rules_text = "\n".join([f"- {rule}" for rule in validation_rules])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a JSON validation expert. Check the provided JSON data against the validation rules and return a corrected version if needed.

VALIDATION RULES:
{rules_text}

RESPONSE FORMAT:
{{
  "valid": true/false,
  "errors": ["list of validation errors"],
  "corrected_data": {{}},  // Include only if corrections were made
  "confidence_score": 0.0  // Rate confidence in the data quality (0-1)
}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

JSON DATA TO VALIDATE:
{json_data}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt


class JSONSchemaEnforcer:
    """Helper class for enforcing JSON schema compliance"""
    
    @staticmethod
    def get_required_fields(schema_class) -> List[str]:
        """
        Get required fields from a Pydantic model.
        
        Args:
            schema_class: Pydantic model class
            
        Returns:
            List of required field names
        """
        required = []
        for field_name, field_info in schema_class.model_fields.items():
            # Check if field is required (no default value and not Optional)
            if field_info.is_required():
                required.append(field_name)
        return required
    
    @staticmethod
    def validate_structure(data: Dict[str, Any], schema_class) -> Dict[str, Any]:
        """
        Validate data structure against schema and provide corrections.
        
        Args:
            data: Data to validate
            schema_class: Target Pydantic model class
            
        Returns:
            Validation results with corrections
        """
        required_fields = JSONSchemaEnforcer.get_required_fields(schema_class)
        missing_fields = []
        corrected_data = data.copy()
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
                # Add default values where possible
                if field == "unit":
                    corrected_data[field] = "mm"
                elif field == "color":
                    corrected_data[field] = "Matte Black"
                elif field == "quantity":
                    corrected_data[field] = 1
        
        # Validate nested structures
        if "dimensions" in data:
            dim_required = JSONSchemaEnforcer.get_required_fields(schema_class.model_fields["dimensions"].annotation)
            for dim_field in dim_required:
                if dim_field not in data.get("dimensions", {}):
                    corrected_data.setdefault("dimensions", {})[dim_field] = None
        
        return {
            "valid": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "corrected_data": corrected_data,
            "confidence_adjustment": -0.1 * len(missing_fields)  # Reduce confidence for missing data
        }


# Global instances
prompt_engineer = PromptEngineer()
schema_enforcer = JSONSchemaEnforcer()


def get_prompt_engineer() -> PromptEngineer:
    """Get the global prompt engineer instance"""
    return prompt_engineer


def get_schema_enforcer() -> JSONSchemaEnforcer:
    """Get the global schema enforcer instance"""
    return schema_enforcer


# Example usage
if __name__ == "__main__":
    # Test prompt creation
    test_text = """Technical specifications for a mounting bracket:
- Dimensions: 150mm x 75mm x 25mm
- Material: Aluminum 6061-T6
- Finish: Anodized black
- Weight: approximately 200g"""
    
    prompt = prompt_engineer.create_extraction_prompt(test_text, "object_spec")
    print("Generated Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Test schema enforcement
    test_data = {
        "name": "Mounting Bracket",
        "dimensions": {
            "length": 150,
            "width": 75,
            "height": 25
            # Missing "unit" field
        },
        "material": {
            "type": "Metal"
            # Missing "color" field
        }
    }
    
    validation = schema_enforcer.validate_structure(test_data, ObjectSpec)
    print("Validation Results:")
    print(json.dumps(validation, indent=2))