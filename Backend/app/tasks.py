import asyncio
import logging
import os
from datetime import datetime
from celery import current_task
from app.worker import celery_app
from app.services.document_parser import document_parser
from app.services.tripo_sr_service import get_tripo_sr_service
from app.services.model_conversion_service import get_conversion_service
from app.services.file_storage_service import get_storage_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_document(self, file_path: str, file_type: str) -> dict:
    """Process uploaded document (PDF/Email) using unstructured library and extract metadata"""
    try:
        # Update task state
        self.update_state(state="PROCESSING", meta={"status": "Initializing document parsing"})
        logger.info(f"Starting document processing for {file_path} ({file_type})")
        
        # Parse document using unstructured library
        self.update_state(state="PROCESSING", meta={"status": "Parsing document with unstructured library"})
        parsed_data = document_parser.parse_document(file_path, file_type)
        
        # Extract and structure metadata for 3D generation
        self.update_state(state="PROCESSING", meta={"status": "Extracting structured metadata"})
        structured_metadata = self._extract_3d_metadata(parsed_data)
        
        # Log processing results
        logger.info(f"Document parsed successfully. Elements: {parsed_data['metadata']['element_count']}")
        logger.info(f"Potential dimensions found: {len(structured_metadata['dimensions'])}")
        logger.info(f"Potential materials found: {len(structured_metadata['materials'])}")
        
        # Prepare final result
        result = {
            "title": structured_metadata.get("title", "Untitled Document"),
            "description": structured_metadata.get("description", "Parsed document content"),
            "dimensions": structured_metadata["dimensions"],
            "materials": structured_metadata["materials"],
            "file_type": file_type,
            "file_name": parsed_data["file_name"],
            "content_length": len(parsed_data["content"]),
            "raw_content": parsed_data["content"][:1000] + "..." if len(parsed_data["content"]) > 1000 else parsed_data["content"]
        }
        
        self.update_state(state="COMPLETED", meta={"status": "Document processed successfully", "metadata": result})
        logger.info(f"Document processing completed for {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error(error_msg)
        self.update_state(state="FAILED", meta={"status": "Processing failed", "error": str(e)})
        raise

def _extract_3d_metadata(self, parsed_data: dict) -> dict:
    """Extract and structure metadata specifically for 3D generation"""
    content = parsed_data["content"]
    metadata = parsed_data["metadata"]
    
    # Extract title from filename or first content lines
    title = metadata["file_name"]
    if "." in title:
        title = ".".join(title.split(".")[:-1])  # Remove extension
    
    # Extract description from first few lines of content
    content_lines = content.split('\n')
    description_lines = [line.strip() for line in content_lines if line.strip()][:3]
    description = " ".join(description_lines)
    if len(description) > 200:
        description = description[:200] + "..."
    
    # Use extracted dimensions and materials from parser
    dimensions = metadata.get("potential_dimensions", [])
    materials = metadata.get("potential_materials", [])
    
    # Additional processing for better 3D metadata
    enhanced_dimensions = self._enhance_dimensions(dimensions, content)
    enhanced_materials = self._enhance_materials(materials, content)
    
    return {
        "title": title,
        "description": description,
        "dimensions": enhanced_dimensions,
        "materials": enhanced_materials
    }

def _enhance_dimensions(self, dimensions: list, content: str) -> list:
    """Enhance and validate dimension data"""
    enhanced = []
    
    # Add common dimension defaults if none found
    if not dimensions:
        # Look for common dimension indicators in content
        import re
        size_patterns = [
            r'size\s*:?\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]?\s*(\d*(?:\.\d+)?)',
            r'model\s*size\s*:?\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, content.lower())
            for match in matches:
                dims = [dim for dim in match if dim]  # Remove empty strings
                if dims:
                    enhanced.append(" × ".join(dims) + " mm")
    
    # Ensure we have reasonable defaults
    if not enhanced and not dimensions:
        enhanced = ["100 × 100 × 100 mm"]  # Default cube dimensions
    
    return dimensions + enhanced

def _enhance_materials(self, materials: list, content: str) -> list:
    """Enhance and validate material data"""
    enhanced = materials.copy()
    
    # Add common materials based on content context
    content_lower = content.lower()
    
    # Technical context materials
    if any(word in content_lower for word in ['machine', 'part', 'component', 'assembly']):
        if 'steel' not in [m.lower() for m in enhanced]:
            enhanced.append('Steel')
    
    # Consumer product context
    if any(word in content_lower for word in ['product', 'device', 'gadget']):
        if 'plastic' not in [m.lower() for m in enhanced]:
            enhanced.append('Plastic')
    
    # Ensure we have at least one material
    if not enhanced:
        enhanced = ['Generic Material']
    
    return enhanced

# Bind helper methods to the task
process_document._extract_3d_metadata = _extract_3d_metadata.__get__(process_document, type(process_document))
process_document._enhance_dimensions = _enhance_dimensions.__get__(process_document, type(process_document))
process_document._enhance_materials = _enhance_materials.__get__(process_document, type(process_document))

@celery_app.task(bind=True)
def generate_3d_model(self, metadata: dict) -> dict:
    """Generate 3D model from extracted metadata using TripoSR"""
    try:
        self.update_state(state="PROCESSING", meta={"status": "Initializing 3D generation services"})
        
        # Get service instances
        tripo_service = get_tripo_sr_service()
        conversion_service = get_conversion_service()
        storage_service = get_storage_service()
        
        # Extract generation parameters from metadata
        self.update_state(state="PROCESSING", meta={"status": "Preparing generation parameters"})
        generation_params = self._prepare_generation_params(metadata)
        
        # Generate text prompt for TripoSR
        self.update_state(state="PROCESSING", meta={"status": "Creating text prompt for 3D generation"})
        text_prompt = self._create_text_prompt(metadata)
        
        # Generate mesh using TripoSR
        self.update_state(state="PROCESSING", meta={"status": "Generating 3D mesh with TripoSR"})
        mesh_data = tripo_service.generate_mesh_from_text(
            text_prompt,
            resolution=generation_params.get('resolution', 256),
            guidance_scale=generation_params.get('guidance_scale', 7.5)
        )
        
        # Save intermediate OBJ file
        self.update_state(state="PROCESSING", meta={"status": "Saving intermediate mesh file"})
        obj_content = self._mesh_to_obj_format(mesh_data)
        temp_obj = storage_service.create_temp_file(obj_content.encode(), '.obj')
        
        # Convert to GLB format
        self.update_state(state="PROCESSING", meta={"status": "Converting to GLB format"})
        glb_filename = f"{metadata.get('title', 'untitled_model')}.glb"
        final_path = storage_service.base_generated_dir / glb_filename
        
        conversion_result = conversion_service.convert_model_format(
            temp_obj['path'],
            str(final_path),
            target_format='.glb',
            optimize_quality='medium'
        )
        
        # Clean up temporary file
        try:
            os.unlink(temp_obj['path'])
        except Exception:
            pass  # Ignore cleanup errors
        
        # Prepare final result
        model_info = {
            "model_id": f"model_{int(datetime.now().timestamp())}",
            "title": metadata.get('title', 'Generated Model'),
            "description": metadata.get('description', ''),
            "file_path": str(final_path),
            "filename": glb_filename,
            "format": "GLB",
            "size": conversion_result['output_stats']['size_bytes'],
            "mesh_stats": conversion_result['mesh_stats'],
            "quality_metrics": conversion_result['quality_metrics'],
            "generation_time": datetime.now().isoformat(),
            "source_metadata": metadata
        }
        
        self.update_state(state="COMPLETED", meta={"status": "3D model generated successfully", "model_info": model_info})
        logger.info(f"3D model generation completed: {glb_filename}")
        return model_info
        
    except Exception as e:
        error_msg = f"3D model generation failed: {str(e)}"
        logger.error(error_msg)
        self.update_state(state="FAILED", meta={"status": "Generation failed", "error": str(e)})
        raise

def _prepare_generation_params(self, metadata: dict) -> dict:
    """Prepare generation parameters from metadata"""
    params = {
        'resolution': 256,
        'guidance_scale': 7.5,
        'num_inference_steps': 50
    }
    
    # Adjust resolution based on complexity
    if metadata.get('dimensions') and len(metadata['dimensions']) > 3:
        params['resolution'] = 384  # Higher resolution for complex objects
    
    return params

def _create_text_prompt(self, metadata: dict) -> str:
    """Create text prompt for TripoSR from metadata"""
    title = metadata.get('title', '3D object')
    description = metadata.get('description', '')
    dimensions = metadata.get('dimensions', [])
    materials = metadata.get('materials', [])
    
    # Build descriptive prompt
    prompt_parts = [f"A detailed 3D model of {title.lower()}"]
    
    if description:
        prompt_parts.append(f"described as {description.lower()}")
    
    if dimensions:
        dim_desc = ", ".join(dimensions[:3])  # Take first 3 dimensions
        prompt_parts.append(f"with dimensions {dim_desc}")
    
    if materials:
        mat_desc = ", ".join(materials[:2])  # Take first 2 materials
        prompt_parts.append(f"made of {mat_desc.lower()}")
    
    # Add quality descriptors
    prompt_parts.extend([
        "high quality",
        "detailed textures",
        "professional 3D rendering"
    ])
    
    return ", ".join(prompt_parts)

def _mesh_to_obj_format(self, mesh_data: dict) -> str:
    """Convert mesh data to OBJ format string"""
    lines = ["# 3D Model Generated by TripoSR", ""]
    
    # Write vertices
    for vertex in mesh_data['vertices']:
        lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
    
    lines.append("")
    
    # Write faces
    for face in mesh_data['faces']:
        # OBJ faces are 1-indexed
        lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
    
    return "\n".join(lines)