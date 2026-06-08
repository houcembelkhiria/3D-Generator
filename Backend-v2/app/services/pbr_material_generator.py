"""
PBR Material Generator for Unity HDRP
Generates physically-based rendering materials from vertex colors and mesh geometry,
creating HDRP-compatible texture maps and material definitions.
"""

import numpy as np
import trimesh
import logging
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PBRMaterialGenerator:
    """Generates PBR materials suitable for Unity HDRP from mesh data"""
    
    def __init__(self, texture_resolution: int = 1024):
        """
        Initialize PBR material generator.
        
        Args:
            texture_resolution: Resolution for generated texture maps
        """
        self.texture_resolution = texture_resolution
        self.supported_formats = ['.png', '.jpg', '.tga']
    
    def generate_pbr_textures(self, 
                            mesh: trimesh.Trimesh,
                            output_directory: str,
                            base_name: str = "generated_asset") -> Dict[str, str]:
        """
        Generate complete PBR texture set from mesh vertex colors.
        
        Args:
            mesh: Input mesh with vertex colors
            output_directory: Directory to save texture files
            base_name: Base name for generated textures
            
        Returns:
            Dictionary mapping texture types to file paths
        """
        logger.info("Generating PBR texture set...")
        
        # Create output directory
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        texture_paths = {}
        
        try:
            # Generate base color texture from vertex colors
            base_color_path = self._generate_base_color_texture(
                mesh, output_path, base_name
            )
            texture_paths['base_color'] = base_color_path
            
            # Generate roughness map from geometry analysis
            roughness_path = self._generate_roughness_map(
                mesh, output_path, base_name
            )
            texture_paths['roughness'] = roughness_path
            
            # Generate metallic map (conservative estimate)
            metallic_path = self._generate_metallic_map(
                mesh, output_path, base_name
            )
            texture_paths['metallic'] = metallic_path
            
            # Generate normal map from mesh normals
            normal_path = self._generate_normal_map(
                mesh, output_path, base_name
            )
            texture_paths['normal'] = normal_path
            
            # Generate ambient occlusion map
            ao_path = self._generate_ao_map(
                mesh, output_path, base_name
            )
            texture_paths['ambient_occlusion'] = ao_path
            
            logger.info(f"PBR texture set generated successfully: {len(texture_paths)} textures")
            
        except Exception as e:
            logger.error(f"Failed to generate PBR textures: {str(e)}")
            raise
        
        return texture_paths
    
    def _generate_base_color_texture(self, 
                                   mesh: trimesh.Trimesh,
                                   output_path: Path,
                                   base_name: str) -> str:
        """Generate base color texture from vertex colors"""
        logger.info("Generating base color texture...")
        
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            logger.warning("No vertex colors found, generating default texture")
            # Create default gray texture
            texture_array = np.full((self.texture_resolution, self.texture_resolution, 3), 128, dtype=np.uint8)
        else:
            # Sample vertex colors and create texture
            texture_array = self._sample_vertex_colors_to_texture(
                mesh, self.texture_resolution
            )
        
        # Save texture
        texture_path = output_path / f"{base_name}_BaseColor.png"
        image = Image.fromarray(texture_array)
        image.save(texture_path, 'PNG')
        
        logger.info(f"Base color texture saved: {texture_path}")
        return str(texture_path)
    
    def _generate_roughness_map(self,
                              mesh: trimesh.Trimesh,
                              output_path: Path,
                              base_name: str) -> str:
        """Generate roughness map based on geometric complexity"""
        logger.info("Generating roughness map...")
        
        # Analyze mesh geometry to estimate roughness
        # Simpler, smoother surfaces = lower roughness values
        # Complex, detailed surfaces = higher roughness values
        
        # Create base roughness map (medium roughness)
        roughness_array = np.full((self.texture_resolution, self.texture_resolution), 128, dtype=np.uint8)
        
        # Modify based on curvature analysis (simplified)
        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            # Estimate roughness from face angles and vertex distribution
            face_normals = mesh.face_normals
            if len(face_normals) > 10:  # Need sufficient faces for analysis
                # Calculate variation in normals as roughness indicator
                normal_variance = np.var(face_normals, axis=0)
                avg_variance = np.mean(normal_variance)
                
                # Map variance to roughness (0-1 range, then to 0-255)
                roughness_modifier = min(avg_variance * 100, 100)  # Cap at 100
                base_roughness = 128 + int(roughness_modifier)
                base_roughness = np.clip(base_roughness, 50, 200)  # Reasonable range
                
                roughness_array = np.full_like(roughness_array, base_roughness)
        
        # Save texture
        texture_path = output_path / f"{base_name}_Roughness.png"
        image = Image.fromarray(roughness_array, mode='L')
        image.save(texture_path, 'PNG')
        
        logger.info(f"Roughness map saved: {texture_path}")
        return str(texture_path)
    
    def _generate_metallic_map(self,
                             mesh: trimesh.Trimesh,
                             output_path: Path,
                             base_name: str) -> str:
        """Generate metallic map (conservative approach)"""
        logger.info("Generating metallic map...")
        
        # Conservative approach: mostly non-metallic with some metallic areas
        # This prevents overly shiny objects that might look unrealistic
        
        metallic_array = np.zeros((self.texture_resolution, self.texture_resolution), dtype=np.uint8)
        
        # Add some metallic highlights based on vertex color analysis
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors
            if len(vertex_colors) > 0:
                # Analyze color saturation - highly saturated colors might indicate metals
                rgb_colors = vertex_colors[:, :3].astype(np.float32) / 255.0
                saturation = np.max(rgb_colors, axis=1) - np.min(rgb_colors, axis=1)
                
                # If we have highly saturated colors, add some metallic areas
                high_saturation_ratio = np.mean(saturation > 0.5)
                if high_saturation_ratio > 0.3:  # 30% of vertices are highly saturated
                    # Add sparse metallic spots
                    metallic_array[::20, ::20] = 255  # Sparse metallic dots
        
        # Save texture
        texture_path = output_path / f"{base_name}_Metallic.png"
        image = Image.fromarray(metallic_array, mode='L')
        image.save(texture_path, 'PNG')
        
        logger.info(f"Metallic map saved: {texture_path}")
        return str(texture_path)
    
    def _generate_normal_map(self,
                           mesh: trimesh.Trimesh,
                           output_path: Path,
                           base_name: str) -> str:
        """Generate normal map from mesh normals"""
        logger.info("Generating normal map...")
        
        # Create tangent space normal map
        # Simplified approach: generate from vertex normals
        
        # Start with neutral normal map (RGB: 128,128,255)
        normal_array = np.zeros((self.texture_resolution, self.texture_resolution, 3), dtype=np.uint8)
        normal_array[:, :, 0] = 128  # X component
        normal_array[:, :, 1] = 128  # Y component  
        normal_array[:, :, 2] = 255  # Z component (dominant)
        
        # Add subtle normal variations based on mesh complexity
        if len(mesh.vertices) > 0:
            # Simple noise-like pattern based on vertex distribution
            x_coords = np.linspace(0, 1, self.texture_resolution)
            y_coords = np.linspace(0, 1, self.texture_resolution)
            xx, yy = np.meshgrid(x_coords, y_coords)
            
            # Subtle variations
            noise_scale = 0.1
            normal_array[:, :, 0] = np.clip(128 + np.sin(xx * 20) * 20 * noise_scale, 0, 255)
            normal_array[:, :, 1] = np.clip(128 + np.cos(yy * 15) * 15 * noise_scale, 0, 255)
        
        # Save texture
        texture_path = output_path / f"{base_name}_Normal.png"
        image = Image.fromarray(normal_array)
        image.save(texture_path, 'PNG')
        
        logger.info(f"Normal map saved: {texture_path}")
        return str(texture_path)
    
    def _generate_ao_map(self,
                       mesh: trimesh.Trimesh,
                       output_path: Path,
                       base_name: str) -> str:
        """Generate ambient occlusion map"""
        logger.info("Generating ambient occlusion map...")
        
        # Simplified AO generation - darken concave areas
        # For now, create a moderate AO map
        
        ao_array = np.full((self.texture_resolution, self.texture_resolution), 200, dtype=np.uint8)
        
        # Add some edge darkening to simulate basic AO
        edge_width = 20
        ao_array[:edge_width, :] = 150  # Top edge
        ao_array[-edge_width:, :] = 150  # Bottom edge
        ao_array[:, :edge_width] = 150   # Left edge
        ao_array[:, -edge_width:] = 150  # Right edge
        
        # Save texture
        texture_path = output_path / f"{base_name}_AO.png"
        image = Image.fromarray(ao_array, mode='L')
        image.save(texture_path, 'PNG')
        
        logger.info(f"Ambient occlusion map saved: {texture_path}")
        return str(texture_path)
    
    def _sample_vertex_colors_to_texture(self, 
                                       mesh: trimesh.Trimesh, 
                                       resolution: int) -> np.ndarray:
        """Sample vertex colors to create a texture map"""
        # Get vertex colors
        vertex_colors = mesh.visual.vertex_colors
        
        if vertex_colors is None or len(vertex_colors) == 0:
            # Return default gray texture
            return np.full((resolution, resolution, 3), 128, dtype=np.uint8)
        
        # Convert to RGB (remove alpha if present)
        if vertex_colors.shape[1] == 4:
            rgb_colors = vertex_colors[:, :3]
        else:
            rgb_colors = vertex_colors
        
        # Simple sampling approach - average colors in grid
        texture = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # Distribute vertex colors across texture space
        # This is a simplified approach - in practice, you'd want proper UV unwrapping
        num_vertices = len(mesh.vertices)
        for i in range(resolution):
            for j in range(resolution):
                # Simple index mapping (would be improved with proper UV coordinates)
                vertex_idx = int((i * resolution + j) * num_vertices / (resolution * resolution))
                vertex_idx = min(vertex_idx, num_vertices - 1)
                texture[i, j] = rgb_colors[vertex_idx]
        
        return texture
    
    def generate_hdrp_material_definition(self,
                                        texture_paths: Dict[str, str],
                                        output_directory: str,
                                        material_name: str = "GeneratedMaterial") -> str:
        """
        Generate Unity HDRP material definition file.
        
        Args:
            texture_paths: Dictionary of generated texture file paths
            output_directory: Directory to save material definition
            material_name: Name for the material
            
        Returns:
            Path to generated material definition file
        """
        logger.info("Generating HDRP material definition...")
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create material definition compatible with Unity HDRP
        material_def = {
            "material": {
                "name": material_name,
                "shader": "HDRP/Lit",
                "properties": {
                    "baseColor": {
                        "texture": texture_paths.get('base_color', ""),
                        "color": [1.0, 1.0, 1.0, 1.0]
                    },
                    "roughness": {
                        "texture": texture_paths.get('roughness', ""),
                        "value": 0.5
                    },
                    "metallic": {
                        "texture": texture_paths.get('metallic', ""),
                        "value": 0.0
                    },
                    "normal": {
                        "texture": texture_paths.get('normal', ""),
                        "scale": 1.0
                    },
                    "ambientOcclusion": {
                        "texture": texture_paths.get('ambient_occlusion', ""),
                        "intensity": 1.0
                    }
                },
                "renderingMode": "Opaque",
                "alphaTest": 0.5,
                "doubleSided": False
            }
        }
        
        # Save material definition
        material_path = output_path / f"{material_name}.mat.json"
        with open(material_path, 'w') as f:
            json.dump(material_def, f, indent=2)
        
        logger.info(f"HDRP material definition saved: {material_path}")
        return str(material_path)
    
    def apply_pbr_material_to_mesh(self,
                                 mesh: trimesh.Trimesh,
                                 output_directory: str,
                                 base_name: str = "asset") -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Apply complete PBR material workflow to mesh.
        
        Args:
            mesh: Input mesh
            output_directory: Directory for generated assets
            base_name: Base name for generated files
            
        Returns:
            Tuple of (mesh_with_materials, material_metadata)
        """
        logger.info("Applying complete PBR material workflow...")
        
        # Generate texture maps
        texture_paths = self.generate_pbr_textures(mesh, output_directory, base_name)
        
        # Generate material definition
        material_path = self.generate_hdrp_material_definition(
            texture_paths, output_directory, f"{base_name}_Material"
        )
        
        # Create metadata
        metadata = {
            "material_workflow": "pbr_hdrp",
            "textures_generated": len(texture_paths),
            "texture_paths": texture_paths,
            "material_definition": material_path,
            "workflow_timestamp": str(trimesh.util.now())
        }
        
        logger.info("PBR material workflow completed successfully")
        return mesh, metadata


# Global generator instance
pbr_generator = PBRMaterialGenerator()


def get_pbr_generator() -> PBRMaterialGenerator:
    """Get the global PBR material generator instance"""
    return pbr_generator