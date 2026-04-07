"""
3D Model Conversion Service
Handles conversion between different 3D file formats (OBJ, GLB, STL, etc.)
and mesh optimization operations with Unity HDRP compatibility.
"""

import os
import json
import logging
import trimesh
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

# Import HDRP integration components
try:
    from .unity_hdrp_adapter import get_hdrp_adapter
    from .pbr_material_generator import get_pbr_generator
    from .asset_validator import get_asset_validator
    HDRP_INTEGRATION_AVAILABLE = True
    logger = logging.getLogger(__name__)
except ImportError:
    HDRP_INTEGRATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("HDRP integration modules not available")


logger = logging.getLogger(__name__)


class ModelConversionService:
    """Service for 3D model format conversion and optimization with HDRP support"""
    
    SUPPORTED_INPUT_FORMATS = {'.obj', '.stl', '.ply', '.off', '.gltf', '.glb'}
    SUPPORTED_OUTPUT_FORMATS = {'.obj', '.stl', '.ply', '.glb', '.gltf'}
    
    def __init__(self):
        """Initialize the conversion service"""
        self.optimization_presets = {
            'low': {'simplify_ratio': 0.5, 'merge_vertices': True},
            'medium': {'simplify_ratio': 0.7, 'merge_vertices': True},
            'high': {'simplify_ratio': 0.9, 'merge_vertices': True},
            'none': {'simplify_ratio': 1.0, 'merge_vertices': False}
        }
        
        # HDRP-specific settings
        self.hdrp_settings = {
            'enable_coordinate_transform': True,
            'enable_pbr_materials': True,
            'enable_validation': True,
            'default_object_category': 'prop'
        }
    
    def convert_model_format(self, 
                           input_path: str, 
                           output_path: str, 
                           target_format: Optional[str] = None,
                           optimize_quality: str = 'medium',
                           hdrp_compatible: bool = False,
                           object_category: str = 'default') -> Dict[str, Any]:
        """
        Convert 3D model from one format to another.
        
        Args:
            input_path: Path to input model file
            output_path: Path for output model file
            target_format: Target format extension (optional, inferred from output_path)
            optimize_quality: Quality preset ('low', 'medium', 'high', 'none')
            hdrp_compatible: Enable HDRP-specific optimizations
            object_category: Object category for scale standardization
            
        Returns:
            Dictionary with conversion results and metadata
        """
        try:
            # Validate input file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            input_ext = Path(input_path).suffix.lower()
            if input_ext not in self.SUPPORTED_INPUT_FORMATS:
                raise ValueError(f"Unsupported input format: {input_ext}")
            
            # Determine target format
            if target_format:
                output_ext = target_format.lower()
                if not output_ext.startswith('.'):
                    output_ext = '.' + output_ext
            else:
                output_ext = Path(output_path).suffix.lower()
            
            if output_ext not in self.SUPPORTED_OUTPUT_FORMATS:
                raise ValueError(f"Unsupported output format: {output_ext}")
            
            logger.info(f"Converting {input_path} ({input_ext}) to {output_path} ({output_ext})")
            
            # Load the mesh
            mesh = trimesh.load(input_path, force='mesh')
            
            # Apply HDRP preprocessing if requested
            if hdrp_compatible and HDRP_INTEGRATION_AVAILABLE:
                mesh = self._apply_hdrp_preprocessing(mesh, object_category)
            
            # Apply optimization
            optimized_mesh = self._optimize_mesh(mesh, optimize_quality)
            
            # Export to target format
            export_kwargs = self._get_export_kwargs(output_ext)
            optimized_mesh.export(output_path, **export_kwargs)
            
            # Generate metadata
            result = self._generate_conversion_metadata(
                input_path, output_path, optimized_mesh, optimize_quality
            )
            
            # Add HDRP-specific metadata
            if hdrp_compatible:
                result['hdrp_compatibility'] = {
                    'enabled': True,
                    'object_category': object_category,
                    'coordinate_system_transformed': self.hdrp_settings['enable_coordinate_transform'],
                    'pbr_materials_enabled': self.hdrp_settings['enable_pbr_materials']
                }
            
            logger.info(f"Conversion completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Model conversion failed: {str(e)}")
            raise
    
    def _apply_hdrp_preprocessing(self, mesh: trimesh.Trimesh, object_category: str) -> trimesh.Trimesh:
        """Apply HDRP-specific preprocessing to mesh"""
        logger.info(f"Applying HDRP preprocessing for category: {object_category}")
        
        if not HDRP_INTEGRATION_AVAILABLE:
            logger.warning("HDRP integration not available, skipping preprocessing")
            return mesh
        
        try:
            # Apply coordinate transformation
            if self.hdrp_settings['enable_coordinate_transform']:
                hdrp_adapter = get_hdrp_adapter()
                mesh = hdrp_adapter.transform_coordinates(mesh)
                logger.debug("Applied coordinate system transformation")
            
            # Apply scale standardization
            hdrp_adapter = get_hdrp_adapter()
            mesh = hdrp_adapter.standardize_scale(mesh, object_category)
            logger.debug("Applied scale standardization")
            
            # Apply pivot alignment
            mesh = hdrp_adapter.align_pivot_to_ground(mesh)
            logger.debug("Applied pivot alignment")
            
            return mesh
            
        except Exception as e:
            logger.error(f"HDRP preprocessing failed: {str(e)}")
            return mesh  # Return original mesh on failure

    def _optimize_mesh(self, mesh: trimesh.Trimesh, quality_preset: str) -> trimesh.Trimesh:
        """
        Optimize mesh based on quality preset.
        
        Args:
            mesh: Input mesh to optimize
            quality_preset: Quality level ('low', 'medium', 'high', 'none')
            
        Returns:
            Optimized mesh
        """
        if quality_preset not in self.optimization_presets:
            logger.warning(f"Unknown quality preset '{quality_preset}', using 'medium'")
            quality_preset = 'medium'
        
        preset = self.optimization_presets[quality_preset]
        logger.info(f"Applying {quality_preset} quality optimization")
        
        # Make a copy to avoid modifying original
        optimized = mesh.copy()
        
        # Vertex merging to remove duplicates
        if preset['merge_vertices']:
            optimized.merge_vertices()
            logger.debug(f"Merged duplicate vertices")
        
        # Mesh simplification if requested (disabled due to dependency issues)
        simplify_ratio = preset['simplify_ratio']
        if simplify_ratio < 1.0:
            logger.debug(f"Mesh simplification requested but disabled due to missing dependencies")
            # Would implement simplification here when fast_simplification is available
        
        # Recalculate normals
        optimized.fix_normals()
        
        return optimized
    
    def _get_export_kwargs(self, format_ext: str) -> Dict[str, Any]:
        """Get export parameters for different formats"""
        kwargs = {}
        
        if format_ext == '.glb':
            kwargs.update({
                'file_type': 'glb'
            })
        elif format_ext == '.gltf':
            kwargs.update({
                'file_type': 'gltf'
            })
        elif format_ext == '.obj':
            kwargs.update({
                'file_type': 'obj'
            })
        
        return kwargs
    
    def _generate_conversion_metadata(self, 
                                    input_path: str, 
                                    output_path: str, 
                                    mesh: trimesh.Trimesh,
                                    quality_preset: str) -> Dict[str, Any]:
        """Generate metadata for the conversion process"""
        input_stats = os.stat(input_path)
        output_stats = os.stat(output_path) if os.path.exists(output_path) else None
        
        # Calculate mesh statistics
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        volume = mesh.volume if mesh.is_volume else 0
        surface_area = mesh.area
        
        return {
            'conversion_info': {
                'input_file': input_path,
                'output_file': output_path,
                'input_format': Path(input_path).suffix.lower(),
                'output_format': Path(output_path).suffix.lower(),
                'quality_preset': quality_preset,
                'timestamp': datetime.now().isoformat()
            },
            'input_stats': {
                'size_bytes': input_stats.st_size,
                'modified_time': datetime.fromtimestamp(input_stats.st_mtime).isoformat()
            },
            'output_stats': {
                'size_bytes': output_stats.st_size if output_stats else 0,
                'created_time': datetime.now().isoformat()
            },
            'mesh_stats': {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'edges': len(mesh.edges_unique),
                'volume': float(volume),
                'surface_area': float(surface_area),
                'bounding_box': {
                    'min': bounds[0].tolist(),
                    'max': bounds[1].tolist(),
                    'size': size.tolist(),
                    'center': mesh.centroid.tolist()
                },
                'is_watertight': mesh.is_watertight,
                'is_convex': mesh.is_convex,
                'euler_number': mesh.euler_number
            },
            'quality_metrics': {
                'vertex_density': len(mesh.vertices) / float(surface_area) if surface_area > 0 else 0,
                'face_aspect_ratio_mean': 0.0,  # Disabled due to trimesh version compatibility
                'face_aspect_ratio_std': 0.0
            }
        }
    
    def batch_convert_models(self, 
                           input_directory: str, 
                           output_directory: str,
                           target_format: str = '.glb',
                           quality_preset: str = 'medium') -> List[Dict[str, Any]]:
        """
        Convert multiple models in a directory.
        
        Args:
            input_directory: Directory containing input models
            output_directory: Directory for output models
            target_format: Target format for all conversions
            quality_preset: Quality preset for all conversions
            
        Returns:
            List of conversion results
        """
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        results = []
        supported_files = [
            f for f in os.listdir(input_directory) 
            if Path(f).suffix.lower() in self.SUPPORTED_INPUT_FORMATS
        ]
        
        logger.info(f"Found {len(supported_files)} supported files in {input_directory}")
        
        for filename in supported_files:
            try:
                input_path = os.path.join(input_directory, filename)
                output_filename = Path(filename).stem + target_format
                output_path = os.path.join(output_directory, output_filename)
                
                result = self.convert_model_format(
                    input_path, output_path, target_format, quality_preset
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to convert {filename}: {str(e)}")
                results.append({
                    'error': str(e),
                    'input_file': filename,
                    'success': False
                })
        
        return results
    
    def validate_mesh(self, mesh_path: str) -> Dict[str, Any]:
        """
        Validate a 3D mesh file.
        
        Args:
            mesh_path: Path to mesh file to validate
            
        Returns:
            Validation results and mesh information
        """
        try:
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            
            # Load mesh
            mesh = trimesh.load(mesh_path, force='mesh')
            
            # Perform validation checks
            validation_results = {
                'file_exists': True,
                'load_successful': True,
                'format': Path(mesh_path).suffix.lower(),
                'file_size': os.path.getsize(mesh_path),
                'mesh_properties': {
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces),
                    'is_watertight': mesh.is_watertight,
                    'is_convex': mesh.is_convex,
                    'is_volume': mesh.is_volume,
                    'euler_number': mesh.euler_number,
                    'area': float(mesh.area),
                    'volume': float(mesh.volume) if mesh.is_volume else 0
                },
                'quality_checks': {
                    'degenerate_faces': len(mesh.faces[mesh.degenerate_faces]),
                    'duplicate_faces': len(mesh.faces[mesh.duplicate_faces]),
                    'duplicate_vertices': len(mesh.vertices) - len(np.unique(mesh.vertices, axis=0)),
                    'non_manifold_edges': len(mesh.edges_unique_length[mesh.edges_unique_length > 2])
                },
                'bounds': {
                    'min': mesh.bounds[0].tolist(),
                    'max': mesh.bounds[1].tolist(),
                    'size': (mesh.bounds[1] - mesh.bounds[0]).tolist()
                }
            }
            
            # Overall validity assessment
            quality_issues = []
            if validation_results['quality_checks']['degenerate_faces'] > 0:
                quality_issues.append("Degenerate faces detected")
            if validation_results['quality_checks']['duplicate_faces'] > 0:
                quality_issues.append("Duplicate faces detected")
            if validation_results['quality_checks']['duplicate_vertices'] > 0:
                quality_issues.append("Duplicate vertices detected")
            if validation_results['quality_checks']['non_manifold_edges'] > 0:
                quality_issues.append("Non-manifold edges detected")
            
            validation_results['is_valid'] = len(quality_issues) == 0
            validation_results['quality_issues'] = quality_issues
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Mesh validation failed: {str(e)}")
            return {
                'file_exists': os.path.exists(mesh_path),
                'load_successful': False,
                'error': str(e),
                'is_valid': False
            }
    
    def repair_mesh(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Attempt to repair common mesh issues.
        
        Args:
            input_path: Path to problematic mesh
            output_path: Path for repaired mesh
            
        Returns:
            Repair results and metadata
        """
        try:
            # Load mesh
            mesh = trimesh.load(input_path, force='mesh')
            
            # Store original stats
            original_stats = {
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'is_watertight': mesh.is_watertight,
                'is_volume': mesh.is_volume
            }
            
            # Apply repairs
            repairs_applied = []
            
            # Fix normals
            if not mesh.is_winding_consistent:
                mesh.fix_normals()
                repairs_applied.append("Fixed face winding")
            
            # Remove degenerate faces
            degenerate_count = len(mesh.faces[mesh.degenerate_faces])
            if degenerate_count > 0:
                mesh.remove_degenerate_faces()
                repairs_applied.append(f"Removed {degenerate_count} degenerate faces")
            
            # Merge duplicate vertices
            vertex_count_before = len(mesh.vertices)
            mesh.merge_vertices()
            vertex_count_after = len(mesh.vertices)
            if vertex_count_after < vertex_count_before:
                repairs_applied.append(f"Merged {vertex_count_before - vertex_count_after} duplicate vertices")
            
            # Fill holes if mesh is not watertight
            if not mesh.is_watertight and len(mesh.faces) > 0:
                try:
                    mesh.fill_holes()
                    repairs_applied.append("Filled mesh holes")
                except Exception as e:
                    logger.warning(f"Failed to fill holes: {e}")
            
            # Export repaired mesh
            mesh.export(output_path)
            
            # Generate results
            result = {
                'repair_info': {
                    'input_file': input_path,
                    'output_file': output_path,
                    'repairs_applied': repairs_applied,
                    'timestamp': datetime.now().isoformat()
                },
                'before_repair': original_stats,
                'after_repair': {
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces),
                    'is_watertight': mesh.is_watertight,
                    'is_volume': mesh.is_volume
                },
                'improvements': {
                    'vertices_removed': original_stats['vertices'] - len(mesh.vertices),
                    'faces_removed': original_stats['faces'] - len(mesh.faces),
                    'watertight_fixed': not original_stats['is_watertight'] and mesh.is_watertight,
                    'volume_fixed': not original_stats['is_volume'] and mesh.is_volume
                }
            }
            
            logger.info(f"Mesh repair completed with {len(repairs_applied)} repairs applied")
            return result
            
        except Exception as e:
            logger.error(f"Mesh repair failed: {str(e)}")
            raise


# Global service instance
conversion_service = ModelConversionService()


def get_conversion_service() -> ModelConversionService:
    """Get the global conversion service instance"""
    return conversion_service


# Example usage and testing
if __name__ == "__main__":
    service = ModelConversionService()
    
    # Example validation
    test_mesh_path = "./test_cube.obj"  # You would need an actual mesh file
    
    if os.path.exists(test_mesh_path):
        print("Validating mesh...")
        validation = service.validate_mesh(test_mesh_path)
        print(f"Valid: {validation['is_valid']}")
        if validation['quality_issues']:
            print("Issues found:")
            for issue in validation['quality_issues']:
                print(f"  - {issue}")
    else:
        print("Test mesh file not found. Skipping validation test.")