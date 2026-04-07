"""
Unity HDRP Adapter Service
Handles coordinate system transformations, scale standardization, and HDRP-specific 
optimizations for 3D assets generated from AI models.
"""

import numpy as np
import trimesh
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class UnityHDRPAdapter:
    """Adapter for preparing 3D assets for Unity HDRP integration"""
    
    # Unity coordinate system constants
    UNITY_COORDINATE_SYSTEM = {
        'handedness': 'left',  # Left-handed coordinate system
        'up_axis': 'y',        # Y-axis is up
        'forward_axis': 'z'    # Z-axis is forward
    }
    
    # Standard scale units for different object categories (in meters)
    SCALE_STANDARDS = {
        'character': 1.8,      # Average human height
        'prop': 0.5,           # Medium-sized prop
        'structure': 10.0,     # Building/structure size
        'vehicle': 4.5,        # Average vehicle length
        'furniture': 2.0,      # Furniture piece
        'default': 1.0         # Default unit scale
    }
    
    def __init__(self):
        """Initialize the Unity HDRP adapter"""
        self.transformation_history = []
    
    def transform_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Transform mesh from source coordinate system (typically TripoSR: Right-handed, Z-up) 
        to Unity coordinate system (Left-handed, Y-up).
        
        Args:
            mesh: Input mesh to transform
            
        Returns:
            Transformed mesh ready for Unity
        """
        logger.info("Applying Unity coordinate transformation...")
        
        # Create transformation matrix: Right-handed Z-up -> Left-handed Y-up
        # This involves:
        # 1. Swapping Y and Z axes (Z-up to Y-up)
        # 2. Inverting one axis to change handedness (X or Z)
        # 3. Applying the transformation
        
        transform_matrix = np.array([
            [1,  0,  0, 0],  # X remains unchanged
            [0,  0,  1, 0],  # Y = Z (up direction swap)  
            [0, -1,  0, 0],  # Z = -Y (handedness flip)
            [0,  0,  0, 1]   # W unchanged
        ], dtype=np.float64)
        
        # Apply transformation
        original_bounds = mesh.bounds.copy()
        mesh.apply_transform(transform_matrix)
        
        # Record transformation for metadata
        self.transformation_history.append({
            'type': 'coordinate_transform',
            'matrix': transform_matrix.tolist(),
            'original_bounds': original_bounds.tolist(),
            'new_bounds': mesh.bounds.tolist()
        })
        
        logger.info(f"Coordinate transformation applied. New bounds: {mesh.bounds}")
        return mesh
    
    def standardize_scale(self, 
                         mesh: trimesh.Trimesh, 
                         target_category: str = 'default',
                         target_size: Optional[float] = None) -> trimesh.Trimesh:
        """
        Standardize mesh scale according to Unity conventions and object category.
        
        Args:
            mesh: Input mesh to scale
            target_category: Object category for scale reference
            target_size: Specific target size in meters (overrides category)
            
        Returns:
            Scaled mesh
        """
        logger.info(f"Standardizing scale for category: {target_category}")
        
        # Get current dimensions
        bounds = mesh.bounds
        current_size = np.max(bounds[1] - bounds[0])  # Maximum extent
        
        # Determine target size
        if target_size is not None:
            desired_size = target_size
        else:
            desired_size = self.SCALE_STANDARDS.get(target_category, 
                                                  self.SCALE_STANDARDS['default'])
        
        # Calculate scale factor
        if current_size > 0:
            scale_factor = desired_size / current_size
        else:
            scale_factor = 1.0
            logger.warning("Mesh has zero size, using unit scale")
        
        # Apply scaling
        mesh.apply_scale(scale_factor)
        
        # Record transformation
        self.transformation_history.append({
            'type': 'scale_standardization',
            'category': target_category,
            'original_size': float(current_size),
            'target_size': float(desired_size),
            'scale_factor': float(scale_factor)
        })
        
        logger.info(f"Scale standardized: {current_size:.3f} -> {desired_size:.3f} (factor: {scale_factor:.3f})")
        return mesh
    
    def align_pivot_to_ground(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Align mesh pivot to ground level (Y=0) and center on XZ plane.
        
        Args:
            mesh: Input mesh to align
            
        Returns:
            Aligned mesh
        """
        logger.info("Aligning pivot to ground level...")
        
        bounds = mesh.bounds
        min_bound = bounds[0]
        max_bound = bounds[1]
        
        # Calculate transformations
        center_x = (min_bound[0] + max_bound[0]) / 2
        center_z = (min_bound[2] + max_bound[2]) / 2
        ground_y = min_bound[1]  # Bottom of mesh
        
        # Translation vector to move pivot to ground center
        translation = np.array([-center_x, -ground_y, -center_z])
        
        # Apply translation
        mesh.apply_translation(translation)
        
        # Record transformation
        self.transformation_history.append({
            'type': 'pivot_alignment',
            'translation': translation.tolist(),
            'original_bounds': bounds.tolist(),
            'new_bounds': mesh.bounds.tolist()
        })
        
        logger.info(f"Pivot aligned to ground. New bounds: {mesh.bounds}")
        return mesh
    
    def validate_unity_compatibility(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Validate mesh for Unity HDRP compatibility.
        
        Args:
            mesh: Mesh to validate
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating Unity HDRP compatibility...")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Check basic mesh properties
        validation_results['metrics'] = {
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'bounds': {
                'min': mesh.bounds[0].tolist(),
                'max': mesh.bounds[1].tolist(),
                'size': (mesh.bounds[1] - mesh.bounds[0]).tolist()
            },
            'is_watertight': mesh.is_watertight,
            'is_manifold': mesh.is_winding_consistent,
            'volume': float(mesh.volume) if mesh.is_volume else 0,
            'surface_area': float(mesh.area)
        }
        
        # Validate coordinate system orientation
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        
        # Check if object is reasonably sized for Unity (not microscopic or astronomical)
        max_dimension = np.max(size)
        if max_dimension < 0.01:
            validation_results['issues'].append("Object is extremely small")
            validation_results['recommendations'].append("Consider increasing scale")
            validation_results['is_valid'] = False
        elif max_dimension > 1000:
            validation_results['issues'].append("Object is extremely large")
            validation_results['recommendations'].append("Consider decreasing scale")
            validation_results['is_valid'] = False
        
        # Check for proper orientation (Y should be up direction)
        height = size[1]  # Y dimension
        width = size[0]   # X dimension  
        depth = size[2]   # Z dimension
        
        if height < width * 0.1 or height < depth * 0.1:
            validation_results['issues'].append("Object appears to be lying on its side")
            validation_results['recommendations'].append("Check coordinate transformation")
        
        # Validate mesh quality
        if not mesh.is_watertight:
            validation_results['issues'].append("Mesh is not watertight")
            validation_results['recommendations'].append("Consider repairing mesh")
        
        if not mesh.is_winding_consistent:
            validation_results['issues'].append("Inconsistent face winding")
            validation_results['recommendations'].append("Fix normals and winding")
        
        # Update overall validity
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        logger.info(f"Validation complete. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def apply_complete_transformation(self, 
                                    mesh: trimesh.Trimesh,
                                    category: str = 'default',
                                    target_size: Optional[float] = None) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Apply complete transformation pipeline for Unity HDRP compatibility.
        
        Args:
            mesh: Input mesh
            category: Object category for scale standardization
            target_size: Optional specific target size
            
        Returns:
            Tuple of (transformed_mesh, transformation_metadata)
        """
        logger.info("Applying complete Unity HDRP transformation pipeline...")
        
        # Reset transformation history
        self.transformation_history = []
        
        # Step 1: Coordinate transformation
        mesh = self.transform_coordinates(mesh)
        
        # Step 2: Scale standardization
        mesh = self.standardize_scale(mesh, category, target_size)
        
        # Step 3: Pivot alignment
        mesh = self.align_pivot_to_ground(mesh)
        
        # Step 4: Validation
        validation_results = self.validate_unity_compatibility(mesh)
        
        # Compile metadata
        metadata = {
            'transformation_pipeline': 'complete_unity_hdrp',
            'steps_applied': [
                'coordinate_transformation',
                'scale_standardization', 
                'pivot_alignment',
                'validation'
            ],
            'transformation_history': self.transformation_history,
            'validation_results': validation_results,
            'final_bounds': mesh.bounds.tolist()
        }
        
        logger.info("Complete transformation pipeline applied successfully")
        return mesh, metadata


# Global adapter instance
hdrp_adapter = UnityHDRPAdapter()


def get_hdrp_adapter() -> UnityHDRPAdapter:
    """Get the global HDRP adapter instance"""
    return hdrp_adapter