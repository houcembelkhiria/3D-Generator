"""
Asset Validator for Unity HDRP Compatibility
Validates 3D assets against Unity HDRP requirements and identifies potential issues
before import into the engine.
"""

import trimesh
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AssetValidator:
    """Validates 3D assets for Unity HDRP compatibility and quality"""
    
    def __init__(self):
        """Initialize the asset validator"""
        # Unity HDRP compatibility thresholds
        self.thresholds = {
            'max_vertices': 65000,        # Unity mesh vertex limit
            'max_triangles': 50000,       # Practical triangle limit for real-time
            'min_scale': 0.01,            # Minimum reasonable size (1cm)
            'max_scale': 1000.0,          # Maximum reasonable size (1km)
            'max_texture_size': 4096,     # Maximum recommended texture resolution
            'recommended_texture_sizes': [64, 128, 256, 512, 1024, 2048, 4096]
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'geometry': 0.35,
            'topology': 0.25,
            'scale': 0.20,
            'materials': 0.20
        }
    
    def validate_asset_comprehensive(self, 
                                   mesh: trimesh.Trimesh,
                                   material_data: Optional[Dict] = None,
                                   metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of 3D asset for Unity HDRP.
        
        Args:
            mesh: 3D mesh to validate
            material_data: Optional material information
            metadata: Optional additional metadata
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Performing comprehensive asset validation...")
        
        validation_report = {
            'overall_score': 0.0,
            'validation_timestamp': str(trimesh.util.now()),
            'tests_performed': [],
            'issues_found': [],
            'recommendations': [],
            'pass_fail_summary': {},
            'detailed_results': {}
        }
        
        # Geometry validation
        geom_results = self._validate_geometry(mesh)
        validation_report['detailed_results']['geometry'] = geom_results
        validation_report['tests_performed'].extend(geom_results['tests'])
        validation_report['issues_found'].extend(geom_results['issues'])
        validation_report['recommendations'].extend(geom_results['recommendations'])
        
        # Topology validation
        topo_results = self._validate_topology(mesh)
        validation_report['detailed_results']['topology'] = topo_results
        validation_report['tests_performed'].extend(topo_results['tests'])
        validation_report['issues_found'].extend(topo_results['issues'])
        validation_report['recommendations'].extend(topo_results['recommendations'])
        
        # Scale validation
        scale_results = self._validate_scale(mesh)
        validation_report['detailed_results']['scale'] = scale_results
        validation_report['tests_performed'].extend(scale_results['tests'])
        validation_report['issues_found'].extend(scale_results['issues'])
        validation_report['recommendations'].extend(scale_results['recommendations'])
        
        # Material validation (if provided)
        if material_data:
            mat_results = self._validate_materials(material_data)
            validation_report['detailed_results']['materials'] = mat_results
            validation_report['tests_performed'].extend(mat_results['tests'])
            validation_report['issues_found'].extend(mat_results['issues'])
            validation_report['recommendations'].extend(mat_results['recommendations'])
        
        # Calculate overall score
        validation_report['overall_score'] = self._calculate_overall_score(validation_report)
        
        # Generate pass/fail summary
        validation_report['pass_fail_summary'] = self._generate_pass_fail_summary(validation_report)
        
        logger.info(f"Validation complete. Overall score: {validation_report['overall_score']:.2f}")
        return validation_report
    
    def _validate_geometry(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Validate mesh geometry quality"""
        results = {
            'tests': ['vertex_count_check', 'triangle_count_check', 'watertight_check', 'manifold_check'],
            'issues': [],
            'recommendations': [],
            'metrics': {},
            'score': 1.0
        }
        
        # Count metrics
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)
        edge_count = len(mesh.edges_unique)
        
        results['metrics'] = {
            'vertices': vertex_count,
            'faces': face_count,
            'edges': edge_count,
            'volume': float(mesh.volume) if mesh.is_volume else 0,
            'surface_area': float(mesh.area),
            'bounds': {
                'min': mesh.bounds[0].tolist(),
                'max': mesh.bounds[1].tolist(),
                'size': (mesh.bounds[1] - mesh.bounds[0]).tolist()
            }
        }
        
        # Vertex count check
        if vertex_count > self.thresholds['max_vertices']:
            results['issues'].append(f"Excessive vertex count: {vertex_count} (max: {self.thresholds['max_vertices']})")
            results['recommendations'].append("Reduce vertex count through decimation")
            results['score'] *= 0.7
        
        # Triangle count check
        if face_count > self.thresholds['max_triangles']:
            results['issues'].append(f"Excessive triangle count: {face_count} (max: {self.thresholds['max_triangles']})")
            results['recommendations'].append("Simplify geometry or use LOD system")
            results['score'] *= 0.7
        
        # Watertight check
        if not mesh.is_watertight:
            results['issues'].append("Mesh is not watertight")
            results['recommendations'].append("Repair mesh to close holes")
            results['score'] *= 0.8
        
        # Manifold check
        if not mesh.is_winding_consistent:
            results['issues'].append("Inconsistent face winding")
            results['recommendations'].append("Fix normals and face orientation")
            results['score'] *= 0.8
        
        # Degenerate geometry check
        degenerate_faces = mesh.faces[mesh.degenerate_faces]
        if len(degenerate_faces) > 0:
            results['issues'].append(f"Found {len(degenerate_faces)} degenerate faces")
            results['recommendations'].append("Remove degenerate faces")
            results['score'] *= 0.9
        
        return results
    
    def _validate_topology(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Validate mesh topology quality"""
        results = {
            'tests': ['duplicate_vertex_check', 'duplicate_face_check', 'non_manifold_edge_check'],
            'issues': [],
            'recommendations': [],
            'metrics': {},
            'score': 1.0
        }
        
        # Duplicate vertex check
        unique_vertices = np.unique(mesh.vertices, axis=0)
        duplicate_vertex_count = len(mesh.vertices) - len(unique_vertices)
        results['metrics']['duplicate_vertices'] = duplicate_vertex_count
        
        if duplicate_vertex_count > 0:
            results['issues'].append(f"Found {duplicate_vertex_count} duplicate vertices")
            results['recommendations'].append("Merge duplicate vertices")
            results['score'] *= 0.9
        
        # Duplicate face check
        unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
        duplicate_face_count = len(mesh.faces) - len(unique_faces)
        results['metrics']['duplicate_faces'] = duplicate_face_count
        
        if duplicate_face_count > 0:
            results['issues'].append(f"Found {duplicate_face_count} duplicate faces")
            results['recommendations'].append("Remove duplicate faces")
            results['score'] *= 0.9
        
        # Non-manifold edge check
        edge_counts = mesh.edges_sorted
        non_manifold_edges = np.sum(edge_counts > 2)
        results['metrics']['non_manifold_edges'] = non_manifold_edges
        
        if non_manifold_edges > 0:
            results['issues'].append(f"Found {non_manifold_edges} non-manifold edges")
            results['recommendations'].append("Repair non-manifold geometry")
            results['score'] *= 0.8
        
        return results
    
    def _validate_scale(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Validate mesh scale appropriateness"""
        results = {
            'tests': ['scale_range_check', 'aspect_ratio_check'],
            'issues': [],
            'recommendations': [],
            'metrics': {},
            'score': 1.0
        }
        
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        max_dimension = np.max(size)
        min_dimension = np.min(size[np.nonzero(size)])  # Non-zero minimum
        
        results['metrics'] = {
            'max_dimension': float(max_dimension),
            'min_dimension': float(min_dimension),
            'aspect_ratio': float(max_dimension / min_dimension) if min_dimension > 0 else float('inf'),
            'bounding_box': size.tolist()
        }
        
        # Scale range check
        if max_dimension < self.thresholds['min_scale']:
            results['issues'].append(f"Object too small: {max_dimension:.4f}m (min: {self.thresholds['min_scale']}m)")
            results['recommendations'].append("Increase scale to appropriate size")
            results['score'] *= 0.6
        elif max_dimension > self.thresholds['max_scale']:
            results['issues'].append(f"Object too large: {max_dimension:.1f}m (max: {self.thresholds['max_scale']}m)")
            results['recommendations'].append("Decrease scale to appropriate size")
            results['score'] *= 0.6
        
        # Aspect ratio check
        aspect_ratio = results['metrics']['aspect_ratio']
        if aspect_ratio > 100:  # Extreme aspect ratio
            results['issues'].append(f"Extreme aspect ratio: {aspect_ratio:.1f}:1")
            results['recommendations'].append("Check if scale is appropriate for object type")
            results['score'] *= 0.8
        
        return results
    
    def _validate_materials(self, material_data: Dict) -> Dict[str, Any]:
        """Validate material data for HDRP compatibility"""
        results = {
            'tests': ['texture_resolution_check', 'material_definition_check'],
            'issues': [],
            'recommendations': [],
            'metrics': {},
            'score': 1.0
        }
        
        if 'textures_generated' in material_data:
            results['metrics']['texture_count'] = material_data.get('textures_generated', 0)
        
        if 'texture_paths' in material_data:
            texture_paths = material_data['texture_paths']
            invalid_resolutions = []
            
            for tex_type, tex_path in texture_paths.items():
                try:
                    if Path(tex_path).exists():
                        from PIL import Image
                        with Image.open(tex_path) as img:
                            width, height = img.size
                            results['metrics'][f'{tex_type}_resolution'] = [width, height]
                            
                            # Check if resolution is power of 2 and within recommended sizes
                            if width != height:
                                results['issues'].append(f"{tex_type} texture is not square: {width}x{height}")
                                results['recommendations'].append("Use square textures for optimal performance")
                                results['score'] *= 0.9
                            
                            if width not in self.thresholds['recommended_texture_sizes']:
                                invalid_resolutions.append(f"{tex_type}: {width}x{height}")
                                
                except Exception as e:
                    results['issues'].append(f"Could not validate {tex_type} texture: {str(e)}")
                    results['score'] *= 0.8
            
            if invalid_resolutions:
                results['issues'].append(f"Non-standard resolutions: {', '.join(invalid_resolutions)}")
                results['recommendations'].append("Use standard texture resolutions (64, 128, 256, 512, 1024, 2048, 4096)")
                results['score'] *= 0.9
        
        return results
    
    def _calculate_overall_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score"""
        scores = []
        weights = []
        
        # Extract scores from detailed results
        for category, weight in self.quality_weights.items():
            if category in validation_report['detailed_results']:
                score = validation_report['detailed_results'][category].get('score', 1.0)
                scores.append(score)
                weights.append(weight)
        
        if not scores:
            return 1.0
            
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def _generate_pass_fail_summary(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple pass/fail summary"""
        overall_score = validation_report['overall_score']
        
        summary = {
            'overall_status': 'PASS' if overall_score >= 0.7 else 'WARNING' if overall_score >= 0.5 else 'FAIL',
            'quality_level': self._get_quality_level(overall_score),
            'critical_issues': len([issue for issue in validation_report['issues_found'] if 'excessive' in issue.lower() or 'extreme' in issue.lower()]),
            'total_issues': len(validation_report['issues_found']),
            'total_recommendations': len(validation_report['recommendations'])
        }
        
        return summary
    
    def _get_quality_level(self, score: float) -> str:
        """Convert numerical score to quality level"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Acceptable"
        elif score >= 0.5:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, Any],
                                 output_path: str) -> str:
        """Generate detailed validation report file"""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation report saved: {report_path}")
        return str(report_path)


# Global validator instance
asset_validator = AssetValidator()


def get_asset_validator() -> AssetValidator:
    """Get the global asset validator instance"""
    return asset_validator