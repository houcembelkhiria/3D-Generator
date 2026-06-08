"""
Pydantic models for 3D model specification validation.
Defines the expected structure for LLM-generated JSON responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class DimensionUnit(str, Enum):
    """Supported dimension units"""
    MM = "mm"
    CM = "cm"
    M = "m"
    INCH = "inch"
    FT = "ft"


class MaterialType(str, Enum):
    """Common material types"""
    PLASTIC = "Plastic"
    METAL = "Metal"
    WOOD = "Wood"
    GLASS = "Glass"
    RUBBER = "Rubber"
    SILICONE = "Silicone"
    CARBON_FIBER = "Carbon Fiber"
    CERAMIC = "Ceramic"


class ColorFinish(str, Enum):
    """Common color/finish options"""
    MATTE_BLACK = "Matte Black"
    GLOSSY_WHITE = "Glossy White"
    BRUSHED_METAL = "Brushed Metal"
    TRANSPARENT = "Transparent"
    CUSTOM = "Custom"


class GeometricShape(str, Enum):
    """Basic geometric shapes"""
    CUBE = "Cube"
    SPHERE = "Sphere"
    CYLINDER = "Cylinder"
    CONE = "Cone"
    PYRAMID = "Pyramid"
    PRISM = "Prism"
    CUSTOM = "Custom"


class DimensionSpec(BaseModel):
    """Specification for object dimensions"""
    length: Optional[float] = Field(None, description="Length dimension", gt=0)
    width: Optional[float] = Field(None, description="Width dimension", gt=0)
    height: Optional[float] = Field(None, description="Height dimension", gt=0)
    diameter: Optional[float] = Field(None, description="Diameter (for circular objects)", gt=0)
    radius: Optional[float] = Field(None, description="Radius (for circular objects)", gt=0)
    unit: DimensionUnit = Field(DimensionUnit.MM, description="Unit of measurement")
    
    @validator('unit', pre=True)
    def validate_unit(cls, v):
        if isinstance(v, str):
            return DimensionUnit(v.lower())
        return v


class MaterialSpec(BaseModel):
    """Specification for object material"""
    type: MaterialType = Field(..., description="Primary material type")
    subtype: Optional[str] = Field(None, description="Specific material variant")
    color: ColorFinish = Field(ColorFinish.MATTE_BLACK, description="Surface finish/color")
    texture: Optional[str] = Field(None, description="Surface texture description")


class ObjectSpec(BaseModel):
    """Complete specification for a 3D object"""
    name: str = Field(..., description="Descriptive name of the object", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Detailed description of the object")
    
    # Geometry
    shape: GeometricShape = Field(GeometricShape.CUSTOM, description="Basic geometric shape")
    dimensions: DimensionSpec = Field(..., description="Physical dimensions")
    
    # Material and appearance
    material: MaterialSpec = Field(..., description="Material specifications")
    
    # Additional properties
    quantity: int = Field(1, description="Number of instances to generate", ge=1, le=100)
    hollow: Optional[bool] = Field(False, description="Whether the object should be hollow")
    weight_estimate: Optional[float] = Field(None, description="Estimated weight in grams", gt=0)
    
    # Technical requirements
    tolerance: Optional[float] = Field(None, description="Manufacturing tolerance in mm", gt=0)
    surface_finish: Optional[str] = Field(None, description="Required surface finish quality")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Smart Vending Machine",
                "description": "Modern vending machine with touch screen interface",
                "shape": "CUSTOM",
                "dimensions": {
                    "length": 1800,
                    "width": 900,
                    "height": 700,
                    "unit": "mm"
                },
                "material": {
                    "type": "Metal",
                    "subtype": "Brushed Aluminum",
                    "color": "Brushed Metal"
                },
                "quantity": 1,
                "hollow": True
            }
        }
    }


class TableData(BaseModel):
    """Structured table data extraction"""
    headers: List[str] = Field(..., description="Table column headers")
    rows: List[List[str]] = Field(..., description="Table data rows")
    caption: Optional[str] = Field(None, description="Table caption or title")


class DocumentAnalysis(BaseModel):
    """Complete document analysis result"""
    objects: List[ObjectSpec] = Field(default_factory=list, description="Extracted 3D object specifications")
    tables: List[TableData] = Field(default_factory=list, description="Extracted table data")
    raw_text: str = Field(..., description="Cleaned raw text from document")
    confidence_score: float = Field(0.0, description="Confidence score for extraction (0-1)", ge=0, le=1)
    warnings: List[str] = Field(default_factory=list, description="Any warnings or issues during extraction")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "objects": [
                    {
                        "name": "Control Panel",
                        "shape": "CUSTOM",
                        "dimensions": {
                            "length": 300,
                            "width": 200,
                            "height": 50,
                            "unit": "mm"
                        },
                        "material": {
                            "type": "Plastic",
                            "color": "Matte Black"
                        }
                    }
                ],
                "tables": [],
                "raw_text": "Technical specifications for the control panel...",
                "confidence_score": 0.85,
                "warnings": []
            }
        }
    }


# Utility functions for validation and conversion
def validate_object_spec(data: Dict[str, Any]) -> ObjectSpec:
    """
    Validate and convert dictionary to ObjectSpec model.
    
    Args:
        data: Dictionary containing object specification data
        
    Returns:
        Validated ObjectSpec instance
        
    Raises:
        ValidationError: If data doesn't match the schema
    """
    return ObjectSpec(**data)


def validate_document_analysis(data: Dict[str, Any]) -> DocumentAnalysis:
    """
    Validate and convert dictionary to DocumentAnalysis model.
    
    Args:
        data: Dictionary containing document analysis data
        
    Returns:
        Validated DocumentAnalysis instance
        
    Raises:
        ValidationError: If data doesn't match the schema
    """
    # Convert nested objects
    if 'objects' in data:
        data['objects'] = [validate_object_spec(obj) for obj in data['objects']]
    
    return DocumentAnalysis(**data)


# Example validation usage
if __name__ == "__main__":
    # Test data
    test_data = {
        "name": "Test Object",
        "shape": "Cube",
        "dimensions": {
            "length": 100,
            "width": 100,
            "height": 100,
            "unit": "mm"
        },
        "material": {
            "type": "Plastic",
            "color": "Matte Black"
        }
    }
    
    try:
        obj_spec = validate_object_spec(test_data)
        print("Validation successful!")
        print(f"Object: {obj_spec.name}")
        print(f"Dimensions: {obj_spec.dimensions.length} x {obj_spec.dimensions.width} x {obj_spec.dimensions.height} {obj_spec.dimensions.unit}")
        print(f"Material: {obj_spec.material.type}")
    except Exception as e:
        print(f"Validation failed: {e}")