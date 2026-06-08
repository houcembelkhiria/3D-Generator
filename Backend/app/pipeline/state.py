import operator
from typing import Annotated, List, Optional, TypedDict


class Pipeline3DState(TypedDict):
    # Input
    file_path: str
    file_type: str

    # Parse stage
    raw_text: str
    parsed_content: dict

    # Spec extraction
    spec: Optional[dict]
    spec_valid: bool
    spec_retry_count: int

    # Generation
    mesh_output: Optional[dict]
    mesh_valid: bool
    mesh_retry_count: int

    # Output
    model_info: Optional[dict]

    # Accumulated errors across all nodes
    errors: Annotated[List[str], operator.add]

    current_step: str
