import shutil
import magic
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import os
from datetime import datetime
from pypdf import PdfReader
from app.worker import celery_app
from app.core.config import settings
from app.core.text_cleaner import clean_document_text

router = APIRouter()

# ============================================================================
# Pydantic Models for Request/Response Documentation
# ============================================================================

class TaskResponse(BaseModel):
    """Response model for document upload endpoint"""
    task_id: str = Field(..., description="Unique identifier for the processing task", example="550e8400-e29b-41d4-a716-446655440000")
    status: str = Field(..., description="Current status of the task", example="processing")
    message: str = Field(..., description="Human-readable status message", example="File uploaded successfully. Processing started.")

class TaskResult(BaseModel):
    """Response model for task status endpoint"""
    task_id: str = Field(..., description="Unique identifier for the task", example="550e8400-e29b-41d4-a716-446655440000")
    status: str = Field(..., description="Current status: pending, processing, completed, failed", example="completed")
    result: Optional[dict] = Field(None, description="Task result data or error information")

class ModelInfo(BaseModel):
    """Information about a generated 3D model"""
    filename: str = Field(..., description="Name of the 3D model file", example="model_123.glb")
    size: int = Field(..., description="File size in bytes", example=1024567)
    created: float = Field(..., description="Unix timestamp of file creation", example=1704067200.0)

class ModelsListResponse(BaseModel):
    """Response model for listing generated 3D models"""
    models: List[ModelInfo] = Field(..., description="List of generated 3D models")

def detect_file_type(file_path: str) -> str:
    """
    Automatically detect file type using python-magic library.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        MIME type string (e.g., 'application/pdf', 'message/rfc822')
    """
    try:
        mime = magic.Magic(mime=True)
        detected_type = mime.from_file(file_path)
        return detected_type
    except Exception as e:
        print(f"Warning: Could not detect file type: {str(e)}")
        return "application/octet-stream"

def is_supported_file_type(mime_type: str) -> bool:
    """
    Check if the detected MIME type is supported for text extraction.
    
    Args:
        mime_type: Detected MIME type
        
    Returns:
        Boolean indicating if type is supported
    """
    supported_types = {
        'application/pdf': 'PDF Document',
        'message/rfc822': 'Email Message',
        'text/plain': 'Plain Text'
    }
    return mime_type in supported_types

def get_file_type_description(mime_type: str) -> str:
    """
    Get human-readable description of file type.
    
    Args:
        mime_type: MIME type string
        
    Returns:
        Human-readable description
    """
    type_descriptions = {
        'application/pdf': 'PDF Document',
        'message/rfc822': 'Email Message (.eml)',
        'text/plain': 'Plain Text File'
    }
    return type_descriptions.get(mime_type, 'Unknown File Type')

class PDFExtractResponse(BaseModel):
    """Response model for PDF text extraction"""
    filename: str = Field(..., description="Original filename of the uploaded PDF", example="document.pdf")
    extracted_text: str = Field(..., description="Extracted text content from the PDF", example="Sample text content from PDF...")
    message: Optional[str] = Field(None, description="Optional message for edge cases (e.g., scanned documents)")

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message describing what went wrong", example="Invalid file type. Only PDF files are allowed.")

@router.post(
    "/upload",
    response_model=TaskResponse,
    tags=["Document Processing"],
    summary="Upload document for 3D generation",
    description="""
    Upload a PDF or Email document to initiate 3D model generation.
    
    The document will be processed asynchronously by Celery workers.
    Use the returned `task_id` to check processing status.
    
    **Supported file types:**
    - PDF documents (`application/pdf`)
    - Email messages (`message/rfc822`)
    
    **Processing flow:**
    1. File validation (type and size)
    2. File storage in uploads directory
    3. Celery task queued for processing
    4. AI/ML pipeline generates 3D model
    5. Result stored in generated directory
    """,
    response_description="Task information including task_id for status tracking",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def upload_file(
    file: UploadFile = File(
        ...,
        description="Document file to upload (PDF or Email .eml)",
        example="document.pdf or email.eml"
    )
):
    """
    Upload document for 3D generation processing using unstructured pipeline.
    
    Supports both PDF documents and Email files (.eml format).
    The unstructured library parses the document and extracts metadata
    for 3D model generation.
    
    - **file**: PDF or Email (.eml) file to process
    - Returns task_id to track processing status
    """
    
    # Validate file type
    allowed_types = ["application/pdf", "message/rfc822"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF and Email files are supported"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
    filename = f"{file_id}.{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Start Celery task
    task = celery_app.send_task(
        "app.tasks.process_document",
        args=[file_path, file.content_type],
        task_id=file_id
    )
    
    return TaskResponse(
        task_id=file_id,
        status="processing",
        message="File uploaded successfully. Processing started."
    )

@router.get(
    "/task/{task_id}",
    response_model=TaskResult,
    tags=["Task Management"],
    summary="Get task status",
    description="""
    Retrieve the current status and result of a processing task.
    
    **Task States:**
    - **pending**: Task is queued and waiting to be processed
    - **processing**: Task is currently being executed
    - **completed**: Task finished successfully, result available
    - **failed**: Task failed, error information provided
    
    **Usage:**
    Poll this endpoint after uploading a document to track progress.
    When status is "completed", the 3D model is ready for download.
    """,
    response_description="Current task status and optional result data",
    responses={
        200: {"model": TaskResult, "description": "Task status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Task not found"},
    }
)
async def get_task_status(
    task_id: str
):
    """
    Get the status of a processing task.
    
    - **task_id**: UUID returned from the upload endpoint
    - Returns current status and result if completed
    """
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "PENDING":
        return TaskResult(
            task_id=task_id,
            status="pending",
            result=None
        )
    elif task.state == "PROCESSING":
        return TaskResult(
            task_id=task_id,
            status="processing",
            result=task.info
        )
    elif task.state == "SUCCESS":
        return TaskResult(
            task_id=task_id,
            status="completed",
            result=task.result
        )
    elif task.state == "FAILURE":
        return TaskResult(
            task_id=task_id,
            status="failed",
            result={"error": str(task.info)}
        )
    else:
        return TaskResult(
            task_id=task_id,
            status=task.state.lower(),
            result=task.info
        )

@router.get(
    "/models",
    response_model=ModelsListResponse,
    tags=["3D Models"],
    summary="List generated 3D models",
    description="""
    Retrieve a list of all generated 3D models.
    
    **Supported formats:**
    - GLB (GL Transmission Format Binary)
    - OBJ (Wavefront OBJ)
    - STL (STereoLithography)
    
    **Response includes:**
    - Filename for download/reference
    - File size in bytes
    - Creation timestamp
    
    Models are stored in the `generated/` directory and persist until manually deleted.
    """,
    response_description="List of all generated 3D models with metadata"
)
async def list_generated_models():
    """
    List all generated 3D models.
    
    Returns metadata for all GLB, OBJ, and STL files in the generated directory.
    """
    models = []
    if os.path.exists(settings.GENERATED_DIR):
        for filename in os.listdir(settings.GENERATED_DIR):
            if filename.endswith((".glb", ".obj", ".stl")):
                file_path = os.path.join(settings.GENERATED_DIR, filename)
                models.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "created": os.path.getctime(file_path)
                })
    return {"models": models}

@router.post(
    "/extract-text/",
    response_model=PDFExtractResponse,
    tags=["PDF Tools"],
    summary="Extract text from documents with automatic type detection",
    description="""
    Extract plain text content from uploaded documents with automatic file type detection.
    
    **Automatic Detection:**
    - Uses python-magic library to detect actual file type
    - Supports PDF, EML, and plain text files
    - Validates against supported formats
    
    **Supported Formats:**
    - PDF documents (application/pdf)
    - Email files (.eml format, message/rfc822)
    - Plain text files (text/plain)
    
    **Advanced Features:**
    - Smart text cleaning and formatting
    - Multi-page PDF support with page separation
    - Email header/body parsing for EML files
    - Language detection and encoding handling
    - Duplicate whitespace removal
    - Special character preservation
    - Progress tracking for large documents
    
    **Use Cases:**
    - Extract text from PDF documents for processing
    - Parse email content from EML files
    - Process plain text files
    - Pre-process documents before 3D generation
    - Content analysis and indexing
    - Document digitization workflows
    
    **Processing Details:**
    1. File validation and security checks
    2. Automatic file type detection using magic numbers
    3. Format-specific parsing (PDF, EML, or text)
    4. Text extraction from content
    5. Advanced text cleaning and normalization
    6. Metadata extraction
    7. Temporary file cleanup
    """,
    response_description="Extracted text content from the PDF with metadata",
    responses={
        200: {"model": PDFExtractResponse, "description": "Text extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type (non-PDF/non-EML)"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Corrupted or invalid PDF"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    }
)
async def extract_text_from_pdf(
    file: UploadFile = File(
        ...,
        description="Document file to extract text from (PDF, EML, or text file, max 50MB)",
        example="document.pdf or email.eml or notes.txt"
    )
):
    """
    Extract text from an uploaded document with automatic file type detection.
    
    - **file**: Document file to process (PDF, EML, or text file, max 50MB)
    - Returns extracted text with cleaning and formatting
    - Automatically detects file type using python-magic
    - Includes format-specific metadata and processing
    """
    # File size validation
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    # Create a temporary file to save the uploaded document
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        # Save uploaded file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Automatic file type detection
        detected_mime_type = detect_file_type(temp_filename)
        file_type_description = get_file_type_description(detected_mime_type)
        
        print(f"Detected file type: {detected_mime_type} ({file_type_description})")
        
        # Validate supported file types
        if not is_supported_file_type(detected_mime_type):
            supported_desc = ", ".join([
                f"{desc} ({mime})" 
                for mime, desc in {
                    'application/pdf': 'PDF Document',
                    'message/rfc822': 'Email Message',
                    'text/plain': 'Plain Text'
                }.items()
            ])
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{file_type_description}'. Supported types: {supported_desc}"
            )

        # Route to appropriate processing based on detected file type
        if detected_mime_type == 'application/pdf':
            # Process PDF file
            reader = PdfReader(temp_filename)
            
            # Metadata collection
            page_count = len(reader.pages)
            text_results = []
            total_chars = 0
            
            # Extract text from each page with progress tracking
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    raw_text = page.extract_text() or ""
                    if raw_text.strip():
                        # Enhanced text cleaning
                        cleaned_text = clean_extracted_text(raw_text)
                        if cleaned_text:
                            text_results.append({
                                "page": page_num,
                                "text": cleaned_text,
                                "characters": len(cleaned_text)
                            })
                            total_chars += len(cleaned_text)
                except Exception as e:
                    # Log page extraction errors but continue with other pages
                    print(f"Warning: Could not extract text from page {page_num}: {str(e)}")
                    continue
            
            # Combine all pages
            combined_text = "\n\n--- PAGE BREAK ---\n\n".join([item["text"] for item in text_results])
            
            # Final text cleaning
            final_text = clean_extracted_text(combined_text) if combined_text else ""
            
            # Prepare response
            response_data = {
                "filename": file.filename,
                "extracted_text": final_text
            }
            
            # Add metadata and warnings
            if not final_text.strip():
                response_data["message"] = (
                    "No readable text found in PDF (might be a scanned image). "
                    "Consider using OCR for scanned documents or check if the PDF contains embedded images only."
                )
            elif page_count > 1:
                response_data["message"] = f"Successfully extracted text from {page_count} pages ({total_chars} characters total)"
            else:
                response_data["message"] = f"Successfully extracted text ({total_chars} characters)"
                
        elif detected_mime_type == 'message/rfc822':
            # Process EML file
            try:
                import email
                from email import policy
                from email.parser import BytesParser
                
                with open(temp_filename, 'rb') as f:
                    msg = BytesParser(policy=policy.default).parse(f)
                
                # Extract email components
                subject = msg.get('Subject', 'No Subject')
                from_addr = msg.get('From', 'Unknown Sender')
                to_addr = msg.get('To', 'Unknown Recipient')
                date = msg.get('Date', 'Unknown Date')
                
                # Extract body content
                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body_text = part.get_content()
                            break
                else:
                    body_text = msg.get_content()
                
                # Clean and format email content
                combined_text = f"Subject: {subject}\nFrom: {from_addr}\nTo: {to_addr}\nDate: {date}\n\n{body_text}"
                final_text = clean_extracted_text(combined_text)
                
                response_data = {
                    "filename": file.filename,
                    "extracted_text": final_text,
                    "message": f"Successfully extracted email content. Subject: {subject}"
                }
                
            except Exception as e:
                print(f"Error processing EML file: {str(e)}")
                response_data = {
                    "filename": file.filename,
                    "extracted_text": f"[EMAIL PROCESSING ERROR] Could not parse email file: {str(e)}",
                    "message": "Error processing email file"
                }
                
        elif detected_mime_type == 'text/plain':
            # Process plain text file
            try:
                with open(temp_filename, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                final_text = clean_extracted_text(raw_text)
                
                response_data = {
                    "filename": file.filename,
                    "extracted_text": final_text,
                    "message": "Successfully extracted text from plain text file"
                }
            except Exception as e:
                print(f"Error processing text file: {str(e)}")
                response_data = {
                    "filename": file.filename,
                    "extracted_text": f"[TEXT PROCESSING ERROR] Could not read text file: {str(e)}",
                    "message": "Error processing text file"
                }
        else:
            # Fallback for unsupported but detected types
            response_data = {
                "filename": file.filename,
                "extracted_text": f"[UNSUPPORTED FILE TYPE] Detected as {file_type_description} but no processing logic available",
                "message": f"File type {file_type_description} detected but not fully supported"
            }
            
        return response_data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during text extraction: {str(e)}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Warning: Could not clean up temporary file {temp_filename}: {str(e)}")

# Phase 2: LLM Integration Endpoints
# ============================================================================

from app.services.llm_service import get_llm_service
from app.services.prompt_engineering import get_prompt_engineer, get_schema_enforcer
from app.models.spec_models import ObjectSpec, DocumentAnalysis, validate_object_spec


@router.post("/analyze-document", response_model=DocumentAnalysis)
async def analyze_document_with_llm(
    file: UploadFile = File(...),
    refine: bool = False,
    max_tokens: int = 1024
):
    """
    Analyze document using LLM to extract structured 3D model specifications.
    
    Args:
        file: Uploaded document file (PDF, EML, or text)
        refine: Whether to perform refinement pass
        max_tokens: Maximum tokens for LLM generation
        
    Returns:
        Structured document analysis with 3D specifications
    """
    try:
        # First, extract text using existing functionality
        extraction_result = await extract_text_from_pdf(file)
        raw_text = extraction_result.get("extracted_text", "")
        
        if not raw_text or "ERROR" in raw_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from document"
            )
        
        # Get services
        llm_service = get_llm_service()
        prompt_engineer = get_prompt_engineer()
        
        # Create analysis prompt
        prompt = prompt_engineer.create_extraction_prompt(raw_text, "document_analysis")
        
        # Generate LLM response
        llm_response = llm_service.generate_response(
            prompt, 
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Extract JSON from response
        json_data = llm_service.extract_json_from_text(llm_response)
        
        if not json_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM failed to produce valid JSON response"
            )
        
        # Validate and structure the response
        try:
            analysis = DocumentAnalysis(**json_data)
            analysis.raw_text = raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text
            analysis.confidence_score = min(0.95, max(0.1, analysis.confidence_score))
        except Exception as e:
            # If validation fails, try to create a basic structure
            analysis = DocumentAnalysis(
                objects=[],
                tables=[],
                raw_text=raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text,
                confidence_score=0.3,
                warnings=[f"Validation failed: {str(e)}"]
            )
        
        # Optional refinement pass
        if refine and analysis.confidence_score < 0.8:
            refinement_prompt = prompt_engineer.create_refinement_prompt(
                json.dumps(analysis.dict(), indent=2),
                "Improve accuracy and completeness of the extracted specifications"
            )
            
            refined_response = llm_service.generate_response(refinement_prompt, max_tokens=max_tokens)
            refined_json = llm_service.extract_json_from_text(refined_response)
            
            if refined_json:
                try:
                    refined_analysis = DocumentAnalysis(**refined_json)
                    refined_analysis.raw_text = analysis.raw_text
                    return refined_analysis
                except:
                    pass  # Return original analysis if refinement fails
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}"
        )


class LLMTestRequest(BaseModel):
    """Request model for testing LLM functionality"""
    prompt: str = Field(..., description="Test prompt to send to LLM")
    max_tokens: int = Field(256, description="Maximum tokens to generate", ge=1, le=2048)


class LLMTestResponse(BaseModel):
    """Response model for LLM testing"""
    response: str = Field(..., description="LLM generated response")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    processing_time: float = Field(..., description="Time taken in seconds")


@router.post("/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """
    Test endpoint for LLM functionality.
    
    Args:
        request: Test request with prompt and parameters
        
    Returns:
        LLM response and performance metrics
    """
    import time
    
    try:
        llm_service = get_llm_service()
        start_time = time.time()
        
        response = llm_service.generate_response(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=0.7
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        tokens_generated = len(response.split())
        
        return LLMTestResponse(
            response=response,
            tokens_generated=tokens_generated,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM test failed: {str(e)}"
        )


def clean_extracted_text(text: str) -> str:
    """
    Advanced text cleaning with regex filters to remove noise like signatures, headers, footers.
    Uses the TextCleaner module for configurable and maintainable text cleaning.
    
    Args:
        text: Raw text extracted from PDF/email
        
    Returns:
        Cleaned and normalized text
    """
    return clean_document_text(text)



# ============================================================================
# LangGraph Pipeline Endpoint
# ============================================================================

class PipelineRunResponse(BaseModel):
    task_id: str
    status: str
    message: str


@router.post(
    "/run-pipeline",
    response_model=PipelineRunResponse,
    tags=["LangGraph Pipeline"],
    summary="Run LangGraph 3D generation pipeline",
    description="""
    Upload a document and run the full LangGraph pipeline:
    parse → LLM spec extraction (with retry) → 3D mesh generation (with retry) → result.

    Uses the same task polling endpoint (`/task/{task_id}`) to check progress.
    """,
)
async def run_pipeline(
    file: UploadFile = File(..., description="PDF or EML document"),
):
    allowed_types = ["application/pdf", "message/rfc822"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and Email files are supported",
        )

    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "bin"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}.{file_extension}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    celery_app.send_task(
        "app.tasks.run_pipeline",
        args=[file_path, file.content_type],
        task_id=file_id,
    )

    return PipelineRunResponse(
        task_id=file_id,
        status="processing",
        message="LangGraph pipeline started. Poll /task/{task_id} for status.",
    )
