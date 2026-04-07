"""
Document parsing service using the unstructured library.
Handles PDF and email (.eml) document parsing with metadata extraction.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from unstructured.staging.base import convert_to_isd
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    """Service for parsing documents using unstructured library"""
    
    def __init__(self):
        self.supported_types = {
            'application/pdf': self._parse_pdf,
            'message/rfc822': self._parse_email
        }
    
    def parse_document(self, file_path: str, content_type: str) -> Dict[str, Any]:
        """
        Parse document and extract structured data.
        
        Args:
            file_path: Path to the document file
            content_type: MIME type of the document
            
        Returns:
            Dictionary containing parsed content and metadata
        """
        if content_type not in self.supported_types:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        try:
            # Parse the document
            elements = self.supported_types[content_type](file_path)
            
            # Extract text content
            text_content = self._extract_text(elements)
            
            # Extract metadata
            metadata = self._extract_metadata(elements, file_path)
            
            # Clean and structure the content
            cleaned_text = self._clean_text(text_content)
            
            return {
                "content": cleaned_text,
                "metadata": metadata,
                "elements": [{"text": getattr(e, 'text', ''), "category": getattr(e, 'category', 'Unknown')} for e in elements],
                "file_type": content_type,
                "file_name": os.path.basename(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    def _parse_pdf(self, file_path: str) -> List[Any]:
        """Parse PDF document using unstructured"""
        logger.info(f"Parsing PDF document: {file_path}")
        elements = partition(filename=file_path, strategy="hi_res")
        return elements
    
    def _parse_email(self, file_path: str) -> List[Any]:
        """Parse email document using unstructured"""
        logger.info(f"Parsing email document: {file_path}")
        elements = partition(filename=file_path)
        return elements
    
    def _extract_text(self, elements: List[Any]) -> str:
        """Extract text content from parsed elements"""
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_parts.append(str(element.text))
        
        return '\n'.join(text_parts)
    
    def _extract_metadata(self, elements: List[Any], file_path: str) -> Dict[str, Any]:
        """Extract metadata from parsed elements"""
        metadata = {
            "file_size": os.path.getsize(file_path),
            "file_name": os.path.basename(file_path),
            "element_count": len(elements),
            "categories": {},
            "potential_dimensions": [],
            "potential_materials": []
        }
        
        # Count element categories
        for element in elements:
            category = getattr(element, 'category', 'Unknown')
            metadata["categories"][category] = metadata["categories"].get(category, 0) + 1
        
        # Extract potential technical information
        full_text = self._extract_text(elements).lower()
        metadata["potential_dimensions"] = self._extract_dimensions(full_text)
        metadata["potential_materials"] = self._extract_materials(full_text)
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = clean_extra_whitespace(text)
        
        # Group broken paragraphs
        text = group_broken_paragraphs(text)
        
        # Additional cleaning
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_dimensions(self, text: str) -> List[str]:
        """Extract potential dimension information from text"""
        import re
        
        # Common dimension patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]?\s*(\d*(?:\.\d+)?)\s*(mm|cm|m|inch|in|ft)',
            r'(length|width|height|depth|diameter)\s*:?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in|ft)',
            r'(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in|ft)\s*(length|width|height|depth|diameter)'
        ]
        
        dimensions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                dimensions.append(' '.join(match))
        
        return list(set(dimensions))  # Remove duplicates
    
    def _extract_materials(self, text: str) -> List[str]:
        """Extract potential material information from text"""
        # Common material keywords
        material_keywords = [
            'steel', 'aluminum', 'plastic', 'wood', 'glass', 'carbon fiber',
            'titanium', 'copper', 'brass', 'bronze', 'rubber', 'silicone',
            'polypropylene', 'abs', 'nylon', 'polycarbonate'
        ]
        
        materials = []
        text_lower = text.lower()
        
        for material in material_keywords:
            if material in text_lower:
                materials.append(material.title())
        
        return list(set(materials))  # Remove duplicates

# Global instance
document_parser = DocumentParser()