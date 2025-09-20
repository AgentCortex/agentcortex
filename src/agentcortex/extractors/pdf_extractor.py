"""PDF text extraction module."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files using multiple backends."""
    
    def __init__(self, backend: str = "pdfplumber"):
        """
        Initialize PDF extractor.
        
        Args:
            backend: Backend to use for extraction ('pdfplumber' or 'pypdf2')
        """
        self.backend = backend
        if backend not in ["pdfplumber", "pypdf2"]:
            raise ValueError("Backend must be 'pdfplumber' or 'pypdf2'")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.backend == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)
    
    def extract_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF content as bytes
            
        Returns:
            Extracted text as string
        """
        if self.backend == "pdfplumber":
            return self._extract_bytes_with_pdfplumber(pdf_bytes)
        else:
            return self._extract_bytes_with_pypdf2(pdf_bytes)
    
    def extract_with_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = self.extract_text(pdf_path)
        
        if self.backend == "pdfplumber":
            return self._extract_metadata_pdfplumber(pdf_path, text)
        else:
            return self._extract_metadata_pypdf2(pdf_path, text)
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        else:
                            logger.warning(f"No text found on page {page_num}")
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error opening PDF with pdfplumber: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        else:
                            logger.warning(f"No text found on page {page_num}")
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error opening PDF with PyPDF2: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_bytes_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text from bytes using pdfplumber."""
        text_parts = []
        
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error opening PDF bytes with pdfplumber: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_bytes_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text from bytes using PyPDF2."""
        text_parts = []
        
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.error(f"Error extracting page {page_num}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error opening PDF bytes with PyPDF2: {e}")
            raise
        
        return "\n\n".join(text_parts)
    
    def _extract_metadata_pdfplumber(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """Extract metadata using pdfplumber."""
        metadata = {
            "text": text,
            "source": str(pdf_path),
            "backend": self.backend,
            "page_count": 0,
            "metadata": {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["page_count"] = len(pdf.pages)
                if pdf.metadata:
                    metadata["metadata"] = pdf.metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_metadata_pypdf2(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """Extract metadata using PyPDF2."""
        metadata = {
            "text": text,
            "source": str(pdf_path),
            "backend": self.backend,
            "page_count": 0,
            "metadata": {}
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata["page_count"] = len(reader.pages)
                if reader.metadata:
                    # Convert metadata to dict
                    metadata["metadata"] = dict(reader.metadata)
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def extract_multiple(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of dictionaries with text and metadata
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.extract_with_metadata(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract from {pdf_path}: {e}")
                # Add error result
                results.append({
                    "text": "",
                    "source": pdf_path,
                    "backend": self.backend,
                    "error": str(e)
                })
        
        return results