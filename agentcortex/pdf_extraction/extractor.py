"""PDF text extraction using multiple libraries for robustness."""

import io
import logging
from pathlib import Path
from typing import List, Optional, Union

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Robust PDF text extraction using multiple libraries.
    
    Tries PyMuPDF first, falls back to pdfplumber, then PyPDF2.
    """
    
    def __init__(self, preserve_layout: bool = True):
        """
        Initialize PDF extractor.
        
        Args:
            preserve_layout: Whether to attempt to preserve text layout
        """
        self.preserve_layout = preserve_layout
    
    def extract_text(self, pdf_path: Union[str, Path, bytes]) -> str:
        """
        Extract text from PDF using the most appropriate method.
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            
        Returns:
            Extracted text as string
            
        Raises:
            ValueError: If no text could be extracted
        """
        text = None
        
        # Try PyMuPDF first (fastest and most reliable)
        if PYMUPDF_AVAILABLE:
            try:
                text = self._extract_with_pymupdf(pdf_path)
                if text and text.strip():
                    logger.info("Successfully extracted text using PyMuPDF")
                    return text
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fall back to pdfplumber (better layout preservation)
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text and text.strip():
                    logger.info("Successfully extracted text using pdfplumber")
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Last resort: PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                text = self._extract_with_pypdf2(pdf_path)
                if text and text.strip():
                    logger.info("Successfully extracted text using PyPDF2")
                    return text
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        raise ValueError("No PDF extraction libraries available or could not extract text from PDF")
    
    def _extract_with_pymupdf(self, pdf_path: Union[str, Path, bytes]) -> str:
        """Extract text using PyMuPDF (fitz)."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) not available")
            
        if isinstance(pdf_path, bytes):
            doc = fitz.open(stream=pdf_path, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)
        
        text_blocks = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            if self.preserve_layout:
                # Get text with layout preservation
                text = page.get_text("text")
            else:
                # Get raw text
                text = page.get_text()
            
            if text.strip():
                text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        return "\n\n".join(text_blocks)
    
    def _extract_with_pdfplumber(self, pdf_path: Union[str, Path, bytes]) -> str:
        """Extract text using pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not available")
            
        if isinstance(pdf_path, bytes):
            pdf_file = io.BytesIO(pdf_path)
        else:
            pdf_file = pdf_path
        
        text_blocks = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_blocks)
    
    def _extract_with_pypdf2(self, pdf_path: Union[str, Path, bytes]) -> str:
        """Extract text using PyPDF2."""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 not available")
            
        if isinstance(pdf_path, bytes):
            pdf_file = io.BytesIO(pdf_path)
        else:
            pdf_file = open(pdf_path, 'rb')
        
        try:
            reader = PdfReader(pdf_file)
            text_blocks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_blocks.append(f"--- Page {page_num + 1} ---\n{text}")
            
            return "\n\n".join(text_blocks)
        finally:
            if not isinstance(pdf_path, bytes):
                pdf_file.close()
    
    def extract_pages(self, pdf_path: Union[str, Path, bytes]) -> List[str]:
        """
        Extract text from each page separately.
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            
        Returns:
            List of text strings, one per page
        """
        if not PYMUPDF_AVAILABLE:
            # Fall back to text-based extraction
            full_text = self.extract_text(pdf_path)
            # Simple page splitting (not ideal but works for demo)
            if "--- Page" in full_text:
                pages = full_text.split("--- Page")[1:]  # Skip first empty element
                return [f"Page {page}" for page in pages]
            else:
                return [full_text]
        
        if isinstance(pdf_path, bytes):
            doc = fitz.open(stream=pdf_path, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)
        
        pages = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text" if self.preserve_layout else None)
            pages.append(text)
        
        doc.close()
        return pages
    
    def get_pdf_info(self, pdf_path: Union[str, Path, bytes]) -> dict:
        """
        Get PDF metadata and basic information.
        
        Args:
            pdf_path: Path to PDF file or PDF bytes
            
        Returns:
            Dictionary with PDF information
        """
        if not PYMUPDF_AVAILABLE:
            # Return basic info without PyMuPDF
            return {
                "page_count": 1,
                "title": "",
                "author": "",
                "subject": "",
                "creator": "",
                "producer": "",
                "creation_date": "",
                "modification_date": "",
            }
            
        if isinstance(pdf_path, bytes):
            doc = fitz.open(stream=pdf_path, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)
        
        metadata = doc.metadata
        info = {
            "page_count": doc.page_count,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }
        
        doc.close()
        return info