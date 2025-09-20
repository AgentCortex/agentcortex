#!/usr/bin/env python3
"""
PDF Processing Example

This example demonstrates how to:
1. Extract text from PDF files using multiple libraries
2. Chunk the extracted text efficiently
3. Handle PDF metadata and page information
4. Process multiple PDFs in batch
"""

import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import agentcortex
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentcortex.pdf_extraction import PDFExtractor
from agentcortex.text_chunking import TextChunker
from agentcortex.utils import setup_logging

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def create_sample_pdf():
    """Create a sample PDF for demonstration."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a simple PDF
        filename = "/tmp/sample_document.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        
        # Page 1
        c.drawString(100, 750, "Sample Document for AgentCortex")
        c.drawString(100, 720, "Page 1: Introduction")
        c.drawString(100, 690, "")
        c.drawString(100, 660, "This is a sample PDF document created for testing the AgentCortex")
        c.drawString(100, 630, "PDF extraction capabilities. The document contains multiple pages")
        c.drawString(100, 600, "with different types of content to demonstrate the robustness")
        c.drawString(100, 570, "of the text extraction pipeline.")
        c.drawString(100, 540, "")
        c.drawString(100, 510, "Machine learning is a powerful tool for data analysis and")
        c.drawString(100, 480, "pattern recognition. It enables computers to learn from data")
        c.drawString(100, 450, "without being explicitly programmed for every task.")
        
        c.showPage()
        
        # Page 2
        c.drawString(100, 750, "Page 2: Technical Content")
        c.drawString(100, 720, "")
        c.drawString(100, 690, "Vector databases are essential for modern AI applications.")
        c.drawString(100, 660, "They store high-dimensional vectors representing data points")
        c.drawString(100, 630, "and enable efficient similarity searches.")
        c.drawString(100, 600, "")
        c.drawString(100, 570, "FAISS (Facebook AI Similarity Search) is a library for")
        c.drawString(100, 540, "efficient similarity search and clustering of dense vectors.")
        c.drawString(100, 510, "It provides several indexing methods for different use cases:")
        c.drawString(100, 480, "- IndexFlatL2: Exact search using L2 distance")
        c.drawString(100, 450, "- IndexIVFFlat: Inverted file with exact post-verification")
        c.drawString(100, 420, "- IndexHNSW: Hierarchical Navigable Small World graphs")
        
        c.showPage()
        
        # Page 3
        c.drawString(100, 750, "Page 3: Applications")
        c.drawString(100, 720, "")
        c.drawString(100, 690, "Retrieval-Augmented Generation (RAG) combines the power of")
        c.drawString(100, 660, "large language models with external knowledge retrieval.")
        c.drawString(100, 630, "This approach helps overcome limitations like:")
        c.drawString(100, 600, "- Knowledge cutoff dates")
        c.drawString(100, 570, "- Hallucination in generated content")
        c.drawString(100, 540, "- Domain-specific knowledge gaps")
        c.drawString(100, 510, "")
        c.drawString(100, 480, "RAG systems typically follow these steps:")
        c.drawString(100, 450, "1. Index documents in a vector database")
        c.drawString(100, 420, "2. Retrieve relevant documents for a query")
        c.drawString(100, 390, "3. Generate responses using retrieved context")
        
        c.save()
        
        logger.info(f"Created sample PDF: {filename}")
        return filename
        
    except ImportError:
        logger.warning("reportlab not available, creating a text file instead")
        filename = "/tmp/sample_document.txt"
        with open(filename, 'w') as f:
            f.write("""Sample Document for AgentCortex
Page 1: Introduction

This is a sample document created for testing the AgentCortex
text processing capabilities. The document contains multiple sections
with different types of content to demonstrate the robustness
of the text processing pipeline.

Machine learning is a powerful tool for data analysis and
pattern recognition. It enables computers to learn from data
without being explicitly programmed for every task.

Page 2: Technical Content

Vector databases are essential for modern AI applications.
They store high-dimensional vectors representing data points
and enable efficient similarity searches.

FAISS (Facebook AI Similarity Search) is a library for
efficient similarity search and clustering of dense vectors.
It provides several indexing methods for different use cases:
- IndexFlatL2: Exact search using L2 distance
- IndexIVFFlat: Inverted file with exact post-verification
- IndexHNSW: Hierarchical Navigable Small World graphs

Page 3: Applications

Retrieval-Augmented Generation (RAG) combines the power of
large language models with external knowledge retrieval.
This approach helps overcome limitations like:
- Knowledge cutoff dates
- Hallucination in generated content
- Domain-specific knowledge gaps

RAG systems typically follow these steps:
1. Index documents in a vector database
2. Retrieve relevant documents for a query
3. Generate responses using retrieved context
""")
        return filename


def main():
    """Run the PDF processing example."""
    logger.info("Starting PDF Processing Example")
    
    # Create a sample document
    sample_file = create_sample_pdf()
    
    if sample_file.endswith('.txt'):
        logger.info("Working with text file (PDF libraries not available)")
        # Read text file directly
        with open(sample_file, 'r') as f:
            text_content = f.read()
        
        # Simulate PDF info
        pdf_info = {
            "page_count": 3,
            "title": "Sample Document",
            "author": "AgentCortex",
            "creation_date": "2024-01-01"
        }
        
        pages = text_content.split("Page ")[1:]  # Split by page markers
        pages = [f"Page {page}" for page in pages]
        
    else:
        # Step 1: Initialize PDF extractor
        logger.info("Initializing PDF extractor...")
        extractor = PDFExtractor(preserve_layout=True)
        
        # Step 2: Extract PDF information
        logger.info("Extracting PDF information...")
        pdf_info = extractor.get_pdf_info(sample_file)
        logger.info("PDF Information:")
        for key, value in pdf_info.items():
            logger.info(f"  {key}: {value}")
        
        # Step 3: Extract text from all pages
        logger.info("Extracting text from PDF...")
        text_content = extractor.extract_text(sample_file)
        logger.info(f"Extracted {len(text_content)} characters of text")
        
        # Step 4: Extract pages separately
        logger.info("Extracting pages separately...")
        pages = extractor.extract_pages(sample_file)
        logger.info(f"Extracted {len(pages)} pages")
        
        for i, page_text in enumerate(pages, 1):
            logger.info(f"Page {i}: {len(page_text)} characters")
            logger.info(f"  Preview: {page_text[:100]}...")
    
    # Step 5: Chunk the text using different strategies
    logger.info("\nTesting different chunking strategies...")
    
    chunking_strategies = [
        ("sentence_aware", 300, 50),
        ("fixed_size", 400, 80),
        ("token_based", 200, 40)
    ]
    
    for strategy, chunk_size, overlap in chunking_strategies:
        logger.info(f"\nChunking with {strategy} strategy:")
        
        chunker = TextChunker(
            chunk_size=chunk_size,
            overlap_size=overlap,
            strategy=strategy
        )
        
        # Chunk the full text
        chunks = chunker.chunk_text(text_content)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Show chunk statistics
        chunk_sizes = [metadata.char_count for _, metadata in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        logger.info(f"  Average chunk size: {avg_size:.1f} characters")
        
        # Show first chunk as example
        if chunks:
            first_chunk, first_metadata = chunks[0]
            logger.info(f"  First chunk preview: {first_chunk[:100]}...")
            logger.info(f"  First chunk metadata: {vars(first_metadata)}")
    
    # Step 6: Process multiple pages with metadata
    logger.info("\nProcessing pages with metadata...")
    chunker = TextChunker(chunk_size=250, overlap_size=25, strategy="sentence_aware")
    
    page_metadata = [
        {"page_number": i + 1, "document": "sample_doc", "section": f"page_{i+1}"}
        for i in range(len(pages))
    ]
    
    all_chunks = chunker.chunk_multiple_texts(
        texts=pages,
        source_pages=list(range(1, len(pages) + 1))
    )
    
    logger.info(f"Total chunks from all pages: {len(all_chunks)}")
    
    # Group chunks by page
    chunks_by_page = {}
    for chunk_text, metadata in all_chunks:
        page_num = metadata.source_page
        if page_num not in chunks_by_page:
            chunks_by_page[page_num] = []
        chunks_by_page[page_num].append((chunk_text, metadata))
    
    for page_num, page_chunks in chunks_by_page.items():
        logger.info(f"  Page {page_num}: {len(page_chunks)} chunks")
    
    # Step 7: Demonstrate error handling
    logger.info("\nTesting error handling...")
    try:
        # Try to extract from non-existent file
        fake_extractor = PDFExtractor()
        fake_extractor.extract_text("/tmp/nonexistent.pdf")
    except Exception as e:
        logger.info(f"  Handled error gracefully: {e}")
    
    # Clean up
    if os.path.exists(sample_file):
        os.remove(sample_file)
        logger.info(f"Cleaned up sample file: {sample_file}")
    
    logger.info("PDF Processing Example completed successfully!")


if __name__ == "__main__":
    main()