#!/usr/bin/env python3
"""
PDF Processing Example

This example demonstrates how to:
1. Extract text from PDF files
2. Process and chunk the extracted text
3. Add to RAG pipeline
4. Query the processed documents

Usage:
    python pdf_processing_example.py [pdf_file1] [pdf_file2] ...
    
If no PDF files are provided, it will create a sample text file and demonstrate
text-based document processing instead.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcortex import RAGPipeline, RAGConfig
from agentcortex.extractors import PDFExtractor
from agentcortex.chunkers import TextChunker


def create_sample_pdf_content():
    """Create sample content to demonstrate PDF processing."""
    return """
    The Future of Artificial Intelligence in Healthcare
    
    Artificial Intelligence (AI) is revolutionizing healthcare by providing innovative 
    solutions for diagnosis, treatment, and patient care. This document explores the 
    current applications and future potential of AI in medical settings.
    
    Chapter 1: AI in Medical Diagnosis
    
    Machine learning algorithms are increasingly being used to analyze medical images, 
    including X-rays, MRIs, and CT scans. These systems can detect patterns that might 
    be missed by human radiologists, potentially leading to earlier and more accurate 
    diagnoses.
    
    Deep learning models, particularly convolutional neural networks (CNNs), have shown 
    remarkable success in image classification tasks. In dermatology, AI systems can 
    identify skin cancers with accuracy comparable to experienced dermatologists.
    
    Chapter 2: AI in Drug Discovery
    
    The pharmaceutical industry is leveraging AI to accelerate drug discovery and 
    development. Machine learning models can predict molecular behavior, identify 
    potential drug candidates, and optimize chemical compounds.
    
    Natural language processing (NLP) is being used to analyze vast amounts of 
    biomedical literature, helping researchers identify new therapeutic targets 
    and understand disease mechanisms.
    
    Chapter 3: Personalized Medicine
    
    AI enables personalized treatment plans by analyzing patient data, genetic 
    information, and medical history. This approach can improve treatment outcomes 
    and reduce adverse effects.
    
    Predictive analytics can help identify patients at risk of developing certain 
    conditions, enabling preventive interventions and early treatment.
    
    Chapter 4: Challenges and Ethical Considerations
    
    While AI offers tremendous potential in healthcare, there are important challenges 
    to address, including data privacy, algorithmic bias, and the need for transparent 
    and explainable AI systems.
    
    Regulatory frameworks must evolve to ensure the safe and effective deployment 
    of AI technologies in clinical settings.
    
    Conclusion
    
    The integration of AI in healthcare is transforming medical practice and improving 
    patient outcomes. Continued research and development, along with careful attention 
    to ethical considerations, will be crucial for realizing the full potential of AI 
    in medicine.
    """


def main():
    """Run PDF processing example."""
    print("=" * 60)
    print("PDF Processing Example")
    print("=" * 60)
    
    # Get PDF files from command line arguments
    pdf_files = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not pdf_files:
        print("No PDF files provided. Creating sample text document instead.")
        print("To use with actual PDFs, run: python pdf_processing_example.py file1.pdf file2.pdf")
        use_sample_text = True
    else:
        print(f"Processing {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        use_sample_text = False
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    
    # Initialize RAG pipeline
    config = RAGConfig()
    config.data_dir = "./pdf_example_data"
    config.faiss.index_path = "./pdf_example_index"
    config.chunker.chunk_size = 800  # Smaller chunks for better granularity
    config.chunker.chunk_overlap = 150
    
    pipeline = RAGPipeline(config)
    print("✓ RAG Pipeline initialized")
    
    # Initialize PDF extractor for demonstration
    pdf_extractor = PDFExtractor(backend="pdfplumber")
    print("✓ PDF Extractor initialized")
    
    # Initialize text chunker for demonstration
    text_chunker = TextChunker(
        chunk_size=config.chunker.chunk_size,
        chunk_overlap=config.chunker.chunk_overlap
    )
    print("✓ Text Chunker initialized")
    
    # 2. Process documents
    if use_sample_text:
        print("\n2. Processing sample text document...")
        
        # Create sample document
        sample_content = create_sample_pdf_content()
        
        # Demonstrate text chunking
        print("Chunking text...")
        documents = [{
            "text": sample_content,
            "source": "AI_Healthcare_Sample.txt",
            "document_type": "sample"
        }]
        
        chunks = text_chunker.chunk_documents(documents, method="recursive")
        print(f"✓ Created {len(chunks)} chunks from sample document")
        
        # Show chunk statistics
        chunk_stats = text_chunker.get_chunk_statistics(chunks)
        print(f"✓ Average chunk size: {chunk_stats['avg_char_count']:.0f} characters")
        print(f"✓ Chunk size range: {chunk_stats['min_char_count']}-{chunk_stats['max_char_count']}")
        
        # Add to pipeline
        stats = pipeline.add_documents_from_text(
            texts=[sample_content],
            sources=["AI_Healthcare_Sample"],
            chunking_method="recursive"
        )
        
    else:
        print("\n2. Processing PDF files...")
        
        # Check if files exist
        valid_pdfs = []
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                valid_pdfs.append(pdf_file)
                print(f"✓ Found: {pdf_file}")
            else:
                print(f"✗ Not found: {pdf_file}")
        
        if not valid_pdfs:
            print("No valid PDF files found. Exiting.")
            return
        
        # Extract text from PDFs (demonstration)
        print("\nExtracting text from PDFs...")
        documents = pdf_extractor.extract_multiple(valid_pdfs)
        
        successful_extractions = [doc for doc in documents if not doc.get("error")]
        failed_extractions = len(documents) - len(successful_extractions)
        
        print(f"✓ Successfully extracted text from {len(successful_extractions)} PDFs")
        if failed_extractions > 0:
            print(f"✗ Failed to extract text from {failed_extractions} PDFs")
        
        # Show extraction details
        for doc in successful_extractions[:3]:  # Show first 3
            source = doc.get("source", "Unknown")
            text_length = len(doc.get("text", ""))
            page_count = doc.get("page_count", 0)
            print(f"  - {source}: {text_length} characters, {page_count} pages")
        
        # Add to pipeline
        stats = pipeline.add_documents_from_pdfs(
            pdf_paths=valid_pdfs,
            chunking_method="recursive",
            filter_toxicity=True
        )
    
    print(f"\n✓ Pipeline processing complete:")
    print(f"  - Documents processed: {stats.get('text_documents_processed', stats.get('successful_extractions', 0))}")
    print(f"  - Total chunks created: {stats['total_chunks']}")
    print(f"  - Chunking method: {stats['chunking_method']}")
    
    # 3. Query the processed documents
    print("\n3. Querying processed documents...")
    
    queries = [
        "What is artificial intelligence in healthcare?",
        "How is AI used in medical diagnosis?",
        "What are the challenges of AI in medicine?",
        "Tell me about drug discovery and AI",
        "What is personalized medicine?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = pipeline.query(query, k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                source = result.get('source', 'Unknown')
                chunk_id = result.get('chunk_id', 'N/A')
                text_preview = result.get('text', '')[:150] + "..."
                
                print(f"  {i}. Source: {source} | Chunk: {chunk_id} | Similarity: {similarity:.3f}")
                print(f"     {text_preview}")
        else:
            print("  No relevant results found.")
    
    # 4. Demonstrate different chunking methods
    print("\n4. Comparing chunking methods...")
    
    if use_sample_text:
        sample_text = create_sample_pdf_content()
        
        methods = ["recursive", "fixed", "sentence"]
        for method in methods:
            chunks = text_chunker.chunk_text(sample_text, method=method)
            stats = text_chunker.get_chunk_statistics(chunks)
            
            print(f"  {method.title()} chunking:")
            print(f"    - Chunks: {len(chunks)}")
            print(f"    - Avg size: {stats['avg_char_count']:.0f} chars")
            print(f"    - Size range: {stats['min_char_count']}-{stats['max_char_count']}")
    
    # 5. Save pipeline
    print("\n5. Saving pipeline...")
    pipeline.save_pipeline()
    print("✓ Pipeline saved successfully")
    
    # 6. Show final statistics
    print("\n6. Final Pipeline Statistics:")
    print("-" * 40)
    
    pipeline_stats = pipeline.get_pipeline_stats()
    vector_stats = pipeline_stats['vector_storage']
    
    print(f"Documents in index: {vector_stats['total_documents']}")
    print(f"Embedding model: {vector_stats['embedding_model']}")
    print(f"Index type: {vector_stats['index_type']}")
    print(f"Toxicity filtering: {pipeline_stats['toxicity_filter_enabled']}")
    
    print("\n" + "=" * 60)
    print("PDF Processing Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()