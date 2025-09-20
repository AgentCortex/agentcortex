#!/usr/bin/env python3
"""
Basic RAG Pipeline Example

This example demonstrates how to:
1. Initialize the RAG pipeline
2. Add text documents
3. Query for relevant information
4. Get pipeline statistics

Usage:
    python basic_rag_example.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcortex import RAGPipeline, RAGConfig


def main():
    """Run basic RAG pipeline example."""
    print("=" * 60)
    print("Basic RAG Pipeline Example")
    print("=" * 60)
    
    # 1. Initialize the RAG pipeline with default configuration
    print("\n1. Initializing RAG Pipeline...")
    config = RAGConfig()
    config.data_dir = "./example_data"
    config.faiss.index_path = "./example_faiss_index"
    
    pipeline = RAGPipeline(config)
    print("✓ RAG Pipeline initialized successfully")
    
    # 2. Add some sample documents
    print("\n2. Adding sample documents...")
    sample_texts = [
        """
        Artificial Intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals.
        """,
        """
        Machine Learning is a subset of artificial intelligence that focuses on 
        algorithms that can learn from and make predictions or decisions based on data. 
        Instead of being explicitly programmed to perform a task, machine learning 
        algorithms build mathematical models based on training data.
        """,
        """
        Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers 
        and human language. It involves programming computers to process and analyze 
        large amounts of natural language data.
        """,
        """
        Computer Vision is a field of artificial intelligence that trains computers 
        to interpret and understand the visual world. Using digital images from 
        cameras and videos and deep learning models, machines can accurately identify 
        and classify objects.
        """,
        """
        Deep Learning is part of a broader family of machine learning methods based 
        on artificial neural networks with representation learning. The adjective "deep" 
        refers to the use of multiple layers in the network.
        """
    ]
    
    sources = [
        "AI_Overview",
        "ML_Basics", 
        "NLP_Introduction",
        "Computer_Vision_Guide",
        "Deep_Learning_Fundamentals"
    ]
    
    # Add documents to the pipeline
    stats = pipeline.add_documents_from_text(
        texts=sample_texts,
        sources=sources,
        chunking_method="recursive",
        filter_toxicity=True
    )
    
    print(f"✓ Added {stats['text_documents_processed']} documents")
    print(f"✓ Created {stats['total_chunks']} chunks")
    print(f"✓ Average chunk size: {stats['avg_char_count']:.0f} characters")
    
    # 3. Query the pipeline
    print("\n3. Querying the pipeline...")
    
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is the difference between AI and ML?",
        "Tell me about deep learning",
        "What is computer vision used for?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = pipeline.query(query, k=3)
        
        print(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            source = result.get('source', 'Unknown')
            text_preview = result.get('text', '')[:100] + "..."
            
            print(f"  {i}. Source: {source} (similarity: {similarity:.3f})")
            print(f"     Preview: {text_preview}")
    
    # 4. Get pipeline statistics
    print("\n4. Pipeline Statistics:")
    print("-" * 40)
    
    pipeline_stats = pipeline.get_pipeline_stats()
    vector_stats = pipeline_stats['vector_storage']
    
    print(f"Total documents in index: {vector_stats['total_documents']}")
    print(f"Embedding model: {vector_stats['embedding_model']}")
    print(f"Vector dimension: {vector_stats['dimension']}")
    print(f"Index type: {vector_stats['index_type']}")
    print(f"Toxicity filtering: {pipeline_stats['toxicity_filter_enabled']}")
    
    # 5. Save pipeline state
    print("\n5. Saving pipeline state...")
    pipeline.save_pipeline()
    print("✓ Pipeline state saved successfully")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()