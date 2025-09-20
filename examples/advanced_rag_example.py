#!/usr/bin/env python3
"""
Advanced RAG Pipeline Example

This example demonstrates advanced features:
1. Custom configuration
2. Language detection and filtering
3. Toxicity filtering
4. Multiple chunking strategies
5. Batch processing

Usage:
    python advanced_rag_example.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcortex import RAGPipeline, RAGConfig
from agentcortex.filters import ToxicityFilter
from agentcortex.utils import LangExtractUtils


def main():
    """Run advanced RAG pipeline example."""
    print("=" * 60)
    print("Advanced RAG Pipeline Example")
    print("=" * 60)
    
    # 1. Load custom configuration
    print("\n1. Loading custom configuration...")
    
    config_path = Path(__file__).parent / "sample_config.json"
    if config_path.exists():
        config = RAGConfig.from_file(str(config_path))
        print("✓ Loaded configuration from sample_config.json")
    else:
        config = RAGConfig()
        print("✓ Using default configuration")
    
    # Customize for this example
    config.data_dir = "./advanced_example_data"
    config.faiss.index_path = "./advanced_example_index"
    config.chunker.chunk_size = 600
    config.toxicity.enabled = True
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    print("✓ Advanced RAG Pipeline initialized")
    
    # 2. Demonstrate language detection
    print("\n2. Language Detection Example...")
    
    lang_utils = LangExtractUtils()
    
    multilingual_texts = [
        "This is an English text about artificial intelligence and machine learning.",
        "Este es un texto en español sobre inteligencia artificial y aprendizaje automático.",
        "Ceci est un texte français sur l'intelligence artificielle et l'apprentissage automatique.",
        "Dies ist ein deutscher Text über künstliche Intelligenz und maschinelles Lernen.",
        "This text contains some problematic content that might be toxic or harmful.",
        "Mixed language text. Este texto tiene múltiples idiomas. C'est un mélange."
    ]
    
    print("Detecting languages...")
    for i, text in enumerate(multilingual_texts, 1):
        lang_info = lang_utils.detect_language(text)
        print(f"  {i}. Language: {lang_info['language_name']} "
              f"(confidence: {lang_info['confidence']:.3f})")
        print(f"     Text: {text[:60]}...")
    
    # Filter for English only
    print("\nFiltering for English documents...")
    documents = [{"text": text, "source": f"doc_{i}"} for i, text in enumerate(multilingual_texts)]
    english_docs = lang_utils.filter_by_language_quality(
        documents, 
        min_confidence=0.8, 
        allowed_languages=['en']
    )
    
    print(f"✓ Filtered to {len(english_docs)} English documents out of {len(documents)} total")
    
    # 3. Demonstrate toxicity filtering
    print("\n3. Toxicity Filtering Example...")
    
    toxicity_filter = ToxicityFilter(threshold=0.6)
    
    test_texts = [
        "Artificial intelligence is a fascinating field of computer science.",
        "Machine learning algorithms can help solve complex problems.",
        "This is completely stupid and useless garbage content.",
        "Deep learning networks are powerful tools for pattern recognition.",
        "I hate this technology, it's going to destroy everything!",
        "Natural language processing enables computers to understand human language."
    ]
    
    print("Checking toxicity...")
    toxicity_results = toxicity_filter.batch_check_toxicity(test_texts)
    
    for result in toxicity_results:
        status = "🔴 TOXIC" if result['is_toxic'] else "🟢 CLEAN"
        score = result['toxicity_score']
        text_preview = result['text'][:50] + "..."
        print(f"  {status} (score: {score:.3f}): {text_preview}")
    
    # Filter out toxic content
    clean_texts = [result['text'] for result in toxicity_results if not result['is_toxic']]
    print(f"✓ Filtered to {len(clean_texts)} clean texts out of {len(test_texts)} total")
    
    # 4. Add documents with different chunking strategies
    print("\n4. Testing Multiple Chunking Strategies...")
    
    sample_documents = [
        """
        Chapter 1: Introduction to Neural Networks
        
        Neural networks are computing systems inspired by biological neural networks. 
        They consist of interconnected nodes (neurons) that process information using 
        a connectionist approach to computation. The connections between neurons have 
        weights that can be adjusted based on experience, making neural networks 
        adaptive and capable of learning.
        
        The basic building block of a neural network is the artificial neuron, also 
        known as a perceptron. Each neuron receives inputs, applies weights to these 
        inputs, sums them up, and passes the result through an activation function 
        to produce an output.
        """,
        """
        Chapter 2: Deep Learning Architectures
        
        Deep learning is a subset of machine learning that uses neural networks with 
        multiple layers (hence "deep") to model and understand complex patterns in data. 
        The depth of these networks allows them to learn hierarchical representations 
        of data, with each layer learning increasingly abstract features.
        
        Common deep learning architectures include:
        - Convolutional Neural Networks (CNNs) for image processing
        - Recurrent Neural Networks (RNNs) for sequential data
        - Transformer models for natural language processing
        - Generative Adversarial Networks (GANs) for data generation
        """
    ]
    
    chunking_methods = ["recursive", "sentence", "fixed"]
    
    for method in chunking_methods:
        print(f"\nTesting {method} chunking...")
        
        # Create temporary pipeline for this method
        temp_config = config.model_copy()
        temp_config.faiss.index_path = f"./temp_{method}_index"
        temp_pipeline = RAGPipeline(temp_config)
        
        # Add documents
        stats = temp_pipeline.add_documents_from_text(
            texts=sample_documents,
            sources=[f"Neural_Networks_Ch{i+1}" for i in range(len(sample_documents))],
            chunking_method=method,
            filter_toxicity=True
        )
        
        print(f"  ✓ Created {stats['total_chunks']} chunks")
        print(f"  ✓ Average chunk size: {stats['avg_char_count']:.0f} characters")
        
        # Test query
        results = temp_pipeline.query("What are neural networks?", k=2)
        print(f"  ✓ Query returned {len(results)} results")
        
        if results:
            best_result = results[0]
            similarity = best_result.get('similarity', 0)
            print(f"    Best match similarity: {similarity:.3f}")
    
    # 5. Demonstrate batch processing
    print("\n5. Batch Processing Example...")
    
    # Use the main pipeline for batch processing
    batch_texts = [
        "Transformer models have revolutionized natural language processing.",
        "Computer vision systems can now recognize objects with high accuracy.",
        "Reinforcement learning enables agents to learn through interaction.",
        "Generative models can create realistic images and text.",
        "Edge computing brings AI processing closer to data sources."
    ]
    
    # Add all texts at once
    print("Adding batch of documents...")
    stats = pipeline.add_documents_from_text(
        texts=batch_texts,
        sources=[f"AI_Topic_{i+1}" for i in range(len(batch_texts))],
        chunking_method="recursive"
    )
    
    print(f"✓ Processed {stats['text_documents_processed']} documents")
    print(f"✓ Created {stats['total_chunks']} chunks")
    
    # Batch queries
    batch_queries = [
        "How do transformers work?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What are generative models?",
        "What is edge computing?"
    ]
    
    print("\nRunning batch queries...")
    for query in batch_queries:
        results = pipeline.query(query, k=1)
        if results:
            best_match = results[0]
            similarity = best_match.get('similarity', 0)
            source = best_match.get('source', 'Unknown')
            print(f"  '{query}' → {source} (similarity: {similarity:.3f})")
    
    # 6. Advanced statistics and analysis
    print("\n6. Advanced Pipeline Analysis...")
    
    # Get comprehensive statistics
    pipeline_stats = pipeline.get_pipeline_stats()
    
    print("Configuration Summary:")
    config_summary = pipeline_stats['config']
    print(f"  - Embedding model: {config_summary['embedding']['model_name']}")
    print(f"  - Chunk size: {config_summary['chunker']['chunk_size']}")
    print(f"  - Chunk overlap: {config_summary['chunker']['chunk_overlap']}")
    print(f"  - Index type: {config_summary['faiss']['index_type']}")
    print(f"  - Toxicity filtering: {config_summary['toxicity']['enabled']}")
    
    print("\nVector Storage Statistics:")
    vector_stats = pipeline_stats['vector_storage']
    print(f"  - Total documents: {vector_stats['total_documents']}")
    print(f"  - Vector dimension: {vector_stats['dimension']}")
    print(f"  - Embedding device: {vector_stats['device']}")
    
    # Language analysis of stored documents
    print("\nStored Document Analysis...")
    if vector_stats['total_documents'] > 0:
        # This is a simplified analysis - in practice you'd need to access the stored texts
        print(f"  - Documents successfully indexed: {vector_stats['total_documents']}")
        print(f"  - Metadata entries: {vector_stats['metadata_count']}")
    
    # 7. Save and cleanup
    print("\n7. Saving pipeline...")
    pipeline.save_pipeline()
    print("✓ Pipeline state saved")
    
    # Show where files were saved
    print(f"\nFiles saved to:")
    print(f"  - Index: {config.faiss.index_path}")
    print(f"  - Data: {config.data_dir}")
    print(f"  - Config: {config.data_dir}/pipeline_config.json")
    
    print("\n" + "=" * 60)
    print("Advanced RAG Pipeline Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()