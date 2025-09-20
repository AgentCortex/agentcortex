#!/usr/bin/env python3
"""
Complete RAG Pipeline Example

This example demonstrates a full end-to-end RAG pipeline including:
1. PDF text extraction
2. Text chunking with multiple strategies
3. Toxicity filtering for content cleaning
4. Language detection and filtering
5. Vector storage and similarity search
6. Hugging Face model integration
7. Complete retrieval system with evaluation
"""

import logging
import sys
from pathlib import Path
import json

# Add the parent directory to the path so we can import agentcortex
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentcortex.pdf_extraction import PDFExtractor
from agentcortex.text_chunking import TextChunker
from agentcortex.vector_storage import VectorStore
from agentcortex.retrieval import RAGRetriever
from agentcortex.toxicity_filter import ToxicityFilter
from agentcortex.langextract import LanguageExtractor
from agentcortex.huggingface_tools import EmbeddingGenerator, ModelEvaluator
from agentcortex.utils import setup_logging, Config

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a diverse sample dataset for testing."""
    return [
        {
            "text": """
            Machine Learning Fundamentals
            
            Machine learning is a method of data analysis that automates analytical model building. 
            It is a branch of artificial intelligence (AI) based on the idea that systems can learn 
            from data, identify patterns and make decisions with minimal human intervention.
            
            There are three main types of machine learning:
            1. Supervised Learning: Learning with labeled examples
            2. Unsupervised Learning: Finding patterns in unlabeled data  
            3. Reinforcement Learning: Learning through interaction and feedback
            
            Popular algorithms include linear regression, decision trees, random forests, 
            support vector machines, and neural networks.
            """,
            "metadata": {
                "topic": "machine_learning",
                "difficulty": "beginner",
                "source": "ml_textbook",
                "language": "en"
            }
        },
        {
            "text": """
            Deep Learning and Neural Networks
            
            Deep learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers to model and understand complex patterns in data. These networks 
            are inspired by the human brain's structure and function.
            
            Key architectures include:
            - Feedforward Neural Networks: Basic multilayer perceptrons
            - Convolutional Neural Networks (CNNs): For image processing
            - Recurrent Neural Networks (RNNs): For sequential data
            - Transformers: For natural language processing
            
            Deep learning has revolutionized fields like computer vision, natural language 
            processing, speech recognition, and game playing.
            """,
            "metadata": {
                "topic": "deep_learning",
                "difficulty": "intermediate",
                "source": "dl_guide",
                "language": "en"
            }
        },
        {
            "text": """
            Vector Databases and Similarity Search
            
            Vector databases are specialized databases designed to store and query 
            high-dimensional vector data efficiently. They are essential for applications 
            like similarity search, recommendation systems, and retrieval-augmented generation.
            
            Popular vector databases include:
            - FAISS: Facebook's library for efficient similarity search
            - Pinecone: Managed vector database service
            - Weaviate: Open-source vector database
            - Chroma: Simple vector database for AI applications
            
            These systems use techniques like approximate nearest neighbor (ANN) search 
            to handle millions of vectors efficiently.
            """,
            "metadata": {
                "topic": "vector_databases",
                "difficulty": "advanced",
                "source": "vector_db_paper",
                "language": "en"
            }
        },
        {
            "text": """
            El Procesamiento de Lenguaje Natural
            
            El procesamiento de lenguaje natural (PLN) es un subcampo de la lingüística, 
            las ciencias de la computación y la inteligencia artificial que se ocupa de 
            las interacciones entre las computadoras y el lenguaje humano.
            
            Las aplicaciones del PLN incluyen:
            - Traducción automática
            - Análisis de sentimientos
            - Resumen de texto
            - Sistemas de pregunta-respuesta
            
            Los modelos modernos de PLN utilizan arquitecturas de transformadores 
            como BERT, GPT y T5 para lograr resultados de vanguardia.
            """,
            "metadata": {
                "topic": "nlp",
                "difficulty": "intermediate",
                "source": "nlp_spanish_guide",
                "language": "es"
            }
        },
        {
            "text": """
            This text contains some inappropriate content and stupid remarks about AI technology. 
            The author is completely wrong about everything and their approach is terrible. 
            This kind of bad content should be filtered out from training data.
            
            However, it also contains some useful information about retrieval systems 
            and how they can be improved with better algorithms.
            """,
            "metadata": {
                "topic": "mixed_content",
                "difficulty": "beginner",
                "source": "problematic_source",
                "language": "en"
            }
        }
    ]


def main():
    """Run the complete RAG pipeline example."""
    logger.info("Starting Complete RAG Pipeline Example")
    
    # Step 1: Initialize configuration
    logger.info("Setting up configuration...")
    config = Config()
    config.load_from_dict({
        "rag": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "chunking_strategy": "sentence_aware",
            "vector_store_type": "flat",
            "toxicity_threshold": 0.6,
            "use_toxicity_filter": True
        }
    })
    
    rag_config = config.get_rag_config()
    logger.info(f"Using configuration: {vars(rag_config)}")
    
    # Step 2: Create sample dataset
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset()
    texts = [item["text"] for item in dataset]
    metadata_list = [item["metadata"] for item in dataset]
    
    logger.info(f"Created dataset with {len(texts)} documents")
    
    # Step 3: Language detection and filtering
    logger.info("Performing language detection...")
    lang_extractor = LanguageExtractor()
    
    language_results = lang_extractor.detect_languages_batch(texts)
    
    # Filter for English content only (for this example)
    english_texts = []
    english_metadata = []
    english_indices = []
    
    for i, (text, metadata, lang_result) in enumerate(zip(texts, metadata_list, language_results)):
        logger.info(f"Document {i+1}: Detected language = {lang_result['language']} "
                   f"(confidence: {lang_result['confidence']:.3f})")
        
        if lang_result['language'] == 'en' and lang_result['confidence'] > 0.7:
            english_texts.append(text)
            english_metadata.append(metadata)
            english_indices.append(i)
    
    logger.info(f"Filtered to {len(english_texts)} English documents")
    
    # Step 4: Toxicity filtering
    logger.info("Applying toxicity filtering...")
    toxicity_filter = ToxicityFilter(
        toxicity_threshold=rag_config.toxicity_threshold,
        use_rule_based=True
    )
    
    # Clean toxic content instead of removing it
    clean_texts, toxicity_analysis = toxicity_filter.filter_dataset(
        english_texts,
        remove_toxic=False,
        clean_toxic=True,
        show_progress=True
    )
    
    # Get toxicity statistics
    toxicity_stats = toxicity_filter.get_statistics(toxicity_analysis)
    logger.info(f"Toxicity filtering results:")
    logger.info(f"  Total texts: {toxicity_stats['total_texts']}")
    logger.info(f"  Toxic texts: {toxicity_stats['toxic_texts']} "
               f"({toxicity_stats['toxic_percentage']:.1f}%)")
    
    # Step 5: Text chunking
    logger.info("Chunking texts...")
    chunker = TextChunker(
        chunk_size=rag_config.chunk_size,
        overlap_size=rag_config.chunk_overlap,
        strategy=rag_config.chunking_strategy
    )
    
    all_chunks = []
    all_chunk_metadata = []
    
    for i, (text, metadata) in enumerate(zip(clean_texts, english_metadata)):
        chunks = chunker.chunk_text(text, source_page=i+1)
        
        for chunk_text, chunk_meta in chunks:
            # Combine original metadata with chunk metadata
            combined_metadata = metadata.copy()
            combined_metadata.update({
                "chunk_id": len(all_chunks),
                "original_doc_id": i,
                "chunk_size": chunk_meta.char_count,
                "word_count": chunk_meta.word_count,
                "start_index": chunk_meta.start_index,
                "end_index": chunk_meta.end_index
            })
            
            all_chunks.append(chunk_text)
            all_chunk_metadata.append(combined_metadata)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(clean_texts)} documents")
    
    # Step 6: Initialize embedding generator
    logger.info("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        model_name=rag_config.embedding_model,
        normalize_embeddings=rag_config.normalize_embeddings
    )
    
    logger.info(f"Embedding dimension: {embedding_generator.get_embedding_dimension()}")
    
    # Step 7: Create vector store and add documents
    logger.info("Creating vector store and adding documents...")
    retriever = RAGRetriever(
        embedding_model=rag_config.embedding_model,
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        chunking_strategy=rag_config.chunking_strategy,
        vector_store_type=rag_config.vector_store_type,
        distance_metric=rag_config.distance_metric
    )
    
    retriever.add_documents_from_text(
        texts=all_chunks,
        metadata=all_chunk_metadata
    )
    
    # Step 8: Test retrieval with various queries
    logger.info("Testing retrieval system...")
    
    test_queries = [
        "What is machine learning?",
        "Tell me about neural networks and deep learning",
        "How do vector databases work?",
        "What are the types of machine learning algorithms?",
        "Explain similarity search and ANN",
    ]
    
    retrieval_results = {}
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        
        # Retrieve relevant documents
        results = retriever.retrieve(query, k=3)
        
        retrieval_results[query] = []
        
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('score', 0.0)
            topic = doc.metadata.get('topic', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            logger.info(f"  Result {i}: Score={score:.3f}, Topic={topic}, Source={source}")
            logger.info(f"    Content: {doc.page_content[:150]}...")
            
            retrieval_results[query].append({
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })
    
    # Step 9: Evaluate retrieval performance
    logger.info("\nEvaluating retrieval performance...")
    
    # Create ground truth for evaluation (simplified)
    ground_truth = {
        "What is machine learning?": ["machine_learning"],
        "Tell me about neural networks and deep learning": ["deep_learning"],
        "How do vector databases work?": ["vector_databases"],
        "What are the types of machine learning algorithms?": ["machine_learning"],
        "Explain similarity search and ANN": ["vector_databases"]
    }
    
    evaluator = ModelEvaluator()
    
    # Prepare data for evaluation
    retrieved_docs = []
    relevant_docs = []
    
    for query, expected_topics in ground_truth.items():
        query_results = retrieval_results.get(query, [])
        
        # Extract topics from retrieved documents
        retrieved_topics = [
            result["metadata"].get("topic", "unknown") 
            for result in query_results
        ]
        
        retrieved_docs.append(retrieved_topics)
        relevant_docs.append(expected_topics)
    
    # Evaluate retrieval metrics
    retrieval_metrics = evaluator.evaluate_retrieval(
        retrieved_docs=retrieved_docs,
        relevant_docs=relevant_docs,
        k_values=[1, 2, 3]
    )
    
    logger.info("Retrieval evaluation metrics:")
    for metric, value in retrieval_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Step 10: Test similarity search with embeddings
    logger.info("\nTesting direct similarity search...")
    
    test_query = "machine learning algorithms"
    query_embedding = embedding_generator.encode(test_query)
    
    # Find most similar documents
    chunk_embeddings = embedding_generator.encode(all_chunks[:10])  # Test with first 10 chunks
    
    similarities = embedding_generator.compute_similarity(
        [test_query], 
        all_chunks[:10], 
        metric="cosine"
    )
    
    logger.info(f"Direct similarity search for: '{test_query}'")
    top_indices = similarities[0].argsort()[::-1][:3]
    
    for i, idx in enumerate(top_indices, 1):
        similarity = similarities[0][idx]
        chunk = all_chunks[idx]
        logger.info(f"  {i}. Similarity: {similarity:.3f}")
        logger.info(f"     Content: {chunk[:100]}...")
    
    # Step 11: Save the complete pipeline
    logger.info("\nSaving RAG pipeline...")
    
    pipeline_path = "/tmp/complete_rag_pipeline"
    retriever.save(pipeline_path)
    
    # Save configuration and metadata
    with open(f"{pipeline_path}/pipeline_info.json", "w") as f:
        pipeline_info = {
            "total_documents": len(clean_texts),
            "total_chunks": len(all_chunks),
            "embedding_model": rag_config.embedding_model,
            "toxicity_stats": toxicity_stats,
            "retrieval_metrics": retrieval_metrics,
            "config": vars(rag_config)
        }
        json.dump(pipeline_info, f, indent=2)
    
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    # Step 12: Test loading and using the saved pipeline
    logger.info("Testing saved pipeline...")
    
    new_retriever = RAGRetriever()
    new_retriever.load(pipeline_path)
    
    test_query = "What are neural networks?"
    loaded_results = new_retriever.retrieve(test_query, k=2)
    
    logger.info(f"Test query on loaded pipeline: '{test_query}'")
    logger.info(f"Retrieved {len(loaded_results)} documents")
    
    for i, doc in enumerate(loaded_results, 1):
        score = doc.metadata.get('score', 0.0)
        logger.info(f"  {i}. Score: {score:.3f}")
        logger.info(f"     Content: {doc.page_content[:100]}...")
    
    # Step 13: Performance summary
    logger.info("\nPipeline Performance Summary:")
    logger.info(f"  Original documents: {len(texts)}")
    logger.info(f"  English documents: {len(english_texts)}")
    logger.info(f"  Clean documents: {len(clean_texts)}")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    logger.info(f"  Embedding dimension: {embedding_generator.get_embedding_dimension()}")
    logger.info(f"  Vector store stats: {retriever.get_retriever_stats()}")
    
    logger.info("Complete RAG Pipeline Example finished successfully!")


if __name__ == "__main__":
    main()