#!/usr/bin/env python3
"""
Basic RAG Pipeline Example

This example demonstrates how to use AgentCortex to:
1. Extract text from PDF documents
2. Chunk the text 
3. Create vector embeddings
4. Perform similarity search
5. Set up a complete RAG retrieval system
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import agentcortex
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentcortex.pdf_extraction import PDFExtractor
from agentcortex.text_chunking import TextChunker
from agentcortex.vector_storage import VectorStore
from agentcortex.retrieval import RAGRetriever
from agentcortex.utils import setup_logging

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Run the basic RAG example."""
    logger.info("Starting Basic RAG Pipeline Example")
    
    # Sample documents (since we don't have actual PDFs in this example)
    sample_documents = [
        """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence (AI) based on the idea that systems can learn 
        from data, identify patterns and make decisions with minimal human intervention.
        Machine learning algorithms build a model based on training data in order to make 
        predictions or decisions without being explicitly programmed to do so.
        """,
        """
        Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language. 
        NLP combines computational linguistics with statistical, machine learning, and deep learning 
        models, enabling computers to process and analyze large amounts of natural language data.
        """,
        """
        Vector databases are specialized databases designed to store and query high-dimensional 
        vector data efficiently. They are essential for applications like similarity search, 
        recommendation systems, and retrieval-augmented generation (RAG). Vector databases 
        use specialized indexing techniques like FAISS, Annoy, or HNSW to enable fast 
        approximate nearest neighbor searches across millions of vectors.
        """,
        """
        Retrieval-Augmented Generation (RAG) is an AI framework that combines the power of 
        large language models with external knowledge retrieval. RAG works by first retrieving 
        relevant documents or passages from a knowledge base, then using this retrieved 
        information to generate more accurate and contextually relevant responses. This 
        approach helps address limitations of pure generative models like hallucination 
        and knowledge cutoffs.
        """
    ]
    
    # Step 1: Initialize RAG Retriever
    logger.info("Initializing RAG retriever...")
    retriever = RAGRetriever(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        chunking_strategy="sentence_aware"
    )
    
    # Step 2: Add documents to the retriever
    logger.info("Adding sample documents to retriever...")
    metadata = [
        {"topic": "machine_learning", "source": "ml_textbook"},
        {"topic": "nlp", "source": "nlp_guide"},
        {"topic": "vector_databases", "source": "vector_db_manual"},
        {"topic": "rag", "source": "rag_paper"}
    ]
    
    retriever.add_documents_from_text(
        texts=sample_documents,
        metadata=metadata
    )
    
    # Step 3: Perform similarity searches
    logger.info("Performing similarity searches...")
    
    queries = [
        "What is machine learning?",
        "How do vector databases work?",
        "What is RAG and how does it work?",
        "Tell me about natural language processing"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        
        # Retrieve relevant documents
        results = retriever.retrieve(query, k=2)
        
        logger.info(f"Found {len(results)} relevant documents:")
        for i, doc in enumerate(results, 1):
            logger.info(f"  {i}. Score: {doc.metadata.get('score', 'N/A'):.3f}")
            logger.info(f"     Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"     Topic: {doc.metadata.get('topic', 'Unknown')}")
            logger.info(f"     Content: {doc.page_content[:200]}...")
    
    # Step 4: Get retriever statistics
    logger.info("\nRetriever Statistics:")
    stats = retriever.get_retriever_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Step 5: Save the retriever for later use
    save_path = "/tmp/basic_rag_example"
    logger.info(f"Saving retriever to {save_path}")
    retriever.save(save_path)
    
    # Step 6: Load the retriever to demonstrate persistence
    logger.info("Loading retriever from disk...")
    new_retriever = RAGRetriever()
    new_retriever.load(save_path)
    
    # Test the loaded retriever
    test_query = "What are the applications of machine learning?"
    results = new_retriever.retrieve(test_query, k=1)
    logger.info(f"\nTest query on loaded retriever: {test_query}")
    logger.info(f"Retrieved {len(results)} documents")
    
    logger.info("Basic RAG Pipeline Example completed successfully!")


if __name__ == "__main__":
    main()