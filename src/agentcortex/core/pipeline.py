"""Main RAG pipeline orchestrating all components."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from ..extractors.pdf_extractor import PDFExtractor
from ..chunkers.text_chunker import TextChunker
from ..storage.faiss_storage import FAISSStorage
from ..retrieval.langchain_retriever import LangChainRetriever
from ..filters.toxicity_filter import ToxicityFilter
from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline with PDF processing, chunking, storage, and retrieval."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG configuration (uses default if None)
        """
        self.config = config or RAGConfig()
        
        # Create necessary directories
        self.config.create_directories()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # PDF Extractor
        self.pdf_extractor = PDFExtractor(backend="pdfplumber")
        
        # Text Chunker
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunker.chunk_size,
            chunk_overlap=self.config.chunker.chunk_overlap,
            separators=[self.config.chunker.separator] if self.config.chunker.separator else None,
            keep_separator=self.config.chunker.keep_separator
        )
        
        # FAISS Storage
        self.vector_storage = FAISSStorage(
            embedding_model=self.config.embedding.model_name,
            index_path=self.config.faiss.index_path,
            dimension=self.config.embedding.dimension,
            index_type=self.config.faiss.index_type,
            normalize_embeddings=self.config.faiss.normalize_embeddings,
            device=self.config.embedding.device
        )
        
        # LangChain Retriever
        self.retriever = LangChainRetriever(
            faiss_storage=self.vector_storage,
            llm=None,  # Will be set later if needed
            retriever_k=4
        )
        
        # Toxicity Filter
        if self.config.toxicity.enabled:
            self.toxicity_filter = ToxicityFilter(
                model_name=self.config.toxicity.model_name,
                threshold=self.config.toxicity.threshold,
                device=self.config.embedding.device
            )
        else:
            self.toxicity_filter = None
    
    def add_documents_from_pdfs(
        self, 
        pdf_paths: List[str], 
        chunking_method: str = "recursive",
        filter_toxicity: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents from PDF files to the RAG pipeline.
        
        Args:
            pdf_paths: List of paths to PDF files
            chunking_method: Method to use for text chunking
            filter_toxicity: Whether to filter toxic content
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {len(pdf_paths)} PDF files")
        
        # Extract text from PDFs
        documents = self.pdf_extractor.extract_multiple(pdf_paths)
        
        # Filter out failed extractions
        valid_documents = [doc for doc in documents if not doc.get("error")]
        failed_extractions = len(documents) - len(valid_documents)
        
        if failed_extractions > 0:
            logger.warning(f"Failed to extract text from {failed_extractions} PDFs")
        
        # Chunk documents
        chunks = self.text_chunker.chunk_documents(valid_documents, method=chunking_method)
        
        # Filter toxicity if enabled
        if filter_toxicity and self.toxicity_filter:
            original_chunk_count = len(chunks)
            chunks = self.toxicity_filter.filter_documents(chunks)
            filtered_count = original_chunk_count - len(chunks)
            
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} toxic chunks")
        
        # Add to vector storage
        if chunks:
            self.vector_storage.add_documents(chunks)
        
        # Calculate statistics
        stats = {
            "pdf_files_processed": len(pdf_paths),
            "successful_extractions": len(valid_documents),
            "failed_extractions": failed_extractions,
            "total_chunks": len(chunks),
            "chunking_method": chunking_method,
            "toxicity_filtering": filter_toxicity and self.toxicity_filter is not None
        }
        
        if chunks:
            chunk_stats = self.text_chunker.get_chunk_statistics(chunks)
            stats.update(chunk_stats)
        
        logger.info(f"Successfully processed {stats['successful_extractions']} PDFs into {stats['total_chunks']} chunks")
        
        return stats
    
    def add_documents_from_text(
        self, 
        texts: List[str], 
        sources: Optional[List[str]] = None,
        chunking_method: str = "recursive",
        filter_toxicity: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents from raw text to the RAG pipeline.
        
        Args:
            texts: List of text strings
            sources: List of source identifiers (optional)
            chunking_method: Method to use for text chunking
            filter_toxicity: Whether to filter toxic content
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {len(texts)} text documents")
        
        # Create document format
        documents = []
        for i, text in enumerate(texts):
            source = sources[i] if sources and i < len(sources) else f"text_{i}"
            documents.append({
                "text": text,
                "source": source,
                "document_type": "text"
            })
        
        # Chunk documents
        chunks = self.text_chunker.chunk_documents(documents, method=chunking_method)
        
        # Filter toxicity if enabled
        if filter_toxicity and self.toxicity_filter:
            original_chunk_count = len(chunks)
            chunks = self.toxicity_filter.filter_documents(chunks)
            filtered_count = original_chunk_count - len(chunks)
            
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} toxic chunks")
        
        # Add to vector storage
        if chunks:
            self.vector_storage.add_documents(chunks)
        
        # Calculate statistics
        stats = {
            "text_documents_processed": len(texts),
            "total_chunks": len(chunks),
            "chunking_method": chunking_method,
            "toxicity_filtering": filter_toxicity and self.toxicity_filter is not None
        }
        
        if chunks:
            chunk_stats = self.text_chunker.get_chunk_statistics(chunks)
            stats.update(chunk_stats)
        
        logger.info(f"Successfully processed {len(texts)} text documents into {stats['total_chunks']} chunks")
        
        return stats
    
    def query(
        self, 
        query: str, 
        k: int = 5, 
        filter_query_toxicity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG pipeline for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_query_toxicity: Whether to filter toxic queries
            
        Returns:
            List of relevant documents with metadata
        """
        # Filter query toxicity if enabled
        if filter_query_toxicity and self.toxicity_filter:
            is_toxic, toxicity_score = self.toxicity_filter.is_toxic(query)
            if is_toxic:
                logger.warning(f"Toxic query detected (score: {toxicity_score})")
                filtered_query = self.toxicity_filter.filter_text(query)
                if filtered_query != query:
                    logger.info("Using filtered query for search")
                    query = filtered_query
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(query, k)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        
        return results
    
    def answer_question(
        self, 
        question: str, 
        k: int = 5,
        filter_question_toxicity: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: Question to answer
            k: Number of context documents to use
            filter_question_toxicity: Whether to filter toxic questions
            
        Returns:
            Dictionary with answer and context
        """
        if self.retriever.llm is None:
            # If no LLM is set, just return relevant documents
            logger.warning("No LLM configured. Returning relevant documents only.")
            results = self.query(question, k, filter_question_toxicity)
            
            return {
                "question": question,
                "answer": "No LLM configured. Please review the relevant documents below.",
                "context_documents": results,
                "has_llm": False
            }
        
        # Use LangChain retriever for full QA
        try:
            result = self.retriever.answer_question(question, k)
            result["has_llm"] = True
            return result
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # Fallback to document retrieval
            results = self.query(question, k, filter_question_toxicity)
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context_documents": results,
                "has_llm": True,
                "error": str(e)
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "config": self.config.model_dump(),
            "vector_storage": self.vector_storage.get_stats(),
            "retriever": self.retriever.get_retriever_stats(),
            "toxicity_filter_enabled": self.toxicity_filter is not None
        }
        
        if self.toxicity_filter:
            stats["toxicity_filter"] = {
                "model_name": self.toxicity_filter.model_name,
                "threshold": self.toxicity_filter.threshold
            }
        
        return stats
    
    def save_pipeline(self) -> None:
        """Save pipeline state to disk."""
        logger.info("Saving pipeline state...")
        
        # Save vector storage
        self.vector_storage.save_index()
        
        # Save configuration
        config_path = Path(self.config.data_dir) / "pipeline_config.json"
        self.config.save_to_file(str(config_path))
        
        logger.info("Pipeline state saved successfully")
    
    def clear_pipeline(self) -> None:
        """Clear all stored data from the pipeline."""
        logger.warning("Clearing pipeline data...")
        
        # Clear vector storage
        self.vector_storage.clear()
        
        logger.info("Pipeline data cleared")
    
    def set_llm(self, llm) -> None:
        """
        Set the language model for question answering.
        
        Args:
            llm: LangChain-compatible LLM instance
        """
        self.retriever.set_llm(llm)
        logger.info("LLM set for question answering")
    
    def update_config(self, new_config: RAGConfig) -> None:
        """
        Update pipeline configuration (requires reinitialization).
        
        Args:
            new_config: New configuration
        """
        logger.info("Updating pipeline configuration...")
        self.config = new_config
        self.config.create_directories()
        self._initialize_components()
        logger.info("Pipeline configuration updated")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "RAGPipeline":
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Initialized RAG pipeline
        """
        config = RAGConfig.from_file(config_path)
        return cls(config)
    
    @classmethod
    def from_env(cls) -> "RAGPipeline":
        """
        Create pipeline from environment variables.
        
        Returns:
            Initialized RAG pipeline
        """
        config = RAGConfig.from_env()
        return cls(config)