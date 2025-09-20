"""FAISS vector storage for RAG pipeline."""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FAISSStorage:
    """FAISS-based vector storage for document embeddings."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "./faiss_index",
        dimension: Optional[int] = None,
        index_type: str = "IndexFlatIP",
        normalize_embeddings: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize FAISS storage.
        
        Args:
            embedding_model: Name of sentence transformer model
            index_path: Path to save/load FAISS index
            dimension: Embedding dimension (auto-detected if None)
            index_type: Type of FAISS index to use
            normalize_embeddings: Whether to normalize embeddings
            device: Device to run embeddings on
        """
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index_type = index_type
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Auto-detect dimension if not provided
        if self.dimension is None:
            test_embedding = self.embedding_model.encode(["test"])
            self.dimension = test_embedding.shape[1]
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store document metadata
        self._create_index()
        
        # Try to load existing index
        self.load_index()
    
    def _create_index(self) -> None:
        """Create FAISS index based on configuration."""
        if self.index_type == "IndexFlatIP":
            # Inner product (cosine similarity if normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            # L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # IVF with flat quantizer (requires training)
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "IndexHNSW":
            # HNSW (Hierarchical Navigable Small World)
            M = 16  # number of connections
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type} with dimension {self.dimension}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        # Normalize embeddings if required
        if self.normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def add_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 32
    ) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
            batch_size: Batch size for embedding generation
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata.extend(chunks)
        
        logger.info(f"Added {len(chunks)} documents to FAISS index")
    
    def search(
        self, 
        query: str, 
        k: int = 5, 
        return_similarities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            return_similarities: Whether to include similarity scores
            
        Returns:
            List of search results with metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embed_texts([query])
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.astype(np.float32), k
        )
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                if return_similarities:
                    result["similarity"] = float(similarity)
                result["rank"] = i + 1
                results.append(result)
        
        return results
    
    def search_batch(
        self, 
        queries: List[str], 
        k: int = 5, 
        return_similarities: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            return_similarities: Whether to include similarity scores
            
        Returns:
            List of search results for each query
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [[] for _ in queries]
        
        # Generate query embeddings
        query_embeddings = self.embed_texts(queries)
        
        # Search
        similarities, indices = self.index.search(
            query_embeddings.astype(np.float32), k
        )
        
        # Prepare results for each query
        all_results = []
        for query_idx in range(len(queries)):
            results = []
            for i, (similarity, idx) in enumerate(
                zip(similarities[query_idx], indices[query_idx])
            ):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    if return_similarities:
                        result["similarity"] = float(similarity)
                    result["rank"] = i + 1
                    results.append(result)
            all_results.append(results)
        
        return all_results
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config_file = self.index_path / "config.pkl"
        config = {
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device
        }
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            config_file = self.index_path / "config.pkl"
            
            if not all(f.exists() for f in [index_file, metadata_file, config_file]):
                logger.info("Index files not found, starting with empty index")
                return False
            
            # Load configuration
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
            
            # Verify configuration compatibility
            if (config["embedding_model"] != self.embedding_model_name or
                config["dimension"] != self.dimension):
                logger.warning("Index configuration mismatch, starting with empty index")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "metadata_count": len(self.metadata)
        }
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self._create_index()
        self.metadata = []
        logger.info("Cleared FAISS index")
    
    def remove_documents(self, doc_ids: List[int]) -> None:
        """
        Remove documents by their IDs (not supported by all FAISS indices).
        
        Args:
            doc_ids: List of document IDs to remove
        """
        logger.warning("Document removal not implemented for FAISS. Consider rebuilding index.")
        # Note: FAISS doesn't support direct document removal
        # Would need to rebuild index without specified documents