"""FAISS-based vector storage for efficient similarity search."""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Supports multiple embedding models and index types.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        dimension: Optional[int] = None,
        metric: str = "cosine"
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model: Sentence transformer model name
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            dimension: Embedding dimension (auto-detected if None)
            metric: Distance metric ("cosine", "euclidean")
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.metric = metric
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get dimension from model if not provided
        if dimension is None:
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.dimension = dimension
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage for documents and metadata
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        logger.info(f"Initialized vector store with dimension {self.dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.index_type == "flat":
            if self.metric == "cosine":
                # Normalize vectors for cosine similarity with L2 distance
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "ivf":
            # IVF index for large datasets
            nlist = 100  # number of clusters
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        elif self.index_type == "hnsw":
            # HNSW index for fast approximate search
            M = 16  # number of connections
            index = faiss.IndexHNSWFlat(self.dimension, M)
            if self.metric == "cosine":
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                index.metric_type = faiss.METRIC_L2
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts
            ids: Optional list of document IDs
        """
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        if ids:
            self.ids.extend(ids)
        else:
            start_id = len(self.ids)
            self.ids.extend([f"doc_{start_id + i}" for i in range(len(documents))])
        
        logger.info(f"Total documents in store: {len(self.documents)}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score, metadata) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            # Apply score threshold
            if score_threshold and score < score_threshold:
                continue
            
            document = self.documents[idx]
            metadata = self.metadata[idx].copy()
            metadata["id"] = self.ids[idx]
            
            results.append((document, float(score), metadata))
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            score_threshold: Minimum similarity score
            
        Returns:
            List of results for each query
        """
        if len(self.documents) == 0:
            return [[] for _ in queries]
        
        # Generate query embeddings
        query_embeddings = self.embedding_model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, k)
        
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue
                
                if score_threshold and score < score_threshold:
                    continue
                
                document = self.documents[idx]
                metadata = self.metadata[idx].copy()
                metadata["id"] = self.ids[idx]
                
                results.append((document, float(score), metadata))
            
            all_results.append(results)
        
        return all_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            (document, metadata) tuple or None if not found
        """
        try:
            idx = self.ids.index(doc_id)
            return self.documents[idx], self.metadata[idx]
        except ValueError:
            return None
    
    def save(self, save_path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            save_path: Directory path to save the store
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save documents and metadata
        with open(save_path / "store_data.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "ids": self.ids,
                "embedding_model_name": self.embedding_model_name,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "metric": self.metric
            }, f)
        
        logger.info(f"Vector store saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            load_path: Directory path to load the store from
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load documents and metadata
        with open(load_path / "store_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.ids = data["ids"]
        self.embedding_model_name = data["embedding_model_name"]
        self.index_type = data["index_type"]
        self.dimension = data["dimension"]
        self.metric = data["metric"]
        
        # Reinitialize embedding model if needed
        if not hasattr(self, "embedding_model") or \
           self.embedding_model_name != data["embedding_model_name"]:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info(f"Vector store loaded from {load_path}")
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Note: This recreates the index, which can be expensive for large datasets.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        # Find indices to keep
        indices_to_keep = []
        new_documents = []
        new_metadata = []
        new_ids = []
        
        for i, doc_id in enumerate(self.ids):
            if doc_id not in doc_ids:
                indices_to_keep.append(i)
                new_documents.append(self.documents[i])
                new_metadata.append(self.metadata[i])
                new_ids.append(self.ids[i])
        
        if len(indices_to_keep) == len(self.documents):
            logger.info("No documents to delete")
            return
        
        # Update storage
        self.documents = new_documents
        self.metadata = new_metadata
        self.ids = new_ids
        
        # Recreate index
        self.index = self._create_index()
        
        if self.documents:
            # Re-add remaining documents
            embeddings = self.embedding_model.encode(
                self.documents,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            if self.index_type == "ivf":
                self.index.train(embeddings)
            
            self.index.add(embeddings)
        
        logger.info(f"Deleted {len(doc_ids)} documents. Remaining: {len(self.documents)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "index_type": self.index_type,
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0
        }