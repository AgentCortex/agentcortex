"""Embedding generation utilities."""

import logging
import numpy as np
import torch
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Flexible embedding generation using various models.
    
    Supports both sentence-transformers and raw transformers models.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_type: str = "sentence_transformer",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Model name or path
            model_type: "sentence_transformer" or "transformer"
            device: Device to use
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.model_type = model_type
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(f"Embedding generator initialized with {model_name} on {self.device}")
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            if self.model_type == "sentence_transformer":
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
            else:
                # Raw transformer model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Add pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Text or list of texts to encode
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == "sentence_transformer":
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=self.normalize_embeddings
            )
        else:
            embeddings = self._encode_with_transformer(texts, convert_to_numpy)
        
        return embeddings
    
    def _encode_with_transformer(
        self,
        texts: List[str],
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode texts using raw transformer model."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs["attention_mask"]
                )
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return final_embeddings.numpy()
        
        return final_embeddings
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Encode documents and add embeddings to metadata.
        
        Args:
            documents: List of document dictionaries
            text_key: Key containing the text to encode
            show_progress: Whether to show progress
            
        Returns:
            Documents with embeddings added
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.encode(texts, show_progress=show_progress)
        
        # Add embeddings to documents
        enriched_docs = []
        for doc, embedding in zip(documents, embeddings):
            enriched_doc = doc.copy()
            enriched_doc["embedding"] = embedding
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        metric: str = "cosine"
    ) -> Union[float, np.ndarray]:
        """
        Compute similarity between texts.
        
        Args:
            texts1: First text or list of texts
            texts2: Second text or list of texts
            metric: Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score(s)
        """
        # Get embeddings
        emb1 = self.encode(texts1, convert_to_numpy=True)
        emb2 = self.encode(texts2, convert_to_numpy=True)
        
        # Ensure 2D arrays
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        # Compute similarity
        if metric == "cosine":
            # Normalize vectors for cosine similarity
            emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            similarity = np.dot(emb1_norm, emb2_norm.T)
        elif metric == "dot":
            similarity = np.dot(emb1, emb2.T)
        elif metric == "euclidean":
            # Convert to similarity (higher is more similar)
            distances = np.linalg.norm(emb1[:, None, :] - emb2[None, :, :], axis=2)
            similarity = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        # Return scalar if single comparison
        if similarity.shape == (1, 1):
            return float(similarity[0, 0])
        
        return similarity
    
    def find_most_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar documents to a query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score, index) tuples
        """
        # Get embeddings
        query_emb = self.encode(query, convert_to_numpy=True)
        doc_embs = self.encode(documents, convert_to_numpy=True)
        
        # Compute similarities
        similarities = self.compute_similarity([query], documents, metric="cosine")
        if similarities.ndim > 1:
            similarities = similarities[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((documents[idx], float(similarities[idx]), int(idx)))
        
        return results
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Cluster embeddings.
        
        Args:
            embeddings: Embeddings to cluster
            n_clusters: Number of clusters
            method: Clustering method ("kmeans")
            
        Returns:
            Cluster labels
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("sklearn required for clustering")
            raise
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        method: str = "umap"
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Target dimensions
            method: Reduction method ("umap", "tsne", "pca")
            
        Returns:
            Reduced embeddings
        """
        try:
            if method == "umap":
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            elif method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
            elif method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
                
            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings
            
        except ImportError as e:
            logger.error(f"Required library not available for {method}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.model_type == "sentence_transformer":
            return self.model.get_sentence_embedding_dimension()
        else:
            # Get dimension from a test encoding
            test_emb = self.encode("test", convert_to_numpy=True)
            return test_emb.shape[-1]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size
        }