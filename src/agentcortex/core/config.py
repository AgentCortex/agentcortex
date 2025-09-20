"""Configuration management for RAG pipeline."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
from pathlib import Path


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384)
    device: str = Field(default="cpu")
    batch_size: int = Field(default=32)


class FAISSConfig(BaseModel):
    """Configuration for FAISS vector storage."""
    
    index_type: str = Field(default="IndexFlatIP")
    index_path: str = Field(default="./faiss_index")
    normalize_embeddings: bool = Field(default=True)


class ChunkerConfig(BaseModel):
    """Configuration for text chunking."""
    
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    separator: str = Field(default="\n\n")
    keep_separator: bool = Field(default=True)


class ToxicityConfig(BaseModel):
    """Configuration for toxicity filtering."""
    
    model_name: str = Field(default="unitary/toxic-bert")
    threshold: float = Field(default=0.7)
    enabled: bool = Field(default=True)


class LLMConfig(BaseModel):
    """Configuration for Language Model."""
    
    provider: str = Field(default="huggingface")
    model_name: str = Field(default="microsoft/DialoGPT-medium")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    api_key: Optional[str] = Field(default=None)


class RAGConfig(BaseModel):
    """Main configuration class for RAG pipeline."""
    
    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    toxicity: ToxicityConfig = Field(default_factory=ToxicityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # General settings
    data_dir: str = Field(default="./data")
    cache_dir: str = Field(default="./cache")
    log_level: str = Field(default="INFO")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "RAGConfig":
        """Load configuration from JSON file."""
        import json
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = os.getenv("EMBEDDING_MODEL")
        
        if os.getenv("FAISS_INDEX_PATH"):
            config.faiss.index_path = os.getenv("FAISS_INDEX_PATH")
        
        if os.getenv("CHUNK_SIZE"):
            config.chunker.chunk_size = int(os.getenv("CHUNK_SIZE"))
        
        if os.getenv("LLM_API_KEY"):
            config.llm.api_key = os.getenv("LLM_API_KEY")
        
        if os.getenv("DATA_DIR"):
            config.data_dir = os.getenv("DATA_DIR")
        
        return config
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            import json
            json.dump(self.model_dump(), f, indent=2)
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.data_dir,
            self.cache_dir,
            self.faiss.index_path,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)