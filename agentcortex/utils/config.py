"""Configuration management utilities."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: Optional[int] = None
    normalize_embeddings: bool = True
    
    # Text chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "sentence_aware"
    
    # Vector store configuration
    vector_store_type: str = "flat"
    distance_metric: str = "cosine"
    
    # Retrieval configuration
    top_k: int = 4
    score_threshold: Optional[float] = None
    
    # Model configuration
    use_quantization: bool = False
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Toxicity filtering
    toxicity_threshold: float = 0.7
    use_toxicity_filter: bool = True
    
    # Language detection
    default_language: str = "en"
    language_confidence_threshold: float = 0.7


class Config:
    """
    Configuration management for AgentCortex.
    
    Supports loading from files, environment variables, and dictionaries.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_data = {}
        self.rag_config = RAGConfig()
        
        # Load environment variables if available
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        # Load configuration file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    try:
                        import yaml
                        self.config_data = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError("PyYAML required for YAML configuration files")
                else:
                    self.config_data = json.load(f)
            
            # Update RAG configuration
            self._update_rag_config()
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {file_path}: {e}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config_data.update(config_dict)
        self._update_rag_config()
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "AGENTCORTEX_EMBEDDING_MODEL": ("rag", "embedding_model"),
            "AGENTCORTEX_CHUNK_SIZE": ("rag", "chunk_size", int),
            "AGENTCORTEX_CHUNK_OVERLAP": ("rag", "chunk_overlap", int),
            "AGENTCORTEX_CHUNKING_STRATEGY": ("rag", "chunking_strategy"),
            "AGENTCORTEX_VECTOR_STORE_TYPE": ("rag", "vector_store_type"),
            "AGENTCORTEX_DISTANCE_METRIC": ("rag", "distance_metric"),
            "AGENTCORTEX_TOP_K": ("rag", "top_k", int),
            "AGENTCORTEX_SCORE_THRESHOLD": ("rag", "score_threshold", float),
            "AGENTCORTEX_USE_QUANTIZATION": ("rag", "use_quantization", bool),
            "AGENTCORTEX_DEVICE": ("rag", "device"),
            "AGENTCORTEX_CACHE_DIR": ("rag", "cache_dir"),
            "AGENTCORTEX_TOXICITY_THRESHOLD": ("rag", "toxicity_threshold", float),
            "AGENTCORTEX_USE_TOXICITY_FILTER": ("rag", "use_toxicity_filter", bool),
            "AGENTCORTEX_DEFAULT_LANGUAGE": ("rag", "default_language"),
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value type
                if len(config_path) > 2:
                    value_type = config_path[2]
                    if value_type == int:
                        value = int(value)
                    elif value_type == float:
                        value = float(value)
                    elif value_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Set nested configuration
                section = config_path[0]
                key = config_path[1]
                
                if section not in self.config_data:
                    self.config_data[section] = {}
                
                self.config_data[section][key] = value
        
        self._update_rag_config()
    
    def _update_rag_config(self) -> None:
        """Update RAG configuration from loaded data."""
        rag_data = self.config_data.get("rag", {})
        
        for key, value in rag_data.items():
            if hasattr(self.rag_config, key):
                setattr(self.rag_config, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Update RAG config if needed
        if keys[0] == "rag":
            self._update_rag_config()
    
    def save_to_file(self, file_path: str, format: str = "json") -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Output file path
            format: File format ("json" or "yaml")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if format.lower() == "yaml":
                try:
                    import yaml
                    yaml.dump(self.config_data, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML output")
            else:
                json.dump(self.config_data, f, indent=2)
    
    def get_rag_config(self) -> RAGConfig:
        """Get RAG configuration object."""
        return self.rag_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config_data.copy()
    
    def merge(self, other_config: Union["Config", Dict[str, Any]]) -> None:
        """
        Merge with another configuration.
        
        Args:
            other_config: Configuration to merge
        """
        if isinstance(other_config, Config):
            other_data = other_config.to_dict()
        else:
            other_data = other_config
        
        self._deep_merge(self.config_data, other_data)
        self._update_rag_config()
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate RAG configuration
        if self.rag_config.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.rag_config.chunk_overlap < 0:
            errors.append("chunk_overlap must be non-negative")
        
        if self.rag_config.chunk_overlap >= self.rag_config.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        if self.rag_config.top_k <= 0:
            errors.append("top_k must be positive")
        
        if self.rag_config.chunking_strategy not in ["sentence_aware", "token_based", "fixed_size"]:
            errors.append("Invalid chunking_strategy")
        
        if self.rag_config.vector_store_type not in ["flat", "ivf", "hnsw"]:
            errors.append("Invalid vector_store_type")
        
        if self.rag_config.distance_metric not in ["cosine", "euclidean"]:
            errors.append("Invalid distance_metric")
        
        if not (0.0 <= self.rag_config.toxicity_threshold <= 1.0):
            errors.append("toxicity_threshold must be between 0 and 1")
        
        if not (0.0 <= self.rag_config.language_confidence_threshold <= 1.0):
            errors.append("language_confidence_threshold must be between 0 and 1")
        
        return errors
    
    def create_default_config_file(self, file_path: str, format: str = "yaml") -> None:
        """
        Create a default configuration file.
        
        Args:
            file_path: Output file path
            format: File format ("json" or "yaml")
        """
        default_config = {
            "rag": asdict(RAGConfig()),
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            "paths": {
                "models_cache": "./models",
                "vector_store": "./vector_store",
                "data": "./data",
                "logs": "./logs"
            }
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if format.lower() == "yaml":
                try:
                    import yaml
                    yaml.dump(default_config, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML output")
            else:
                json.dump(default_config, f, indent=2)