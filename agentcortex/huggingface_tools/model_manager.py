"""Hugging Face model management with bitsandbytes optimization."""

import logging
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages Hugging Face models with optimization support.
    
    Supports quantization with bitsandbytes and efficient loading.
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "auto",
        use_quantization: bool = False,
        quantization_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model manager.
        
        Args:
            model_name: Hugging Face model name or path
            model_type: Model type ("auto", "causal_lm", "sequence_classification", "embedding")
            use_quantization: Whether to use quantization
            quantization_config: Custom quantization configuration
            device: Device to load model on
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.use_quantization = use_quantization
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Setup quantization config
        self.quantization_config = None
        if use_quantization and self.device == "cuda":
            self.quantization_config = self._create_quantization_config(quantization_config)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(f"Model manager initialized for {model_name} on {self.device}")
    
    def _create_quantization_config(
        self,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> BitsAndBytesConfig:
        """Create quantization configuration."""
        default_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
        
        if custom_config:
            default_config.update(custom_config)
        
        return BitsAndBytesConfig(**default_config)
    
    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Common loading arguments
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        # Add quantization config if enabled
        if self.quantization_config:
            model_kwargs["quantization_config"] = self.quantization_config
            logger.info("Using quantization for model loading")
        else:
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model based on type
        try:
            if self.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            elif self.model_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:  # auto or embedding
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model_type not in ["causal_lm", "auto"]:
            raise ValueError("Text generation requires a causal language model")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_embeddings(
        self,
        texts: Union[str, list],
        pooling_strategy: str = "mean"
    ) -> torch.Tensor:
        """
        Get embeddings for text(s).
        
        Args:
            texts: Input text or list of texts
            pooling_strategy: Pooling strategy ("mean", "cls", "max")
            
        Returns:
            Embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply pooling
        if pooling_strategy == "mean":
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif pooling_strategy == "cls":
            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0]
        elif pooling_strategy == "max":
            # Max pooling
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embeddings
    
    def classify_text(
        self,
        texts: Union[str, list],
        return_scores: bool = False
    ) -> Union[list, tuple]:
        """
        Classify text(s) using sequence classification model.
        
        Args:
            texts: Input text or list of texts
            return_scores: Whether to return confidence scores
            
        Returns:
            Classifications or (classifications, scores) if return_scores=True
        """
        if self.model_type != "sequence_classification":
            raise ValueError("Text classification requires a sequence classification model")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to labels if available
        if hasattr(self.model.config, "id2label"):
            labels = [self.model.config.id2label[pred.item()] for pred in predictions]
        else:
            labels = predictions.tolist()
        
        if return_scores:
            scores = probabilities.max(dim=-1)[0].tolist()
            return labels, scores
        
        return labels
    
    def create_pipeline(
        self,
        task: str,
        **kwargs
    ) -> pipeline:
        """
        Create a Hugging Face pipeline.
        
        Args:
            task: Pipeline task (e.g., "text-generation", "text-classification")
            **kwargs: Additional pipeline arguments
            
        Returns:
            Hugging Face pipeline
        """
        return pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "quantized": self.quantization_config is not None,
            "parameters": None,
            "memory_usage": None
        }
        
        if self.model:
            try:
                info["parameters"] = sum(p.numel() for p in self.model.parameters())
                
                if torch.cuda.is_available() and self.device == "cuda":
                    info["memory_usage"] = torch.cuda.memory_allocated() / 1024**3  # GB
                    
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
        
        return info
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded from memory")
    
    def save_model(self, save_path: str) -> None:
        """Save model and tokenizer to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(save_path)
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")