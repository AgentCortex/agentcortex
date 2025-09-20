"""Toxicity filtering for prompt cleaning."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """Filter toxic content from text using transformer models."""
    
    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        threshold: float = 0.7,
        device: str = "cpu",
        use_pipeline: bool = True
    ):
        """
        Initialize toxicity filter.
        
        Args:
            model_name: HuggingFace model for toxicity detection
            threshold: Threshold for toxicity classification (0-1)
            device: Device to run model on
            use_pipeline: Whether to use HuggingFace pipeline (simpler) or direct model
        """
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.use_pipeline = use_pipeline
        
        # Initialize model
        self._load_model()
        
        # Predefined toxic patterns (as backup/supplement)
        self.toxic_patterns = [
            r'\b(?:hate|kill|die|stupid|idiot|moron)\b',
            r'\b(?:fuck|shit|damn|hell)\w*\b',
            r'\b(?:racist|sexist|homophobic)\b',
        ]
        
        # Common profanity list (basic filter)
        self.profanity_words = {
            'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron',
            'hate', 'kill', 'die', 'murder', 'violence'
        }
    
    def _load_model(self) -> None:
        """Load the toxicity detection model."""
        try:
            if self.use_pipeline:
                # Use HuggingFace pipeline (easier)
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                self.tokenizer = None
                self.model = None
            else:
                # Load model and tokenizer directly
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.classifier = None
            
            logger.info(f"Loaded toxicity model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading toxicity model: {e}")
            # Fallback to basic pattern matching
            self.classifier = None
            self.tokenizer = None
            self.model = None
            logger.warning("Falling back to pattern-based toxicity detection")
    
    def is_toxic(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is toxic.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_toxic, confidence_score)
        """
        if not text.strip():
            return False, 0.0
        
        # Try model-based detection first
        if self.classifier is not None:
            try:
                return self._model_based_detection(text)
            except Exception as e:
                logger.warning(f"Model-based detection failed: {e}")
        
        # Fallback to pattern-based detection
        return self._pattern_based_detection(text)
    
    def _model_based_detection(self, text: str) -> Tuple[bool, float]:
        """Use transformer model for toxicity detection."""
        if self.use_pipeline:
            # Using pipeline
            results = self.classifier(text)
            
            # Find toxicity score (different models have different label formats)
            toxic_score = 0.0
            for result in results[0]:  # results is a list with one element
                label = result['label'].lower()
                score = result['score']
                
                if 'toxic' in label or 'negative' in label or label == 'LABEL_1':
                    toxic_score = max(toxic_score, score)
            
            return toxic_score > self.threshold, toxic_score
        
        else:
            # Using direct model
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # Assuming binary classification: [non-toxic, toxic]
                toxic_score = probabilities[0][1].item()
                
                return toxic_score > self.threshold, toxic_score
    
    def _pattern_based_detection(self, text: str) -> Tuple[bool, float]:
        """Use pattern matching for toxicity detection."""
        text_lower = text.lower()
        
        # Check for profanity words
        profanity_count = sum(1 for word in self.profanity_words if word in text_lower)
        
        # Check for toxic patterns
        pattern_matches = sum(1 for pattern in self.toxic_patterns if re.search(pattern, text_lower, re.IGNORECASE))
        
        # Simple scoring
        total_words = len(text.split())
        if total_words == 0:
            return False, 0.0
        
        # Score based on ratio of problematic content
        profanity_ratio = profanity_count / total_words
        pattern_score = min(pattern_matches * 0.3, 1.0)  # Each pattern match adds 0.3
        
        combined_score = min(profanity_ratio * 2 + pattern_score, 1.0)
        
        return combined_score > self.threshold, combined_score
    
    def filter_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """
        Filter toxic content from text.
        
        Args:
            text: Text to filter
            replacement: Replacement string for toxic content
            
        Returns:
            Filtered text
        """
        is_toxic, score = self.is_toxic(text)
        
        if is_toxic:
            # If entire text is toxic, replace it
            if score > 0.9:  # Very high toxicity
                return replacement
            
            # Try to filter specific parts
            return self._filter_toxic_parts(text, replacement)
        
        return text
    
    def _filter_toxic_parts(self, text: str, replacement: str) -> str:
        """Filter specific toxic parts of text."""
        # Split into sentences and filter each
        sentences = re.split(r'[.!?]+', text)
        filtered_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                is_toxic, _ = self.is_toxic(sentence.strip())
                if is_toxic:
                    filtered_sentences.append(replacement)
                else:
                    filtered_sentences.append(sentence.strip())
        
        return '. '.join(s for s in filtered_sentences if s != replacement)
    
    def filter_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter toxic content from a list of documents.
        
        Args:
            documents: List of documents with 'text' field
            
        Returns:
            List of filtered documents with toxicity metadata
        """
        filtered_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                filtered_docs.append(doc)
                continue
            
            is_toxic, score = self.is_toxic(text)
            
            # Create filtered document
            filtered_doc = doc.copy()
            filtered_doc["text"] = self.filter_text(text)
            filtered_doc["toxicity_score"] = score
            filtered_doc["is_toxic"] = is_toxic
            filtered_doc["was_filtered"] = is_toxic
            
            # Only include if not too toxic
            if score < 0.95:  # Exclude extremely toxic content
                filtered_docs.append(filtered_doc)
        
        return filtered_docs
    
    def batch_check_toxicity(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Check toxicity for multiple texts.
        
        Args:
            texts: List of texts to check
            
        Returns:
            List of toxicity results
        """
        results = []
        
        for i, text in enumerate(texts):
            is_toxic, score = self.is_toxic(text)
            results.append({
                "index": i,
                "text": text,
                "is_toxic": is_toxic,
                "toxicity_score": score
            })
        
        return results
    
    def get_toxicity_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get toxicity statistics for a collection of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with toxicity statistics
        """
        results = self.batch_check_toxicity(texts)
        
        toxic_count = sum(1 for r in results if r["is_toxic"])
        scores = [r["toxicity_score"] for r in results]
        
        return {
            "total_texts": len(texts),
            "toxic_count": toxic_count,
            "toxic_percentage": toxic_count / len(texts) * 100 if texts else 0,
            "avg_toxicity_score": sum(scores) / len(scores) if scores else 0,
            "max_toxicity_score": max(scores) if scores else 0,
            "min_toxicity_score": min(scores) if scores else 0,
            "threshold": self.threshold,
            "model_name": self.model_name
        }
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update toxicity threshold.
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = new_threshold
        logger.info(f"Updated toxicity threshold to {new_threshold}")
    
    def add_custom_patterns(self, patterns: List[str]) -> None:
        """
        Add custom toxic patterns.
        
        Args:
            patterns: List of regex patterns to add
        """
        self.toxic_patterns.extend(patterns)
        logger.info(f"Added {len(patterns)} custom patterns")
    
    def add_profanity_words(self, words: List[str]) -> None:
        """
        Add custom profanity words.
        
        Args:
            words: List of words to add to profanity list
        """
        self.profanity_words.update(word.lower() for word in words)
        logger.info(f"Added {len(words)} profanity words")