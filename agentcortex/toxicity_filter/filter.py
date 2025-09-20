"""Toxicity filtering for prompt cleaning using detoxify and custom filters."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """
    Comprehensive toxicity filtering for text cleaning.
    
    Uses detoxify library for ML-based detection and rule-based filters.
    """
    
    def __init__(
        self,
        model_name: str = "original",
        toxicity_threshold: float = 0.7,
        use_rule_based: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize toxicity filter.
        
        Args:
            model_name: Detoxify model name ('original', 'unbiased', 'multilingual')
            toxicity_threshold: Threshold for toxicity classification
            use_rule_based: Whether to use rule-based filtering
            device: Device to run model on
        """
        self.model_name = model_name
        self.toxicity_threshold = toxicity_threshold
        self.use_rule_based = use_rule_based
        
        # Initialize detoxify model
        self.detoxify_model = None
        if DETOXIFY_AVAILABLE:
            try:
                self.detoxify_model = Detoxify(model_name, device=device)
                logger.info(f"Loaded detoxify model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load detoxify model: {e}")
        else:
            logger.warning("Detoxify library not available, using rule-based filtering only")
        
        # Rule-based filters
        self.explicit_words = self._load_explicit_words()
        self.toxic_patterns = self._compile_toxic_patterns()
        
        logger.info("Toxicity filter initialized")
    
    def _load_explicit_words(self) -> set:
        """Load explicit words list (basic implementation)."""
        # This is a minimal list - in production, you'd want a comprehensive list
        explicit_words = {
            # Add explicit words here - keeping minimal for example
            "profanity1", "profanity2", "hate_word1", "offensive_term1"
        }
        return explicit_words
    
    def _compile_toxic_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for toxic content detection."""
        patterns = [
            # Personal attacks
            re.compile(r'\b(you\s+are\s+(?:stupid|idiot|moron))\b', re.IGNORECASE),
            
            # Hate speech patterns (very basic examples)
            re.compile(r'\b(hate\s+(?:all|every)\s+\w+)\b', re.IGNORECASE),
            
            # Violent threats (basic patterns)
            re.compile(r'\b(kill\s+(?:you|yourself|him|her))\b', re.IGNORECASE),
            re.compile(r'\b(die\s+(?:you|yourself))\b', re.IGNORECASE),
            
            # Discriminatory language patterns
            re.compile(r'\b(all\s+\w+\s+are\s+(?:bad|evil|stupid))\b', re.IGNORECASE),
        ]
        
        return patterns
    
    def detect_toxicity(
        self,
        text: Union[str, List[str]],
        return_scores: bool = False
    ) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Detect toxicity in text(s).
        
        Args:
            text: Text or list of texts to analyze
            return_scores: Whether to return detailed scores
            
        Returns:
            Boolean(s) indicating toxicity or detailed scores if requested
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        results = []
        
        for txt in texts:
            result = self._analyze_single_text(txt, return_scores)
            results.append(result)
        
        return results[0] if single_input else results
    
    def _analyze_single_text(
        self,
        text: str,
        return_scores: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """Analyze a single text for toxicity."""
        # Initialize scores
        scores = {
            "toxicity": 0.0,
            "severe_toxicity": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_attack": 0.0,
            "rule_based_flags": []
        }
        
        # ML-based detection with detoxify
        if self.detoxify_model:
            try:
                detox_scores = self.detoxify_model.predict(text)
                scores.update(detox_scores)
            except Exception as e:
                logger.warning(f"Detoxify prediction failed: {e}")
        
        # Rule-based detection
        if self.use_rule_based:
            rule_flags = self._apply_rule_based_filters(text)
            scores["rule_based_flags"] = rule_flags
            
            # Boost toxicity score if rule-based filters triggered
            if rule_flags:
                scores["toxicity"] = max(scores["toxicity"], 0.8)
        
        # Determine if toxic
        is_toxic = (
            scores["toxicity"] > self.toxicity_threshold or
            len(scores["rule_based_flags"]) > 0
        )
        
        if return_scores:
            scores["is_toxic"] = is_toxic
            return scores
        
        return is_toxic
    
    def _apply_rule_based_filters(self, text: str) -> List[str]:
        """Apply rule-based toxicity filters."""
        flags = []
        text_lower = text.lower()
        
        # Check explicit words
        for word in self.explicit_words:
            if word in text_lower:
                flags.append(f"explicit_word: {word}")
        
        # Check toxic patterns
        for pattern in self.toxic_patterns:
            matches = pattern.findall(text)
            for match in matches:
                flags.append(f"toxic_pattern: {match}")
        
        # Check for excessive capitalization (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.7 and len(text) > 10:
            flags.append("excessive_caps")
        
        # Check for repeated characters (spam-like)
        if re.search(r'(.)\1{4,}', text):
            flags.append("repeated_chars")
        
        return flags
    
    def clean_text(
        self,
        text: str,
        replacement_strategy: str = "remove",
        replacement_text: str = "[FILTERED]"
    ) -> str:
        """
        Clean toxic content from text.
        
        Args:
            text: Text to clean
            replacement_strategy: "remove", "replace", or "mask"
            replacement_text: Text to use for replacement
            
        Returns:
            Cleaned text
        """
        cleaned_text = text
        
        # Remove/replace explicit words
        for word in self.explicit_words:
            if word in cleaned_text.lower():
                if replacement_strategy == "remove":
                    # Remove the word and surrounding whitespace
                    pattern = rf'\b{re.escape(word)}\b'
                    cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                elif replacement_strategy == "replace":
                    pattern = rf'\b{re.escape(word)}\b'
                    cleaned_text = re.sub(pattern, replacement_text, cleaned_text, flags=re.IGNORECASE)
                elif replacement_strategy == "mask":
                    pattern = rf'\b{re.escape(word)}\b'
                    masked_word = word[0] + '*' * (len(word) - 2) + word[-1] if len(word) > 2 else '*' * len(word)
                    cleaned_text = re.sub(pattern, masked_word, cleaned_text, flags=re.IGNORECASE)
        
        # Apply pattern-based cleaning
        for pattern in self.toxic_patterns:
            if replacement_strategy == "remove":
                cleaned_text = pattern.sub('', cleaned_text)
            else:
                cleaned_text = pattern.sub(replacement_text, cleaned_text)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def filter_dataset(
        self,
        texts: List[str],
        remove_toxic: bool = True,
        clean_toxic: bool = False,
        show_progress: bool = True
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Filter a dataset of texts.
        
        Args:
            texts: List of texts to filter
            remove_toxic: Whether to remove toxic texts
            clean_toxic: Whether to clean toxic texts instead of removing
            show_progress: Whether to show progress
            
        Returns:
            Tuple of (filtered_texts, analysis_results)
        """
        filtered_texts = []
        analysis_results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                texts_iter = tqdm(texts, desc="Filtering texts")
            except ImportError:
                texts_iter = texts
        else:
            texts_iter = texts
        
        for text in texts_iter:
            analysis = self._analyze_single_text(text, return_scores=True)
            analysis_results.append(analysis)
            
            if analysis["is_toxic"]:
                if clean_toxic:
                    # Clean the text
                    cleaned_text = self.clean_text(text)
                    filtered_texts.append(cleaned_text)
                elif not remove_toxic:
                    # Keep toxic text as-is
                    filtered_texts.append(text)
                # If remove_toxic=True, skip adding the text
            else:
                # Non-toxic text, keep as-is
                filtered_texts.append(text)
        
        return filtered_texts, analysis_results
    
    def batch_detect(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch toxicity detection for efficiency.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            List of analysis results
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                range_iter = tqdm(range(0, len(texts), batch_size), desc="Processing batches")
            except ImportError:
                range_iter = range(0, len(texts), batch_size)
        else:
            range_iter = range(0, len(texts), batch_size)
        
        for i in range_iter:
            batch_texts = texts[i:i + batch_size]
            
            # Process batch with detoxify if available
            if self.detoxify_model:
                try:
                    batch_scores = self.detoxify_model.predict(batch_texts)
                    
                    # Convert to list of dicts
                    for j, text in enumerate(batch_texts):
                        scores = {key: float(batch_scores[key][j]) for key in batch_scores.keys()}
                        
                        # Add rule-based analysis
                        if self.use_rule_based:
                            rule_flags = self._apply_rule_based_filters(text)
                            scores["rule_based_flags"] = rule_flags
                            
                            if rule_flags:
                                scores["toxicity"] = max(scores["toxicity"], 0.8)
                        
                        scores["is_toxic"] = (
                            scores["toxicity"] > self.toxicity_threshold or
                            len(scores.get("rule_based_flags", [])) > 0
                        )
                        
                        results.append(scores)
                        
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
                    # Fall back to individual processing
                    for text in batch_texts:
                        result = self._analyze_single_text(text, return_scores=True)
                        results.append(result)
            else:
                # Process individually without detoxify
                for text in batch_texts:
                    result = self._analyze_single_text(text, return_scores=True)
                    results.append(result)
        
        return results
    
    def get_statistics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics from analysis results.
        
        Args:
            analysis_results: Results from toxicity analysis
            
        Returns:
            Statistics dictionary
        """
        if not analysis_results:
            return {}
        
        total_texts = len(analysis_results)
        toxic_count = sum(1 for r in analysis_results if r.get("is_toxic", False))
        
        # Average scores
        avg_scores = {}
        score_keys = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
        
        for key in score_keys:
            scores = [r.get(key, 0.0) for r in analysis_results]
            avg_scores[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0.0
        
        # Rule-based flags statistics
        all_flags = []
        for r in analysis_results:
            all_flags.extend(r.get("rule_based_flags", []))
        
        flag_counts = {}
        for flag in all_flags:
            flag_type = flag.split(":")[0] if ":" in flag else flag
            flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1
        
        return {
            "total_texts": total_texts,
            "toxic_texts": toxic_count,
            "toxic_percentage": (toxic_count / total_texts) * 100 if total_texts > 0 else 0,
            "clean_texts": total_texts - toxic_count,
            "average_scores": avg_scores,
            "rule_based_flags": flag_counts
        }
    
    def update_filters(
        self,
        new_explicit_words: Optional[List[str]] = None,
        new_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Update filter rules.
        
        Args:
            new_explicit_words: New explicit words to add
            new_patterns: New regex patterns to add
        """
        if new_explicit_words:
            self.explicit_words.update(new_explicit_words)
            logger.info(f"Added {len(new_explicit_words)} new explicit words")
        
        if new_patterns:
            new_compiled_patterns = []
            for pattern_str in new_patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    new_compiled_patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")
            
            self.toxic_patterns.extend(new_compiled_patterns)
            logger.info(f"Added {len(new_compiled_patterns)} new patterns")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current filter configuration."""
        return {
            "model_name": self.model_name,
            "toxicity_threshold": self.toxicity_threshold,
            "use_rule_based": self.use_rule_based,
            "explicit_words": list(self.explicit_words),
            "patterns": [p.pattern for p in self.toxic_patterns]
        }