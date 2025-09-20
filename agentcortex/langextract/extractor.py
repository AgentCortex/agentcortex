"""Language detection and extraction utilities."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

try:
    import langdetect
    from langdetect import detect, detect_langs, DetectorFactory
    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageExtractor:
    """
    Language detection and text extraction utilities.
    
    Provides language detection, multilingual text processing,
    and language-specific text extraction capabilities.
    """
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize language extractor.
        
        Args:
            default_language: Default language code to assume
        """
        self.default_language = default_language
        
        # Language code mappings
        self.language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish'
        }
        
        # Common language-specific patterns
        self.language_patterns = {
            'en': [
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
                r'\b(is|are|was|were|been|being|have|has|had)\b'
            ],
            'es': [
                r'\b(el|la|los|las|un|una|y|o|pero|en|de|con|por|para)\b',
                r'\b(es|son|fue|fueron|está|están|ser|estar|tener|tiene)\b'
            ],
            'fr': [
                r'\b(le|la|les|un|une|et|ou|mais|dans|de|avec|par|pour)\b',
                r'\b(est|sont|était|étaient|être|avoir|a|ont)\b'
            ],
            'de': [
                r'\b(der|die|das|ein|eine|und|oder|aber|in|von|mit|für)\b',
                r'\b(ist|sind|war|waren|sein|haben|hat|hatte)\b'
            ]
        }
        
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect library not available, using rule-based detection only")
        
        logger.info("Language extractor initialized")
    
    def detect_language(
        self,
        text: str,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with language detection results
        """
        if not text.strip():
            return {
                "language": self.default_language,
                "confidence": 0.0,
                "method": "default",
                "all_probabilities": []
            }
        
        result = {
            "language": self.default_language,
            "confidence": 0.0,
            "method": "rule_based",
            "all_probabilities": []
        }
        
        # Try ML-based detection first
        if LANGDETECT_AVAILABLE:
            try:
                # Get all language probabilities
                lang_probs = detect_langs(text)
                
                if lang_probs:
                    best_lang = lang_probs[0]
                    result.update({
                        "language": best_lang.lang,
                        "confidence": best_lang.prob,
                        "method": "ml_detection",
                        "all_probabilities": [
                            {"language": lp.lang, "confidence": lp.prob}
                            for lp in lang_probs
                        ]
                    })
                    
                    # If confidence is high enough, return result
                    if best_lang.prob >= confidence_threshold:
                        return result
                        
            except Exception as e:
                logger.warning(f"ML language detection failed: {e}")
        
        # Fall back to rule-based detection
        rule_result = self._detect_language_rule_based(text)
        if rule_result["confidence"] > result["confidence"]:
            result.update(rule_result)
        
        return result
    
    def _detect_language_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based language detection using patterns."""
        text_lower = text.lower()
        language_scores = {}
        
        for lang_code, patterns in self.language_patterns.items():
            score = 0
            total_matches = 0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
                total_matches += matches
            
            # Normalize by text length
            text_words = len(text_lower.split())
            if text_words > 0:
                language_scores[lang_code] = score / text_words
            else:
                language_scores[lang_code] = 0
        
        # Find best match
        if language_scores:
            best_lang = max(language_scores.keys(), key=lambda k: language_scores[k])
            confidence = language_scores[best_lang]
            
            return {
                "language": best_lang,
                "confidence": min(confidence * 2, 1.0),  # Scale confidence
                "method": "rule_based",
                "all_probabilities": [
                    {"language": lang, "confidence": score}
                    for lang, score in sorted(
                        language_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                ]
            }
        
        return {
            "language": self.default_language,
            "confidence": 0.0,
            "method": "default",
            "all_probabilities": []
        }
    
    def detect_languages_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect languages for multiple texts.
        
        Args:
            texts: List of texts to analyze
            show_progress: Whether to show progress bar
            
        Returns:
            List of language detection results
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                texts_iter = tqdm(texts, desc="Detecting languages")
            except ImportError:
                texts_iter = texts
        else:
            texts_iter = texts
        
        for text in texts_iter:
            result = self.detect_language(text)
            results.append(result)
        
        return results
    
    def filter_by_language(
        self,
        texts: List[str],
        target_languages: List[str],
        min_confidence: float = 0.7
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """
        Filter texts by language.
        
        Args:
            texts: List of texts to filter
            target_languages: List of target language codes
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (filtered_texts, original_indices, detection_results)
        """
        filtered_texts = []
        original_indices = []
        detection_results = []
        
        for i, text in enumerate(texts):
            detection = self.detect_language(text)
            detection_results.append(detection)
            
            if (detection["language"] in target_languages and 
                detection["confidence"] >= min_confidence):
                filtered_texts.append(text)
                original_indices.append(i)
        
        return filtered_texts, original_indices, detection_results
    
    def extract_multilingual_content(
        self,
        text: str,
        segment_by_language: bool = True
    ) -> Dict[str, Any]:
        """
        Extract and analyze multilingual content.
        
        Args:
            text: Text potentially containing multiple languages
            segment_by_language: Whether to segment text by language
            
        Returns:
            Dictionary with multilingual analysis
        """
        result = {
            "overall_language": self.detect_language(text),
            "segments": [],
            "language_distribution": {},
            "is_multilingual": False
        }
        
        if segment_by_language:
            # Split text into sentences/paragraphs for analysis
            segments = self._split_text_into_segments(text)
            
            language_counts = Counter()
            
            for segment in segments:
                if len(segment.strip()) < 10:  # Skip very short segments
                    continue
                
                seg_detection = self.detect_language(segment)
                seg_info = {
                    "text": segment,
                    "language": seg_detection["language"],
                    "confidence": seg_detection["confidence"],
                    "start_pos": text.find(segment),
                    "length": len(segment)
                }
                
                result["segments"].append(seg_info)
                language_counts[seg_detection["language"]] += 1
            
            # Calculate language distribution
            total_segments = sum(language_counts.values())
            if total_segments > 0:
                result["language_distribution"] = {
                    lang: count / total_segments 
                    for lang, count in language_counts.items()
                }
                
                # Check if multilingual (more than one language with >10% presence)
                significant_langs = [
                    lang for lang, ratio in result["language_distribution"].items()
                    if ratio > 0.1
                ]
                result["is_multilingual"] = len(significant_langs) > 1
        
        return result
    
    def _split_text_into_segments(self, text: str) -> List[str]:
        """Split text into segments for language analysis."""
        # Split by sentences first
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # If sentences are too short, try paragraphs
        if len(sentences) > 1 and all(len(s.strip()) < 50 for s in sentences):
            paragraphs = text.split('\n\n')
            return [p.strip() for p in paragraphs if p.strip()]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def get_language_statistics(
        self,
        detection_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics from language detection results.
        
        Args:
            detection_results: List of detection results
            
        Returns:
            Statistics dictionary
        """
        if not detection_results:
            return {}
        
        # Count languages
        language_counts = Counter()
        method_counts = Counter()
        confidence_scores = []
        
        for result in detection_results:
            language_counts[result["language"]] += 1
            method_counts[result["method"]] += 1
            confidence_scores.append(result["confidence"])
        
        total_texts = len(detection_results)
        
        # Language distribution
        language_distribution = {
            lang: {
                "count": count,
                "percentage": (count / total_texts) * 100,
                "name": self.language_names.get(lang, lang)
            }
            for lang, count in language_counts.most_common()
        }
        
        # Confidence statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        low_confidence_count = sum(1 for c in confidence_scores if c < 0.5)
        
        return {
            "total_texts": total_texts,
            "unique_languages": len(language_counts),
            "language_distribution": language_distribution,
            "detection_methods": dict(method_counts),
            "confidence_stats": {
                "average": avg_confidence,
                "min": min(confidence_scores),
                "max": max(confidence_scores),
                "low_confidence_count": low_confidence_count,
                "low_confidence_percentage": (low_confidence_count / total_texts) * 100
            }
        }
    
    def translate_language_codes(
        self,
        language_codes: List[str]
    ) -> Dict[str, str]:
        """
        Translate language codes to human-readable names.
        
        Args:
            language_codes: List of language codes
            
        Returns:
            Dictionary mapping codes to names
        """
        return {
            code: self.language_names.get(code, code)
            for code in language_codes
        }
    
    def is_latin_script(self, text: str) -> bool:
        """Check if text primarily uses Latin script."""
        if not text:
            return True
        
        latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 256)
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return True
        
        return (latin_chars / total_alpha) > 0.8
    
    def extract_non_latin_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract non-Latin script segments from text."""
        segments = []
        current_segment = ""
        current_start = 0
        is_in_non_latin = False
        
        for i, char in enumerate(text):
            if char.isalpha():
                char_is_non_latin = ord(char) >= 256
                
                if char_is_non_latin and not is_in_non_latin:
                    # Starting non-Latin segment
                    if current_segment.strip():
                        # Save previous Latin segment
                        pass
                    current_segment = char
                    current_start = i
                    is_in_non_latin = True
                    
                elif not char_is_non_latin and is_in_non_latin:
                    # Ending non-Latin segment
                    if current_segment.strip():
                        segments.append({
                            "text": current_segment,
                            "start": current_start,
                            "end": i,
                            "language": self.detect_language(current_segment)
                        })
                    current_segment = char
                    current_start = i
                    is_in_non_latin = False
                    
                else:
                    # Continue current segment
                    current_segment += char
            else:
                # Non-alphabetic character
                current_segment += char
        
        # Handle final segment
        if is_in_non_latin and current_segment.strip():
            segments.append({
                "text": current_segment,
                "start": current_start,
                "end": len(text),
                "language": self.detect_language(current_segment)
            })
        
        return segments
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {"code": code, "name": name}
            for code, name in self.language_names.items()
        ]