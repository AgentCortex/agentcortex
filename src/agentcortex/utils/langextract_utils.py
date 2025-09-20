"""Language extraction utilities (Google langextract-like functionality)."""

import logging
import re
import requests
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import langdetect
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)


class LangExtractUtils:
    """Utilities for language detection and text extraction."""
    
    def __init__(self):
        """Initialize language extraction utilities."""
        # Common language codes and names
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'th': 'Thai',
            'vi': 'Vietnamese'
        }
        
        # Language-specific text processing patterns
        self.language_patterns = {
            'en': {
                'sentence_endings': r'[.!?]+',
                'word_separators': r'\s+',
                'common_words': {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            },
            'es': {
                'sentence_endings': r'[.!?]+',
                'word_separators': r'\s+',
                'common_words': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'}
            },
            'fr': {
                'sentence_endings': r'[.!?]+',
                'word_separators': r'\s+',
                'common_words': {'le', 'la', 'les', 'de', 'et', 'à', 'un', 'une', 'que', 'qui', 'ne', 'se'}
            },
            'de': {
                'sentence_endings': r'[.!?]+',
                'word_separators': r'\s+',
                'common_words': {'der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'ist', 'auf', 'für'}
            }
        }
    
    def detect_language(self, text: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Detect the language of given text.
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with language information
        """
        if not text.strip():
            return {
                'language': 'unknown',
                'language_name': 'Unknown',
                'confidence': 0.0,
                'is_confident': False,
                'all_possibilities': []
            }
        
        try:
            # Get all possible languages with probabilities
            lang_probs = detect_langs(text)
            
            # Get the most likely language
            primary_lang = lang_probs[0]
            
            result = {
                'language': primary_lang.lang,
                'language_name': self.language_names.get(primary_lang.lang, primary_lang.lang.upper()),
                'confidence': primary_lang.prob,
                'is_confident': primary_lang.prob >= confidence_threshold,
                'all_possibilities': [
                    {
                        'language': lang.lang,
                        'language_name': self.language_names.get(lang.lang, lang.lang.upper()),
                        'confidence': lang.prob
                    }
                    for lang in lang_probs
                ]
            }
            
            return result
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                'language': 'unknown',
                'language_name': 'Unknown',
                'confidence': 0.0,
                'is_confident': False,
                'all_possibilities': [],
                'error': str(e)
            }
    
    def detect_multiple_languages(self, text: str) -> Dict[str, Any]:
        """
        Detect multiple languages in text (basic implementation).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language information for text segments
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_languages = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only analyze sentences with sufficient content
                lang_info = self.detect_language(sentence)
                if lang_info['is_confident']:
                    sentence_languages.append({
                        'text': sentence,
                        'language': lang_info['language'],
                        'confidence': lang_info['confidence']
                    })
        
        # Count languages
        lang_counter = Counter(item['language'] for item in sentence_languages)
        
        # Determine primary language
        if lang_counter:
            primary_lang = lang_counter.most_common(1)[0][0]
            primary_name = self.language_names.get(primary_lang, primary_lang.upper())
        else:
            primary_lang = 'unknown'
            primary_name = 'Unknown'
        
        return {
            'primary_language': primary_lang,
            'primary_language_name': primary_name,
            'language_distribution': dict(lang_counter),
            'sentence_languages': sentence_languages,
            'is_multilingual': len(lang_counter) > 1,
            'total_sentences_analyzed': len(sentence_languages)
        }
    
    def extract_by_language(self, text: str, target_language: str) -> Dict[str, Any]:
        """
        Extract text segments in a specific language.
        
        Args:
            text: Text to analyze
            target_language: Language code to extract (e.g., 'en', 'es')
            
        Returns:
            Dictionary with extracted segments
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        extracted_sentences = []
        other_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                lang_info = self.detect_language(sentence)
                
                if (lang_info['language'] == target_language and 
                    lang_info['is_confident']):
                    extracted_sentences.append({
                        'text': sentence,
                        'confidence': lang_info['confidence']
                    })
                else:
                    other_sentences.append({
                        'text': sentence,
                        'detected_language': lang_info['language'],
                        'confidence': lang_info['confidence']
                    })
        
        # Join extracted sentences
        extracted_text = '. '.join(item['text'] for item in extracted_sentences)
        
        return {
            'target_language': target_language,
            'target_language_name': self.language_names.get(target_language, target_language.upper()),
            'extracted_text': extracted_text,
            'extracted_sentences': extracted_sentences,
            'other_sentences': other_sentences,
            'extraction_stats': {
                'total_sentences': len(sentences),
                'extracted_count': len(extracted_sentences),
                'other_count': len(other_sentences),
                'extraction_ratio': len(extracted_sentences) / len(sentences) if sentences else 0
            }
        }
    
    def analyze_text_language_features(self, text: str) -> Dict[str, Any]:
        """
        Analyze language-specific features of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language feature analysis
        """
        lang_info = self.detect_language(text)
        language = lang_info['language']
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Language-specific analysis
        features = {
            'basic_stats': {
                'word_count': word_count,
                'character_count': char_count,
                'sentence_count': sentence_count,
                'avg_word_length': char_count / word_count if word_count > 0 else 0,
                'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
            },
            'language_info': lang_info,
            'language_features': {}
        }
        
        # Add language-specific features if we have patterns for this language
        if language in self.language_patterns:
            patterns = self.language_patterns[language]
            
            # Count common words
            words = text.lower().split()
            common_word_count = sum(1 for word in words if word in patterns['common_words'])
            
            features['language_features'] = {
                'common_word_count': common_word_count,
                'common_word_ratio': common_word_count / word_count if word_count > 0 else 0,
                'uses_language_patterns': True
            }
        else:
            features['language_features'] = {
                'uses_language_patterns': False
            }
        
        return features
    
    def filter_by_language_quality(
        self, 
        documents: List[Dict[str, Any]], 
        min_confidence: float = 0.8,
        allowed_languages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter documents based on language detection quality.
        
        Args:
            documents: List of documents with 'text' field
            min_confidence: Minimum language detection confidence
            allowed_languages: List of allowed language codes (None = all)
            
        Returns:
            Filtered list of documents with language metadata
        """
        filtered_docs = []
        
        for doc in documents:
            text = doc.get('text', '')
            if not text.strip():
                continue
            
            # Detect language
            lang_info = self.detect_language(text, min_confidence)
            
            # Check confidence
            if not lang_info['is_confident']:
                continue
            
            # Check allowed languages
            if allowed_languages and lang_info['language'] not in allowed_languages:
                continue
            
            # Add language metadata to document
            filtered_doc = doc.copy()
            filtered_doc['language_info'] = lang_info
            filtered_docs.append(filtered_doc)
        
        return filtered_docs
    
    def translate_language_codes(self, language_codes: List[str]) -> List[Dict[str, str]]:
        """
        Translate language codes to human-readable names.
        
        Args:
            language_codes: List of language codes
            
        Returns:
            List of dictionaries with code and name
        """
        return [
            {
                'code': code,
                'name': self.language_names.get(code, code.upper())
            }
            for code in language_codes
        ]
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {'code': code, 'name': name}
            for code, name in self.language_names.items()
        ]
    
    def batch_detect_languages(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect languages for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of language detection results
        """
        results = []
        
        for i, text in enumerate(texts):
            lang_info = self.detect_language(text)
            result = {
                'index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                **lang_info
            }
            results.append(result)
        
        return results