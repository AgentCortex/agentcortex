#!/usr/bin/env python3
"""
Basic functionality tests for AgentCortex components.

These tests verify the core functionality without requiring 
internet connections or model downloads.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import components for testing
from agentcortex.text_chunking import TextChunker, ChunkMetadata
from agentcortex.toxicity_filter import ToxicityFilter
from agentcortex.langextract import LanguageExtractor
from agentcortex.utils import Config


class TestTextChunking:
    """Test text chunking functionality."""
    
    def test_fixed_size_chunking(self):
        """Test fixed-size text chunking."""
        chunker = TextChunker(
            chunk_size=50,
            overlap_size=10,
            strategy="fixed_size"
        )
        
        text = "This is a test document. " * 10
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, tuple) for chunk in chunks)
        assert all(len(chunk) == 2 for chunk in chunks)
        
        # Check chunk structure
        chunk_text, metadata = chunks[0]
        assert isinstance(chunk_text, str)
        assert isinstance(metadata, ChunkMetadata)
        assert metadata.char_count > 0
        assert metadata.word_count > 0
    
    def test_sentence_aware_chunking(self):
        """Test sentence-aware chunking (without NLTK if not available)."""
        chunker = TextChunker(
            chunk_size=100,
            overlap_size=20,
            strategy="sentence_aware"
        )
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        chunk_text, metadata = chunks[0]
        assert len(chunk_text) > 0
        assert metadata.char_count == len(chunk_text)
    
    def test_multiple_texts_chunking(self):
        """Test chunking multiple texts."""
        chunker = TextChunker(chunk_size=30, overlap_size=5)
        
        texts = [
            "First document text here.",
            "Second document content.",
            "Third document material."
        ]
        
        chunks = chunker.chunk_multiple_texts(texts, source_pages=[1, 2, 3])
        
        assert len(chunks) >= len(texts)
        for chunk_text, metadata in chunks:
            assert metadata.source_page in [1, 2, 3]


class TestToxicityFilter:
    """Test toxicity filtering functionality."""
    
    def test_basic_detection(self):
        """Test basic toxicity detection."""
        toxicity_filter = ToxicityFilter(
            toxicity_threshold=0.7,
            use_rule_based=True
        )
        
        safe_text = "This is a nice and helpful message."
        result = toxicity_filter.detect_toxicity(safe_text)
        
        # Should be False (not toxic) for safe text
        assert isinstance(result, bool)
    
    def test_rule_based_filtering(self):
        """Test rule-based toxicity detection."""
        toxicity_filter = ToxicityFilter(use_rule_based=True)
        
        # Add custom toxic words for testing
        toxicity_filter.update_filters(new_explicit_words=["badword"])
        
        toxic_text = "This contains a badword in it."
        safe_text = "This is a perfectly fine message."
        
        toxic_result = toxicity_filter.detect_toxicity(toxic_text, return_scores=True)
        safe_result = toxicity_filter.detect_toxicity(safe_text, return_scores=True)
        
        assert toxic_result["is_toxic"] == True
        assert safe_result["is_toxic"] == False
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        toxicity_filter = ToxicityFilter()
        toxicity_filter.update_filters(new_explicit_words=["badword"])
        
        text = "This contains a badword that should be cleaned."
        cleaned = toxicity_filter.clean_text(text, replacement_strategy="replace", 
                                            replacement_text="[FILTERED]")
        
        assert "badword" not in cleaned
        assert "[FILTERED]" in cleaned
    
    def test_dataset_filtering(self):
        """Test dataset filtering."""
        toxicity_filter = ToxicityFilter()
        toxicity_filter.update_filters(new_explicit_words=["toxic"])
        
        texts = [
            "This is a good message.",
            "This contains toxic content.",
            "Another good message."
        ]
        
        clean_texts, analysis = toxicity_filter.filter_dataset(
            texts, remove_toxic=True
        )
        
        assert len(clean_texts) <= len(texts)
        assert len(analysis) == len(texts)


class TestLanguageExtractor:
    """Test language detection functionality."""
    
    def test_basic_detection(self):
        """Test basic language detection."""
        lang_extractor = LanguageExtractor()
        
        english_text = "This is an English sentence."
        result = lang_extractor.detect_language(english_text)
        
        assert isinstance(result, dict)
        assert "language" in result
        assert "confidence" in result
        assert "method" in result
    
    def test_batch_detection(self):
        """Test batch language detection."""
        lang_extractor = LanguageExtractor()
        
        texts = [
            "This is English text.",
            "Hello world, how are you?",
            "Machine learning is interesting."
        ]
        
        results = lang_extractor.detect_languages_batch(texts, show_progress=False)
        
        assert len(results) == len(texts)
        assert all("language" in result for result in results)
    
    def test_language_filtering(self):
        """Test language-based filtering."""
        lang_extractor = LanguageExtractor()
        
        texts = [
            "This is English text.",
            "Hola, como estas?",  # Spanish
            "Bonjour, comment allez-vous?",  # French
            "Another English sentence."
        ]
        
        english_texts, indices, results = lang_extractor.filter_by_language(
            texts, target_languages=["en"], min_confidence=0.1
        )
        
        assert len(english_texts) <= len(texts)
        assert len(indices) == len(english_texts)
        assert len(results) == len(texts)


class TestConfiguration:
    """Test configuration management."""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = Config()
        
        # Test default RAG config
        rag_config = config.get_rag_config()
        assert rag_config.chunk_size > 0
        assert rag_config.chunk_overlap >= 0
        assert rag_config.chunking_strategy in ["sentence_aware", "token_based", "fixed_size"]
    
    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        config = Config()
        test_config = {
            "rag": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model": "test-model"
            }
        }
        
        config.load_from_dict(test_config)
        rag_config = config.get_rag_config()
        
        assert rag_config.chunk_size == 500
        assert rag_config.chunk_overlap == 50
        assert rag_config.embedding_model == "test-model"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Set invalid configuration
        config.set("rag.chunk_size", -1)
        config.set("rag.chunk_overlap", 2000)  # Larger than chunk_size
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("chunk_size must be positive" in error for error in errors)
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files."""
        config = Config()
        config.set("rag.chunk_size", 800)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_path, format="json")
            assert os.path.exists(temp_path)
            
            # Load configuration
            new_config = Config(temp_path)
            assert new_config.get("rag.chunk_size") == 800
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_imports():
    """Test that all major components can be imported."""
    # Test basic imports
    from agentcortex import PDFExtractor, TextChunker, VectorStore, RAGRetriever, ToxicityFilter
    from agentcortex.utils import setup_logging, Config
    from agentcortex.langextract import LanguageExtractor
    from agentcortex.huggingface_tools import ModelEvaluator
    
    # If we can import them, the structure is correct
    assert PDFExtractor is not None
    assert TextChunker is not None
    assert VectorStore is not None
    assert RAGRetriever is not None
    assert ToxicityFilter is not None
    assert setup_logging is not None
    assert Config is not None
    assert LanguageExtractor is not None
    assert ModelEvaluator is not None


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])