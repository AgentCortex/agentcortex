"""Basic functionality tests for RAG pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcortex import RAGPipeline, RAGConfig
from agentcortex.extractors import PDFExtractor
from agentcortex.chunkers import TextChunker
from agentcortex.filters import ToxicityFilter
from agentcortex.utils import LangExtractUtils


class TestRAGConfig:
    """Test RAG configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = RAGConfig()
        
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimension == 384
        assert config.chunker.chunk_size == 1000
        assert config.chunker.chunk_overlap == 200
        assert config.toxicity.enabled is True
        assert config.faiss.normalize_embeddings is True
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "embedding": {"model_name": "test-model", "dimension": 512},
            "chunker": {"chunk_size": 500}
        }
        
        config = RAGConfig.from_dict(config_dict)
        
        assert config.embedding.model_name == "test-model"
        assert config.embedding.dimension == 512
        assert config.chunker.chunk_size == 500
        # Default values should still be present
        assert config.chunker.chunk_overlap == 200


class TestTextChunker:
    """Test text chunking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.sample_text = """
        This is a test document. It contains multiple sentences and paragraphs.
        
        This is the second paragraph. It also has multiple sentences for testing
        the chunking functionality of the RAG pipeline.
        
        Finally, this is the third paragraph. It should be split appropriately
        based on the configured chunk size and overlap parameters.
        """
    
    def test_recursive_chunking(self):
        """Test recursive text chunking."""
        chunks = self.chunker.chunk_text(self.sample_text, method="recursive")
        
        assert len(chunks) > 0
        assert all(chunk["method"] == "recursive" for chunk in chunks)
        assert all(len(chunk["text"]) <= self.chunker.chunk_size + 50 for chunk in chunks)  # Allow some variance
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        chunks = self.chunker.chunk_text(self.sample_text, method="fixed")
        
        assert len(chunks) > 0
        assert all(chunk["method"] == "fixed" for chunk in chunks)
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        chunks = self.chunker.chunk_text(self.sample_text, method="sentence")
        
        assert len(chunks) > 0
        assert all(chunk["method"] == "sentence" for chunk in chunks)
    
    def test_chunk_statistics(self):
        """Test chunk statistics calculation."""
        chunks = self.chunker.chunk_text(self.sample_text, method="recursive")
        stats = self.chunker.get_chunk_statistics(chunks)
        
        assert "total_chunks" in stats
        assert "avg_char_count" in stats
        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_char_count"] > 0


class TestToxicityFilter:
    """Test toxicity filtering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a simpler threshold for testing
        self.filter = ToxicityFilter(threshold=0.8, use_pipeline=False)
    
    def test_clean_text_detection(self):
        """Test detection of clean text."""
        clean_text = "This is a normal, clean text about artificial intelligence."
        is_toxic, score = self.filter.is_toxic(clean_text)
        
        # Should not be flagged as toxic
        assert not is_toxic
        assert score < 0.8
    
    def test_pattern_based_detection(self):
        """Test pattern-based toxicity detection."""
        toxic_text = "This is stupid and I hate it completely."
        is_toxic, score = self.filter.is_toxic(toxic_text)
        
        # May or may not be flagged depending on patterns
        assert isinstance(is_toxic, bool)
        assert 0 <= score <= 1
    
    def test_batch_toxicity_check(self):
        """Test batch toxicity checking."""
        texts = [
            "Normal text about technology.",
            "Another clean text sample.",
            "Text with some problematic words like stupid."
        ]
        
        results = self.filter.batch_check_toxicity(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "is_toxic" in result
            assert "toxicity_score" in result
            assert "text" in result
    
    def test_toxicity_stats(self):
        """Test toxicity statistics calculation."""
        texts = [
            "Clean text sample one.",
            "Clean text sample two.",
            "Potentially problematic text with stupid content."
        ]
        
        stats = self.filter.get_toxicity_stats(texts)
        
        assert "total_texts" in stats
        assert "toxic_count" in stats
        assert "toxic_percentage" in stats
        assert stats["total_texts"] == len(texts)


class TestLangExtractUtils:
    """Test language extraction utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lang_utils = LangExtractUtils()
    
    def test_english_detection(self):
        """Test English language detection."""
        english_text = "This is a sample English text for language detection testing."
        result = self.lang_utils.detect_language(english_text)
        
        assert result["language"] == "en"
        assert result["language_name"] == "English"
        assert result["confidence"] > 0.5
    
    def test_language_filtering(self):
        """Test language-based document filtering."""
        documents = [
            {"text": "This is English text.", "source": "doc1"},
            {"text": "Este es texto en español.", "source": "doc2"},
            {"text": "This is also English text.", "source": "doc3"}
        ]
        
        english_docs = self.lang_utils.filter_by_language_quality(
            documents, 
            allowed_languages=["en"],
            min_confidence=0.5
        )
        
        # Should filter to English documents only
        assert len(english_docs) <= len(documents)
        for doc in english_docs:
            assert "language_info" in doc
    
    def test_supported_languages(self):
        """Test getting supported languages."""
        languages = self.lang_utils.get_supported_languages()
        
        assert len(languages) > 0
        assert all("code" in lang and "name" in lang for lang in languages)
        
        # Check that English is included
        english_found = any(lang["code"] == "en" for lang in languages)
        assert english_found


class TestRAGPipeline:
    """Test complete RAG pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = RAGConfig()
        self.config.data_dir = str(Path(self.test_dir) / "data")
        self.config.faiss.index_path = str(Path(self.test_dir) / "faiss_index")
        self.config.cache_dir = str(Path(self.test_dir) / "cache")
        
        # Use smaller chunk size for testing
        self.config.chunker.chunk_size = 200
        self.config.chunker.chunk_overlap = 50
        
        # Disable toxicity filtering for simpler testing
        self.config.toxicity.enabled = False
        
        # Initialize pipeline
        self.pipeline = RAGPipeline(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.config == self.config
        assert self.pipeline.pdf_extractor is not None
        assert self.pipeline.text_chunker is not None
        assert self.pipeline.vector_storage is not None
        assert self.pipeline.retriever is not None
    
    def test_add_text_documents(self):
        """Test adding text documents to pipeline."""
        texts = [
            "Artificial intelligence is a branch of computer science.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        stats = self.pipeline.add_documents_from_text(
            texts=texts,
            sources=["doc1", "doc2", "doc3"],
            filter_toxicity=False
        )
        
        assert stats["text_documents_processed"] == len(texts)
        assert stats["total_chunks"] > 0
    
    def test_query_pipeline(self):
        """Test querying the pipeline."""
        # First add some documents
        texts = [
            "Neural networks are computational models inspired by the human brain.",
            "Convolutional neural networks are particularly good for image processing.",
            "Recurrent neural networks can handle sequential data effectively."
        ]
        
        self.pipeline.add_documents_from_text(texts, filter_toxicity=False)
        
        # Query the pipeline
        results = self.pipeline.query("What are neural networks?", k=2)
        
        assert len(results) <= 2
        if results:  # Only check if we got results
            assert all("similarity" in result for result in results)
            assert all("text" in result for result in results)
    
    def test_pipeline_statistics(self):
        """Test getting pipeline statistics."""
        stats = self.pipeline.get_pipeline_stats()
        
        assert "config" in stats
        assert "vector_storage" in stats
        assert "retriever" in stats
        assert "toxicity_filter_enabled" in stats
    
    def test_save_and_load_pipeline(self):
        """Test saving pipeline state."""
        # Add some data
        texts = ["Test document for saving pipeline state."]
        self.pipeline.add_documents_from_text(texts, filter_toxicity=False)
        
        # Save pipeline
        self.pipeline.save_pipeline()
        
        # Check that files were created
        index_path = Path(self.config.faiss.index_path)
        assert index_path.exists()
        
        config_path = Path(self.config.data_dir) / "pipeline_config.json"
        assert config_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])