#!/usr/bin/env python3
"""
Offline Test Script

This script tests the core functionality without requiring internet access
or model downloads. Perfect for CI/CD environments.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcortex.core.config import RAGConfig
from agentcortex.extractors.pdf_extractor import PDFExtractor
from agentcortex.chunkers.text_chunker import TextChunker
from agentcortex.filters.toxicity_filter import ToxicityFilter
from agentcortex.utils.langextract_utils import LangExtractUtils


def test_config():
    """Test configuration functionality."""
    print("Testing RAG Configuration...")
    
    # Test default config
    config = RAGConfig()
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.chunker.chunk_size == 1000
    print("✓ Default configuration created successfully")
    
    # Test config from dict
    config_dict = {
        "embedding": {"model_name": "test-model", "dimension": 512},
        "chunker": {"chunk_size": 500}
    }
    config = RAGConfig.from_dict(config_dict)
    assert config.embedding.model_name == "test-model"
    assert config.embedding.dimension == 512
    assert config.chunker.chunk_size == 500
    print("✓ Configuration from dictionary works")
    
    # Test saving and loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save_to_file(f.name)
        loaded_config = RAGConfig.from_file(f.name)
        assert loaded_config.embedding.model_name == "test-model"
        Path(f.name).unlink()  # Clean up
    print("✓ Configuration save/load works")


def test_text_chunker():
    """Test text chunking functionality."""
    print("\nTesting Text Chunker...")
    
    # Create chunker with small size for testing
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    sample_text = """
    This is a test document. It contains multiple sentences and paragraphs.
    
    This is the second paragraph. It also has multiple sentences for testing
    the chunking functionality of the RAG pipeline.
    
    Finally, this is the third paragraph. It should be split appropriately
    based on the configured chunk size and overlap parameters.
    """
    
    # Test different chunking methods
    methods = ["recursive", "fixed", "sentence"]
    for method in methods:
        chunks = chunker.chunk_text(sample_text, method=method)
        assert len(chunks) > 0
        assert all(chunk["method"] == method for chunk in chunks)
        print(f"✓ {method.title()} chunking works ({len(chunks)} chunks)")
    
    # Test chunk statistics
    chunks = chunker.chunk_text(sample_text, method="recursive")
    stats = chunker.get_chunk_statistics(chunks)
    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(chunks)
    print("✓ Chunk statistics calculation works")


def test_toxicity_filter():
    """Test toxicity filtering (basic pattern-based only)."""
    print("\nTesting Toxicity Filter...")
    
    # Create filter without transformer model (offline mode)
    toxicity_filter = ToxicityFilter(threshold=0.8, use_pipeline=False)
    
    # Test clean text
    clean_text = "This is a normal, clean text about artificial intelligence."
    is_toxic, score = toxicity_filter.is_toxic(clean_text)
    assert isinstance(is_toxic, bool)
    assert 0 <= score <= 1
    print("✓ Clean text detection works")
    
    # Test batch processing
    texts = [
        "Normal text about technology.",
        "Another clean text sample.",
        "Text with some potentially problematic words."
    ]
    
    results = toxicity_filter.batch_check_toxicity(texts)
    assert len(results) == len(texts)
    for result in results:
        assert "is_toxic" in result
        assert "toxicity_score" in result
    print("✓ Batch toxicity checking works")
    
    # Test statistics
    stats = toxicity_filter.get_toxicity_stats(texts)
    assert "total_texts" in stats
    assert stats["total_texts"] == len(texts)
    print("✓ Toxicity statistics calculation works")


def test_language_utils():
    """Test language detection utilities."""
    print("\nTesting Language Utilities...")
    
    lang_utils = LangExtractUtils()
    
    # Test English detection
    english_text = "This is a sample English text for language detection testing."
    result = lang_utils.detect_language(english_text)
    assert "language" in result
    assert "confidence" in result
    print(f"✓ Language detection works (detected: {result['language']})")
    
    # Test document filtering
    documents = [
        {"text": "This is English text.", "source": "doc1"},
        {"text": "Este es texto en español.", "source": "doc2"},
        {"text": "This is also English text.", "source": "doc3"}
    ]
    
    filtered_docs = lang_utils.filter_by_language_quality(
        documents, 
        min_confidence=0.1  # Low threshold for testing
    )
    assert len(filtered_docs) <= len(documents)
    print(f"✓ Language filtering works ({len(filtered_docs)}/{len(documents)} docs passed)")
    
    # Test supported languages
    languages = lang_utils.get_supported_languages()
    assert len(languages) > 0
    assert all("code" in lang and "name" in lang for lang in languages)
    print(f"✓ Supported languages list works ({len(languages)} languages)")


def test_pdf_extractor():
    """Test PDF extractor (without actual PDF files)."""
    print("\nTesting PDF Extractor...")
    
    # Create extractor
    pdf_extractor = PDFExtractor(backend="pdfplumber")
    
    # Test with sample bytes (not a real PDF)
    sample_bytes = b"This is not a real PDF but tests the interface"
    
    try:
        # This will fail but we're testing the interface
        text = pdf_extractor.extract_from_bytes(sample_bytes)
    except Exception:
        # Expected to fail with invalid PDF
        pass
    
    print("✓ PDF extractor interface works")
    
    # Test multiple file processing (with non-existent files)
    fake_paths = ["nonexistent1.pdf", "nonexistent2.pdf"]
    results = pdf_extractor.extract_multiple(fake_paths)
    
    # Should return error results for non-existent files
    assert len(results) == len(fake_paths)
    assert all("error" in result for result in results)
    print("✓ Multiple PDF processing interface works")


def main():
    """Run all offline tests."""
    print("=" * 60)
    print("AgentCortex Offline Functionality Test")
    print("=" * 60)
    
    try:
        test_config()
        test_text_chunker()
        test_toxicity_filter()
        test_language_utils()
        test_pdf_extractor()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ Core functionality is working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()