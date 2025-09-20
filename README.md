# AgentCortex

A comprehensive RAG (Retrieval-Augmented Generation) pipeline with PDF processing, vector storage, and LLM integration.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

✅ **Complete RAG Pipeline**: End-to-end retrieval-augmented generation system
✅ **PDF Text Extraction**: Advanced PDF processing with multiple backends
✅ **Intelligent Text Chunking**: Multiple chunking strategies (recursive, fixed, sentence-based)
✅ **FAISS Vector Storage**: High-performance similarity search and storage
✅ **LangChain Integration**: Seamless integration with LangChain ecosystem
✅ **Hugging Face Tools**: Transformers, embeddings, and model integration
✅ **Toxicity Filtering**: Automatic content filtering for safe AI applications
✅ **Language Detection**: Multi-language support with automatic detection
✅ **Flexible Configuration**: JSON-based configuration system
✅ **Production Ready**: Comprehensive error handling and logging

## Quick Start

### Installation

```bash
# Install uv (Python package manager)
pip install uv

# Clone the repository
git clone https://github.com/AgentCortex/agentcortex.git
cd agentcortex

# Install dependencies
uv sync
```

### Basic Usage

```python
from agentcortex import RAGPipeline, RAGConfig

# Initialize pipeline with default configuration
pipeline = RAGPipeline()

# Add documents from text
texts = [
    "Artificial intelligence is transforming healthcare through advanced diagnostics.",
    "Machine learning algorithms can predict patient outcomes with high accuracy.",
    "Natural language processing helps extract insights from medical records."
]

# Process and add to vector store
stats = pipeline.add_documents_from_text(
    texts=texts,
    sources=["healthcare_ai", "ml_medicine", "nlp_medical"],
    chunking_method="recursive"
)

# Query the pipeline
results = pipeline.query("How is AI used in healthcare?", k=3)

for result in results:
    print(f"Source: {result['source']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['text'][:100]}...")
    print("-" * 50)
```

### PDF Processing

```python
# Process PDF files
pdf_files = ["document1.pdf", "document2.pdf"]
stats = pipeline.add_documents_from_pdfs(
    pdf_paths=pdf_files,
    chunking_method="sentence",
    filter_toxicity=True
)

print(f"Processed {stats['successful_extractions']} PDFs")
print(f"Created {stats['total_chunks']} chunks")
```

## Configuration

### Using JSON Configuration

```python
from agentcortex import RAGPipeline, RAGConfig

# Load from configuration file
config = RAGConfig.from_file("config.json")
pipeline = RAGPipeline(config)

# Or from environment variables
config = RAGConfig.from_env()
pipeline = RAGPipeline(config)
```

### Sample Configuration

```json
{
  "embedding": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu",
    "batch_size": 32
  },
  "faiss": {
    "index_type": "IndexFlatIP",
    "index_path": "./faiss_index",
    "normalize_embeddings": true
  },
  "chunker": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n",
    "keep_separator": true
  },
  "toxicity": {
    "model_name": "unitary/toxic-bert",
    "threshold": 0.7,
    "enabled": true
  }
}
```

## Advanced Features

### Language Detection and Filtering

```python
from agentcortex.utils import LangExtractUtils

lang_utils = LangExtractUtils()

# Detect language
result = lang_utils.detect_language("This is English text")
print(f"Language: {result['language_name']} (confidence: {result['confidence']:.3f})")

# Filter documents by language
english_docs = lang_utils.filter_by_language_quality(
    documents,
    allowed_languages=['en'],
    min_confidence=0.8
)
```

### Toxicity Filtering

```python
from agentcortex.filters import ToxicityFilter

toxicity_filter = ToxicityFilter(threshold=0.7)

# Check if text is toxic
is_toxic, score = toxicity_filter.is_toxic("Some text to check")
print(f"Toxic: {is_toxic}, Score: {score:.3f}")

# Filter documents
clean_docs = toxicity_filter.filter_documents(documents)
```

### Multiple Chunking Strategies

```python
from agentcortex.chunkers import TextChunker

chunker = TextChunker(chunk_size=800, chunk_overlap=200)

# Different chunking methods
recursive_chunks = chunker.chunk_text(text, method="recursive")
sentence_chunks = chunker.chunk_text(text, method="sentence")
fixed_chunks = chunker.chunk_text(text, method="fixed")
token_chunks = chunker.chunk_text(text, method="token")
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_rag_example.py` - Simple RAG pipeline usage
- `pdf_processing_example.py` - PDF document processing
- `advanced_rag_example.py` - Advanced features demonstration

Run examples:

```bash
cd examples
python basic_rag_example.py
python pdf_processing_example.py sample.pdf
python advanced_rag_example.py
```

## API Reference

### RAGPipeline

Main pipeline class that orchestrates all components.

#### Methods

- `add_documents_from_pdfs(pdf_paths, chunking_method, filter_toxicity)` - Process PDF files
- `add_documents_from_text(texts, sources, chunking_method, filter_toxicity)` - Add text documents
- `query(query, k, filter_query_toxicity)` - Query for relevant documents
- `answer_question(question, k, filter_question_toxicity)` - Answer questions using LLM
- `get_pipeline_stats()` - Get comprehensive statistics
- `save_pipeline()` - Save pipeline state
- `clear_pipeline()` - Clear all data

### Components

#### PDFExtractor
- Supports multiple backends (pdfplumber, PyPDF2)
- Extracts text and metadata from PDF files
- Handles batch processing

#### TextChunker
- Multiple chunking strategies
- Configurable chunk size and overlap
- Token-aware chunking with tiktoken

#### FAISSStorage
- High-performance vector similarity search
- Multiple index types (Flat, IVF, HNSW)
- Persistent storage and loading

#### ToxicityFilter
- Transformer-based toxicity detection
- Pattern-based fallback filtering
- Configurable thresholds

#### LangExtractUtils
- Automatic language detection
- Multi-language document filtering
- Language-specific text processing

## Testing

Run the test suite:

```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/agentcortex
```

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/AgentCortex/agentcortex.git
cd agentcortex

# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Project Structure

```
agentcortex/
├── src/agentcortex/           # Main package
│   ├── core/                  # Core pipeline components
│   ├── extractors/            # Text extraction modules
│   ├── chunkers/              # Text chunking modules
│   ├── storage/               # Vector storage modules
│   ├── retrieval/             # Retrieval and QA modules
│   ├── filters/               # Content filtering modules
│   └── utils/                 # Utility modules
├── examples/                  # Usage examples
├── tests/                     # Test suite
├── docs/                      # Documentation
└── pyproject.toml            # Project configuration
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 **Documentation**: [Coming Soon]
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/AgentCortex/agentcortex/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/AgentCortex/agentcortex/discussions)
- 📧 **Email**: hello@agentcortex.com

## Roadmap

- [ ] Web interface for pipeline management
- [ ] Additional vector database backends (Pinecone, Weaviate)
- [ ] More LLM integrations (OpenAI, Anthropic, Cohere)
- [ ] Advanced query optimization
- [ ] Distributed processing support
- [ ] Real-time document ingestion
- [ ] Multi-modal support (images, audio)

---

Built with ❤️ by the AgentCortex team
