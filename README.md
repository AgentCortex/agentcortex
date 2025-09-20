# AgentCortex

AgentCortex is a comprehensive RAG (Retrieval-Augmented Generation) pipeline that provides robust text extraction, processing, and retrieval capabilities. It integrates multiple state-of-the-art libraries and techniques to create a powerful and flexible system for AI-powered document analysis and question answering.

## Features

- **PDF Text Extraction**: Robust PDF text extraction using multiple libraries (PyMuPDF, pdfplumber, PyPDF2)
- **Intelligent Text Chunking**: Multiple chunking strategies (sentence-aware, token-based, fixed-size) with overlap support
- **Vector Storage**: FAISS-based vector storage with multiple index types and distance metrics
- **LangChain Integration**: Seamless integration with LangChain for retrieval and QA chains
- **Hugging Face Tools**: Model management, embeddings, and evaluation with bitsandbytes optimization
- **Toxicity Filtering**: ML-based and rule-based toxicity detection and content cleaning
- **Language Detection**: Multi-language support with Google langextract functionality
- **Configuration Management**: Flexible configuration system with environment variable support

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/AgentCortex/agentcortex.git
cd agentcortex

# Install with uv
uv sync

# For GPU support (optional)
uv sync --extra gpu

# For additional NLP features (optional)
uv sync --extra nlp
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Basic RAG Pipeline

```python
from agentcortex import RAGRetriever

# Initialize retriever
retriever = RAGRetriever(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000,
    chunk_overlap=200
)

# Add documents
documents = [
    "Machine learning is a method of data analysis...",
    "Natural language processing enables computers..."
]

retriever.add_documents_from_text(documents)

# Search for relevant content
results = retriever.retrieve("What is machine learning?", k=3)

for doc in results:
    print(f"Score: {doc.metadata['score']:.3f}")
    print(f"Content: {doc.page_content[:200]}...")
```

### PDF Processing

```python
from agentcortex import PDFExtractor, TextChunker

# Extract text from PDF
extractor = PDFExtractor(preserve_layout=True)
text = extractor.extract_text("document.pdf")

# Chunk the text
chunker = TextChunker(
    chunk_size=500,
    overlap_size=50,
    strategy="sentence_aware"
)

chunks = chunker.chunk_text(text)
print(f"Created {len(chunks)} chunks")
```

### Toxicity Filtering

```python
from agentcortex import ToxicityFilter

# Initialize filter
toxicity_filter = ToxicityFilter(
    toxicity_threshold=0.7,
    use_rule_based=True
)

# Filter content
texts = ["This is a great article!", "This content is inappropriate..."]
clean_texts, analysis = toxicity_filter.filter_dataset(
    texts,
    clean_toxic=True
)

print(f"Cleaned {len(texts)} texts, flagged {len(texts) - len(clean_texts)} as toxic")
```

## Architecture

AgentCortex is built with a modular architecture:

```
agentcortex/
├── pdf_extraction/     # PDF text extraction
├── text_chunking/      # Text chunking strategies
├── vector_storage/     # FAISS vector storage
├── retrieval/          # RAG retrieval system
├── huggingface_tools/  # HF model integration
├── toxicity_filter/    # Content filtering
├── langextract/        # Language detection
└── utils/              # Utilities and config
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_rag_example.py`: Basic RAG pipeline usage
- `pdf_processing_example.py`: PDF extraction and chunking
- `toxicity_filtering_example.py`: Content moderation
- `complete_rag_pipeline.py`: End-to-end pipeline with all features

Run an example:

```bash
cd examples
python basic_rag_example.py
```

## Configuration

AgentCortex supports flexible configuration through files and environment variables:

### Configuration File (YAML)

```yaml
rag:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 1000
  chunk_overlap: 200
  chunking_strategy: "sentence_aware"
  vector_store_type: "flat"
  distance_metric: "cosine"
  toxicity_threshold: 0.7
  use_toxicity_filter: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

```bash
export AGENTCORTEX_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export AGENTCORTEX_CHUNK_SIZE=1000
export AGENTCORTEX_TOXICITY_THRESHOLD=0.7
```

### Python Configuration

```python
from agentcortex.utils import Config

config = Config("config.yaml")
rag_config = config.get_rag_config()

# Override specific settings
config.set("rag.chunk_size", 800)
```

## Advanced Features

### Vector Storage Options

```python
from agentcortex import VectorStore

# Flat index (exact search)
store = VectorStore(
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat",
    metric="cosine"
)

# IVF index (fast approximate search)
store = VectorStore(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    index_type="ivf",
    metric="euclidean"
)

# HNSW index (memory efficient)
store = VectorStore(
    embedding_model="all-MiniLM-L6-v2",
    index_type="hnsw",
    metric="cosine"
)
```

### Hugging Face Model Management

```python
from agentcortex.huggingface_tools import ModelManager

# Load model with quantization
model_manager = ModelManager(
    model_name="microsoft/DialoGPT-medium",
    model_type="causal_lm",
    use_quantization=True
)

# Generate text
response = model_manager.generate_text(
    "Hello, how are you?",
    max_length=100,
    temperature=0.7
)
```

### Language Detection

```python
from agentcortex.langextract import LanguageExtractor

lang_extractor = LanguageExtractor()

# Detect language
result = lang_extractor.detect_language("Hello world")
print(f"Language: {result['language']}, Confidence: {result['confidence']}")

# Filter by language
english_texts, indices, results = lang_extractor.filter_by_language(
    texts=multilingual_texts,
    target_languages=["en"],
    min_confidence=0.8
)
```

### Evaluation and Metrics

```python
from agentcortex.huggingface_tools import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate retrieval performance
retrieval_metrics = evaluator.evaluate_retrieval(
    retrieved_docs=retrieved_documents,
    relevant_docs=ground_truth_documents,
    k_values=[1, 3, 5, 10]
)

print(f"MAP: {retrieval_metrics['map']:.3f}")
print(f"MRR: {retrieval_metrics['mrr']:.3f}")
```

## Performance Tips

1. **Use appropriate chunk sizes**: Smaller chunks (200-500 tokens) for precise retrieval, larger chunks (1000+ tokens) for context-rich retrieval.

2. **Choose the right vector index**: 
   - Flat index for small datasets (<10K documents)
   - IVF index for medium datasets (10K-1M documents)
   - HNSW index for large datasets (>1M documents)

3. **Enable quantization for large models**:
   ```python
   model_manager = ModelManager(
       model_name="microsoft/DialoGPT-large",
       use_quantization=True
   )
   ```

4. **Batch processing for efficiency**:
   ```python
   # Process multiple texts at once
   results = toxicity_filter.batch_detect(texts, batch_size=32)
   ```

## Development

### Setting up for Development

```bash
# Clone and install in development mode
git clone https://github.com/AgentCortex/agentcortex.git
cd agentcortex

# Install with development dependencies
uv sync --group dev

# Run tests
pytest

# Format code
black agentcortex/
flake8 agentcortex/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_pdf_extraction.py
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=agentcortex --cov-report=html
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **FAISS**: For efficient similarity search
- **LangChain**: For retrieval and LLM integration
- **Hugging Face**: For transformer models and tools
- **Sentence Transformers**: For embedding generation
- **PyMuPDF, pdfplumber, PyPDF2**: For PDF processing
- **Detoxify**: For toxicity detection

## Citation

If you use AgentCortex in your research, please cite:

```bibtex
@software{agentcortex2024,
  title={AgentCortex: A Comprehensive RAG Pipeline},
  author={AgentCortex Team},
  year={2024},
  url={https://github.com/AgentCortex/agentcortex}
}
```

---

For more information, visit our [documentation](https://agentcortex.readthedocs.io) or join our [community discussions](https://github.com/AgentCortex/agentcortex/discussions).
