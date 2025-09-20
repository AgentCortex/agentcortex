# AgentCortex Quick Start Guide

Welcome to AgentCortex! This guide will get you up and running with the RAG pipeline in minutes.

## Prerequisites

- Python 3.9 or higher
- Internet connection (for downloading models on first run)

## Installation

```bash
# Clone the repository
git clone https://github.com/AgentCortex/agentcortex.git
cd agentcortex

# Install dependencies with uv
pip install uv
uv sync

# Or install with pip
pip install -e .
```

## 5-Minute Tutorial

### 1. Basic Text Processing

```python
from agentcortex import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Add some documents
texts = [
    "Python is a powerful programming language for AI development.",
    "Machine learning models require large datasets for training.",
    "Natural language processing helps computers understand human language."
]

# Process and store documents
stats = pipeline.add_documents_from_text(texts)
print(f"Added {stats['total_chunks']} chunks from {stats['text_documents_processed']} documents")

# Query the pipeline
results = pipeline.query("What is Python used for?", k=2)
for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['text']}")
    print("-" * 50)
```

### 2. PDF Processing

```python
# Process PDF files
pdf_files = ["document1.pdf", "research_paper.pdf"]
stats = pipeline.add_documents_from_pdfs(pdf_files)

# Query across all documents
results = pipeline.query("Tell me about the research findings", k=3)
```

### 3. Custom Configuration

```python
from agentcortex import RAGConfig

# Create custom configuration
config = RAGConfig()
config.chunker.chunk_size = 500  # Smaller chunks
config.embedding.model_name = "all-mpnet-base-v2"  # Better embeddings

# Initialize with custom config
pipeline = RAGPipeline(config)
```

## Common Use Cases

### Document Q&A System

```python
# Load your documents
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
pipeline.add_documents_from_pdfs(documents)

# Ask questions
question = "What are the main conclusions of the study?"
results = pipeline.query(question, k=5)

# Process results
for i, result in enumerate(results, 1):
    print(f"Result {i}: {result['source']}")
    print(f"Content: {result['text'][:200]}...")
```

### Content Safety

```python
from agentcortex.filters import ToxicityFilter

# Initialize toxicity filter
filter = ToxicityFilter(threshold=0.7)

# Check content safety
texts = ["This is a normal text", "This contains harmful content"]
for text in texts:
    is_toxic, score = filter.is_toxic(text)
    print(f"Toxic: {is_toxic}, Score: {score:.3f}")
```

### Multi-language Support

```python
from agentcortex.utils import LangExtractUtils

lang_utils = LangExtractUtils()

# Detect language
text = "This is an English sentence"
result = lang_utils.detect_language(text)
print(f"Language: {result['language_name']} ({result['confidence']:.3f})")

# Filter by language
english_docs = lang_utils.filter_by_language_quality(
    documents, 
    allowed_languages=['en']
)
```

## Example Scripts

Run the included examples to see more features:

```bash
# Basic RAG example
python examples/basic_rag_example.py

# PDF processing example
python examples/pdf_processing_example.py sample.pdf

# Advanced features demonstration
python examples/advanced_rag_example.py

# Offline functionality test (no internet required)
python examples/offline_test.py
```

## Configuration Options

### Quick Config with JSON

Create a `config.json` file:

```json
{
  "chunker": {"chunk_size": 800, "chunk_overlap": 100},
  "embedding": {"model_name": "all-MiniLM-L6-v2"},
  "toxicity": {"enabled": true, "threshold": 0.8}
}
```

Load it:

```python
config = RAGConfig.from_file("config.json")
pipeline = RAGPipeline(config)
```

### Environment Variables

```bash
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
export CHUNK_SIZE="1200"
export DATA_DIR="./my_data"
```

```python
config = RAGConfig.from_env()
pipeline = RAGPipeline(config)
```

## Troubleshooting

### Model Download Issues

If you get connection errors:

1. Check your internet connection
2. Try running in offline mode first: `python examples/offline_test.py`
3. Set up proxy if needed: `export https_proxy=your_proxy`

### Memory Issues

For large documents:

```python
config = RAGConfig()
config.embedding.batch_size = 16  # Reduce batch size
config.chunker.chunk_size = 500   # Smaller chunks
```

### Performance Optimization

```python
config = RAGConfig()
config.faiss.index_type = "IndexHNSW"  # Faster search
config.embedding.device = "cuda"       # Use GPU if available
```

## Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Explore the examples/** directory for advanced use cases
3. **Check the tests/** directory to understand the API
4. **Customize the configuration** for your specific needs
5. **Integrate with your application** using the simple API

## Getting Help

- 📖 Documentation: Check README.md and docstrings
- 🐛 Issues: [GitHub Issues](https://github.com/AgentCortex/agentcortex/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/AgentCortex/agentcortex/discussions)

Happy coding with AgentCortex! 🚀