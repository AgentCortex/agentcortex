"""
AgentCortex: Comprehensive RAG Pipeline

A powerful toolkit for building Retrieval-Augmented Generation (RAG) systems
with PDF processing, vector storage, and LLM integration.
"""

__version__ = "0.1.0"
__author__ = "AgentCortex"
__email__ = "hello@agentcortex.com"

# Core imports
from .core.pipeline import RAGPipeline
from .core.config import RAGConfig

# Component imports
from .extractors.pdf_extractor import PDFExtractor
from .chunkers.text_chunker import TextChunker
from .storage.faiss_storage import FAISSStorage
from .retrieval.langchain_retriever import LangChainRetriever
from .filters.toxicity_filter import ToxicityFilter

__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "PDFExtractor",
    "TextChunker",
    "FAISSStorage",
    "LangChainRetriever",
    "ToxicityFilter",
]