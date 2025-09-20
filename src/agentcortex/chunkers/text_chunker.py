"""Text chunking functionality for RAG pipeline."""

import logging
import re
from typing import List, Dict, Any, Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
    # Create a simple fallback text splitter
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None, keep_separator=True):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]
            self.keep_separator = keep_separator
        
        def split_text(self, text: str) -> List[str]:
            """Simple text splitting fallback."""
            chunks = []
            current_chunk = ""
            
            # Simple word-based splitting
            words = text.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                    current_chunk += (" " if current_chunk else "") + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


class TextChunker:
    """Intelligent text chunking with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        tokenizer_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
            keep_separator: Whether to keep separators in chunks
            tokenizer_name: Name of tokenizer for token-based chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.tokenizer_name = tokenizer_name
        
        # Default separators for recursive splitting
        if separators is None:
            self.separators = [
                "\n\n",      # Paragraphs
                "\n",        # Lines
                " ",         # Words
                ".",         # Sentences
                "!",
                "?",
                ";",
                ",",
                ""           # Characters
            ]
        else:
            self.separators = separators
        
        # Initialize LangChain text splitter
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=keep_separator
        )
        
        # Initialize tokenizer for token-based chunking
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_name)
            except Exception as e:
                logger.warning(f"Could not load tokenizer {tokenizer_name}: {e}")
                self.tokenizer = None
        else:
            logger.warning("tiktoken not available. Token-based chunking disabled.")
            self.tokenizer = None
    
    def chunk_text(self, text: str, method: str = "recursive") -> List[Dict[str, Any]]:
        """
        Chunk text using specified method.
        
        Args:
            text: Text to chunk
            method: Chunking method ('recursive', 'fixed', 'sentence', 'token')
            
        Returns:
            List of chunks with metadata
        """
        if not text.strip():
            return []
        
        if method == "recursive":
            return self._chunk_recursive(text)
        elif method == "fixed":
            return self._chunk_fixed(text)
        elif method == "sentence":
            return self._chunk_sentence(text)
        elif method == "token":
            return self._chunk_token(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _chunk_recursive(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using recursive character text splitter."""
        chunks = self.langchain_splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "chunk_id": i,
                "method": "recursive",
                "char_count": len(chunk),
                "token_count": self._count_tokens(chunk) if self.tokenizer else None,
                "start_pos": text.find(chunk) if chunk in text else None
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def _chunk_fixed(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text using fixed-size windows."""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "chunk_id": chunk_id,
                    "method": "fixed",
                    "char_count": len(chunk),
                    "token_count": self._count_tokens(chunk) if self.tokenizer else None,
                    "start_pos": i
                })
                chunk_id += 1
        
        return chunks
    
    def _chunk_sentence(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by sentences with size constraints."""
        # Split into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk,
                        "chunk_id": chunk_id,
                        "method": "sentence",
                        "char_count": len(current_chunk),
                        "token_count": self._count_tokens(current_chunk) if self.tokenizer else None,
                        "start_pos": start_pos
                    })
                    chunk_id += 1
                    start_pos = text.find(current_chunk, start_pos) + len(current_chunk)
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk,
                "chunk_id": chunk_id,
                "method": "sentence",
                "char_count": len(current_chunk),
                "token_count": self._count_tokens(current_chunk) if self.tokenizer else None,
                "start_pos": start_pos
            })
        
        return chunks
    
    def _chunk_token(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text based on token count."""
        if not TIKTOKEN_AVAILABLE or not self.tokenizer:
            logger.error("Tokenizer not available for token-based chunking, falling back to recursive")
            return self._chunk_recursive(text)
        
        # Encode text to tokens
        tokens = self.tokenizer.encode(text)
        token_chunk_size = self.chunk_size // 4  # Rough estimate: 4 chars per token
        token_overlap = self.chunk_overlap // 4
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(tokens), token_chunk_size - token_overlap):
            chunk_tokens = tokens[i:i + token_chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "method": "token",
                    "char_count": len(chunk_text),
                    "token_count": len(chunk_tokens),
                    "start_pos": None  # Difficult to calculate with tokenization
                })
                chunk_id += 1
        
        return chunks
    
    def _count_tokens(self, text: str) -> Optional[int]:
        """Count tokens in text."""
        if not self.tokenizer:
            return None
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return None
    
    def chunk_documents(
        self, 
        documents: List[Dict[str, Any]], 
        method: str = "recursive"
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'text' field
            method: Chunking method to use
            
        Returns:
            List of chunks with document metadata
        """
        all_chunks = []
        
        for doc_id, document in enumerate(documents):
            text = document.get("text", "")
            if not text.strip():
                continue
            
            chunks = self.chunk_text(text, method)
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    "document_id": doc_id,
                    "source": document.get("source", "unknown"),
                    "document_metadata": {
                        k: v for k, v in document.items() 
                        if k not in ["text"]
                    }
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        char_counts = [chunk["char_count"] for chunk in chunks]
        token_counts = [chunk["token_count"] for chunk in chunks if chunk["token_count"]]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_char_count": sum(char_counts) / len(char_counts),
            "min_char_count": min(char_counts),
            "max_char_count": max(char_counts),
            "total_chars": sum(char_counts)
        }
        
        if token_counts:
            stats.update({
                "avg_token_count": sum(token_counts) / len(token_counts),
                "min_token_count": min(token_counts),
                "max_token_count": max(token_counts),
                "total_tokens": sum(token_counts)
            })
        
        return stats