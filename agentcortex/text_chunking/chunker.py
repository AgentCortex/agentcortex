"""Text chunking functionality with multiple strategies."""

import re
import logging
from typing import List, Optional, Callable
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    start_index: int
    end_index: int
    word_count: int
    char_count: int
    token_count: Optional[int] = None
    source_page: Optional[int] = None


class TextChunker:
    """
    Flexible text chunking with multiple strategies.
    
    Supports sentence-aware chunking, token-based chunking,
    and fixed-size chunking with overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        strategy: str = "sentence_aware",
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk (tokens or characters)
            overlap_size: Overlap between consecutive chunks
            strategy: Chunking strategy ("sentence_aware", "token_based", "fixed_size")
            encoding_name: Tokenizer encoding name for token counting
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.strategy = strategy
        self.encoding_name = encoding_name
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {encoding_name}: {e}")
        
        # Download NLTK data if needed
        if NLTK_AVAILABLE and strategy == "sentence_aware":
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
    
    def chunk_text(self, text: str, source_page: Optional[int] = None) -> List[tuple]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Text to chunk
            source_page: Optional source page number
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        if self.strategy == "sentence_aware":
            return self._chunk_sentence_aware(text, source_page)
        elif self.strategy == "token_based":
            return self._chunk_token_based(text, source_page)
        elif self.strategy == "fixed_size":
            return self._chunk_fixed_size(text, source_page)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_sentence_aware(self, text: str, source_page: Optional[int] = None) -> List[tuple]:
        """Chunk text while respecting sentence boundaries."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, falling back to fixed-size chunking")
            return self._chunk_fixed_size(text, source_page)
        
        chunks = []
        sentences = sent_tokenize(text)
        
        current_chunk = ""
        current_start = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._get_size(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunk_start = text.find(current_chunk, current_start)
                    chunk_end = chunk_start + len(current_chunk)
                    
                    metadata = ChunkMetadata(
                        start_index=chunk_start,
                        end_index=chunk_end,
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        token_count=self._count_tokens(current_chunk),
                        source_page=source_page
                    )
                    chunks.append((current_chunk, metadata))
                    
                    # Start new chunk with overlap
                    if self.overlap_size > 0:
                        overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                        current_chunk = overlap_text + " " + sentence
                        current_start = chunk_end - len(overlap_text)
                    else:
                        current_chunk = sentence
                        current_start = chunk_end
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk_start = text.find(current_chunk, current_start)
            chunk_end = chunk_start + len(current_chunk)
            
            metadata = ChunkMetadata(
                start_index=chunk_start,
                end_index=chunk_end,
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                token_count=self._count_tokens(current_chunk),
                source_page=source_page
            )
            chunks.append((current_chunk, metadata))
        
        return chunks
    
    def _chunk_token_based(self, text: str, source_page: Optional[int] = None) -> List[tuple]:
        """Chunk text based on token count."""
        if not self.tokenizer:
            logger.warning("Tokenizer not available, falling back to character-based chunking")
            return self._chunk_fixed_size(text, source_page)
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find actual text positions
            chunk_start = text.find(chunk_text)
            if chunk_start == -1:
                # Fallback to approximate position
                chunk_start = int((start_idx / len(tokens)) * len(text))
            chunk_end = chunk_start + len(chunk_text)
            
            metadata = ChunkMetadata(
                start_index=chunk_start,
                end_index=chunk_end,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                token_count=len(chunk_tokens),
                source_page=source_page
            )
            chunks.append((chunk_text, metadata))
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_size
            if start_idx >= end_idx:
                break
        
        return chunks
    
    def _chunk_fixed_size(self, text: str, source_page: Optional[int] = None) -> List[tuple]:
        """Chunk text into fixed-size pieces based on characters."""
        chunks = []
        
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            
            # Try to break at word boundary
            if end_idx < len(text) and not text[end_idx].isspace():
                # Find last space in chunk
                last_space = chunk_text.rfind(' ')
                if last_space > len(chunk_text) * 0.8:  # Only if it's near the end
                    end_idx = start_idx + last_space
                    chunk_text = text[start_idx:end_idx]
            
            metadata = ChunkMetadata(
                start_index=start_idx,
                end_index=end_idx,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                token_count=self._count_tokens(chunk_text),
                source_page=source_page
            )
            chunks.append((chunk_text, metadata))
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_size
            if start_idx >= end_idx:
                break
        
        return chunks
    
    def _get_size(self, text: str) -> int:
        """Get size of text based on current strategy."""
        if self.strategy == "token_based" and self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text)
    
    def _count_tokens(self, text: str) -> Optional[int]:
        """Count tokens in text if tokenizer is available."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return None
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if self.strategy == "token_based" and self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= overlap_size:
                return text
            overlap_tokens = tokens[-overlap_size:]
            return self.tokenizer.decode(overlap_tokens)
        else:
            if len(text) <= overlap_size:
                return text
            return text[-overlap_size:]
    
    def chunk_multiple_texts(
        self,
        texts: List[str],
        source_pages: Optional[List[int]] = None
    ) -> List[tuple]:
        """
        Chunk multiple texts (e.g., from different pages).
        
        Args:
            texts: List of texts to chunk
            source_pages: Optional list of source page numbers
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        all_chunks = []
        
        for i, text in enumerate(texts):
            page_num = source_pages[i] if source_pages else None
            chunks = self.chunk_text(text, page_num)
            all_chunks.extend(chunks)
        
        return all_chunks