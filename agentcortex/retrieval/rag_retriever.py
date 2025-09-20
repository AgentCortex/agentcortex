"""RAG retriever with LangChain integration."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores.base import VectorStore as LangChainVectorStore
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..vector_storage import VectorStore
from ..text_chunking import TextChunker, ChunkMetadata

logger = logging.getLogger(__name__)


class CustomVectorStoreRetriever(BaseRetriever):
    """Custom retriever that wraps our VectorStore for LangChain compatibility."""
    
    def __init__(self, vector_store: VectorStore, search_kwargs: Optional[Dict] = None):
        """
        Initialize retriever.
        
        Args:
            vector_store: Our custom vector store
            search_kwargs: Additional search parameters
        """
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {}
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = self.search_kwargs.get("k", 4)
        score_threshold = self.search_kwargs.get("score_threshold")
        
        results = self.vector_store.search(
            query=query,
            k=k,
            score_threshold=score_threshold
        )
        
        documents = []
        for content, score, metadata in results:
            # Create LangChain Document
            doc_metadata = metadata.copy()
            doc_metadata["score"] = score
            
            doc = Document(
                page_content=content,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        return documents


class RAGRetriever:
    """
    Complete RAG retriever with document processing, storage, and retrieval.
    
    Integrates PDF extraction, text chunking, vector storage, and LangChain retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "sentence_aware",
        vector_store_type: str = "flat",
        distance_metric: str = "cosine"
    ):
        """
        Initialize RAG retriever.
        
        Args:
            embedding_model: Embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Text chunking strategy
            vector_store_type: FAISS index type
            distance_metric: Distance metric for similarity
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            overlap_size=chunk_overlap,
            strategy=chunking_strategy
        )
        
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            index_type=vector_store_type,
            metric=distance_metric
        )
        
        # LangChain components
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.retriever = CustomVectorStoreRetriever(self.vector_store)
        
        logger.info("RAG retriever initialized")
    
    def add_documents_from_text(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        source_pages: Optional[List[int]] = None
    ) -> None:
        """
        Add documents from raw text.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each document
            source_pages: Optional source page numbers
        """
        logger.info(f"Processing {len(texts)} documents")
        
        all_chunks = []
        all_metadata = []
        
        for i, text in enumerate(texts):
            # Get base metadata
            base_metadata = metadata[i] if metadata else {}
            source_page = source_pages[i] if source_pages else None
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(text, source_page)
            
            for chunk_text, chunk_metadata in chunks:
                # Combine base metadata with chunk metadata
                combined_metadata = base_metadata.copy()
                combined_metadata.update({
                    "chunk_start": chunk_metadata.start_index,
                    "chunk_end": chunk_metadata.end_index,
                    "word_count": chunk_metadata.word_count,
                    "char_count": chunk_metadata.char_count,
                    "token_count": chunk_metadata.token_count,
                    "source_page": chunk_metadata.source_page,
                    "original_doc_index": i
                })
                
                all_chunks.append(chunk_text)
                all_metadata.append(combined_metadata)
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=all_chunks,
            metadata=all_metadata
        )
        
        logger.info(f"Added {len(all_chunks)} chunks to vector store")
    
    def add_pdf_documents(
        self,
        pdf_paths: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add PDF documents to the retriever.
        
        Args:
            pdf_paths: List of PDF file paths
            metadata: Optional metadata for each PDF
        """
        from ..pdf_extraction import PDFExtractor
        
        extractor = PDFExtractor(preserve_layout=True)
        texts = []
        all_metadata = []
        all_source_pages = []
        
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text from PDF
            try:
                pages = extractor.extract_pages(pdf_path)
                pdf_info = extractor.get_pdf_info(pdf_path)
                
                # Get base metadata
                base_metadata = metadata[i] if metadata else {}
                base_metadata.update({
                    "source_file": pdf_path,
                    "pdf_title": pdf_info.get("title", ""),
                    "pdf_author": pdf_info.get("author", ""),
                    "total_pages": len(pages)
                })
                
                # Add each page
                for page_num, page_text in enumerate(pages):
                    if page_text.strip():  # Only add non-empty pages
                        texts.append(page_text)
                        page_metadata = base_metadata.copy()
                        page_metadata["page_number"] = page_num + 1
                        all_metadata.append(page_metadata)
                        all_source_pages.append(page_num + 1)
                
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")
                continue
        
        if texts:
            self.add_documents_from_text(texts, all_metadata, all_source_pages)
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filter_metadata: Metadata filters (not implemented yet)
            
        Returns:
            List of relevant documents
        """
        # Update retriever search parameters
        self.retriever.search_kwargs = {
            "k": k,
            "score_threshold": score_threshold
        }
        
        # Retrieve documents
        documents = self.retriever._get_relevant_documents(query, run_manager=None)
        
        # Apply metadata filtering if specified
        if filter_metadata:
            filtered_docs = []
            for doc in documents:
                match = True
                for key, value in filter_metadata.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            documents = filtered_docs
        
        return documents
    
    def create_qa_chain(self, llm, chain_type: str = "stuff") -> RetrievalQA:
        """
        Create a question-answering chain.
        
        Args:
            llm: Language model for question answering
            chain_type: Type of QA chain ("stuff", "map_reduce", "refine")
            
        Returns:
            RetrievalQA chain
        """
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def create_compression_retriever(self, llm) -> ContextualCompressionRetriever:
        """
        Create a contextual compression retriever.
        
        Args:
            llm: Language model for compression
            
        Returns:
            Contextual compression retriever
        """
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        
        return compression_retriever
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.search(query, k)
        
        doc_score_pairs = []
        for content, score, metadata in results:
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model_name
        }
    
    def save(self, save_path: str) -> None:
        """Save the retriever to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(str(save_path / "vector_store"))
        
        # Save configuration
        import json
        config = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_strategy": self.text_chunker.strategy
        }
        
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"RAG retriever saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """Load the retriever from disk."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Retriever not found at {load_path}")
        
        # Load configuration
        import json
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Update configuration
        self.embedding_model_name = config["embedding_model"]
        self.chunk_size = config["chunk_size"] 
        self.chunk_overlap = config["chunk_overlap"]
        
        # Reinitialize components
        self.text_chunker = TextChunker(
            chunk_size=self.chunk_size,
            overlap_size=self.chunk_overlap,
            strategy=config["chunking_strategy"]
        )
        
        # Load vector store
        self.vector_store.load(str(load_path / "vector_store"))
        
        # Update retriever
        self.retriever = CustomVectorStoreRetriever(self.vector_store)
        
        logger.info(f"RAG retriever loaded from {load_path}")