"""LangChain integration for RAG retrieval."""

import logging
from typing import List, Dict, Any, Optional

try:
    from langchain.schema import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain_core.language_models import LLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        from langchain.retrievers.base import BaseRetriever
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
        from langchain.vectorstores.base import VectorStore
        from langchain.embeddings.base import Embeddings
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.llms.base import LLM
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        # Create dummy classes if LangChain is not available
        LANGCHAIN_AVAILABLE = False
        
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}
        
        class BaseRetriever:
            pass
        
        class CallbackManagerForRetrieverRun:
            pass
        
        class VectorStore:
            pass
        
        class Embeddings:
            pass
        
        class RetrievalQA:
            pass
        
        class PromptTemplate:
            pass
        
        class LLM:
            pass

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """Wrapper to make FAISSStorage compatible with LangChain VectorStore."""
    
    def __init__(self, faiss_storage):
        """Initialize with FAISSStorage instance."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. VectorStore functionality limited.")
        self.faiss_storage = faiss_storage
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ) -> List[str]:
        """Add texts to the vector store."""
        # Convert to chunks format expected by FAISSStorage
        chunks = []
        for i, text in enumerate(texts):
            chunk = {
                "text": text,
                "chunk_id": i,
                "method": "external"
            }
            if metadatas and i < len(metadatas):
                chunk.update(metadatas[i])
            chunks.append(chunk)
        
        self.faiss_storage.add_documents(chunks)
        return [str(i) for i in range(len(texts))]
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs
    ) -> List[Document]:
        """Search for similar documents."""
        results = self.faiss_storage.search(query, k, return_similarities=False)
        
        documents = []
        for result in results:
            # Extract text and metadata
            text = result.pop("text", "")
            metadata = result  # Everything else is metadata
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        return documents
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs
    ) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        results = self.faiss_storage.search(query, k, return_similarities=True)
        
        documents_with_scores = []
        for result in results:
            # Extract text, score, and metadata
            text = result.pop("text", "")
            score = result.pop("similarity", 0.0)
            metadata = result  # Everything else is metadata
            
            document = Document(
                page_content=text,
                metadata=metadata
            )
            documents_with_scores.append((document, score))
        
        return documents_with_scores
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ):
        """Create vector store from texts (not implemented)."""
        raise NotImplementedError("Use FAISSStorage directly for initialization")


class CustomRetriever(BaseRetriever):
    """Custom retriever using FAISSStorage."""
    
    def __init__(self, faiss_storage, k: int = 4):
        """Initialize retriever."""
        super().__init__()
        self.faiss_storage = faiss_storage
        self.k = k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents."""
        results = self.faiss_storage.search(query, self.k, return_similarities=False)
        
        documents = []
        for result in results:
            # Extract text and metadata
            text = result.pop("text", "")
            metadata = result  # Everything else is metadata
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        return documents


class LangChainRetriever:
    """LangChain integration for RAG retrieval and QA."""
    
    def __init__(
        self,
        faiss_storage,
        llm: Optional[LLM] = None,
        prompt_template: Optional[str] = None,
        retriever_k: int = 4
    ):
        """
        Initialize LangChain retriever.
        
        Args:
            faiss_storage: FAISSStorage instance
            llm: Language model for QA
            prompt_template: Custom prompt template
            retriever_k: Number of documents to retrieve
        """
        self.faiss_storage = faiss_storage
        self.llm = llm
        self.retriever_k = retriever_k
        
        # Create vector store wrapper
        self.vector_store = FAISSVectorStore(faiss_storage)
        
        # Create retriever
        self.retriever = CustomRetriever(faiss_storage, k=retriever_k)
        
        # Default prompt template
        if prompt_template is None:
            self.prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        else:
            self.prompt_template = prompt_template
        
        # Initialize QA chain if LLM is provided
        self.qa_chain = None
        if llm is not None:
            self._initialize_qa_chain()
    
    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain with LLM."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. QA chain functionality disabled.")
            return
            
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            List of relevant documents with metadata
        """
        if k is None:
            k = self.retriever_k
        
        return self.faiss_storage.search(query, k, return_similarities=True)
    
    def retrieve_and_format(self, query: str, k: Optional[int] = None) -> str:
        """
        Retrieve documents and format them as context.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            source = result.get("source", "Unknown")
            similarity = result.get("similarity", 0.0)
            
            context_parts.append(
                f"Document {i} (similarity: {similarity:.3f}, source: {source}):\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using retrieved context.
        
        Args:
            question: Question to answer
            k: Number of documents to retrieve for context
            
        Returns:
            Dictionary with answer and source documents
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Returning documents only.")
            results = self.retrieve(question, k)
            return {
                "question": question,
                "answer": "LangChain not available. Please review the relevant documents.",
                "source_documents": results
            }
            
        if self.qa_chain is None:
            raise ValueError("No LLM provided. Cannot generate answers without LLM.")
        
        # Override retriever k if specified
        if k is not None:
            original_k = self.retriever.k
            self.retriever.k = k
        
        try:
            result = self.qa_chain({"query": question})
            
            # Format source documents
            source_docs = []
            for doc in result.get("source_documents", []):
                source_docs.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "question": question,
                "answer": result.get("result", ""),
                "source_documents": source_docs
            }
        
        finally:
            # Restore original k
            if k is not None:
                self.retriever.k = original_k
    
    def batch_answer(self, questions: List[str], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            k: Number of documents to retrieve per question
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        for question in questions:
            try:
                answer = self.answer_question(question, k)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                answers.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "source_documents": []
                })
        
        return answers
    
    def update_prompt_template(self, new_template: str) -> None:
        """
        Update the prompt template.
        
        Args:
            new_template: New prompt template string
        """
        self.prompt_template = new_template
        
        if self.llm is not None:
            self._initialize_qa_chain()
    
    def set_llm(self, llm: LLM) -> None:
        """
        Set or update the language model.
        
        Args:
            llm: New language model instance
        """
        self.llm = llm
        self._initialize_qa_chain()
    
    def get_similar_questions(self, query: str, k: int = 5) -> List[str]:
        """
        Find documents that might contain similar questions or topics.
        
        Args:
            query: Query to find similar content for
            k: Number of similar documents to retrieve
            
        Returns:
            List of text snippets from similar documents
        """
        results = self.retrieve(query, k)
        return [result.get("text", "") for result in results]
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        faiss_stats = self.faiss_storage.get_stats()
        
        return {
            "retriever_k": self.retriever_k,
            "has_llm": self.llm is not None,
            "has_qa_chain": self.qa_chain is not None,
            "prompt_template": self.prompt_template,
            **faiss_stats
        }