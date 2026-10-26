"""
Tesla Integrity Bot - Core RAG Engine Module
Orchestrates document loading, text chunking, embedding generation, vector storage, and Groq LLM synthesis.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from config import RAGConfig
from core.retriever import ThresholdRetriever, document_to_chunk
from ingestion.loaders import DocumentLoader
from ingestion.chunker import TextChunker
from models.embeddings import EmbeddingManager
from models.generator import GroqGenerator
from storage.vector_store import VectorStore


class RAGEngine:
    """
    RAGEngine orchestrates the core RAG pipeline:
    Document Ingestion -> Chunking -> Embeddings -> ChromaDB Vector Store -> Retrieval -> Groq Generation.
    """

    def __init__(
        self,
        data_dir: str = "data",
        vector_dir: str = "./vector_store",
        embedding_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
    ):
        defaults = RAGConfig()
        emb_model = embedding_model_name or defaults.embedding_model_name
        model_llm = llm_model_name or defaults.llm_model_name

        self.config = RAGConfig(
            data_dir=Path(data_dir),
            vector_dir=vector_dir,
            embedding_model_name=emb_model,
            llm_model_name=model_llm,
        )
        self.data_dir = self.config.data_dir
        self.vector_dir = self.config.vector_dir

        self.loader = DocumentLoader(data_dir=self.data_dir)
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.embedding_manager = EmbeddingManager(model_name=emb_model)
        self.generator = GroqGenerator(model_name=model_llm)
        self._vector_store = None
        # LangChain chain: query -> retriever (Top-K + threshold) -> Documents -> generate (prompt | Groq LLM)
        self._rag_pipeline = (
            RunnablePassthrough.assign(documents=RunnableLambda(self._retrieve_documents))
            | RunnableLambda(self._generate)
        )

    @property
    def vector_store(self) -> VectorStore:
        """Lazy loads ChromaDB VectorStore instance."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                embedding_function=self.embedding_manager,
                persist_directory=self.vector_dir,
                collection_name=self.config.collection_name,
            )
        return self._vector_store

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Delegates to DocumentLoader."""
        return self.loader.load_all_documents()

    def chunk_documents(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """Delegates to TextChunker."""
        if documents is None:
            documents = self.load_all_documents()
        return self.chunker.chunk_documents(
            documents, chunk_size=chunk_size, overlap=overlap
        )

    def build_or_update_index(self, chunk_size: int = 500, overlap: int = 100) -> int:
        """Runs the complete ingestion pipeline: Load -> Chunk -> Embed -> Add to ChromaDB."""
        print("--- RAG ENGINE: INDEXING PIPELINE STARTED ---")
        docs = self.load_all_documents()
        if not docs:
            print("No documents found to index.")
            return 0

        chunks = self.chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
        count = self.vector_store.add_chunks(chunks)
        print(f"--- RAG ENGINE: INDEXING COMPLETED ({count} chunks indexed) ---")
        return count

    def _retrieve_documents(self, inputs: Dict[str, Any]) -> List[Document]:
        """Pipeline steps 1-3: auto-index if empty, then Top-K retrieval from ChromaDB with the similarity threshold."""
        if self.vector_store.count() == 0:
            print("Vector store is empty. Automatically building index from data directory...")
            self.build_or_update_index()

        retriever = ThresholdRetriever(
            vector_store=self.vector_store,
            top_k=inputs["top_k"],
            similarity_threshold=inputs["similarity_threshold"],
            metadata_filter=inputs["metadata_filter"],
        )
        return retriever.invoke(inputs["query"])

    def _generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step 4: generate the answer via the Groq LLM and assemble the response dictionary."""
        relevant_chunks = [document_to_chunk(d) for d in state["documents"]]
        gen_result = self.generator.generate_answer(
            query=state["query"],
            retrieved_chunks=relevant_chunks,
            temperature=state["temperature"],
            top_p=state["top_p"],
        )

        metadata_list = [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "document_id": c.get("document_id"),
                "section": c.get("section"),
                "classification": c.get("classification"),
                "effective_date": c.get("effective_date"),
                "similarity_score": c.get("similarity_score"),
            }
            for c in relevant_chunks
        ]

        return {
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "retrieved_chunks": relevant_chunks,
            "metadata": metadata_list,
        }

    def answer_query(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.1,
        similarity_threshold: float = 0.35,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline:
        1. Embed user query string
        2. Retrieve Top-K semantically similar chunks from ChromaDB
        3. Filter chunks by similarity threshold
        4. Generate grounded answer via Groq LLM
        """
        return self._rag_pipeline.invoke(
            {
                "query": query,
                "top_k": top_k,
                "temperature": temperature,
                "top_p": top_p,
                "similarity_threshold": similarity_threshold,
                "metadata_filter": metadata_filter,
            }
        )
