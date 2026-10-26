"""
Tesla Integrity Bot - Retriever Module
LangChain retriever that runs Top-K cosine search on ChromaDB and then applies the similarity threshold.
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation
from storage.vector_store import VectorStore


def chunk_to_document(chunk: Dict[str, Any]) -> Document:
    """Converts a retrieved chunk dict into a LangChain Document (chunk_id -> id, retrieved_text -> page_content)."""
    metadata = {k: v for k, v in chunk.items() if k not in ("chunk_id", "retrieved_text")}
    return Document(id=chunk["chunk_id"], page_content=chunk["retrieved_text"], metadata=metadata)


def document_to_chunk(document: Document) -> Dict[str, Any]:
    """Inverse of chunk_to_document; returns the chunk dict shape used by the generator and the Streamlit UI."""
    return {"chunk_id": document.id, "retrieved_text": document.page_content, **document.metadata}


class ThresholdRetriever(BaseRetriever):
    """Top-K retrieval followed by a hard similarity cut-off (applied after Top-K, keeping scores >= threshold)."""

    model_config = {"arbitrary_types_allowed": True}

    vector_store: SkipValidation[VectorStore]  # duck-typed: anything with query_similar_chunks()
    top_k: int = 3
    similarity_threshold: float = 0.35
    metadata_filter: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        candidates = self.vector_store.query_similar_chunks(
            query=query,
            top_k=self.top_k,
            metadata_filter=self.metadata_filter,
        )
        return [
            chunk_to_document(c)
            for c in candidates
            if c.get("similarity_score", 0.0) >= self.similarity_threshold
        ]
