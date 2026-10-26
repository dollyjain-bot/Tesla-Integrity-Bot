"""
Tesla Integrity Bot - Embedding Manager Module
Wraps LangChain HuggingFaceEmbeddings (sentence-transformers) to generate dense 384-dimensional vector embeddings.
"""

from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager(Embeddings):
    """
    EmbeddingManager converts unstructured text chunks and user queries
    into dense numerical vector embeddings for semantic similarity search.
    It is itself a LangChain Embeddings object, so vector stores can use it directly while the model stays lazy.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy loads the LangChain embeddings model into memory (also used by the Chroma vector store)."""
        if self._embeddings is None:
            print(f"Loading embedding model '{self.model_name}'...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            print(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Converts text chunks into numerical vectors in batch."""
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Converts user query into a single 384-dimensional vector."""
        if not query or not query.strip():
            return []
        return self.embeddings.embed_query(query)
