"""
Tesla Integrity Bot - Embedding Manager Module
Wraps SentenceTransformer to generate dense 384-dimensional vector embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    EmbeddingManager converts unstructured text chunks and user queries
    into dense numerical vector embeddings for semantic similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loads SentenceTransformer model into memory."""
        if self._model is None:
            print(f"Loading embedding model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Converts text chunks into numerical vectors in batch."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return [arr.tolist() for arr in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Converts user query into a single 384-dimensional vector."""
        if not query or not query.strip():
            return []
        query_embedding = self.model.encode(
            query, show_progress_bar=False, convert_to_numpy=True
        )
        return query_embedding.tolist()
