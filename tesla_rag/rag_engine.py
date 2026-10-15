"""
Tesla Integrity Bot - RAG Engine Module (Modular Facade)
Re-exports core RAG components from modular subpackages (ingestion, models, storage, core).
Maintains 100% backwards compatibility with all existing scripts and tests.
"""

from config import RAGConfig
from ingestion.loaders import DocumentLoader
from ingestion.chunker import TextChunker
from models.embeddings import EmbeddingManager
from models.generator import GroqGenerator
from storage.vector_store import VectorStore
from core.engine import RAGEngine

__all__ = [
    "RAGConfig",
    "DocumentLoader",
    "TextChunker",
    "EmbeddingManager",
    "GroqGenerator",
    "VectorStore",
    "RAGEngine",
]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = RAGEngine(data_dir="data", vector_dir="./vector_store")
    test_query = "What are the rules regarding spatial hashing for customer video feeds?"
    print("\n--- TESTING MODULAR RAG ENGINE ---")
    res = engine.answer_query(query=test_query, top_k=3)
    try:
        print("Answer:\n", res["answer"])
    except UnicodeEncodeError:
        print("Answer:\n", res["answer"].encode("ascii", "replace").decode("ascii"))
