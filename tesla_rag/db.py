"""
Tesla Integrity Bot - Vector Database Module (ChromaDB Integration)
Facade re-exporting VectorStore from the modular storage package.
"""

from storage.vector_store import VectorStore

__all__ = ["VectorStore"]
