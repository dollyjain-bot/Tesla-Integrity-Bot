"""
Ingestion package: Document loading and text chunking.
"""

from ingestion.loaders import DocumentLoader
from ingestion.chunker import TextChunker

__all__ = ["DocumentLoader", "TextChunker"]
