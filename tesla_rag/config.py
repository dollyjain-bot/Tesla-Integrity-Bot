"""
Tesla Integrity Bot - Centralized Configuration Module
Maintains global settings, model parameters, similarity thresholds, and directory paths.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    """Centralized configuration for the RAG pipeline."""
    # Chunking configuration
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval configuration
    top_k: int = 3
    similarity_threshold: float = 0.35

    # Model names
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model_name: str = "openai/gpt-oss-120b"

    # Generation defaults
    temperature: float = 0.0
    top_p: float = 0.1

    # File and directory paths
    data_dir: Path = Path("data")
    vector_dir: str = "./vector_store"
    collection_name: str = "tesla_compliance_policies"
