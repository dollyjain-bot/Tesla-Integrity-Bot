"""
Tesla Integrity Bot - Vector Database Module (ChromaDB Integration)
Manages persistent ChromaDB vector store collections, indexing chunk embeddings, and querying top-K chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb


class VectorStore:
    """
    VectorStore wraps ChromaDB to manage collection initialization,
    chunk indexing with vector embeddings, and similarity-based retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "./vector_store",
        collection_name: str = "tesla_compliance_policies",
    ):
        """
        Initialize the persistent ChromaDB client and collection.

        Args:
            persist_directory (str): Path to directory where ChromaDB files are saved.
            collection_name (str): Name of the vector collection.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection configured with Cosine Similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"ChromaDB VectorStore initialized. Collection: '{self.collection_name}' | Storage: '{self.persist_directory}'"
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Indexes text chunks, metadata, and numerical embeddings into ChromaDB.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries containing chunk_id, source, document_id, etc.
            embeddings (List[List[float]]): Corresponding numerical vector embeddings.

        Returns:
            int: Total number of chunks indexed.
        """
        if not chunks or not embeddings:
            print("No chunks or embeddings provided for indexing.")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: Got {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        ids = []
        documents = []
        metadatas = []
        valid_embeddings = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["chunk_text"]

            metadata_entry = {
                "source": str(chunk.get("source", "unknown")),
                "document_id": str(chunk.get("document_id") or chunk.get("doc_id") or "UNKNOWN"),
                "section": str(chunk.get("section", "Full Document")),
                "classification": str(chunk.get("classification", "CONFIDENTIAL")),
                "effective_date": str(chunk.get("effective_date", "N/A")),
            }

            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append(metadata_entry)
            valid_embeddings.append(embedding)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=valid_embeddings,
            metadatas=metadatas,
        )

        count = len(ids)
        print(f"Successfully indexed {count} chunk vectors into ChromaDB collection '{self.collection_name}'.")
        return count

    def query_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for the top-K most semantically similar chunks.

        Args:
            query_embedding (List[float]): 384-dimensional vector embedding of the query.
            top_k (int): Number of top results to retrieve (default: 3).
            metadata_filter (Optional[Dict[str, Any]]): Metadata filtering dictionary.

        Returns:
            List[Dict[str, Any]]: List of retrieved result dictionaries with similarity scores.
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if metadata_filter:
            query_params["where"] = metadata_filter

        results = self.collection.query(**query_params)

        retrieved_chunks = []
        if (
            results
            and "documents" in results
            and results["documents"]
            and len(results["documents"][0]) > 0
        ):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for chunk_id, doc_text, meta, distance in zip(
                ids, docs, metas, distances
            ):
                # Calculate Cosine Similarity score from Cosine Distance: similarity = 1.0 - distance
                similarity_score = max(0.0, min(1.0, 1.0 - distance))

                retrieved_chunk = {
                    "chunk_id": chunk_id,
                    "retrieved_text": doc_text,
                    "source": meta.get("source", "N/A"),
                    "document_id": meta.get("document_id", "N/A"),
                    "section": meta.get("section", "N/A"),
                    "classification": meta.get("classification", "N/A"),
                    "effective_date": meta.get("effective_date", "N/A"),
                    "cosine_distance": round(distance, 4),
                    "similarity_score": round(similarity_score, 4),
                }
                retrieved_chunks.append(retrieved_chunk)

        return retrieved_chunks

    def count(self) -> int:
        """Returns the total number of chunks stored in the ChromaDB collection."""
        return self.collection.count()
