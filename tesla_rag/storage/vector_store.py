"""
Tesla Integrity Bot - Vector Database Module (ChromaDB Integration)
Manages the persistent ChromaDB collection through the LangChain Chroma integration:
indexing chunks and querying top-K chunks with cosine similarity scores.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStore:
    """
    VectorStore wraps langchain_chroma.Chroma to manage collection initialization,
    chunk indexing with vector embeddings, and similarity-based retrieval.
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str = "./vector_store",
        collection_name: str = "tesla_compliance_policies",
    ):
        """
        Attach to the persistent ChromaDB collection (created only if it does not exist yet).

        Args:
            embedding_function (Embeddings): LangChain embeddings used to embed queries and new chunks.
            persist_directory (str): Path to directory where ChromaDB files are saved.
            collection_name (str): Name of the vector collection.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Get or create collection configured with Cosine Similarity space
        self.store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(
            f"ChromaDB VectorStore initialized. Collection: '{self.collection_name}' | Storage: '{self.persist_directory}'"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and indexes text chunks and metadata into ChromaDB.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries containing chunk_id, source, document_id, etc.

        Returns:
            int: Total number of chunks indexed.
        """
        if not chunks:
            print("No chunks provided for indexing.")
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            metadata_entry = {
                "source": str(chunk.get("source", "unknown")),
                "document_id": str(chunk.get("document_id") or chunk.get("doc_id") or "UNKNOWN"),
                "section": str(chunk.get("section", "Full Document")),
                "classification": str(chunk.get("classification", "CONFIDENTIAL")),
                "effective_date": str(chunk.get("effective_date", "N/A")),
            }

            ids.append(chunk["chunk_id"])
            documents.append(chunk["chunk_text"])
            metadatas.append(metadata_entry)

        self.store.add_texts(texts=documents, metadatas=metadatas, ids=ids)

        count = len(ids)
        print(f"Successfully indexed {count} chunk vectors into ChromaDB collection '{self.collection_name}'.")
        return count

    def query_similar_chunks(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for the top-K most semantically similar chunks.

        Args:
            query (str): User query text (embedded with the store's embedding function).
            top_k (int): Number of top results to retrieve (default: 3).
            metadata_filter (Optional[Dict[str, Any]]): Metadata filtering dictionary.

        Returns:
            List[Dict[str, Any]]: List of retrieved result dictionaries with similarity scores.
        """
        results = self.store.similarity_search_with_score(
            query, k=top_k, filter=metadata_filter or None
        )

        retrieved_chunks = []
        for doc, distance in results:
            meta = doc.metadata
            # Calculate Cosine Similarity score from Cosine Distance: similarity = 1.0 - distance
            similarity_score = max(0.0, min(1.0, 1.0 - distance))

            retrieved_chunks.append(
                {
                    "chunk_id": doc.id,
                    "retrieved_text": doc.page_content,
                    "source": meta.get("source", "N/A"),
                    "document_id": meta.get("document_id", "N/A"),
                    "section": meta.get("section", "N/A"),
                    "classification": meta.get("classification", "N/A"),
                    "effective_date": meta.get("effective_date", "N/A"),
                    "cosine_distance": round(distance, 4),
                    "similarity_score": round(similarity_score, 4),
                }
            )

        return retrieved_chunks

    def count(self) -> int:
        """Returns the total number of chunks stored in the ChromaDB collection."""
        return self.client.get_collection(self.collection_name).count()
