"""Tests the LangChain ThresholdRetriever: Top-K, the >= 0.35 cut-off, Document conversion and round-tripping."""

import unittest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from core.retriever import ThresholdRetriever, chunk_to_document, document_to_chunk


def hit(cid, score):
    return {
        "chunk_id": cid, "retrieved_text": f"text {cid}", "source": "corporate_ethics.txt",
        "document_id": "TSLA-ETH-2026-01", "section": "Full Document", "classification": "INTERNAL",
        "effective_date": "May 10, 2026", "cosine_distance": round(1 - score, 4), "similarity_score": score,
    }


class FakeStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def query_similar_chunks(self, query, top_k, metadata_filter=None):
        self.calls.append((query, top_k, metadata_filter))
        return self.hits[:top_k]


class ThresholdRetrieverTests(unittest.TestCase):
    def test_is_a_langchain_retriever_returning_documents(self):
        r = ThresholdRetriever(vector_store=FakeStore([hit("a", 0.9)]))
        self.assertIsInstance(r, BaseRetriever)
        docs = r.invoke("question")
        self.assertIsInstance(docs[0], Document)
        self.assertEqual((docs[0].id, docs[0].page_content), ("a", "text a"))
        self.assertEqual(docs[0].metadata["similarity_score"], 0.9)

    def test_defaults_and_arguments_are_forwarded(self):
        store = FakeStore([hit(str(i), 0.9) for i in range(5)])
        default = ThresholdRetriever(vector_store=store)
        self.assertEqual((default.top_k, default.similarity_threshold), (3, 0.35))
        self.assertEqual(len(default.invoke("q")), 3)
        ThresholdRetriever(vector_store=store, top_k=2, metadata_filter={"document_id": "X"}).invoke("q2")
        self.assertEqual(store.calls[-1], ("q2", 2, {"document_id": "X"}))

    def test_threshold_is_inclusive_and_applied_after_top_k(self):
        store = FakeStore([hit("a", 0.8), hit("b", 0.35), hit("c", 0.3499), hit("d", 0.99)])
        docs = ThresholdRetriever(vector_store=store, top_k=4).invoke("q")
        self.assertEqual([d.id for d in docs], ["a", "b", "d"])  # order preserved, only c dropped

    def test_all_below_threshold_returns_empty(self):
        self.assertEqual(ThresholdRetriever(vector_store=FakeStore([hit("a", 0.1)])).invoke("q"), [])

    def test_document_round_trip_preserves_chunk_dict(self):
        chunk = hit("TSLA-ETH-2026-01_chunk_3", 0.6123)
        self.assertEqual(document_to_chunk(chunk_to_document(chunk)), chunk)


if __name__ == "__main__":
    unittest.main()
