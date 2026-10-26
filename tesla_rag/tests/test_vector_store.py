"""Tests the LangChain Chroma wrapper on a throwaway directory (never the real vector_store) with deterministic fake embeddings."""

import hashlib
import math
import shutil
import tempfile
import unittest
from typing import List
from langchain_core.embeddings import Embeddings
from storage.vector_store import VectorStore


class FakeEmbeddings(Embeddings):
    """Deterministic 8-d bag-of-hash vectors: identical text -> identical vector, similar words -> closer vectors."""

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * 8
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 8] += 1.0
        return v if any(v) else [1.0] + [0.0] * 7

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def chunk(cid, text, doc="DOC-1", classification="INTERNAL"):
    return {
        "chunk_id": cid,
        "source": "file.txt",
        "document_id": doc,
        "section": "Full Document",
        "classification": classification,
        "effective_date": "May 10, 2026",
        "chunk_text": text,
    }


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = VectorStore(FakeEmbeddings(), persist_directory=self.dir, collection_name="tesla_compliance_policies")
        self.store.add_chunks(
            [
                chunk("DOC-1_chunk_1", "gift limit fifty dollars approval"),
                chunk("DOC-1_chunk_2", "telemetry deletion thirty business days", doc="DOC-2", classification="PUBLIC"),
                chunk("DOC-1_chunk_3", "rotate api keys every ninety days"),
            ]
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_collection_uses_cosine_space(self):
        meta = self.store.client.get_collection("tesla_compliance_policies").metadata
        self.assertEqual(meta.get("hnsw:space"), "cosine")
        self.assertEqual(self.store.count(), 3)

    def test_exact_match_scores_one_and_fields_are_preserved(self):
        res = self.store.query_similar_chunks("gift limit fifty dollars approval", top_k=3)
        top = res[0]
        self.assertEqual(top["chunk_id"], "DOC-1_chunk_1")
        self.assertEqual(top["similarity_score"], 1.0)
        self.assertEqual(top["cosine_distance"], 0.0)
        self.assertEqual(top["retrieved_text"], "gift limit fifty dollars approval")
        self.assertEqual(
            {k: top[k] for k in ("source", "document_id", "section", "classification", "effective_date")},
            {"source": "file.txt", "document_id": "DOC-1", "section": "Full Document",
             "classification": "INTERNAL", "effective_date": "May 10, 2026"},
        )

    def test_score_is_one_minus_cosine_distance_rounded_to_4dp(self):
        q = "gift limit"
        emb = FakeEmbeddings()
        qv, dv = emb.embed_query(q), emb.embed_query("gift limit fifty dollars approval")
        cos = sum(a * b for a, b in zip(qv, dv)) / (math.hypot(*qv) * math.hypot(*dv))
        res = {r["chunk_id"]: r for r in self.store.query_similar_chunks(q, top_k=3)}
        self.assertAlmostEqual(res["DOC-1_chunk_1"]["similarity_score"], round(cos, 4), places=3)

    def test_top_k_limits_results_and_orders_best_first(self):
        res = self.store.query_similar_chunks("rotate api keys", top_k=2)
        self.assertEqual(len(res), 2)
        self.assertGreaterEqual(res[0]["similarity_score"], res[1]["similarity_score"])
        self.assertEqual(res[0]["chunk_id"], "DOC-1_chunk_3")

    def test_metadata_filter(self):
        res = self.store.query_similar_chunks("days", top_k=3, metadata_filter={"document_id": "DOC-2"})
        self.assertEqual([r["chunk_id"] for r in res], ["DOC-1_chunk_2"])

    def test_reopening_existing_collection_keeps_data(self):
        again = VectorStore(FakeEmbeddings(), persist_directory=self.dir, collection_name="tesla_compliance_policies")
        self.assertEqual(again.count(), 3)

    def test_no_chunks_indexes_nothing(self):
        self.assertEqual(self.store.add_chunks([]), 0)
        self.assertEqual(self.store.count(), 3)


if __name__ == "__main__":
    unittest.main()
