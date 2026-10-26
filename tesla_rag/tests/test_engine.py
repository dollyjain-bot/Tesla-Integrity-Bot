"""Tests the RAGEngine LangChain pipeline (threshold, Top-K, short-circuit, defaults) with fakes: no model, DB or API access."""

import inspect
import unittest
from core.engine import RAGEngine
from config import RAGConfig


def hit(cid, score, doc="TSLA-ETH-2026-01"):
    return {
        "chunk_id": cid, "retrieved_text": f"text of {cid}", "source": "corporate_ethics.txt", "document_id": doc,
        "section": "Full Document", "classification": "INTERNAL", "effective_date": "May 10, 2026",
        "cosine_distance": round(1 - score, 4), "similarity_score": score,
    }


class FakeStore:
    def __init__(self, hits, count=250):
        self.hits, self._count, self.queries = hits, count, []

    def count(self):
        return self._count

    def query_similar_chunks(self, query, top_k, metadata_filter=None):
        self.queries.append((query, top_k, metadata_filter))
        return self.hits[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, retrieved_chunks, temperature, top_p):
        self.calls.append((query, retrieved_chunks, temperature, top_p))
        if not retrieved_chunks:
            return {"answer": "REFUSAL", "sources": []}
        return {"answer": "ANSWER", "sources": [{"document_id": retrieved_chunks[0]["document_id"]}]}


def make_engine(hits, count=250):
    engine = RAGEngine()
    engine._vector_store = FakeStore(hits, count)
    engine.generator = FakeGenerator()
    return engine


class EngineDefaultsTests(unittest.TestCase):
    def test_documented_defaults(self):
        sig = inspect.signature(RAGEngine.answer_query).parameters
        self.assertEqual(
            (sig["top_k"].default, sig["temperature"].default, sig["top_p"].default, sig["similarity_threshold"].default),
            (3, 0.0, 0.1, 0.35),
        )
        cfg = RAGConfig()
        self.assertEqual((cfg.chunk_size, cfg.chunk_overlap, cfg.top_k, cfg.similarity_threshold), (500, 100, 3, 0.35))
        self.assertEqual((cfg.embedding_model_name, cfg.llm_model_name), ("all-MiniLM-L6-v2", "openai/gpt-oss-120b"))
        self.assertEqual((cfg.temperature, cfg.top_p, cfg.collection_name), (0.0, 0.1, "tesla_compliance_policies"))


class EnginePipelineTests(unittest.TestCase):
    def test_threshold_keeps_equal_and_drops_below(self):
        engine = make_engine([hit("a", 0.9), hit("b", 0.35), hit("c", 0.3499)])
        res = engine.answer_query("q", top_k=3)
        self.assertEqual([c["chunk_id"] for c in res["retrieved_chunks"]], ["a", "b"])
        self.assertEqual([m["chunk_id"] for m in res["metadata"]], ["a", "b"])

    def test_result_shape_and_parameters_forwarded(self):
        engine = make_engine([hit("a", 0.9)])
        res = engine.answer_query("what?", top_k=2, temperature=0.8, top_p=0.9, metadata_filter={"document_id": "X"})
        self.assertEqual(set(res), {"answer", "sources", "retrieved_chunks", "metadata"})
        self.assertEqual(res["answer"], "ANSWER")
        self.assertEqual(engine.vector_store.queries, [("what?", 2, {"document_id": "X"})])
        self.assertEqual(engine.generator.calls[0][2:], (0.8, 0.9))
        self.assertEqual(
            set(res["metadata"][0]),
            {"chunk_id", "source", "document_id", "section", "classification", "effective_date", "similarity_score"},
        )

    def test_default_call_uses_top_k_3_and_top_p_0_1(self):
        engine = make_engine([hit(str(i), 0.9) for i in range(5)])
        res = engine.answer_query("q")
        self.assertEqual(len(res["retrieved_chunks"]), 3)
        self.assertEqual(engine.generator.calls[0][2:], (0.0, 0.1))

    def test_all_below_threshold_gives_refusal_path_with_no_chunks(self):
        engine = make_engine([hit("a", 0.2), hit("b", 0.1)])
        res = engine.answer_query("q")
        self.assertEqual(res["retrieved_chunks"], [])
        self.assertEqual(res["sources"], [])
        self.assertEqual(engine.generator.calls[0][1], [])

    def test_empty_store_triggers_automatic_indexing(self):
        engine = make_engine([hit("a", 0.9)], count=0)
        built = []
        engine.build_or_update_index = lambda: built.append(True) or 250
        engine.answer_query("q")
        self.assertEqual(built, [True])

    def test_populated_store_is_not_reindexed(self):
        engine = make_engine([hit("a", 0.9)], count=250)
        engine.build_or_update_index = lambda: self.fail("must not re-index an existing collection")
        engine.answer_query("q")


if __name__ == "__main__":
    unittest.main()
