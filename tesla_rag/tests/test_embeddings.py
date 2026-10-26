"""Checks the LangChain embedding wrapper keeps lazy loading and produces the same vectors as sentence-transformers."""

import unittest
from langchain_core.embeddings import Embeddings
from models.embeddings import EmbeddingManager


class EmbeddingManagerTests(unittest.TestCase):
    def test_is_lazy_and_langchain_compatible(self):
        em = EmbeddingManager()
        self.assertIsInstance(em, Embeddings)
        self.assertIsNone(em._embeddings)  # model must not load until first use
        self.assertEqual(em.model_name, "all-MiniLM-L6-v2")

    def test_empty_inputs_do_not_load_model(self):
        em = EmbeddingManager()
        self.assertEqual(em.embed_query(""), [])
        self.assertEqual(em.embed_query("   "), [])
        self.assertEqual(em.embed_documents([]), [])
        self.assertIsNone(em._embeddings)

    def test_vectors_identical_to_sentence_transformers(self):
        from sentence_transformers import SentenceTransformer

        texts = ["What is the gift limit?", "Section 4.2\n  Board Memberships\nand Advisory Roles"]
        em = EmbeddingManager()
        raw = SentenceTransformer("all-MiniLM-L6-v2")
        new_docs = em.embed_documents(texts)
        self.assertEqual(len(new_docs[0]), 384)
        for text, vec in zip(texts, new_docs):
            ref = raw.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()
            self.assertLess(max(abs(a - b) for a, b in zip(vec, ref)), 1e-6)
        ref_q = raw.encode(texts[0], show_progress_bar=False, convert_to_numpy=True).tolist()
        self.assertLess(max(abs(a - b) for a, b in zip(em.embed_query(texts[0]), ref_q)), 1e-6)


if __name__ == "__main__":
    unittest.main()
