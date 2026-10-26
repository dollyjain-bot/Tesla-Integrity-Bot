"""Locks the existing loading/chunking behaviour (500/100) so the LangChain migration cannot change it."""

import hashlib
import unittest
from tests import PROJECT_ROOT
from ingestion.chunker import TextChunker
from ingestion.loaders import DocumentLoader

# Fingerprint of the chunks currently stored in ChromaDB (verified identical to the stored text, IDs and order).
GOLDEN_CHUNK_COUNT = 250
GOLDEN_SHA256 = "1afd955b7ae7db362cd0b1e035bf7c92f2f8cf027cf8dad3457d5ce22a650616"
GOLDEN_PER_DOC = {
    "TSLA-ETH-2026-01": 64,
    "TSLA-ETH-2026-02": 51,
    "TSLA-ETH-2026-05": 43,
    "TSLA-ETH-2026-03": 49,
    "TSLA-ETH-2026-04": 43,
}


class CorpusChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = DocumentLoader(PROJECT_ROOT / "data").load_all_documents()
        cls.chunks = TextChunker(500, 100).chunk_documents(cls.docs)

    def test_five_documents_in_filename_order(self):
        self.assertEqual(
            [d["source"] for d in self.docs],
            [
                "corporate_ethics.txt",
                "customer_data_privacy.txt",
                "information_security.txt",
                "vehicle_confidentiality.txt",
                "whistleblower_policy.txt",
            ],
        )
        self.assertEqual(
            [d["doc_id"] for d in self.docs],
            ["TSLA-ETH-2026-01", "TSLA-ETH-2026-02", "TSLA-ETH-2026-05", "TSLA-ETH-2026-03", "TSLA-ETH-2026-04"],
        )

    def test_chunk_counts(self):
        self.assertEqual(len(self.chunks), GOLDEN_CHUNK_COUNT)
        per_doc = {}
        for c in self.chunks:
            per_doc[c["document_id"]] = per_doc.get(c["document_id"], 0) + 1
        self.assertEqual(per_doc, GOLDEN_PER_DOC)

    def test_chunk_content_ids_and_order_unchanged(self):
        h = hashlib.sha256()
        for c in self.chunks:
            h.update("\x00".join([c["chunk_id"], c["source"], c["section"], c["chunk_text"]]).encode("utf-8"))
            h.update(b"\x01")
        self.assertEqual(h.hexdigest(), GOLDEN_SHA256)

    def test_chunk_metadata_shape(self):
        first = self.chunks[0]
        self.assertEqual(first["chunk_id"], "TSLA-ETH-2026-01_chunk_1")
        self.assertEqual(first["section"], "Full Document")
        self.assertEqual(first["effective_date"], "May 10, 2026")
        self.assertTrue(first["classification"].startswith("CONFIDENTIAL"))
        self.assertEqual(
            set(first), {"chunk_id", "source", "document_id", "section", "classification", "effective_date", "chunk_text"}
        )


class ChunkerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.chunker = TextChunker(500, 100)

    def test_empty_and_whitespace(self):
        self.assertEqual(self.chunker.chunk_text(""), [])
        self.assertEqual(self.chunker.chunk_text("  \n\n  "), [])

    def test_short_text_is_one_stripped_chunk(self):
        self.assertEqual(self.chunker.chunk_text("  hello world  \n"), ["hello world"])

    def test_exactly_chunk_size_is_not_split(self):
        text = "a" * 500
        self.assertEqual(self.chunker.chunk_text(text), [text])

    def test_paragraph_boundaries_are_preferred(self):
        para = "word " * 60  # 300 chars
        chunks = self.chunker.chunk_text(para.strip() + "\n\n" + para.strip())
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(c.startswith("word") for c in chunks))

    def test_no_separator_text_uses_hard_split_with_overlap(self):
        text = "x" * 1200
        chunks = self.chunker.chunk_text(text)
        self.assertEqual([len(c) for c in chunks], [500, 500, 400])

    def test_chunk_ids_restart_per_document(self):
        docs = [
            {"text": "alpha " * 200, "source": "a.txt", "doc_id": "DOC-1"},
            {"text": "beta " * 200, "source": "b.txt", "doc_id": "DOC-2"},
        ]
        chunks = self.chunker.chunk_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(ids[0], "DOC-1_chunk_1")
        self.assertIn("DOC-2_chunk_1", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_known_limitation_same_doc_id_collides(self):
        # Characterisation of CURRENT behaviour (PDF pages share a doc_id): IDs repeat. Not fixed in this migration.
        docs = [
            {"text": "alpha " * 10, "source": "p.pdf", "doc_id": "DOC-9", "section_info": "Page 1 of 2"},
            {"text": "beta " * 10, "source": "p.pdf", "doc_id": "DOC-9", "section_info": "Page 2 of 2"},
        ]
        ids = [c["chunk_id"] for c in self.chunker.chunk_documents(docs)]
        self.assertEqual(ids, ["DOC-9_chunk_1", "DOC-9_chunk_1"])

    def test_missing_metadata_defaults(self):
        chunk = self.chunker.chunk_documents([{"text": "some text"}])[0]
        self.assertEqual(
            (chunk["document_id"], chunk["classification"], chunk["section"], chunk["effective_date"], chunk["source"]),
            ("UNKNOWN_DOC_ID", "CONFIDENTIAL", "Full Document", "N/A", "unknown_source"),
        )


if __name__ == "__main__":
    unittest.main()
