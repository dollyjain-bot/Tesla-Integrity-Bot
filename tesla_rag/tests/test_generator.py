"""Checks the LangChain prompt/Groq layer reproduces the original prompts, parameters and fallback behaviour offline."""

import unittest
from unittest.mock import patch
from models.generator import GroqGenerator

CHUNKS = [
    {
        "chunk_id": "TSLA-ETH-2026-01_chunk_3",
        "retrieved_text": "  Gifts above $50.00 require approval.  ",
        "source": "corporate_ethics.txt",
        "document_id": "TSLA-ETH-2026-01",
        "section": "Full Document",
        "classification": "CONFIDENTIAL",
        "similarity_score": 0.61,
    },
    {
        "chunk_id": "TSLA-ETH-2026-01_chunk_4",
        "retrieved_text": "Braces {like} this must survive templating.",
        "source": "corporate_ethics.txt",
        "document_id": "TSLA-ETH-2026-01",
        "section": "Full Document",
        "classification": "CONFIDENTIAL",
        "similarity_score": 0.55,
    },
    {
        "chunk_id": "TSLA-ETH-2026-05_chunk_9",
        "retrieved_text": "Rotate keys every 90 days.",
        "source": "information_security.txt",
        "document_id": "TSLA-ETH-2026-05",
        "section": "Full Document",
        "classification": "INTERNAL",
        "similarity_score": 0.5,
    },
]
QUESTION = "What is the gift limit?"


def original_user_prompt(context: str, query: str) -> str:
    """The user prompt exactly as the pre-migration f-string built it."""
    return (
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"INSTRUCTIONS:\nProvide your final compliance audit answer strictly based on the context above. "
        f"Format your answer cleanly using Markdown (bold headings, bullet points, clear line breaks). "
        f"Include inline citations referencing the Document ID and Section."
    )


def fake_groq_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {}}


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.gen = GroqGenerator(api_key="gsk_dummy_for_tests")

    def test_context_block_format(self):
        ctx = self.gen._format_context(CHUNKS[:1])
        self.assertEqual(
            ctx,
            "[Retrieved Context Chunk 1]\n- Source File: corporate_ethics.txt\n- Document ID: TSLA-ETH-2026-01\n"
            "- Section: Full Document\n- Classification: CONFIDENTIAL\n- Text Content:\nGifts above $50.00 require approval.\n",
        )
        self.assertIn("\n---\n", self.gen._format_context(CHUNKS))
        self.assertEqual(self.gen._format_context([]), "No relevant context retrieved.")

    def test_rendered_messages_match_original_prompt(self):
        ctx = self.gen._format_context(CHUNKS)
        system, human = self.gen.prompt.format_messages(context=ctx, question=QUESTION)
        self.assertEqual(system.content, GroqGenerator.SYSTEM_PROMPT)
        self.assertEqual(human.content, original_user_prompt(ctx, QUESTION))

    def test_sources_deduplicated_per_document(self):
        sources = self.gen._build_sources_metadata(CHUNKS)
        self.assertEqual([s["document_id"] for s in sources], ["TSLA-ETH-2026-01", "TSLA-ETH-2026-05"])
        self.assertEqual(sources[0]["similarity_score"], 0.61)  # first chunk's score is kept


class GroqCallTests(unittest.TestCase):
    def setUp(self):
        self.gen = GroqGenerator(api_key="gsk_dummy_for_tests")

    def test_request_payload_matches_original_call(self):
        llm = self.gen.llm
        with patch.object(llm.client, "create", return_value=fake_groq_response("  The answer.  ")) as create:
            out = self.gen.generate_answer(QUESTION, CHUNKS, temperature=0.0, top_p=0.1)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(kwargs["temperature"], 0.0)  # not ChatGroq's 1e-08 substitution
        self.assertEqual(kwargs["top_p"], 0.1)
        ctx = self.gen._format_context(CHUNKS)
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "system", "content": GroqGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": original_user_prompt(ctx, QUESTION)},
            ],
        )
        self.assertEqual(out["answer"], "The answer.")
        self.assertEqual(len(out["sources"]), 2)

    def test_custom_temperature_and_top_p_are_forwarded(self):
        with patch.object(self.gen.llm.client, "create", return_value=fake_groq_response("x")) as create:
            self.gen.generate_answer(QUESTION, CHUNKS, temperature=0.8, top_p=0.9)
        self.assertEqual((create.call_args.kwargs["temperature"], create.call_args.kwargs["top_p"]), (0.8, 0.9))

    def test_no_chunks_short_circuits_without_calling_llm(self):
        with patch.object(self.gen.llm.client, "create") as create:
            out = self.gen.generate_answer(QUESTION, [])
        create.assert_not_called()
        self.assertEqual(out["sources"], [])
        self.assertTrue(out["answer"].startswith("The provided policy documents do not contain sufficient information"))

    def test_model_not_found_triggers_fallback_ladder(self):
        calls = []

        class FakeLLM:
            def __init__(self, model):
                self.model = model

        def fake_invoke(llm, context, query, temperature, top_p):
            calls.append(llm.model if hasattr(llm, "model") else "primary")
            if len(calls) == 1:
                raise RuntimeError("Error code: 404 - model_not_found")
            if len(calls) == 2:
                raise RuntimeError("still failing")
            return "from third model"

        with patch.object(self.gen, "_invoke_chain", side_effect=fake_invoke), patch.object(
            self.gen, "_make_llm", side_effect=lambda m: FakeLLM(m)
        ):
            out = self.gen.generate_answer(QUESTION, CHUNKS)
        # Primary is gpt-oss-120b, so the ladder retries gpt-oss-20b then qwen (the primary is skipped)
        self.assertEqual(calls[1:], ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
        self.assertEqual(out["answer"], "from third model")

    def test_other_errors_do_not_trigger_ladder(self):
        with patch.object(self.gen, "_invoke_chain", side_effect=RuntimeError("rate limit 429")) as inv:
            out = self.gen.generate_answer(QUESTION, CHUNKS)
        self.assertEqual(inv.call_count, 1)
        self.assertTrue(out["answer"].startswith("GROQ_API is not configured or model is not loaded."))
        self.assertEqual(len(out["sources"]), 2)


class ApiKeyTests(unittest.TestCase):
    def test_missing_or_placeholder_key_returns_fallback_message(self):
        for key in ("", "your_key_here", "   "):
            gen = GroqGenerator(api_key=key)
            gen.api_key = key  # bypass .env lookup
            self.assertIsNone(gen.llm)
            out = gen.generate_answer(QUESTION, CHUNKS)
            self.assertTrue(out["answer"].startswith("GROQ_API is not configured"))
            self.assertEqual(len(out["sources"]), 2)

    def test_key_is_read_from_groq_api_env_var(self):
        with patch.dict("os.environ", {"GROQ_API": "gsk_from_env"}):
            self.assertEqual(GroqGenerator().api_key, "gsk_from_env")


if __name__ == "__main__":
    unittest.main()
