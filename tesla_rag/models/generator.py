"""
Tesla Integrity Bot - LLM Generation Module
Handles prompt formatting, Groq Cloud API execution, and zero-hallucination guardrails.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq


class GroqGenerator:
    """
    GroqGenerator handles answer generation using the Groq LLM API,
    enforcing strict persona guidelines and zero-hallucination rules.
    """

    SYSTEM_PROMPT = """You are an enterprise corporate ethics compliance assistant for a fictional company demonstration.

You must answer ONLY using the supplied retrieved context.

Do not invent policies, penalties, procedures, deadlines, or requirements.

If the retrieved context does not contain enough information, say:
'The provided policy documents do not contain sufficient information to answer this question.'

Every factual claim must reference the source document and section.

Do not use external knowledge."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API")
        self.model_name = model_name
        self._client = None
        self._logged_key_status = False

    @property
    def client(self) -> Optional[Groq]:
        """Lazy loads and validates Groq client connection."""
        if self._client is None:
            raw_key = (self.api_key or "").strip()
            if raw_key and raw_key.startswith("gsk_"):
                try:
                    self._client = Groq(api_key=raw_key)
                    if not self._logged_key_status:
                        print("[GroqGenerator] [SUCCESS] GROQ_API successfully fetched.")
                        self._logged_key_status = True
                except Exception as e:
                    if not self._logged_key_status:
                        print(f"[GroqGenerator] [WARNING] Failed to initialize Groq client: {e}. Returning fallback answer.")
                        self._logged_key_status = True
                    self._client = None
            else:
                if not self._logged_key_status:
                    print("[GroqGenerator] [WARNING] GROQ_API variable not found or invalid (placeholder key detected). Returning fallback answer.")
                    self._logged_key_status = True
                self._client = None
        return self._client

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunk dictionaries into formatted prompt context blocks."""
        if not retrieved_chunks:
            return "No relevant context retrieved."

        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "N/A")
            doc_id = chunk.get("document_id") or chunk.get("doc_id") or "N/A"
            section = chunk.get("section", "N/A")
            classification = chunk.get("classification", "N/A")
            text = chunk.get("retrieved_text") or chunk.get("chunk_text") or ""

            block = (
                f"[Retrieved Context Chunk {idx}]\n"
                f"- Source File: {source}\n"
                f"- Document ID: {doc_id}\n"
                f"- Section: {section}\n"
                f"- Classification: {classification}\n"
                f"- Text Content:\n{text.strip()}\n"
            )
            context_blocks.append(block)

        return "\n---\n".join(context_blocks)

    def _build_sources_metadata(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicates citations based on document ID and section."""
        sources = []
        seen = set()
        for chunk in retrieved_chunks:
            doc_id = chunk.get("document_id") or chunk.get("doc_id") or "UNKNOWN"
            section = chunk.get("section", "N/A")
            source_file = chunk.get("source", "N/A")
            classification = chunk.get("classification", "N/A")
            sim_score = chunk.get("similarity_score", None)

            key = (doc_id, section, source_file)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "document_id": doc_id,
                        "section": section,
                        "source": source_file,
                        "classification": classification,
                        "similarity_score": sim_score,
                    }
                )
        return sources

    def _generate_demo_fallback_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float,
    ) -> str:
        """Generates fallback message when GROQ_API is not configured or unavailable."""
        if not retrieved_chunks:
            return "The provided policy documents do not contain sufficient information to answer this question."

        return (
            "GROQ_API is not configured or model is not loaded. "
            "Please configure a valid GROQ_API in your .env file to enable LLM answer generation."
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        """Synthesizes answer via Groq LLM using strict context instructions."""
        sources = self._build_sources_metadata(retrieved_chunks)

        if not retrieved_chunks:
            return {
                "answer": "The provided policy documents do not contain sufficient information to answer this question.",
                "sources": [],
            }

        if self.client is not None:
            formatted_context = self._format_context(retrieved_chunks)
            user_prompt = (
                f"RETRIEVED CONTEXT:\n{formatted_context}\n\n"
                f"USER QUESTION:\n{query}\n\n"
                f"INSTRUCTIONS:\nProvide your final compliance audit answer strictly based on the context above. "
                f"Format your answer cleanly using Markdown (bold headings, bullet points, clear line breaks). "
                f"Include inline citations referencing the Document ID and Section."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                )
                final_answer = response.choices[0].message.content.strip()
                return {
                    "answer": final_answer,
                    "sources": sources,
                }
            except Exception as e:
                err_msg = str(e)
                print(f"[GroqGenerator] API Execution Exception with {self.model_name}: {err_msg}")
                # Fallback model retry if primary model returns 404 or decommissioned
                if "404" in err_msg or "model_not_found" in err_msg or "model_decommissioned" in err_msg:
                    fallback_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
                    for alt_model in fallback_models:
                        if alt_model != self.model_name:
                            try:
                                print(f"[GroqGenerator] Retrying with alternative model: {alt_model}...")
                                response = self.client.chat.completions.create(
                                    model=alt_model,
                                    messages=[
                                        {"role": "system", "content": self.SYSTEM_PROMPT},
                                        {"role": "user", "content": user_prompt},
                                    ],
                                    temperature=temperature,
                                    top_p=top_p,
                                )
                                final_answer = response.choices[0].message.content.strip()
                                return {
                                    "answer": final_answer,
                                    "sources": sources,
                                }
                            except Exception:
                                pass

        fallback_answer = self._generate_demo_fallback_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature,
        )

        return {
            "answer": fallback_answer,
            "sources": sources,
        }
