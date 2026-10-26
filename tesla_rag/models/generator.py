"""
Tesla Integrity Bot - LLM Generation Module
Handles prompt formatting, Groq Cloud API execution (via LangChain), and zero-hallucination guardrails.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


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

    USER_PROMPT_TEMPLATE = (
        "RETRIEVED CONTEXT:\n{context}\n\n"
        "USER QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\nProvide your final compliance audit answer strictly based on the context above. "
        "Format your answer cleanly using Markdown (bold headings, bullet points, clear line breaks). "
        "Include inline citations referencing the Document ID and Section."
    )

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API")
        self.model_name = model_name
        self._llm = None
        self._logged_key_status = False
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_PROMPT), ("human", self.USER_PROMPT_TEMPLATE)]
        )

    def _make_llm(self, model_name: str) -> ChatGroq:
        """Builds a ChatGroq model; temperature/top_p are supplied per call via bind()."""
        return ChatGroq(model=model_name, api_key=(self.api_key or "").strip(), timeout=60.0)

    @property
    def llm(self) -> Optional[ChatGroq]:
        """Lazy loads and validates the Groq chat model (None when no valid key is configured)."""
        if self._llm is None:
            raw_key = (self.api_key or "").strip()
            if raw_key and raw_key.startswith("gsk_"):
                try:
                    self._llm = self._make_llm(self.model_name)
                    if not self._logged_key_status:
                        print("[GroqGenerator] [SUCCESS] GROQ_API successfully fetched.")
                        self._logged_key_status = True
                except Exception as e:
                    if not self._logged_key_status:
                        print(f"[GroqGenerator] [WARNING] Failed to initialize Groq client: {e}. Returning fallback answer.")
                        self._logged_key_status = True
                    self._llm = None
            else:
                if not self._logged_key_status:
                    print("[GroqGenerator] [WARNING] GROQ_API variable not found or invalid (placeholder key detected). Returning fallback answer.")
                    self._logged_key_status = True
                self._llm = None
        return self._llm

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

    def _invoke_chain(
        self,
        llm: ChatGroq,
        context: str,
        query: str,
        temperature: float,
        top_p: float,
    ) -> str:
        """Runs prompt -> ChatGroq -> parser. bind() sends temperature as given (ChatGroq's own field would turn 0.0 into 1e-08)."""
        chain = self.prompt | llm.bind(temperature=temperature, top_p=top_p) | StrOutputParser()
        return chain.invoke({"context": context, "question": query}).strip()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 0.1,
    ) -> Dict[str, Any]:
        """Synthesizes answer via Groq LLM using strict context instructions."""
        sources = self._build_sources_metadata(retrieved_chunks)

        if not retrieved_chunks:
            return {
                "answer": "The provided policy documents do not contain sufficient information to answer this question.",
                "sources": [],
            }

        if self.llm is not None:
            formatted_context = self._format_context(retrieved_chunks)
            try:
                final_answer = self._invoke_chain(
                    self.llm, formatted_context, query, temperature, top_p
                )
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
                                final_answer = self._invoke_chain(
                                    self._make_llm(alt_model),
                                    formatted_context,
                                    query,
                                    temperature,
                                    top_p,
                                )
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
