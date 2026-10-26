# ⚡ Tesla Integrity Bot — Enterprise Compliance RAG Architecture

> **Disclaimer**: This repository contains synthetic, fictional corporate ethics and compliance policy documents created strictly for academic, technical demonstration, and portfolio purposes. These documents do not represent actual internal policies of Tesla, Inc.

---

## 📌 Executive Overview

**Tesla Integrity Bot** is an enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for corporate ethics, data privacy, and confidentiality governance. Built with a modular Python architecture, it allows employees, managers, and compliance officers to query complex internal policies in plain English and receive instant, verifiable answers with exact source citations.

The system is engineered to solve a critical enterprise problem: **eliminating hallucinations in legal and compliance domains**. By combining dense semantic search, similarity threshold filtering, and strict negative-constraint prompt engineering, the bot reliably synthesizes multi-document rules and refuses to answer when information is absent.

---

## 🚀 Key Features

* **Grounded Policy Synthesis:** Answers questions strictly from approved policy text, synthesizing multi-hop rules spanning ethics, prototype confidentiality, and telemetry privacy.
* **Zero-Hallucination Guardrails:** Explicit refusal prompts force the model to decline out-of-scope or unanswerable queries (*0% hallucination rate in empirical benchmarks*).
* **Two-Stage Quality Gate:** Combines top-$K$ nearest-neighbor retrieval with a hard **0.35 cosine similarity cutoff**, discarding weak or irrelevant matches before prompt generation.
* **Verifiable Provenance & Citation Badges:** Every claim links to an official Document ID (`TSLA-ETH-2026-XX`) and section.
* **Interactive Source Inspector:** Streamlit expander reveals verbatim retrieved chunks, metadata tags, and exact cosine similarity scores.
* **LangChain RAG Pipeline:** Embeddings (`HuggingFaceEmbeddings`), vector store (`langchain_chroma.Chroma`), a custom `ThresholdRetriever` (`BaseRetriever`), `ChatPromptTemplate`, `ChatGroq` and a runnable chain, all behind the original `RAGEngine.answer_query()` interface.
* **Multi-Model Failover Ladder:** Primary inference on `openai/gpt-oss-120b` via Groq LPUs, with automated fallback retries across `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`.
* **Automated Empirical Benchmarking:** Built-in evaluation harness testing 15 compliance questions across single-doc, multi-doc, and unanswerable edge cases.

---

## 🏛️ Modular Project Architecture

```
tesla_rag/
├── config.py                 # Centralized RAGConfig dataclass (parameters, models, paths)
├── requirements.txt          # Python package dependencies
├── .env                      # Environment variables and API keys
├── README.md                 # Project documentation
│
├── ingestion/                # Document loading & text preprocessing
│   ├── __init__.py
│   ├── loaders.py            # DocumentLoader: regex header parser for .txt & .pdf
│   └── chunker.py            # TextChunker: recursive character splitter with overlap
│
├── models/                   # Embedding & generation models
│   ├── __init__.py
│   ├── embeddings.py         # EmbeddingManager: LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2), lazy-loaded
│   └── generator.py          # GroqGenerator: ChatPromptTemplate | ChatGroq | StrOutputParser, persona prompt & fallbacks
│
├── storage/                  # Vector database layer
│   ├── __init__.py
│   └── vector_store.py       # VectorStore: langchain_chroma.Chroma on a persistent ChromaDB client (cosine space)
│
├── core/                     # Core orchestrator
│   ├── __init__.py
│   ├── retriever.py          # ThresholdRetriever (LangChain BaseRetriever): Top-K search + similarity cut-off
│   └── engine.py             # RAGEngine: LangChain chain retriever -> generator
│
├── tests/                    # unittest suite (offline: no Groq calls, never touches the real vector_store/)
│
├── rag_engine.py             # Backwards-compatible facade re-exporting core modules
├── db.py                     # Backwards-compatible facade re-exporting VectorStore
├── streamlit_app.py          # Streamlit web frontend with dark theme & mock auth
│
├── benchmark.py              # Automated 15-question evaluation suite (Config A vs B)
├── benchmark_results.csv     # Recorded benchmark outputs of the original (pre-LangChain) implementation
├── benchmark_results_langchain.csv  # Benchmark outputs of the LangChain implementation (same 15 questions)
│
├── data/                     # Ingested synthetic policy corpus
│   ├── corporate_ethics.txt       # TSLA-ETH-2026-01: Gifts, bribery, conflict of interest
│   ├── customer_data_privacy.txt  # TSLA-ETH-2026-02: Spatial hashing & 30-day deletion SLA
│   ├── vehicle_confidentiality.txt# TSLA-ETH-2026-03: Optimus/Robotaxi prototype NDAs
│   ├── whistleblower_policy.txt   # TSLA-ETH-2026-04: 24/7 hotline & non-retaliation rules
│   └── information_security.txt   # TSLA-ETH-2026-05: SSO/MFA & 90-day secret rotation SLA
│
├── vector_store/             # ChromaDB persistent on-disk database storage
└── Tesla_Integrity_Bot_Business_Proposal_Final.pptx # Executive presentation deck
```

---

## ⚙️ Technical Specifications

| Component | Technology / Parameter | Description |
| :--- | :--- | :--- |
| **Generation Model (LLM)** | `openai/gpt-oss-120b` | High-reasoning 120B parameter model hosted on Groq LPUs. |
| **LLM Failover Ladder** | `gpt-oss-120b` $\rightarrow$ `gpt-oss-20b` $\rightarrow$ `qwen3.8-27b` | Automated sequential retry if primary model returns 404/decommission. |
| **Embedding Model** | `all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings` | Dense 384-dimensional vector embeddings, local CPU inference. |
| **Vector Database** | **ChromaDB** via LangChain `langchain_chroma.Chroma` | On-disk persistence (`./vector_store`), collection `tesla_compliance_policies`. |
| **Retriever** | `ThresholdRetriever` (LangChain `BaseRetriever`) | Top-K search, then drops chunks below the similarity threshold. Returns LangChain `Document`s. |
| **Prompting & LLM client** | `ChatPromptTemplate` + `ChatGroq` (`langchain-groq`) | Prompt text unchanged; temperature and Top-P are bound per call. |
| **HNSW Distance Space** | **Cosine Similarity** (`{"hnsw:space": "cosine"}`) | Evaluates similarity score: $S = \max(0, \min(1, 1 - d))$. |
| **Chunking Configuration** | **500 characters** window, **100 characters** overlap | Recursive splitting on `["\n\n", "\n", ". ", "? ", "! ", " ", ""]`. |
| **Retrieval Strategy** | Dense top-$K$ with hard threshold cutoff | Default **`top_k = 3`** (UI adjustable 1–5), **`similarity_threshold = 0.35`**. |
| **Decoding Parameters** | `temperature = 0.0`, `top_p = 0.1` | Default for the engine, config and UI sliders. Temperature 0.0 is sent to Groq exactly (not `ChatGroq`'s 1e-08 substitution). |
| **Web UI Framework** | **Streamlit** (Custom Tesla Dark Theme) | Custom CSS (`#0E1117`, red `#E82127`), mock login gate, chunk inspector. |

---

## 📚 Ingested Policy Corpus (`data/`)

| Document ID | Policy Title | Scope & Key Regulations |
| :--- | :--- | :--- |
| **`TSLA-ETH-2026-01`** | Corporate Ethics & Anti-Bribery Code | $50.00 USD de minimis gift limit, FCPA rules, supplier conflict-of-interest prohibitions. |
| **`TSLA-ETH-2026-02`** | Customer Data Privacy & Telemetry | Local edge zero-knowledge spatial hashing (Sec. 2.4) and 30-day statutory deletion SLA (Sec. 5.2). |
| **`TSLA-ETH-2026-03`** | Vehicle Confidentiality & Trade Secrets | Optimus & Robotaxi unreleased builds, proving ground smartphone camera tamper seals. |
| **`TSLA-ETH-2026-04`** | Whistleblower Protection & Hotline | 24/7 compliance portal (`compliance.tesla.com`), 16-digit case keys, zero-retaliation mandate. |
| **`TSLA-ETH-2026-05`** | Information Security & Access Controls | Mandatory SSO/MFA credentials, 90-day automated API key rotation, RBAC vector store controls. |

---

## 📊 Empirical Benchmark Evaluation

The system was evaluated using [`benchmark.py`](file:///c:/Users/Bot/rag/tesla_rag/benchmark.py) across 15 test questions (7 single-doc, 4 multi-doc, and 4 out-of-scope negative tests) comparing **Config A (Deterministic: Temp 0.0, Top-P 0.1)** against **Config B (Exploratory: Temp 0.8, Top-P 0.9)**.

Recorded execution data from [`benchmark_results.csv`](file:///c:/Users/Bot/rag/tesla_rag/benchmark_results.csv):

```
+--------------------------------+--------------------------------+--------------------------------+
| Evaluation Metric (15 Queries) | Config A (Temp 0.0, Top-P 0.1) | Config B (Temp 0.8, Top-P 0.9) |
+--------------------------------+--------------------------------+--------------------------------+
| Average Latency (seconds)      | 0.52s                          | 0.09s                          |
| Retrieval Accuracy (%)         | 100.0%                         | 100.0%                         |
| Citation Accuracy (%)          | 100.0%                         | 100.0%                         |
| Answer Correctness (%)         | 67.2%                          | 40.5%                          |
| Groundedness Score (%)         | 88.0%                          | 61.3%                          |
| Observed Hallucination Rate (%)| 10.0%                          | 36.7%                          |
+--------------------------------+--------------------------------+--------------------------------+
```

> **Key Finding:** On the 4 out-of-scope questions (e.g., ergonomic chair stipends, 2027 delivery targets, pet insurance), **Config A had a 0.0% hallucination rate** (correctly declined all 4). In contrast, Config B hallucinated 100% of the time on unanswerable queries, fabricating non-existent policies. This provides empirical proof that `temperature = 0.0` is essential for enterprise compliance.

### Original vs LangChain implementation (same 15 questions, `top_k = 3`)

Results of the LangChain implementation are in [`benchmark_results_langchain.csv`](benchmark_results_langchain.csv). The closest like-for-like baseline is the original code's re-run from 2026-10-14 (`deck_evidence/benchmark_rerun_2026-10-14.csv`). The table above comes from the older `benchmark_results.csv`, whose scores differ from that re-run, so it is not used for this comparison.

| Metric | Original re-run, Config A | LangChain, Config A | Original re-run, Config B | LangChain, Config B |
| :--- | :--- | :--- | :--- | :--- |
| Average latency (s, network dependent) | 9.87 | 4.36 | 9.94 | 7.07 |
| Retrieval accuracy | 100% | 100% | 100% | 100% |
| Citation accuracy | 93% | 93% | 100% | 93% |
| Answer correctness | 78.9% | 78.9% | 84.5% | 82.8% |
| Groundedness | 94.0% | 94.0% | 94.0% | 94.0% |
| Hallucination rate | 0.0% | 0.0% | 0.0% | 0.0% |

Config A (the deterministic default) matches on every quality metric. Config B differs on 2 of 15 questions (Q1 and Q10), which is expected at temperature 0.8. Retrieved chunks and scores are identical to the original code (verified separately on a copy of the vector store). Latency depends on the Groq API and network, so it is not a like-for-like speed comparison.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10 – 3.12 (Python 3.14+ supported with standard wheel dependencies)
* Valid Groq API Key (`GROQ_API`)

### 2. Clone & Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd tesla_rag

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> `langchain-groq` requires `groq<1.0.0`, so pip installs `groq` 0.37.x. Use a virtual environment so this does not affect other projects. Tested with Python 3.14.

### 4. Configure Environment Variables
Create or edit your `.env` file in the `tesla_rag/` directory:
```env
# Required: Groq Cloud API Key
GROQ_API=gsk_your_groq_api_key_here

# Optional: Vector Store & Directory Settings
CHROMA_DB_DIR=./vector_store
DATA_DIR=data
```

---

## 🖥️ Running the Application

### Option A: Launch Streamlit Web UI (Recommended)
Launch the interactive Tesla-branded web application:
```bash
streamlit run streamlit_app.py
```
* **Local URL:** `http://localhost:8501`
* **Mock Sign-In:** The form is pre-filled with `admin@tesla.demo` / `demo123`. Any non-empty username signs in; the password is **not validated**.
* **Features:** Sidebar controls for Top-$K$ (1–5, default 3), Temperature (0.0–1.0, default 0.0) and Top-P (0.0–1.0, default 0.1), dynamic chat interface, citation badges, and expandable source chunk inspector.

### Option B: Test RAG Engine CLI
Run an end-to-end question answering check in the terminal:
```bash
python rag_engine.py
```
* Automatically indexes `data/*.txt` into ChromaDB if the collection is uninitialized.
* Executes a test query regarding spatial hashing and prints the synthesized response.

### Option C: Run Automated Benchmark Suite
Execute the empirical evaluation suite across all 15 test questions:
```bash
python benchmark.py
```
* Evaluates Config A vs. Config B across latency, retrieval, citation, correctness, and hallucination.
* Exports detailed per-query results to `benchmark_results.csv` and prints a summary comparison table. (This overwrites the recorded original results, so copy the file elsewhere first if you want to keep it.)

### Option D: Run the Tests
```bash
python -m unittest discover -s tests -t .
```
* Offline suite covering chunking, the Chroma wrapper, prompts and request payloads, the retriever, the engine chain, embeddings and the Streamlit login/controls. It uses temporary directories, never the real `vector_store/`.

---

## 🔄 End-to-End Query Execution Lifecycle

```
[ User Query via Streamlit / CLI ]
                │
                ▼
[ RAGEngine.answer_query() ]  ── LangChain chain: RunnablePassthrough.assign(documents=retrieve) | generate
                │
                ▼
[ ThresholdRetriever (BaseRetriever) ]
      ├─ HuggingFaceEmbeddings embeds the query (384-d, all-MiniLM-L6-v2)
      ├─ langchain_chroma.Chroma cosine search ──► Top-K candidate chunks (score = 1 - distance)
      └─ Similarity threshold filter ──► keeps only chunks with similarity >= 0.35
                │
                ▼
      LangChain Documents (id = chunk_id, metadata = source, document_id, section, scores)
                │
                ├──► [ If 0 chunks survive ] ──► Return: "The provided policy documents do not contain..." (LLM not called)
                │
                ▼
[ GroqGenerator ]  ChatPromptTemplate | ChatGroq(openai/gpt-oss-120b, temperature, top_p) | StrOutputParser
                │   (Structures [Retrieved Context Chunk X] blocks; cites Document ID & Section)
                ▼
[ Streamlit UI Output ]  ──► Renders Markdown + Source Badges + Chunk Inspector
```

**Parts that are intentionally not LangChain:** document loading and the 500/100 recursive chunker (LangChain's splitter produces different chunk boundaries, which would no longer match the stored vectors), the similarity score/threshold logic, citation badge building, and the model-fallback loop (it only retries on 404 / model-not-found errors).

---

## 📄 License & Attribution

This project is developed for educational, architectural demonstration, and portfolio purposes. All company names, logos, and policy identifiers are synthetic demonstrations of enterprise compliance retrieval architectures.
