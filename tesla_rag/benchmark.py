"""
Tesla Integrity Bot - RAG System Benchmarking Module
Evaluates and benchmarks two generation configurations (A vs B) across 15 test questions.
Generates benchmark_results.csv and prints a comparative summary metrics table.
"""

import csv
import time
from typing import List, Dict, Any
from rag_engine import RAGEngine


# ------------------------------------------------------------------------------
# 1. BENCHMARK TEST DATASET (15 Questions: Single-Doc, Multi-Doc, Out-of-Scope)
# ------------------------------------------------------------------------------
TEST_DATASET = [
    # Single-Document Answerable Questions
    {
        "id": 1,
        "type": "single_doc",
        "question": "What is the maximum monetary value for promotional gifts an employee can accept without prior written approval?",
        "expected_doc_ids": ["TSLA-ETH-2026-01"],
        "key_facts": ["50", "50.00", "de minimis", "gift"],
        "is_answerable": True,
    },
    {
        "id": 2,
        "type": "single_doc",
        "question": "What cryptographic protocol is used to anonymize customer vehicle telemetry and cabin camera feeds on local edge hardware?",
        "expected_doc_ids": ["TSLA-ETH-2026-02"],
        "key_facts": ["zero-knowledge", "spatial hashing"],
        "is_answerable": True,
    },
    {
        "id": 3,
        "type": "single_doc",
        "question": "What is the statutory SLA for processing customer vehicle telemetry deletion requests under global data protection rules?",
        "expected_doc_ids": ["TSLA-ETH-2026-02"],
        "key_facts": ["30", "business days", "deletion"],
        "is_answerable": True,
    },
    {
        "id": 4,
        "type": "single_doc",
        "question": "What physical camera security seal procedures are enforced for smartphones entering prototype proving grounds?",
        "expected_doc_ids": ["TSLA-ETH-2026-03"],
        "key_facts": ["tamper-evident", "security seal", "tape"],
        "is_answerable": True,
    },
    {
        "id": 5,
        "type": "single_doc",
        "question": "What specific reporting portal and confidential hotline are available for safety whistleblowers?",
        "expected_doc_ids": ["TSLA-ETH-2026-04"],
        "key_facts": ["compliance.tesla.com", "1-800-TSLA-ETHICS"],
        "is_answerable": True,
    },
    {
        "id": 6,
        "type": "single_doc",
        "question": "What secondary authentication factor is mandatory for high-privilege engineering accounts accessing single sign-on portals?",
        "expected_doc_ids": ["TSLA-ETH-2026-05"],
        "key_facts": ["fido2", "yubikey", "hardware security key"],
        "is_answerable": True,
    },
    {
        "id": 7,
        "type": "single_doc",
        "question": "What is the automated rotation schedule for service-to-service API keys and vector store tokens?",
        "expected_doc_ids": ["TSLA-ETH-2026-05"],
        "key_facts": ["90", "calendar days", "vault"],
        "is_answerable": True,
    },

    # Multi-Document Questions
    {
        "id": 8,
        "type": "multi_doc",
        "question": "How do whistleblower non-retaliation rules protect an employee who reports a breach of unreleased vehicle prototype secrecy?",
        "expected_doc_ids": ["TSLA-ETH-2026-03", "TSLA-ETH-2026-04"],
        "key_facts": ["retaliation", "whistleblower", "protected"],
        "is_answerable": True,
    },
    {
        "id": 9,
        "type": "multi_doc",
        "question": "What permissions and security controls govern exporting unhashed customer telemetry for fine-tuning external generative AI models?",
        "expected_doc_ids": ["TSLA-ETH-2026-02", "TSLA-ETH-2026-05"],
        "key_facts": ["prohibited", "enclave", "shadow ai"],
        "is_answerable": True,
    },
    {
        "id": 10,
        "type": "multi_doc",
        "question": "If a supplier offers an expensive dinner to a manager during an active contract negotiation for unreleased vehicle tooling, what combined policy rules apply?",
        "expected_doc_ids": ["TSLA-ETH-2026-01", "TSLA-ETH-2026-03"],
        "key_facts": ["gift", "entertainment", "rfp", "100"],
        "is_answerable": True,
    },
    {
        "id": 11,
        "type": "multi_doc",
        "question": "What role-based access control (RBAC) rules and encryption standards apply to querying internal vector database stores containing confidential ethics logs?",
        "expected_doc_ids": ["TSLA-ETH-2026-05", "TSLA-ETH-2026-01"],
        "key_facts": ["rbac", "role", "token", "vector"],
        "is_answerable": True,
    },

    # Out-of-Scope / Unanswerable Questions
    {
        "id": 12,
        "type": "out_of_scope",
        "question": "What is Tesla's annual financial stipend policy for remote workers purchasing ergonomic office chairs?",
        "expected_doc_ids": [],
        "key_facts": ["sufficient information", "not contain"],
        "is_answerable": False,
    },
    {
        "id": 13,
        "type": "out_of_scope",
        "question": "What are the exact quarterly vehicle delivery target numbers for Model Y in 2027?",
        "expected_doc_ids": [],
        "key_facts": ["sufficient information", "not contain"],
        "is_answerable": False,
    },
    {
        "id": 14,
        "type": "out_of_scope",
        "question": "Which private health insurance company manages dental vision claims for factory technicians in Gigafactory Berlin?",
        "expected_doc_ids": [],
        "key_facts": ["sufficient information", "not contain"],
        "is_answerable": False,
    },
    {
        "id": 15,
        "type": "out_of_scope",
        "question": "What is the company policy regarding employee pet adoption financial reimbursement packages?",
        "expected_doc_ids": [],
        "key_facts": ["sufficient information", "not contain"],
        "is_answerable": False,
    },
]


# ------------------------------------------------------------------------------
# 2. EVALUATION HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def evaluate_response(
    test_case: Dict[str, Any],
    rag_response: Dict[str, Any],
    latency_sec: float,
    config_name: str,
) -> Dict[str, Any]:
    """
    Evaluates answer correctness, groundedness, citation accuracy, retrieval accuracy,
    and hallucination rate for a single test run.
    """
    answer_text = rag_response.get("answer", "")
    sources = rag_response.get("sources", [])
    retrieved_doc_ids = [s.get("document_id") for s in sources if s.get("document_id")]

    # 1. Retrieval Accuracy
    if test_case["is_answerable"]:
        retrieval_correct = any(
            exp_id in retrieved_doc_ids for exp_id in test_case["expected_doc_ids"]
        )
    else:
        # Out of scope: retrieval accuracy is 1.0 if system handles unanswerable gracefully
        retrieval_correct = True

    # 2. Citation Accuracy
    # Check if answer contains explicit citations (e.g., TSLA-ETH-2026 or Section)
    has_citation = (
        "TSLA-ETH" in answer_text or "Section" in answer_text or "Doc" in answer_text
    )
    if test_case["is_answerable"]:
        citation_acc = 1.0 if has_citation else 0.0
    else:
        citation_acc = 1.0  # Not applicable/naturally cited if insufficient info

    # 3. Answer Correctness & Hallucination Rate
    answer_lower = answer_text.lower()
    insufficient_info_phrase = "sufficient information" in answer_lower or "do not contain" in answer_lower

    if not test_case["is_answerable"]:
        # Out-of-scope question:
        if insufficient_info_phrase:
            correctness = 1.0
            groundedness = 1.0
            hallucination = 0.0  # 0% hallucination for correct rejection
        else:
            correctness = 0.0
            groundedness = 0.0
            hallucination = 1.0  # Hallucinated a non-existent policy!
    else:
        # Answerable question:
        matched_facts = sum(
            1 for fact in test_case["key_facts"] if fact.lower() in answer_lower
        )
        fact_ratio = matched_facts / len(test_case["key_facts"]) if test_case["key_facts"] else 1.0
        correctness = round(fact_ratio, 2)
        groundedness = 1.0 if has_citation and correctness > 0.5 else 0.7

        # Check for hallucination in answerable questions (e.g. fabricated facts)
        if correctness < 0.2 and not insufficient_info_phrase:
            hallucination = 0.5
        else:
            hallucination = 0.0

    return {
        "config": config_name,
        "question_id": test_case["id"],
        "type": test_case["type"],
        "question": test_case["question"],
        "answer": answer_text.replace("\n", " "),
        "latency_sec": round(latency_sec, 3),
        "retrieval_accuracy": round(1.0 if retrieval_correct else 0.0, 2),
        "citation_accuracy": round(citation_acc, 2),
        "correctness": round(correctness, 2),
        "groundedness": round(groundedness, 2),
        "hallucination": round(hallucination, 2),
    }


# ------------------------------------------------------------------------------
# 3. BENCHMARK EXECUTION ENGINE
# ------------------------------------------------------------------------------
def run_benchmark():
    print("=" * 80)
    print("TESLA INTEGRITY BOT - GENERATION PARAMETER BENCHMARK SUITE")
    print("=" * 80)

    # Initialize RAGEngine
    engine = RAGEngine(data_dir="data", vector_dir="./vector_store")

    # Define Configurations
    configs = [
        {"name": "Config A (Temp 0.0, Top-P 0.1)", "temp": 0.0, "top_p": 0.1},
        {"name": "Config B (Temp 0.8, Top-P 0.9)", "temp": 0.8, "top_p": 0.9},
    ]

    all_results = []

    for cfg in configs:
        print(f"\n" + "#" * 80)
        print(f"RUNNING BENCHMARK FOR: {cfg['name']}")
        print("#" * 80)

        for item in TEST_DATASET:
            t0 = time.perf_counter()

            try:
                response = engine.answer_query(
                    query=item["question"],
                    top_k=3,
                    temperature=cfg["temp"],
                    top_p=cfg["top_p"],
                )
            except Exception as e:
                response = {
                    "answer": f"API Execution Error: {str(e)}",
                    "sources": [],
                }

            latency = time.perf_counter() - t0

            result = evaluate_response(
                test_case=item,
                rag_response=response,
                latency_sec=latency,
                config_name=cfg["name"],
            )

            all_results.append(result)
            print(
                f"  Q{item['id']:02d} [{item['type']:<12}] | Latency: {latency:.2f}s | "
                f"Correctness: {result['correctness']*100:.0f}% | Hallucination: {result['hallucination']*100:.0f}%"
            )

    # --------------------------------------------------------------------------
    # 4. SAVE CSV REPORT (benchmark_results.csv)
    # --------------------------------------------------------------------------
    csv_file = "benchmark_results.csv"
    csv_headers = [
        "config",
        "question_id",
        "type",
        "question",
        "latency_sec",
        "retrieval_accuracy",
        "citation_accuracy",
        "correctness",
        "groundedness",
        "hallucination",
        "answer",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSuccessfully generated CSV report: '{csv_file}'")

    # --------------------------------------------------------------------------
    # 5. PRINT SUMMARY COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY COMPARISON TABLE")
    print("=" * 80)

    summary_by_config = {}
    for cfg in configs:
        name = cfg["name"]
        cfg_rows = [r for r in all_results if r["config"] == name]
        n = len(cfg_rows)

        avg_latency = sum(r["latency_sec"] for r in cfg_rows) / n
        avg_retrieval = (sum(r["retrieval_accuracy"] for r in cfg_rows) / n) * 100
        avg_citation = (sum(r["citation_accuracy"] for r in cfg_rows) / n) * 100
        avg_correctness = (sum(r["correctness"] for r in cfg_rows) / n) * 100
        avg_groundedness = (sum(r["groundedness"] for r in cfg_rows) / n) * 100
        avg_hallucination = (sum(r["hallucination"] for r in cfg_rows) / n) * 100

        summary_by_config[name] = {
            "avg_latency": avg_latency,
            "retrieval_acc": avg_retrieval,
            "citation_acc": avg_citation,
            "correctness": avg_correctness,
            "groundedness": avg_groundedness,
            "hallucination": avg_hallucination,
        }

    header_line = f"{'Metric / Indicator':<30} | {'Config A (Temp 0.0, Top-P 0.1)':<32} | {'Config B (Temp 0.8, Top-P 0.9)':<32}"
    print("-" * 100)
    print(header_line)
    print("-" * 100)

    name_a = configs[0]["name"]
    name_b = configs[1]["name"]
    res_a = summary_by_config[name_a]
    res_b = summary_by_config[name_b]

    print(f"{'Average Latency (seconds)':<30} | {res_a['avg_latency']:<32.2f} | {res_b['avg_latency']:<32.2f}")
    print(f"{'Retrieval Accuracy (%)':<30} | {res_a['retrieval_acc']:<32.1f}% | {res_b['retrieval_acc']:<32.1f}%")
    print(f"{'Citation Accuracy (%)':<30} | {res_a['citation_acc']:<32.1f}% | {res_b['citation_acc']:<32.1f}%")
    print(f"{'Answer Correctness (%)':<30} | {res_a['correctness']:<32.1f}% | {res_b['correctness']:<32.1f}%")
    print(f"{'Groundedness Score (%)':<30} | {res_a['groundedness']:<32.1f}% | {res_b['groundedness']:<32.1f}%")
    print(f"{'Observed Hallucination Rate (%)':<30} | {res_a['hallucination']:<32.1f}% | {res_b['hallucination']:<32.1f}%")
    print("-" * 100)
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
