"""Deterministic A/B evaluation for dense-only versus hybrid + RRF.

The metrics are transparent token-overlap proxies intended for an offline lab run.
Use RAGAS with a fixed evaluator model for a production-grade semantic evaluation.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import statistics
import time

from .task10_generation import extractive_answer, reorder_for_llm
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task4_chunking_indexing import EMBEDDING_MODEL
from .text_utils import token_set


ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
REPORT_PATH = ROOT / "group_project" / "evaluation" / "RESULT.md"
TOP_K = 5

STOPWORDS = {
    "va", "la", "cua", "cho", "duoc", "trong", "khi", "tai", "theo", "mot",
    "nhung", "cac", "co", "khong", "sinh", "vien", "phai", "ve", "voi",
}


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in token_set(text) if token not in STOPWORDS and len(token) > 2}


def overlap_recall(reference: str, candidate: str) -> float:
    reference_tokens = meaningful_tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & meaningful_tokens(candidate)) / len(reference_tokens)


def evaluate_case(item: dict, results: list[dict], latency_ms: float) -> dict:
    contexts = [result["content"] for result in results]
    joined_context = " ".join(contexts)
    answer = extractive_answer(item["question"], reorder_for_llm(results))
    faithfulness = overlap_recall(answer, joined_context)
    relevance = overlap_recall(item["expected_answer"], answer)
    context_recall = overlap_recall(item["expected_context"], joined_context)
    expected_tokens = meaningful_tokens(item["expected_context"])
    relevant_chunks = sum(
        1
        for context in contexts
        if len(expected_tokens & meaningful_tokens(context)) >= 2
    )
    context_precision = relevant_chunks / len(contexts) if contexts else 0.0
    return {
        "question": item["question"],
        "answer": answer,
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "latency_ms": latency_ms,
    }


def run_configuration(dataset: list[dict], hybrid: bool) -> list[dict]:
    evaluations = []
    for item in dataset:
        started = time.perf_counter()
        dense = semantic_search(item["question"], top_k=TOP_K * 2)
        if hybrid:
            sparse = lexical_search(item["question"], top_k=TOP_K * 2)
            results = rerank_rrf([dense, sparse], top_k=TOP_K)
        else:
            results = dense[:TOP_K]
        latency_ms = (time.perf_counter() - started) * 1000
        evaluations.append(evaluate_case(item, results, latency_ms))
    return evaluations


def summarize(rows: list[dict]) -> dict[str, float]:
    keys = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
    summary = {key: statistics.mean(row[key] for row in rows) for key in keys}
    summary["average"] = statistics.mean(summary.values())
    summary["latency_ms"] = statistics.mean(row["latency_ms"] for row in rows)
    return summary


def write_report(dense_rows: list[dict], hybrid_rows: list[dict]) -> None:
    dense = summarize(dense_rows)
    hybrid = summarize(hybrid_rows)
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
        "average": "**Average**",
    }
    score_rows = []
    for key, label in labels.items():
        score_rows.append(
            f"| {label} | {dense[key]:.3f} | {hybrid[key]:.3f} | "
            f"{hybrid[key] - dense[key]:+.3f} |"
        )

    combined = []
    for config, rows in (("A", dense_rows), ("B", hybrid_rows)):
        for row in rows:
            mean_score = statistics.mean(
                row[key]
                for key in ("faithfulness", "answer_relevance", "context_recall", "context_precision")
            )
            combined.append((mean_score, config, row))
    combined.sort(key=lambda item: item[0])
    worst_rows = []
    for index, (_, config, row) in enumerate(combined[:3], 1):
        stage = "retrieval" if row["context_recall"] < 0.5 else "generation"
        cause = (
            "Từ khóa kỳ vọng chưa xuất hiện trong top-k"
            if stage == "retrieval"
            else "Bộ sinh trích xuất chưa tổng hợp đủ ý"
        )
        question = row["question"].replace("|", "/")
        worst_rows.append(
            f"| {index} | {question} | {config} | {row['faithfulness']:.3f} | "
            f"{row['answer_relevance']:.3f} | {row['context_recall']:.3f} | "
            f"{row['context_precision']:.3f} | {stage} | {cause} |"
        )

    winner = "Config B — hybrid + RRF" if hybrid["average"] >= dense["average"] else "Config A — dense-only"
    report = f"""# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | {date.today().isoformat()} |
| Framework and version | Offline overlap evaluator v1 (`src/evaluate.py`) |
| Evaluator model | Deterministic Vietnamese token-overlap proxies; no API |
| Generator model | Grounded extractive baseline; no API |
| Embedding model | {EMBEDDING_MODEL} |
| Corpus version/commit | `feat.hodinhtuankiet` working tree |
| Golden dataset size | {len(dense_rows)} |
| `top_k` | {TOP_K} |
| Fallback threshold and calibration | 0.39; golden queries scored 0.410–0.673 and five unrelated queries scored 0.196–0.367; fallback excluded from A/B |

## Configurations

- **Config A — dense-only:** shared hashing embedding, cosine search, top-{TOP_K}.
- **Config B — hybrid + RRF:** same dense candidates plus BM25, fused once with RRF `k=60`, top-{TOP_K}.

Both configurations use the same corpus, golden dataset, extractive generator and `top_k`.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
{chr(10).join(score_rows)}

These are reproducible offline proxy scores in `[0, 1]`, not LLM-judged RAGAS scores. They make the A/B retrieval change auditable without an API key.

## A/B comparison

- Better configuration: **{winner}**.
- Evidence: average changed from `{dense['average']:.3f}` to `{hybrid['average']:.3f}`; context recall changed by `{hybrid['context_recall'] - dense['context_recall']:+.3f}`.
- Latency/cost trade-off: dense averaged `{dense['latency_ms']:.1f} ms/query`; hybrid averaged `{hybrid['latency_ms']:.1f} ms/query`. Both runs used no paid API.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|--:|---|:---:|---:|---:|---:|---:|---|---|
{chr(10).join(worst_rows)}

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Replace hashing baseline with a Vietnamese/multilingual sentence-transformer | Dense misses paraphrases in low-recall cases | Higher semantic recall | Re-run the same 17 cases and compare recall |
| 2 | Add heading-aware chunking and parent-section context | Some rules span adjacent chunks | Better context completeness | Measure recall and inspect the three worst cases |
| 3 | Use a fixed LLM judge with RAGAS in CI/nightly evaluation | Token overlap does not capture semantic equivalence | More valid generation metrics | Pin evaluator model/prompt and compare with this baseline |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Hybrid BM25 + RRF | Dense-only | `{hybrid['average'] - dense['average']:+.3f}` average | `{hybrid['latency_ms'] - dense['latency_ms']:+.1f} ms/query`, no API cost | Use the better configuration reported above; rerun after upgrading embeddings. |
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dense_rows = run_configuration(dataset, hybrid=False)
    hybrid_rows = run_configuration(dataset, hybrid=True)
    write_report(dense_rows, hybrid_rows)
    print(f"Evaluated {len(dataset)} cases; report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
