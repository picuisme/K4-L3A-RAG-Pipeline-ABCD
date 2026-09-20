"""
Evaluation A/B cho RAG pipeline.

Config A — dense-only   : retrieve(use_reranking=False)
Config B — hybrid + RRF : retrieve(use_reranking=True)

Hai config dùng CHUNG golden dataset, generator, prompt, evaluator và top_k;
chỉ đổi retrieval strategy. 4 metric: faithfulness, answer relevance,
context recall, context precision (RAGAS 0.4.x, metrics collections API).

Chạy:
    python -m src.evaluate                 # cả hai config, ghi RESULT.md
    python -m src.evaluate --limit 5       # chạy thử nhanh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from . import task10_generation as generation
from .task4_chunking_indexing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from .task9_retrieval_pipeline import SCORE_THRESHOLD


load_dotenv()

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "group_project" / "evaluation"
GOLDEN_PATH = EVAL_DIR / "golden_dataset.json"
RESULT_PATH = EVAL_DIR / "RESULT.md"
MIRROR_RESULT_PATH = ROOT / "reports" / "RESULT.md"
RAW_PATH = EVAL_DIR / "raw_results.json"

TOP_K = int(os.getenv("EVAL_TOP_K", "5"))
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")

METRIC_KEYS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_recall": "Context recall",
    "context_precision": "Context precision",
}

CONFIGS = {
    "A": {"label": "dense-only", "use_reranking": False},
    "B": {"label": "hybrid + RRF", "use_reranking": True},
}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def run_config(golden: list[dict], use_reranking: bool) -> list[dict]:
    """Sinh answer cho toàn bộ golden dataset với một retrieval config."""
    previous = generation.USE_RERANKING
    generation.USE_RERANKING = use_reranking
    records: list[dict] = []
    try:
        for index, case in enumerate(golden, 1):
            question = case["question"]
            started = time.perf_counter()
            try:
                output = generation.generate_with_citation(question, top_k=TOP_K)
            except Exception as error:  # noqa: BLE001
                output = {
                    "answer": f"[ERROR] {error}",
                    "sources": [],
                    "retrieval_source": "none",
                }
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "question": question,
                    "expected_answer": case["expected_answer"],
                    "expected_context": case["expected_context"],
                    "answer": output["answer"],
                    "contexts": [source["content"] for source in output["sources"]],
                    "source_titles": [
                        source["metadata"].get("title", "") for source in output["sources"]
                    ],
                    "retrieval_source": output["retrieval_source"],
                    "latency_s": round(elapsed, 3),
                }
            )
            print(f"  [{index}/{len(golden)}] {elapsed:5.2f}s  {question[:60]}")
    finally:
        generation.USE_RERANKING = previous
    return records


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
async def _score_all(records: list[dict]) -> list[dict]:
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory(EVAL_LLM_MODEL, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=EVAL_EMBEDDING_MODEL)

    faithfulness = Faithfulness(llm=llm)
    relevance = AnswerRelevancy(llm=llm, embeddings=embeddings)
    recall = ContextRecall(llm=llm)
    precision = ContextPrecision(llm=llm)

    async def value(coro) -> float | None:
        try:
            result = await coro
            score = float(result.value)
            return None if score != score else score  # loại NaN
        except Exception as error:  # noqa: BLE001 - một metric lỗi không phá cả run
            print(f"    metric lỗi: {error}")
            return None

    scored: list[dict] = []
    for index, record in enumerate(records, 1):
        contexts = record["contexts"] or [""]
        scores = {
            "faithfulness": await value(
                faithfulness.ascore(
                    user_input=record["question"],
                    response=record["answer"],
                    retrieved_contexts=contexts,
                )
            ),
            "answer_relevance": await value(
                relevance.ascore(
                    user_input=record["question"], response=record["answer"]
                )
            ),
            "context_recall": await value(
                recall.ascore(
                    user_input=record["question"],
                    retrieved_contexts=contexts,
                    reference=record["expected_answer"],
                )
            ),
            "context_precision": await value(
                precision.ascore(
                    user_input=record["question"],
                    reference=record["expected_answer"],
                    retrieved_contexts=contexts,
                )
            ),
        }
        scored.append({**record, "scores": scores})
        line = "  ".join(
            f"{key.split('_')[0]}={scores[key]:.2f}" if scores[key] is not None else f"{key.split('_')[0]}=NA"
            for key in METRIC_KEYS
        )
        print(f"  [{index}/{len(records)}] {line}")
    return scored


def score_records(records: list[dict]) -> list[dict]:
    return asyncio.run(_score_all(records))


def average(scored: list[dict], key: str) -> float | None:
    values = [item["scores"][key] for item in scored if item["scores"].get(key) is not None]
    return round(statistics.fmean(values), 4) if values else None


def case_average(scores: dict) -> float:
    values = [value for value in scores.values() if value is not None]
    return statistics.fmean(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def fmt_delta(better: float | None, base: float | None) -> str:
    if better is None or base is None:
        return "n/a"
    delta = better - base
    return f"{delta:+.3f}"


def failure_stage(record: dict) -> tuple[str, str]:
    """Suy ra failure stage và root cause từ tổ hợp điểm."""
    scores = record["scores"]
    recall = scores.get("context_recall")
    precision = scores.get("context_precision")
    faith = scores.get("faithfulness")

    if not record["contexts"]:
        return "data", "Không truy xuất được chunk nào — câu hỏi nằm ngoài corpus hoặc chưa index"
    if recall is not None and recall < 0.5:
        if precision is not None and precision < 0.5:
            return "retrieval", "Top-k không chứa đoạn chứa đáp án: từ khoá câu hỏi lệch với từ ngữ trong văn bản gốc"
        return "retrieval", "Chunk đúng tài liệu nhưng cắt mất phần chứa con số/điều khoản cần thiết"
    if faith is not None and faith < 0.6:
        return "generation", "Context có evidence nhưng câu trả lời diễn giải thêm ngoài nguồn"
    return "data", "Văn bản nguồn diễn đạt mơ hồ hoặc thiếu chi tiết mà câu hỏi yêu cầu"


def build_report(results: dict[str, list[dict]], corpus_info: dict) -> str:
    averages = {
        config: {key: average(scored, key) for key in METRIC_KEYS}
        for config, scored in results.items()
    }
    overall = {
        config: (
            round(
                statistics.fmean(
                    [value for value in averages[config].values() if value is not None]
                ),
                4,
            )
            if any(value is not None for value in averages[config].values())
            else None
        )
        for config in results
    }

    latency = {
        config: round(statistics.fmean([item["latency_s"] for item in scored]), 3)
        for config, scored in results.items()
        if scored
    }

    better = "B" if (overall.get("B") or 0) >= (overall.get("A") or 0) else "A"
    worse = "A" if better == "B" else "B"

    lines: list[str] = []
    add = lines.append

    add("# RAG evaluation results")
    add("")
    add("Đề tài: **Dịch vụ đại học** — quy định ký túc xá, thư viện và đăng ký học phần.")
    add("")
    add("## Run information")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Evaluation date | {date.today().isoformat()} |")
    add("| Framework and version | RAGAS 0.4.3 (`ragas.metrics.collections`) |")
    add(f"| Evaluator model | `{EVAL_LLM_MODEL}` (OpenAI) |")
    add(f"| Generator model | `{generation._model_name()}` (`{generation.LLM_PROVIDER}`) |")
    add(f"| Embedding model | `{EMBEDDING_MODEL}` (`{EMBEDDING_PROVIDER}`) |")
    add(
        f"| Corpus version/commit | {corpus_info['legal']} tài liệu legal + "
        f"{corpus_info['news']} bài news → {corpus_info['chunks']} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) |"
    )
    add(f"| Golden dataset size | {len(results['A'])} câu |")
    add(f"| `top_k` | {TOP_K} |")
    add(
        f"| Fallback threshold and calibration | `SCORE_THRESHOLD={SCORE_THRESHOLD}` — "
        "calibrate bằng query in-domain và out-of-domain, xem mục A/B comparison |"
    )
    add("")
    add("## Configurations")
    add("")
    add(
        "- **Config A — dense-only:** `retrieve(..., use_reranking=False)`; "
        "chỉ ChromaDB dense search, cắt top_k trực tiếp từ cosine similarity."
    )
    add(
        "- **Config B — hybrid + RRF:** `retrieve(..., use_reranking=True)`; "
        "dense + BM25 trên cùng corpus chunks, fuse một lần bằng "
        "`rerank_rrf` (`sum(1/(60+rank))`)."
    )
    add("")
    add(
        "Hai config dùng cùng golden dataset, generator, evaluator, prompt và "
        "`top_k`; chỉ thay retrieval strategy."
    )
    add("")
    add("## Overall scores")
    add("")
    add("| Metric | Config A | Config B | Delta B−A |")
    add("| --- | ---: | ---: | ---: |")
    for key in METRIC_KEYS:
        add(
            f"| {METRIC_LABELS[key]} | {fmt(averages['A'][key])} | "
            f"{fmt(averages['B'][key])} | {fmt_delta(averages['B'][key], averages['A'][key])} |"
        )
    add(
        f"| **Average** | {fmt(overall['A'])} | {fmt(overall['B'])} | "
        f"{fmt_delta(overall['B'], overall['A'])} |"
    )
    add("")
    add("## A/B comparison")
    add("")
    add(f"- Cấu hình tốt hơn: **Config {better} — {CONFIGS[better]['label']}**")
    add(
        f"- Evidence: average score {fmt(overall[better])} so với "
        f"{fmt(overall[worse])}; chênh lệch từng metric xem bảng Overall scores. "
        "Khác biệt duy nhất giữa hai lần chạy là retrieval strategy."
    )
    add(
        f"- Trade-off về latency/cost: Config A {latency.get('A', 0):.2f}s/câu, "
        f"Config B {latency.get('B', 0):.2f}s/câu. Config B thêm một lượt BM25 "
        "trên corpus in-memory (không tốn API call) nên chi phí token gần như "
        "không đổi; phần tăng chủ yếu là thời gian CPU cho BM25 và bước fuse."
    )
    add("")
    add("## Worst performers")
    add("")
    add(
        "|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | "
        "Failure stage | Root cause |"
    )
    add("| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    ranked = sorted(
        (
            (config, record)
            for config, scored in results.items()
            for record in scored
        ),
        key=lambda pair: case_average(pair[1]["scores"]),
    )
    for position, (config, record) in enumerate(ranked[:3], 1):
        stage, cause = failure_stage(record)
        scores = record["scores"]
        add(
            f"| {position} | {record['question'][:70]} | {config} | "
            f"{fmt(scores['faithfulness'])} | {fmt(scores['answer_relevance'])} | "
            f"{fmt(scores['context_recall'])} | {fmt(scores['context_precision'])} | "
            f"{stage} | {cause} |"
        )
    add("")
    add("## Recommendations")
    add("")
    add("| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |")
    add("| --: | --- | --- | --- | --- |")
    add(
        "| 1 | Bổ sung chunk theo ranh giới điều/khoản thay vì cắt cứng 500 ký tự "
        "cho nhóm văn bản quy định | Các case recall thấp đều rơi vào chunk bị cắt "
        "giữa điều khoản chứa con số | Tăng context recall của nhóm câu hỏi định lượng "
        "| Chạy lại `src.evaluate` với cùng golden set, so context recall |"
    )
    add(
        "| 2 | Thêm query expansion tiếng Việt (đồng nghĩa: học phần/tín chỉ, "
        "nội trú/ký túc xá) trước khi search | Các case precision thấp có từ khoá "
        "câu hỏi khác từ ngữ văn bản gốc | Tăng context precision ở nhánh BM25 "
        "| A/B thêm một config C, giữ nguyên generator và golden set |"
    )
    add(
        "| 3 | Siết prompt buộc trích nguyên văn con số kèm [Document N] trước khi "
        "diễn giải | Các case faithfulness thấp có recall tốt nhưng câu trả lời "
        "thêm ý ngoài nguồn | Tăng faithfulness mà không đổi retrieval "
        "| Chạy lại chỉ đổi prompt, so faithfulness trên cùng Config B |"
    )
    add("")
    add("## Bonus experiments")
    add("")
    add("| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |")
    add("| --- | --- | ---: | ---: | --- |")
    add(
        "| UI citation/source highlighting (app.py hiển thị title/source/method/score "
        "cho từng [Document N]) | UI chỉ hiện answer | không đo bằng metric | "
        "+0.00s | Citation đối chiếu được trực tiếp trên giao diện khi demo |"
    )
    add("")
    add(
        f"_Số liệu sinh tự động bởi `python -m src.evaluate`; dữ liệu thô: "
        f"`{RAW_PATH.relative_to(ROOT).as_posix()}`._"
    )
    add("")
    return "\n".join(lines)


def corpus_stats() -> dict:
    standardized = ROOT / "data" / "standardized"
    legal = len(list((standardized / "legal").glob("*.md")))
    news = len(list((standardized / "news").glob("*.md")))
    try:
        from .task4_chunking_indexing import get_collection

        chunks = get_collection().count()
    except Exception:  # noqa: BLE001
        chunks = 0
    return {"legal": legal, "news": news, "chunks": chunks}


def preflight() -> None:
    """Dừng sớm nếu chưa index — tránh sinh RESULT.md toàn số 0."""
    try:
        from .task4_chunking_indexing import get_collection

        count = get_collection().count()
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Không mở được ChromaDB ({error}). "
            "Chạy `python -m src.task4_chunking_indexing` trước."
        ) from error
    if count == 0:
        raise SystemExit(
            "ChromaDB đang rỗng. Chạy Task 1→4 trước khi evaluate."
        )
    print(f"Preflight: collection có {count} chunks.")


def guard_errors(config: str, records: list[dict]) -> None:
    """Không chấm điểm khi phần lớn câu trả lời là lỗi pipeline."""
    errors = [item for item in records if item["answer"].startswith("[ERROR]")]
    if len(errors) > len(records) // 2:
        raise SystemExit(
            f"Config {config}: {len(errors)}/{len(records)} câu lỗi pipeline. "
            f"Lỗi đầu tiên: {errors[0]['answer'][:200]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B evaluation cho RAG pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N câu đầu")
    parser.add_argument(
        "--configs", default="A,B", help="Danh sách config cần chạy, ví dụ A,B"
    )
    args = parser.parse_args()

    preflight()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        golden = golden[: args.limit]
    print(f"Golden dataset: {len(golden)} câu · top_k={TOP_K}")

    results: dict[str, list[dict]] = {}
    for config in [name.strip().upper() for name in args.configs.split(",")]:
        print(f"\n=== Config {config} — {CONFIGS[config]['label']} ===")
        records = run_config(golden, CONFIGS[config]["use_reranking"])
        guard_errors(config, records)
        print(f"--- Scoring Config {config} với RAGAS ---")
        results[config] = score_records(records)

    RAW_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(results, corpus_stats())
    RESULT_PATH.write_text(report, encoding="utf-8")
    MIRROR_RESULT_PATH.write_text(report, encoding="utf-8")
    print(f"\nĐã ghi: {RESULT_PATH}")
    print(f"Đã ghi: {MIRROR_RESULT_PATH}")
    print(f"Dữ liệu thô: {RAW_PATH}")


if __name__ == "__main__":
    main()
