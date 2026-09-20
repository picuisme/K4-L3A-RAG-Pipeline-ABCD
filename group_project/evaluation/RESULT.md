# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-20 |
| Framework and version | Offline overlap evaluator v1 (`src/evaluate.py`) |
| Evaluator model | Deterministic Vietnamese token-overlap proxies; no API |
| Generator model | Grounded extractive baseline; no API |
| Embedding model | hashing-vi-1024 |
| Corpus version/commit | `feature/legal-rag-individual-report` working tree |
| Golden dataset size | 17 |
| `top_k` | 5 |
| Fallback threshold and calibration | 0.39; golden queries scored 0.410–0.673 and five unrelated queries scored 0.196–0.367; fallback excluded from A/B |

## Configurations

- **Config A — dense-only:** shared hashing embedding, cosine search, top-5.
- **Config B — hybrid + RRF:** same dense candidates plus BM25, fused once with RRF `k=60`, top-5.

Both configurations use the same corpus, golden dataset, extractive generator and `top_k`.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer relevance | 0.745 | 0.772 | +0.028 |
| Context recall | 0.954 | 0.971 | +0.016 |
| Context precision | 0.988 | 1.000 | +0.012 |
| **Average** | 0.922 | 0.936 | +0.014 |

These are reproducible offline proxy scores in `[0, 1]`, not LLM-judged RAGAS scores. They make the A/B retrieval change auditable without an API key.

## A/B comparison

- Better configuration: **Config B — hybrid + RRF**.
- Evidence: average changed from `0.922` to `0.936`; context recall changed by `+0.016`.
- Latency/cost trade-off: dense averaged `249.6 ms/query`; hybrid averaged `265.0 ms/query`. Both runs used no paid API.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|--:|---|:---:|---:|---:|---:|---:|---|---|
| 1 | Khi ra vào KTX Đại học Cần Thơ, sinh viên phải mang theo gì? | A | 1.000 | 0.429 | 0.750 | 1.000 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |
| 2 | Sinh viên nội trú KTX Đại học Cần Thơ phải giữ yên lặng vào những khung giờ nào? | A | 1.000 | 0.600 | 0.600 | 1.000 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |
| 3 | Nhóm ưu tiên số 1 khi xét nội trú KTX HUIT gồm những ai? | A | 1.000 | 0.235 | 1.000 | 1.000 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Replace hashing baseline with a Vietnamese/multilingual sentence-transformer | Dense misses paraphrases in low-recall cases | Higher semantic recall | Re-run the same 17 cases and compare recall |
| 2 | Add heading-aware chunking and parent-section context | Some rules span adjacent chunks | Better context completeness | Measure recall and inspect the three worst cases |
| 3 | Use a fixed LLM judge with RAGAS in CI/nightly evaluation | Token overlap does not capture semantic equivalence | More valid generation metrics | Pin evaluator model/prompt and compare with this baseline |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Hybrid BM25 + RRF | Dense-only | `+0.014` average | `+15.5 ms/query`, no API cost | Use the better configuration reported above; rerun after upgrading embeddings. |
