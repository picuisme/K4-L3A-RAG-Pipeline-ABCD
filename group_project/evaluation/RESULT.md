# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-20 |
| Framework and version | Offline overlap evaluator v1 (`src/evaluate.py`) |
| Evaluator model | Deterministic Vietnamese token-overlap proxies; no API |
| Generator model | Grounded extractive baseline; no API |
| Embedding model | text-embedding-3-small |
| Corpus version/commit | `main` working tree |
| Golden dataset size | 15 |
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
| Answer relevance | 0.814 | 0.731 | -0.083 |
| Context recall | 0.989 | 0.989 | +0.000 |
| Context precision | 0.987 | 0.947 | -0.040 |
| **Average** | 0.948 | 0.917 | -0.031 |

These are reproducible offline proxy scores in `[0, 1]`, not LLM-judged RAGAS scores. They make the A/B retrieval change auditable without an API key.

## A/B comparison

- Better configuration: **Config A — dense-only**.
- Evidence: average changed from `0.948` to `0.917`; context recall changed by `+0.000`.
- Latency/cost trade-off: dense averaged `1281.8 ms/query`; hybrid averaged `1717.6 ms/query`. Both runs used no paid API.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|--:|---|:---:|---:|---:|---:|---:|---|---|
| 1 | Đối tượng nào thuộc nhóm ưu tiên 1 khi xét duyệt vào ở ký túc xá của Trường Đại học Công nghiệp Thực phẩm TP.HCM (HUIT)? | B | 1.000 | 0.050 | 1.000 | 0.600 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |
| 2 | Giờ nghỉ ngơi tại Ký túc xá Trường Đại học Cần Thơ được quy định vào những khung giờ nào mà sinh viên không được làm ồn ào? | B | 1.000 | 0.300 | 1.000 | 0.800 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |
| 3 | Quy định về việc nấu ăn trong phòng ở Ký túc xá Trường Đại học Cần Thơ như thế nào? | A | 1.000 | 0.231 | 1.000 | 1.000 | generation | Bộ sinh trích xuất chưa tổng hợp đủ ý |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Replace hashing baseline with a Vietnamese/multilingual sentence-transformer | Dense misses paraphrases in low-recall cases | Higher semantic recall | Re-run the same 15 cases and compare recall |
| 2 | Add heading-aware chunking and parent-section context | Some rules span adjacent chunks | Better context completeness | Measure recall and inspect the three worst cases |
| 3 | Use a fixed LLM judge with RAGAS in CI/nightly evaluation | Token overlap does not capture semantic equivalence | More valid generation metrics | Pin evaluator model/prompt and compare with this baseline |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
|---|---|---:|---:|---|
| Hybrid BM25 + RRF | Dense-only | `-0.031` average | `+435.8 ms/query`, no API cost | Use the better configuration reported above; rerun after upgrading embeddings. |
