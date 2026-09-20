"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score (hai thang đo khác nhau, cộng thẳng là sai).

    RRF(d) = sum(1 / (k + rank)),  rank bắt đầu từ 1

Lưu ý: RRF score luôn rất nhỏ (~0.03) và chỉ phản ánh thứ hạng. Không bao giờ
so nó với SCORE_THRESHOLD — Task 9 dùng cosine gốc của dense cho việc đó.
"""

from __future__ import annotations


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    best_rank: dict[str, int] = {}

    for ranked_list in ranked_lists or []:
        for rank, item in enumerate(ranked_list or [], 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
            items.setdefault(item_id, item)

    ranked_ids = sorted(
        scores,
        key=lambda item_id: (-scores[item_id], best_rank[item_id], item_id),
    )

    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        result = dict(items[item_id])
        result["metadata"] = dict(result["metadata"])
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)
    return results


if __name__ == "__main__":
    print("Implement xong rerank_rrf — chạy: pytest tests/test_contracts.py -k rrf -q")
