"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.
"""

from typing import Any
from .contracts import validate_search_results


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult theo thứ tự score giảm dần."""
    if top_k <= 0 or not ranked_lists:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results: list[dict] = []

    for item_id in ranked_ids[:top_k]:
        base_item = items[item_id]
        clean_meta: dict[str, Any] = dict(base_item.get("metadata", {}))
        if "chunk_index" in clean_meta:
            clean_meta["chunk_index"] = int(clean_meta["chunk_index"])
        if clean_meta.get("url") == "":
            clean_meta["url"] = None

        result = {
            "id": item_id,
            "content": base_item["content"],
            "score": float(scores[item_id]),
            "metadata": clean_meta,
            "retrieval_method": "hybrid",
        }
        results.append(result)

    validate_search_results(results, top_k=top_k, expected_method="hybrid")
    return results


if __name__ == "__main__":
    test_dense = [
        {"id": "chunk-0", "content": "Sample content 0", "score": 0.9, "metadata": {"source": "doc.md", "title": "Doc", "doc_type": "legal", "url": None, "chunk_index": 0}, "retrieval_method": "dense"},
        {"id": "chunk-1", "content": "Sample content 1", "score": 0.8, "metadata": {"source": "doc.md", "title": "Doc", "doc_type": "legal", "url": None, "chunk_index": 1}, "retrieval_method": "dense"},
    ]
    test_bm25 = [
        {"id": "chunk-1", "content": "Sample content 1", "score": 7.0, "metadata": {"source": "doc.md", "title": "Doc", "doc_type": "legal", "url": None, "chunk_index": 1}, "retrieval_method": "bm25"},
        {"id": "chunk-2", "content": "Sample content 2", "score": 5.0, "metadata": {"source": "doc.md", "title": "Doc", "doc_type": "legal", "url": None, "chunk_index": 2}, "retrieval_method": "bm25"},
    ]
    fused = rerank_rrf([test_dense, test_bm25], top_k=3)
    for f in fused:
        print(f"[{f['score']:.6f}] {f['id']}: {f['content']}")
