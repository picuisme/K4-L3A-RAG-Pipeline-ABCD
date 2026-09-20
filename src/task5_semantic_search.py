"""
Task 5 — Semantic search (dense).

Query được embed bằng chính embed_texts() của Task 4 nên corpus và query luôn
cùng model, cùng dimension — khác model là dense search sai âm thầm mà không
báo lỗi. Chroma trả cosine distance, ta đổi sang similarity = 1 - distance và
giữ nguyên giá trị này: Task 9 dùng đúng cosine gốc để quyết định fallback.
"""

from __future__ import annotations

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if item_id in seen or not content:
            continue
        seen.add(item_id)
        metadata = dict(metadata or {})
        metadata.setdefault("chunk_index", 0)
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": metadata,
                "retrieval_method": "dense",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Phí ký túc xá đóng theo kỳ hay theo tháng?", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"    {result['content'][:120]}...")
