"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    query_vectors = embed_texts([query])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    collection = get_collection()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0] if response.get("ids") else []
    documents = response.get("documents", [[]])[0] if response.get("documents") else []
    metadatas = response.get("metadatas", [[]])[0] if response.get("metadatas") else []
    distances = response.get("distances", [[]])[0] if response.get("distances") else []

    results = []
    seen_ids = set()

    for item_id, content, meta, distance in zip(ids, documents, metadatas, distances):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        clean_meta = dict(meta) if meta else {}
        if "chunk_index" in clean_meta:
            clean_meta["chunk_index"] = int(clean_meta["chunk_index"])
        if clean_meta.get("url") == "":
            clean_meta["url"] = None

        # Đổi cosine distance thành cosine similarity score [0.0, 1.0]
        score = float(max(0.0, 1.0 - float(distance)))

        results.append({
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": clean_meta,
            "retrieval_method": "dense",
        })

    # Sắp xếp giảm dần theo điểm số similarity
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    test_results = semantic_search("Ký túc xá Đại học Cần Thơ quy định gì về giờ giấc?", top_k=3)
    for res in test_results:
        print(f"[{res['score']:.4f}] {res['id']}: {res['content'][:120]}...")
