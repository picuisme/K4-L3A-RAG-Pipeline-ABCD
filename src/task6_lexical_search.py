"""
Task 6 — Lexical search bằng BM25.

BM25 chạy trên **cùng corpus chunks** mà Task 4 đã index: nếu hai nhánh dùng
hai corpus khác nhau thì RRF ở Task 7 fuse theo ID sẽ không khớp. Dense bắt ý
nghĩa gần đúng, BM25 bắt chính xác từ khoá, số quyết định và tên riêng — hai
loại bổ sung nhau, không thay thế.
"""

from __future__ import annotations

import re


# Rỗng lúc import; nạp lười từ vector store ở lần search đầu tiên.
# Test contract monkeypatch trực tiếp biến này nên mọi truy cập phải ở runtime.
CORPUS: list[dict] = []

_bm25_cache: dict[tuple[int, int], object] = {}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenizer đơn giản, giữ dấu tiếng Việt và chữ số."""
    return _TOKEN_RE.findall(text.lower())


def ensure_corpus() -> list[dict]:
    """Nạp corpus từ ChromaDB nếu CORPUS chưa được gán sẵn."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import load_corpus_from_store

        CORPUS = load_corpus_from_store()
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo (và cache) BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    key = (id(corpus), len(corpus))
    cached = _bm25_cache.get(key)
    if cached is not None:
        return cached

    tokenized = [tokenize(item["content"]) or ["_"] for item in corpus]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache[key] = bm25
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    corpus = CORPUS if CORPUS else ensure_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    order = sorted(range(len(corpus)), key=lambda i: float(scores[i]), reverse=True)
    query_tokens = set(tokens)

    results: list[dict] = []
    seen: set[str] = set()
    for index in order:
        score = float(scores[index])
        if score < 0:
            break
        item = corpus[index]
        # BM25Okapi cho idf = 0 khi một từ xuất hiện ở đúng một nửa số document
        # (corpus rất nhỏ). Lúc đó vẫn giữ document nếu nó thực sự chứa từ khoá,
        # và loại document không khớp từ nào.
        if score <= 0 and not (query_tokens & set(tokenize(item["content"]))):
            continue
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        metadata = dict(item["metadata"])
        metadata.setdefault("chunk_index", 0)
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": metadata,
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("mượn tài liệu thư viện quá hạn", top_k=3):
        print(f"{result['score']:.4f}  {result['id']}")
        print(f"    {result['content'][:120]}...")
