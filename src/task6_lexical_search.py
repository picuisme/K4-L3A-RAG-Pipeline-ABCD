"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 4. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import math
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi


class RobustBM25(BM25Okapi):
    """BM25Okapi cải tiến với công thức Lucene IDF để tránh IDF = 0 trên tập dữ liệu nhỏ."""

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


CORPUS: list[dict] = []
_bm25_index: RobustBM25 | None = None
_cached_corpus_id: int | None = None


def get_corpus() -> list[dict]:
    """Nạp chunks từ Task 4 nếu CORPUS chưa được khởi tạo."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents

        documents = load_documents()
        CORPUS = chunk_documents(documents)
    return CORPUS


def build_bm25_index(corpus: list[dict]) -> RobustBM25:
    """Tạo BM25 index từ danh sách corpus chunks."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return RobustBM25(tokenized)


def get_bm25_index(corpus: list[dict]) -> RobustBM25:
    """Lấy hoặc tạo mới BM25 index nếu corpus thay đổi."""
    global _bm25_index, _cached_corpus_id
    current_id = id(corpus)
    if _bm25_index is None or _cached_corpus_id != current_id:
        _bm25_index = build_bm25_index(corpus)
        _cached_corpus_id = current_id
    return _bm25_index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    corpus = CORPUS if CORPUS else get_corpus()
    if not corpus:
        return []

    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    bm25 = get_bm25_index(corpus)
    scores = bm25.get_scores(tokenized_query)
    indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()

    for index in indices:
        score = float(scores[index])
        if score <= 0:
            continue

        item = corpus[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        clean_meta: dict[str, Any] = dict(item.get("metadata", {}))
        if "chunk_index" in clean_meta:
            clean_meta["chunk_index"] = int(clean_meta["chunk_index"])
        if clean_meta.get("url") == "":
            clean_meta["url"] = None

        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": clean_meta,
            "retrieval_method": "bm25",
        })

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("KTX Đại học Cần Thơ", top_k=3):
        print(f"[{result['score']:.4f}] {result['id']}: {result['content'][:100]}...")
