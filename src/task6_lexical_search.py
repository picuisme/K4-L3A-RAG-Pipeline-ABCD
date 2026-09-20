"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

from __future__ import annotations

import math

from .task4_chunking_indexing import chunk_documents, load_documents
from .text_utils import tokenize


CORPUS: list[dict] = []


class BM25Index:
    """Compact BM25 implementation so lexical retrieval also works offline."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in tokenized_corpus]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequencies: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for document in tokenized_corpus:
            counts: dict[str, int] = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            self.frequencies.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = len(tokenized_corpus)
        self.idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for index, counts in enumerate(self.frequencies):
            score = 0.0
            length = self.lengths[index]
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    return BM25Index([tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []
    corpus = CORPUS or chunk_documents(load_documents())
    if not corpus:
        return []
    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    indices = sorted(range(len(scores)), key=lambda i: (-scores[i], corpus[i]["id"]))
    results = []
    for index in indices:
        if scores[index] <= 0 or len(results) >= top_k:
            break
        item = corpus[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
