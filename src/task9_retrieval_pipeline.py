"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search trên cùng corpus chunks.
    2. Fuse hai danh sách bằng RRF **đúng một lần**.
    3. Lấy best cosine score GỐC từ dense results (không phải RRF score).
    4. Nếu score dưới threshold → thử PageIndex fallback.
    5. Fallback lỗi hoặc rỗng → trả hybrid results thay vì crash.

SCORE_THRESHOLD phải calibrate bằng query in-domain và out-of-domain; xem
group_project/evaluation/RESULT.md mục "Fallback threshold and calibration".
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.35)
DEFAULT_TOP_K = 5

# Mỗi nhánh lấy top_k * 5 ứng viên trước khi fuse. Với corpus ~900 chunk và
# nhiều nhiễu từ bài báo, lấy 10 ứng viên (multiplier=2) là quá hẹp: chunk
# đúng thường nằm hạng 12-20 ở một nhánh và bị cắt trước khi RRF kịp cộng
# điểm từ nhánh còn lại. RRF vẫn chỉ chạy một lần, chi phí thêm không đáng kể
# vì cả hai nhánh đều chạy in-memory sau khi đã có query embedding.
CANDIDATE_MULTIPLIER = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid (hoặc dense khi tắt rerank) hoặc pageindex SearchResult."""
    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    dense = semantic_search(query, top_k=candidate_k)
    sparse = lexical_search(query, top_k=candidate_k)

    # RRF chỉ được gọi đúng một lần trong toàn pipeline.
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else dense[:top_k]
    )

    # Ngưỡng so với cosine GỐC của dense, không phải RRF score.
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as error:  # noqa: BLE001 - provider lỗi không được crash UI
            print(f"PageIndex fallback không dùng được: {error}")

    return hybrid[:top_k]


if __name__ == "__main__":
    for question in (
        "Sinh viên được mượn tối đa bao nhiêu cuốn tài liệu ở thư viện?",
        "Công thức nấu phở bò Nam Định gồm những gì?",
    ):
        print(f"\n### {question}")
        for result in retrieve(question, top_k=3):
            print(
                f"  [{result['retrieval_method']}] {result['score']:.4f}  "
                f"{result['metadata']['title'][:50]}"
            )
