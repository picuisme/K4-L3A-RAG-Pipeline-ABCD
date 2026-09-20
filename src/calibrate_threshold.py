"""
Calibrate SCORE_THRESHOLD cho fallback.

Ngưỡng fallback so với **cosine gốc của dense search**, không phải RRF score.
Script này in best dense cosine cho một nhóm query in-domain và một nhóm
out-of-domain rồi đề xuất ngưỡng nằm giữa hai phân bố. Copy kết quả vào
group_project/evaluation/RESULT.md mục "Fallback threshold and calibration".

Chạy:
    python -m src.calibrate_threshold
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search


IN_DOMAIN = [
    "Sinh viên được mượn tối đa bao nhiêu cuốn giáo trình?",
    "Phí ký túc xá nộp theo học kỳ hay theo tháng?",
    "Một học kỳ chính được đăng ký tối đa bao nhiêu tín chỉ?",
    "Khi nào sinh viên bị cảnh báo học tập?",
    "Giờ giữ yên tĩnh trong ký túc xá là mấy giờ?",
]

OUT_OF_DOMAIN = [
    "Công thức nấu phở bò Nam Định gồm những nguyên liệu gì?",
    "Giá vé xem trận chung kết Champions League là bao nhiêu?",
    "Cách cài đặt Kubernetes trên Ubuntu 24.04?",
    "Thời tiết Đà Lạt tuần sau thế nào?",
    "Ai là người sáng lập hãng xe Ferrari?",
]


def best_score(query: str) -> float:
    results = semantic_search(query, top_k=5)
    return results[0]["score"] if results else 0.0


def main() -> None:
    print("=== IN-DOMAIN ===")
    in_scores = []
    for query in IN_DOMAIN:
        score = best_score(query)
        in_scores.append(score)
        print(f"  {score:.4f}  {query}")

    print("\n=== OUT-OF-DOMAIN ===")
    out_scores = []
    for query in OUT_OF_DOMAIN:
        score = best_score(query)
        out_scores.append(score)
        print(f"  {score:.4f}  {query}")

    low_in = min(in_scores) if in_scores else 0.0
    high_out = max(out_scores) if out_scores else 0.0
    suggested = round((low_in + high_out) / 2, 2)

    print("\n=== KẾT LUẬN ===")
    print(f"  min(in-domain)  = {low_in:.4f}")
    print(f"  max(out-domain) = {high_out:.4f}")
    if low_in > high_out:
        print(f"  Hai phân bố tách rời → đặt SCORE_THRESHOLD = {suggested}")
    else:
        print(
            f"  Hai phân bố chồng lấn → chọn {suggested} và chấp nhận đánh đổi; "
            "cân nhắc chunk lại hoặc đổi embedding model."
        )
    print(f"\n  Ghi vào .env:  SCORE_THRESHOLD={suggested}")


if __name__ == "__main__":
    main()
