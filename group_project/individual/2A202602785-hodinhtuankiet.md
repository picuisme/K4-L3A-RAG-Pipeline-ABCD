# Báo cáo đóng góp cá nhân

## Thông tin

- **Họ và tên:** Hồ Đình Tuấn Kiệt
- **Mã học viên:** 2A202602785
- **Nhóm:** K4-L3A — RAG Pipeline ABCD
- **Repository/branch:** `K4-L3A-RAG-Pipeline-ABCD` / `feat.hodinhtuankiet`
- **Commit triển khai:** `7f5e189` — `feat: complete university services RAG pipeline`
- **Commit báo cáo:** `e8bdbd6` — `docs: complete individual contribution report`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit | Trạng thái |
|---|---|---|---|
| Data & chuẩn hóa | Kiểm tra 5 PDF, crawl 5 trang trường đại học có metadata, chuyển thành 10 Markdown không rỗng | `data/`, `src/task1_*`–`task3_*`; `7f5e189` | Done |
| Chunk, embedding, index | Chunk có overlap, ID ổn định, hashing embedding 1024 chiều dùng chung; ChromaDB và local fallback đều upsert idempotent | `src/task4_chunking_indexing.py`; `7f5e189` | Done |
| Hybrid retrieval | Hoàn thiện cosine search, BM25 tiếng Việt, RRF theo ID và page/section fallback; fallback quyết định bằng cosine gốc | `src/task5_*`–`task9_*`; `7f5e189` | Done |
| Generation & UI | Dispatch 3 LLM provider, extractive fallback offline, safe refusal và citation `[S1]`; UI Streamlit hiển thị nguồn/điểm | `src/task10_generation.py`, `app.py`; `7f5e189` | Done |
| Evaluation | Tạo 17 golden Q&A, evaluator 4 metric và A/B dense-only với hybrid + RRF | `src/evaluate.py`, `group_project/evaluation/`; `7f5e189` | Done |
| Tài liệu chạy lại | Cập nhật quick start Python 3.11, chế độ offline, lệnh index/evaluate/demo | `README.md`, `.env.example`; `7f5e189` | Done |

## Quyết định kỹ thuật quan trọng

1. **Dùng hashing embedding làm baseline mặc định, cho phép chuyển sang Sentence Transformers.**
   **Lý do/evidence:** Pipeline cần chạy được không API/model download; cùng `embed_texts()` được dùng khi index và query. Trên corpus thực tế, 17 query in-domain có cosine cao nhất `0.410–0.673`, trong khi 5 query ngoài domain nằm ở `0.196–0.367`.
   **Trade-off:** Tái lập nhanh và không tốn phí nhưng hiểu paraphrase kém hơn embedding ngữ nghĩa; báo cáo đề xuất nâng cấp multilingual model.

2. **Chỉ fuse dense + BM25 một lần bằng RRF; dùng cosine gốc để kích hoạt fallback ở ngưỡng `0.39`.**
   **Lý do/evidence:** Đúng module contract và tránh so sánh RRF score với cosine. Query ngoài domain trả safe refusal; query “Mức phạt trả tài liệu thư viện trễ hạn…” trả đúng quy định 2.000 đồng/ngày kèm citation.
   **Trade-off:** Ngưỡng được hiệu chỉnh cho corpus hiện tại và phải chạy lại calibration khi thay dữ liệu/model.

## Kiểm thử và kết quả

- `python -m pytest tests/test_contracts.py -q`: **15 passed**.
- `python -m pytest tests/test_acceptance.py -q`: **5 passed**.
- `python -m pytest -q`: **20 passed**.
- `python -m src.task4_chunking_indexing`: **297 chunks** từ 10 tài liệu chuẩn hóa.
- `python -m src.evaluate`: **17 cases**, 4 metric. Dense-only `0.879`; hybrid + RRF `0.890` (`+0.012`). Context recall tăng từ `0.954` lên `0.971`; độ trễ trung bình tăng khoảng `47 ms/query` trên lần chạy ghi nhận.
- Lỗi đã xử lý: URL HCMUS trả 404 được thay bằng trang HUIT công khai hoạt động; query ngoài domain từng khớp n-gram chung đã được chặn bằng calibration và domain-aware vectorless fallback; provider lỗi không làm UI crash.

## Điều còn hạn chế

- Bốn metric hiện là proxy token-overlap offline, chưa phải đánh giá ngữ nghĩa bằng một evaluator LLM cố định; extractive generator đôi khi giữ lỗi xuống dòng từ PDF.
- Ưu tiên tiếp theo: dùng multilingual sentence-transformer, chunk theo heading/điều khoản, sau đó chạy RAGAS với model/prompt được pin và so sánh lại trên cùng 17 câu.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc trong commit `7f5e189` và có thể chạy lại theo `README.md` trong buổi demo.

- **Ngày:** 20/09/2026
- **Tên thành viên:** Hồ Đình Tuấn Kiệt
