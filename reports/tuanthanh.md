# Individual contribution report

- File: `reports/tuanthanh.md`

---

## Thông tin

- Họ và tên: Tuấn Thành
- Mã học viên: HV-TUANTHANH (vui lòng cập nhật mã chính xác nếu cần)
- Nhóm: K4 - L3A (Nhóm ABCD)
- Repository/branch: `tuanthanh_dev`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Data Ingestion & Standardization (Task 1, 2, 3) | Thu nạp 5 tài liệu pháp lý PDF và 7 tin tức JSON; viết pipeline chuyển đổi sang Markdown chuẩn hóa giữ nguyên metadata | `src/task3_convert_markdown.py`, `data/standardized/` | Done |
| Chunking & Vector Indexing (Task 4) | Phân đoạn 854 chunks bằng `RecursiveCharacterTextSplitter`, tạo vector embedding (1536 dims) và lập chỉ mục vào ChromaDB | `src/task4_chunking_indexing.py`, `.gitignore` | Done |
| Dense Semantic Search (Task 5) | Truy vấn ChromaDB, chuyển đổi khoảng cách Cosine sang Cosine Similarity `[0.0, 1.0]`, đảm bảo đúng contract `SearchResult` | `src/task5_semantic_search.py` | Done |
| Lexical Search BM25 (Task 6) | Xây dựng bộ tìm kiếm từ khóa với BM25; thiết kế class `RobustBM25` khắc phục lỗi Zero-IDF trên các corpus nhỏ | `src/task6_lexical_search.py` | Done |
| Reranking & Hybrid Fusion (Task 7) | Triển khai thuật toán Reciprocal Rank Fusion (RRF $k=60$) gộp kết quả Dense và BM25, khử trùng lặp theo ID | `src/task7_reranking.py` | Done |
| PageIndex Fallback & Retrieval (Task 8, 9) | Tích hợp cơ chế fallback không vector; phối hợp Dense, BM25, RRF và kích hoạt fallback khi cosine score < 0.30 | `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py` | Done |
| Generation có Citation (Task 10) | Triển khai Lost-in-the-middle Document Reordering, prompt grounding chống bịa đặt (safe refusal), hỗ trợ OpenAI / Gemini / Claude | `src/task10_generation.py` | Done |
| Chatbot Streamlit (UI) | Xây dựng giao diện chat trực quan, hiển thị câu trả lời kèm thẻ Accordion xem chi tiết nguồn trích dẫn, điểm số và phương thức tìm kiếm | `app.py` | Done |
| Golden Dataset & Evaluation | Xây dựng bộ 15 câu hỏi - câu trả lời - context chuẩn; đo lường 4 metrics và viết báo cáo phân tích A/B chi tiết | `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/RESULT.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Thay thế công thức tính IDF mặc định của `BM25Okapi` bằng công thức chuẩn Lucene: $\text{IDF} = \ln\left(1.0 + \frac{N - n + 0.5}{n + 0.5}\right)$.  
   **Lý do/evidence:** Thư viện `rank_bm25` nguyên bản tính $\text{IDF} = \ln\left(\frac{N - n + 0.5}{n + 0.5}\right)$. Khi chạy test fixture nhỏ ($N=2, n=1$), IDF bị triệt tiêu về $\ln(1.0) = 0.0$ khiến toàn bộ chunk bị điểm 0 và làm fail contract test. Áp dụng `RobustBM25` giúp hệ thống luôn gán trọng số dương cho từ khóa xuất hiện.  
   **Trade-off:** Cần viết một subclass nhỏ kế thừa từ `BM25Okapi` thay vì dùng trực tiếp class gốc, nhưng đảm bảo độ ổn định 100% trên mọi quy mô dữ liệu.

2. **Quyết định:** Áp dụng kỹ thuật *Document Reordering* (`reorder_for_llm`) đưa các chunk có điểm liên quan cao nhất về đầu và cuối ngữ cảnh trước khi đưa vào LLM.  
   **Lý do/evidence:** Hiện tượng "Lost-in-the-middle" của các LLM khiến thông tin quan trọng nằm ở giữa context dài dễ bị bỏ sót. Kết quả đo lường thực tế cho thấy điểm Faithfulness tăng từ 0.91 lên 0.965.  
   **Trade-off:** Tăng thêm một thao tác tráo đổi mảng $O(N)$ trong bộ nhớ trước khi format prompt (tốn thời gian xử lý < 0.1ms, hoàn toàn không đáng kể).

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** 
  - Toàn bộ 20 test cases trong `tests/test_contracts.py` và `tests/test_acceptance.py`.
  - Bộ 15 truy vấn kiểm thử thực tế trong `golden_dataset.json`.
- **Kết quả trước/sau:** 
  - Đạt **20/20 test cases PASSED (100%)**.
  - So sánh A/B cho thấy mô hình Hybrid + RRF vượt trội so với Dense-only: Top-1 Accuracy tăng từ 60.0% lên 73.3%, Context Precision tăng từ 0.733 lên 0.850, điểm trung bình 4 metrics tăng từ 0.865 lên 0.923.
- **Lỗi đã phát hiện và cách xử lý:**
  - ChromaDB tải ngầm mô hình ONNX khi khởi tạo: Khắc phục bằng cách truyền rõ ràng tham số `embedding_function=None` khi `get_or_create_collection()`.
  - Giữ bảo mật: Thêm `chroma_db/` vào `.gitignore` để tránh commit nhầm database vector nặng và file `.env`.

## Điều còn hạn chế

- **Hạn chế:** Chiến lược chunking hiện tại dựa trên kích thước cố định (`chunk_size=500, chunk_overlap=50`). Với các bảng biểu phức tạp (như bảng giờ phân bố tiết học), các dòng và tiêu đề cột có thể bị cắt đôi giữa các chunk.
- **Nếu có thêm thời gian:** Tôi sẽ triển khai kỹ thuật *Parent-Document Retriever* (lưu chunk nhỏ để search chính xác nhưng trả về cả section lớn cho LLM) hoặc Markdown Table-Aware Splitter để giữ trọn vẹn cấu trúc bảng biểu.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Tuấn Thành
