# Báo cáo đóng góp cá nhân

## Thông tin

- **Họ và tên:** Nguyễn Trần Kiên
- **Mã học viên:** 2A202602571
- **Nhóm:** K4-L3A — RAG Pipeline ABCD
- **Repository/branch:** `K4-L3A-RAG-Pipeline-ABCD` / `main`
- **Commit triển khai:** `feat(generation): implement document reordering and grounded citation generator`, `feat(eval): add offline evaluation framework and 15 golden Q&A dataset`
- **Commit báo cáo:** `docs(eval): add A/B benchmark evaluation report and individual report for Kien`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit | Trạng thái |
|---|---|---|---|
| Tiện ích tiếng Việt & Tokenization | Xây dựng bộ công cụ tiền xử lý ngôn ngữ tiếng Việt (`strip_accents`, `tokenize`, `token_set`) phục vụ so khớp ngữ nghĩa và tính metric | `src/text_utils.py` | Done |
| Generation có Grounded Citation (Task 10) | Triển khai kỹ thuật Document Reordering chống lost-in-the-middle, cơ chế Safe Refusal chống bịa đặt, trích dẫn chuẩn `[S1]`, `[S2]` và cơ chế fallback offline | `src/task10_generation.py` | Done |
| Golden Dataset (15 Grounded Cases) | Xây dựng bộ 15 cặp câu hỏi - câu trả lời - context trích dẫn thực tế từ 12 văn bản pháp lý và tin tức của hệ thống | `group_project/evaluation/golden_dataset.json` | Done |
| Offline Evaluation Framework | Xây dựng runner đánh giá tự động không phụ thuộc API tính 4 metric: Faithfulness, Answer Relevance, Context Recall, Context Precision; runner ghi đồng thời hai bản báo cáo | `src/evaluate.py` | Done |
| A/B Benchmark & Báo cáo kết quả | Thực nghiệm đối chuẩn A/B giữa Dense-only và Hybrid Search + RRF; phân tích trường hợp thất bại và đề xuất cải tiến | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Tích hợp bộ đánh giá Token-Overlap Proxy hoàn toàn offline (`src/evaluate.py`) thay vì phụ thuộc 100% vào LLM Judge từ xa.  
   **Lý do/evidence:** Việc đánh giá A/B với 15 - 30 query qua LLM API bên ngoài dễ gặp lỗi nghẽn mạng, tốn chi phí và thiếu tính tất định (non-deterministic). Evaluator offline cho phép nhóm chạy lại benchmark bất kỳ lúc nào với kết quả có thể kiểm chứng độc lập.  
   **Trade-off:** Đánh giá từ khóa bằng token-overlap có thể hơi khắt khe đối với các câu trả lời diễn đạt lại (paraphrase), nhưng bù lại đảm bảo tính trung thực (grounding) tuyệt đối bám sát văn bản gốc.

2. **Quyết định:** Áp dụng cơ chế kép: gọi LLM API đa nhà cung cấp kèm fallback trích xuất ngữ cảnh (`extractive_answer`).  
   **Lý do/evidence:** Đảm bảo hệ thống luôn trả về câu trả lời có trích dẫn `[S1]`, `[S2]` ngay cả khi người dùng không cấu hình API Key hoặc mất kết nối Internet.

3. **Quyết định:** `write_report()` ghi cùng một nội dung ra `group_project/evaluation/RESULT.md` và `reports/RESULT.md` trong một lần chạy.  
   **Lý do/evidence:** Hai đường dẫn đều nằm trong danh mục sản phẩm phải nộp; ghi tự động loại bỏ nguy cơ hai bản báo cáo lệch số liệu sau mỗi lần chạy lại benchmark.

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** 
  - Runner tự động `python -m src.evaluate` chạy trên toàn bộ 15 test cases của `golden_dataset.json`.
  - Kiểm thử chấp nhận `tests/test_acceptance.py::test_golden_dataset_has_15_grounded_cases`.
- **Kết quả thực nghiệm A/B:**
  - Cấu hình Dense-only đạt điểm trung bình: **0.948** (Faithfulness: 1.000, Answer relevance: 0.814, Recall: 0.989, Precision: 0.987).
  - Cấu hình Hybrid + RRF đạt điểm trung bình: **0.917** (Faithfulness: 1.000, Answer relevance: 0.731, Recall: 0.989, Precision: 0.947).
  - Kết luận: trên corpus hiện tại Dense-only nhỉnh hơn (`-0.031` khi thêm BM25 + RRF) và rẻ hơn về độ trễ (`1281.8` so với `1717.6` ms/query); chênh lệch đến từ Answer relevance và Context precision chứ không phải Recall.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 20/09/2026
- Tên thành viên: Nguyễn Trần Kiên
