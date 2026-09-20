# Individual contribution report

Họ và tên: Huỳnh Tấn Trung
Mã học viên: 2A202602742
Nhóm: ABCD
Repository/branch: `main`

```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---
| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Data ingestion | Thu thập tài liệu pháp lý/news và chuyển sang Markdown chuẩn hóa | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py` | Done |
| Retrieval pipeline | Chunking/indexing, dense search, BM25, reranking/RRF và PageIndex fallback | `src/task4_chunking_indexing.py`–`src/task9_retrieval_pipeline.py` | Done |
| Generation/UI | Prompt có citation, safe refusal, nhiều provider và giao diện Streamlit hiển thị nguồn | `src/task10_generation.py`, `app.py` | Done |
| Evaluation | Golden dataset, A/B evaluation và calibration threshold | `group_project/evaluation/`, `src/evaluate.py`, `src/calibrate_threshold.py` | Done |

## Thông tin

- Họ và tên:
- Mã học viên:
- Nhóm:
1. **Quyết định:** Dùng hybrid retrieval (dense + BM25 + RRF) làm cấu hình chính
    **Lý do/evidence:** Config B thắng Config A trên cả 4 metric và tốt hơn ở 15/21 câu; context precision tăng `+0.060`
    **Trade-off:** Tăng CPU và độ phức tạp xử lý, nhưng gần như không tăng chi phí API
- Repository/branch:

## Phần việc đã thực hiện

2. **Quyết định:** Bắt buộc citation theo `[Document N]` và safe refusal khi thiếu evidence
    **Lý do/evidence:** Generation giữ mapping citation với nguồn sau khi reorder context và không bịa khi provider/context lỗi
    **Trade-off:** Có thể từ chối câu hỏi ngoài phạm vi thay vì cố trả lời
| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| | | | Done / Partial / Blocked |
- Test hoặc query tôi đã dùng: `pytest -q`; chạy evaluation trên golden dataset 21 câu.
- Kết quả trước/sau nếu có: Hybrid + RRF thắng dense-only trên cả 4 metric; ngưỡng fallback được calibrate thành `0.41`.
- Lỗi đã phát hiện và cách xử lý: phát hiện nhầm nguồn giữa các trường; bổ sung institution trong context/prompt và citation theo document.
## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:
- Một hạn chế cụ thể của phần tôi làm: PageIndex chưa được kiểm thử thật vì thiếu `PAGEINDEX_API_KEY`; latency hiện chưa phải phép đo tuần tự công bằng.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: chạy lại benchmark tuần tự và bổ sung test cho các câu ngoài miền.
   **Lý do/evidence:**
   **Trade-off:**

2. **Quyết định:**
   **Lý do/evidence:**
- Ngày: 20/09/2026
- Tên thành viên: Huỳnh Tấn Trung
## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng:
- Kết quả trước/sau nếu có:
- Lỗi đã phát hiện và cách xử lý:

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm:
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày:
- Tên thành viên:
```
