# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Pytest 9.1.1, LangChain Text Splitters, ChromaDB 0.6.3, rank-bm25 0.2.2 |
| Evaluator model                    | OpenAI gpt-4o-mini |
| Generator model                    | OpenAI gpt-4o-mini (temperature=0.3, top_p=0.9) |
| Embedding model                    | OpenAI text-embedding-3-small (1536 dims, cosine space) |
| Corpus version/commit              | tuanthanh_dev (12 tài liệu quy chế, nội quy KTX và tin tức đại học) |
| Golden dataset size                | 15 Q&A pairs có ground-truth citation |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (Cosine similarity score trên dense retrieval) |

## Configurations

- **Config A — dense-only:** Sử dụng ChromaDB Semantic Search đơn lẻ dựa trên embedding vector của câu truy vấn (`text-embedding-3-small`), lấy top-5 chunks theo cosine similarity.
- **Config B — hybrid + RRF:** Kết hợp song song Dense Semantic Search (ChromaDB) và Sparse Lexical Search (RobustBM25 Okapi) trên cùng 854 chunks, sau đó hợp nhất danh sách xếp hạng bằng thuật toán Reciprocal Rank Fusion (RRF với hằng số làm mượt $k=60$).

Hai config dùng chung 100% tập dữ liệu `golden_dataset.json`, generator model (`gpt-4o-mini`), prompt template có citation, tham số `top_k=5` và cơ chế document reordering chống hiện tượng lost-in-the-middle. Điểm khác biệt duy nhất là phương thức trích xuất (retrieval strategy).

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                 0.910 |                   0.965 |    +0.055 |
| Answer relevance  |                 0.885 |                   0.942 |    +0.057 |
| Context recall    |                 0.933 |                   0.933 |    +0.000 |
| Context precision |                 0.733 |                   0.850 |    +0.117 |
| **Average**       |             **0.865** |               **0.923** | **+0.058** |

## A/B comparison

- **Cấu hình tốt hơn:** **Config B (Hybrid + RRF)** vượt trội hoàn toàn so với Config A trên cả độ chuẩn xác tổng thể (Average: 0.923 vs 0.865) lẫn Context Precision (+11.7%) và Faithfulness (+5.5%).
- **Evidence:**
  - Trong các câu hỏi chứa định danh thực thể, tên riêng, số hiệu quyết định hoặc con số tuyệt đối (ví dụ: *Quyết định 1944/QĐ-ĐHCT*, *phòng E3, E5*, *cơ sở 227 Nguyễn Văn Cừ*, *phí phạt 2.000 đ/ngày*), BM25 tìm chính xác từ khóa và kéo chunk chứa thông tin cốt lõi lên vị trí số 1 trong danh sách RRF, giúp Top-1 Retrieval Accuracy tăng từ 60.0% lên 73.3%.
  - Ngược lại, Config A (Dense-only) đôi khi gán điểm cosine cao cho các đoạn văn có nội dung tổng quan hoặc ngữ cảnh rộng nhưng thiếu số liệu cụ thể của điều khoản. Khi đó, LLM phải đọc nhiều chunk phụ trước khi tới chunk chính xác, làm giảm nhẹ điểm Faithfulness do bị phân tán sự chú ý.
- **Trade-off về latency/cost:**
  - **Latency:** Config A chỉ tốn 1 lượt embedding câu hỏi (~180ms) và 1 query ChromaDB (~15ms), tổng thời gian retrieval khoảng 200ms. Config B bổ sung thêm bước tokenized split và BM25 scoring (~12ms) cùng RRF dict aggregation (<1ms). Tổng thời gian truy xuất của Config B là ~215ms (chỉ tăng ~7.5% latency, hoàn toàn không đáng kể so với thời gian gọi LLM ~800–1200ms).
  - **Cost:** Hoàn toàn tương đồng vì cả hai cấu hình chỉ tốn 1 lượt embedding API call và 1 lượt LLM generation API call trên cùng số lượng `top_k=5` chunks đầu vào.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Quy định về việc nấu ăn trong phòng ở Ký túc xá Trường Đại học Cần Thơ như thế nào? | Config A | 0.80 | 0.85 | 0.80 | 0.50 | retrieval | Dense search xếp đoạn quy định sinh hoạt chung lên trước đoạn cấm nấu ăn cụ thể ở KTX-A do ngữ nghĩa gần nhau. |
|   2 | Tiết 1 buổi sáng tại cơ sở 227 Nguyễn Văn Cừ của Trường Đại học Khoa học Tự nhiên bắt đầu và kết thúc vào mấy giờ? | Config A | 0.85 | 0.88 | 1.00 | 0.60 | retrieval | Bảng thời khóa biểu phân bố tiết học có nhiều số và giờ (6g40-7g30) khiến vector search nhầm với thời gian học kỳ hè (5 tuần). BM25 ở Config B đã giải quyết triệt để vấn đề này. |
|   3 | Khi làm mất hoặc làm hư hỏng tài liệu mượn tại Thư viện Trường ĐH KHXH&NV - ĐHQG-HCM, bạn đọc bị xử lý ra sao? | Config B | 0.90 | 0.90 | 1.00 | 0.70 | generation | Văn bản gốc bị ngắt đoạn giữa trang 1 và trang 2 khi scan PDF, khiến câu văn về bồi thường thiếu một phần chi tiết phụ về trường hợp không tìm thấy sách trên thị trường. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung Parent-Document Retriever hoặc Small-to-Big Chunking | Các bảng quy định thời khóa biểu và khung giờ học tập thường bị chia cắt khi chunk cố định 500 ký tự, làm mất ngữ cảnh tiêu đề cột/hàng. | Tăng Context Precision lên > 0.90 và cải thiện khả năng trả lời chính xác các bảng biểu chi tiết. | Chạy lại evaluation trên các câu hỏi liên quan đến thời khóa biểu và đo lường tỷ lệ chunk chứa đầy đủ header bảng. |
|        2 | Nâng cấp Text Extraction cho các tài liệu PDF scan/dạng cột | Lỗi ở ca worst performer #3 cho thấy tài liệu PDF nội quy thư viện có chỗ bị ngắt trang làm rách đoạn văn giữa chừng. | Loại bỏ các câu bị đứt đoạn, giúp Faithfulness đạt > 0.98. | Kiểm tra các file `.md` sau chuyển đổi không có câu kết thúc dang dở ở ranh giới trang. |
|        3 | Thử nghiệm Query Rewriting hoặc HyDE (Hypothetical Document Embeddings) | Các câu hỏi quá ngắn hoặc dùng từ địa phương/từ lóng (như "giờ giới nghiêm", "tiền phạt trễ sách") đôi khi khó khớp với từ ngữ pháp lý chính thức. | Tăng Recall cho các câu hỏi mở từ sinh viên từ 0.93 lên 0.98. | So sánh A/B giữa pipeline có HyDE và baseline trên tập 30 câu hỏi mở rộng. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Advanced Reranker (Cross-Encoder / Rerank Model so với RRF) | Baseline: Reciprocal Rank Fusion ($k=60$) | MRR tăng +0.03, Context Precision đạt 0.88 | Latency tăng thêm ~85ms do chạy cross-encoder inference | Cross-encoder cho kết quả tinh chỉnh cao hơn ở các câu hỏi phức tạp, tuy nhiên RRF vượt trội về tốc độ (chỉ <1ms) và không tốn chi phí GPU hay API phụ. |
| Lost-in-the-middle Document Reordering (`reorder_for_llm`) | Baseline: Sắp xếp tuyến tính theo score giảm dần thông thường | Faithfulness tăng +0.04, giảm thiểu trích dẫn sai nguồn | Latency delta = 0ms, Cost delta = 0 | Việc đảo các chunk điểm cao nhất về hai đầu (đầu prompt và cuối context) giúp LLM tiếp nhận thông tin quan trọng nhất mà không bị lãng quên ở giữa context dài. |
