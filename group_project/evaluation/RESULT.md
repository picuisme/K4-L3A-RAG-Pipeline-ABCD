# RAG evaluation results

Đề tài: **Dịch vụ đại học** — quy định ký túc xá, thư viện và đăng ký học phần,
tra cứu và đối chiếu giữa nhiều trường.

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-20 |
| Framework and version | RAGAS 0.4.3 (`ragas.metrics.collections`) |
| Evaluator model | `gpt-4o-mini` (OpenAI) |
| Generator model | `gpt-4o-mini` (`openai`), `temperature=0.3`, `top_p=0.9` |
| Embedding model | `text-embedding-3-small` (`openai`, 1536 chiều) |
| Corpus version/commit | 5 văn bản quy định (5 trường) + 7 bài viết → 12 document → **674 chunks** (`chunk_size=500`, `overlap=50`, recursive) |
| Golden dataset size | 21 câu |
| `top_k` | 5 |
| Fallback threshold and calibration | Xem mục Fallback calibration bên dưới |

## Configurations

- **Config A — dense-only:** `retrieve(..., use_reranking=False)`. Chỉ dense
  search trên ChromaDB (cosine), lấy thẳng `dense[:top_k]`.
- **Config B — hybrid + RRF:** `retrieve(..., use_reranking=True)`. Dense và
  BM25 chạy trên **cùng corpus chunks**, mỗi nhánh lấy `top_k × 5 = 25` ứng
  viên, fuse đúng một lần bằng `rerank_rrf` (`Σ 1/(60+rank)`).

Hai config dùng chung golden dataset, generator, evaluator, prompt, `top_k` và
cùng một index ChromaDB. Biến duy nhất thay đổi là retrieval strategy.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.890 | 0.923 | **+0.033** |
| Answer relevance | 0.466 | 0.485 | **+0.019** |
| Context recall | 0.819 | 0.876 | **+0.057** |
| Context precision | 0.707 | 0.767 | **+0.060** |
| **Average** | **0.721** | **0.763** | **+0.042** |

Không có metric nào lỗi hoặc NA: 21/21 câu × 4 metric × 2 config đều có điểm.

## A/B comparison

**Cấu hình tốt hơn: Config B — hybrid + RRF**, thắng trên **cả 4 metric**.

Ba bằng chứng độc lập với nhau:

1. **Thắng ở cấp câu hỏi, không chỉ ở trung bình.** Tính điểm trung bình 4
   metric cho từng câu: B cao hơn A ở **15/21 câu**, thấp hơn ở 6 câu. Không
   phải một vài câu bứt phá kéo trung bình lên.
2. **Giảm số lần từ chối sai.** Đếm câu trả lời mang tính từ chối: A có
   **4/21**, B còn **1/21**. Ba câu mà A không tìm được bằng chứng thì B tìm
   được — đúng vai trò BM25: bắt từ khoá chính xác (“điện nước”, “tiếp khách”,
   “cảnh báo học tập”) mà dense bỏ sót vì ngữ nghĩa quá gần nhau giữa các đoạn.
3. **Context precision tăng nhiều nhất (+0.060).** BM25 đẩy chunk chứa đúng từ
   khoá lên hạng cao, RRF cộng dồn hai nguồn tín hiệu nên chunk vừa gần nghĩa
   vừa khớp từ khoá thắng chunk chỉ gần nghĩa.

**Trade-off về latency/cost.** Phải nói rõ giới hạn của phép đo: cả hai config
chạy với `EVAL_GEN_CONCURRENCY=4`, nên `latency_s` của mỗi câu bao gồm cả thời
gian chờ do 4 luồng cùng gọi API. Con số thu được — A trung bình 15.79s/câu,
B 9.52s/câu — **không** chứng minh B nhanh hơn A; A chạy trước nên gánh phần
khởi động và các lần retry do rate limit. Về mặt chi phí thật:

- B tốn thêm **0 lời gọi API** so với A. BM25 chạy in-memory trên corpus đã nạp
  sẵn, RRF là một vòng lặp cộng số.
- B tốn thêm CPU cho BM25 (674 chunk) và bước fuse, cộng chi phí nạp corpus lần
  đầu từ ChromaDB.
- Token gửi lên LLM giữa hai config là như nhau vì `top_k` không đổi.

Kết luận: B tốt hơn A trên mọi metric mà gần như không tốn thêm chi phí. Để đo
latency cho tử tế thì phải chạy lại tuần tự (`EVAL_GEN_CONCURRENCY=1`).

## Fallback calibration

`src/calibrate_threshold.py` chạy 5 câu in-domain và 5 câu ngoài domain, đo
**cosine gốc** của dense (không phải điểm RRF):

| Nhóm | Thấp nhất | Cao nhất |
| --- | ---: | ---: |
| In-domain (5 câu) | **0.4426** | 0.6113 |
| Out-of-domain (5 câu) | 0.2905 | **0.3822** |

Hai phân bố **tách rời**, khoảng trống 0.4426 − 0.3822 = 0.0604. Ngưỡng đề xuất
là điểm giữa: **`SCORE_THRESHOLD = 0.41`**.

Lần chạy evaluate này thực hiện với `SCORE_THRESHOLD = 0.35` (giá trị mặc định
trước khi calibrate). Vì 0.35 thấp hơn mọi câu in-domain, không câu nào trong
golden set kích hoạt fallback, nên kết quả A/B ở trên không bị ảnh hưởng. Ngưỡng
0.41 chặt hơn, chỉ ảnh hưởng tới câu ngoài domain — đúng mục đích của fallback.

PageIndex không được cấu hình (`PAGEINDEX_API_KEY` rỗng), nên nhánh fallback
luôn ném lỗi và `retrieve()` trả hybrid. Đây là đường đi đã được test
`test_retrieve_survives_fallback_provider_error` bao phủ.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Phí ký túc xá của Trường ĐH Cần Thơ nộp theo kỳ hay theo tháng? | B | 0.50 | 0.00 | 0.00 | 0.00 | retrieval | Cả 5 nguồn trả về đều là bài báo "Mức phí ký túc xá các trường đại học 2025". Bài báo dày đặc từ "phí ký túc xá" kèm nhiều con số nên thắng cả dense lẫn BM25, trong khi quy định gốc của ĐHCT chỉ có một dòng ngắn "Phí KTX nộp theo học kỳ". Nguồn tin tức cạnh tranh và đè bẹp nguồn quy định trên cùng một chủ đề |
| 2 | Quy định giờ giấc ra vào KTX của ĐHCT và HUIT khác nhau thế nào? | B | 0.50 | 0.62 | 0.67 | 0.00 | retrieval | Câu đối chiếu hai trường nhưng `top_k=5` trả về 4 chunk của HUIT và chỉ 1 của ĐHCT. Câu trả lời vì vậy đầy đủ phía HUIT, sơ sài phía ĐHCT. `top_k` cố định không đủ cho câu hỏi cần bằng chứng từ nhiều nguồn |
| 3 | Khi nào sinh viên bị cảnh báo học tập cuối học kỳ chính? (ĐH KHTN) | B | 1.00 | 0.00 | 1.00 | 0.00 | generation | Cả 5 nguồn đều đúng văn bản (Quy chế QĐ 1175/QĐ-KHTN), `context_recall = 1.00`, nhưng model vẫn trả lời "trong tài liệu không có thông tin". Retrieval lấy đúng **văn bản** nhưng không lấy đúng **điều khoản**; bảng điều kiện cảnh báo bị chunk 500 ký tự cắt rời khỏi tiêu đề mục nên model không nhận ra |

### Hai phát hiện xuyên suốt, quan trọng hơn ba câu trên

**Answer relevance 0.47–0.49 không phản ánh chất lượng câu trả lời.** RAGAS
`AnswerRelevancy` gán **0 tuyệt đối** cho mọi câu trả lời bị đánh dấu
*noncommittal* — kể cả khi câu trả lời đúng nhưng có kèm một mệnh đề dè dặt.
Loại 3 câu bị gán 0 ra khỏi Config B thì trung bình các câu còn lại là **0.566**,
và câu trả lời đúng trọn vẹn đạt 0.94. Ví dụ rõ nhất là câu "chậm nộp phí điện
nước": model trả lời đúng mức xử lý là cảnh cáo, `faithfulness = 1.00`,
`context_recall = 1.00`, nhưng vì viết thêm "tài liệu không nêu rõ số ngày cụ
thể" nên bị chấm relevance = 0.00. Đây là đặc tính của metric, không phải lỗi
pipeline — nhưng nó kéo trung bình tổng xuống đáng kể và phải được nói rõ khi
so sánh với nhóm khác.

**Prefix tên trường hoạt động đúng thiết kế.** Đếm nguồn trả về: **12/21 câu có
cả 5 nguồn đến từ đúng một văn bản** của đúng trường được hỏi, và **0/42 kết quả
chứa chunk trùng lặp**. Trước khi thêm prefix, câu hỏi về KTX Cần Thơ còn trả về
bài viết về thư viện Đại học Nam California. Vấn đề còn lại không phải "sai
trường" mà là "đúng văn bản, sai điều khoản" — một bài toán khác và khó hơn.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| --: | --- | --- | --- | --- |
| 1 | Chunk văn bản quy định theo ranh giới `Điều \d+` / `Khoản` thay vì cắt cứng 500 ký tự | Worst performer #3: recall = 1.00 nhưng model vẫn từ chối, vì điều khoản bị cắt rời khỏi tiêu đề mục. Cùng dạng lỗi với câu "chậm nộp phí điện nước" mất mốc "7 ngày" | Tăng context precision và giảm số câu từ chối sai trên nhóm câu hỏi định lượng | Chạy lại `src/evaluate.py` với cùng golden set và cùng `top_k`, so `context_precision` và số câu relevance = 0 |
| 2 | Ưu tiên `doc_type=legal` khi câu hỏi nhắm vào một trường cụ thể (boost điểm hoặc lọc metadata) | Worst performer #1: 5/5 nguồn là bài báo, quy định gốc không lọt top-5. Thống kê nguồn cho thấy riêng bài "Mức phí KTX các trường 2025" chiếm 14/105 slot | Đưa văn bản quy định lên trước tin tức khi hai bên cùng chủ đề | Chạy lại, kiểm riêng nhóm câu hỏi có tên trường: tỉ lệ nguồn `doc_type=legal` trong top-5 |
| 3 | Cho `top_k` co giãn theo số trường được nhắc trong câu hỏi (2 trường → `top_k=8`) | Worst performer #2: câu đối chiếu 2 trường nhận 4 chunk HUIT + 1 chunk ĐHCT, precision = 0.00 | Câu đối chiếu có đủ bằng chứng cả hai phía | Thêm 2–3 câu đối chiếu vào golden set, so `context_recall` trước/sau |
| 4 | Báo cáo answer relevance kèm số liệu đã loại câu noncommittal | 3/21 câu bị gán 0 tuyệt đối kéo trung bình từ 0.566 xuống 0.485 | Số liệu phản ánh đúng chất lượng, tránh so lệch với nhóm khác | Tính cả hai con số trong `src/evaluate.py` và in song song |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Prefix tên trường vào nội dung chunk + `institution` trong metadata | Chunk không có tên trường | Không đo bằng A/B (đổi index nên không so trực tiếp được); đo gián tiếp: 12/21 câu có 5/5 nguồn đúng văn bản của đúng trường, 0 chunk trùng | 0 lời gọi API; ngân sách ký tự chunk giảm đúng độ dài prefix | Giải được lỗi trả về sai trường trên corpus đa nguồn |
| Lọc boilerplate cho bài crawl (`strip_web_boilerplate`) | Markdown thô từ Crawl4AI | Không đo bằng metric | Corpus giảm **885 → 674 chunk** (−24%); bài Tuổi Trẻ 33.411 → 11.305 ký tự | Bớt 211 chunk menu/tin liên quan cạnh tranh với văn bản quy định |
| UI citation highlighting (`app.py` hiện institution/title/source/method/score cho từng `[Document N]`) | UI chỉ hiện answer | Không đo bằng metric | +0.00s | Citation đối chiếu trực tiếp trên màn hình khi demo |

---

_Số liệu tổng hợp sinh bởi `python -m src.evaluate`; phân tích worst performer và
hai phát hiện xuyên suốt đọc trực tiếp từ `raw_results.json` (42 record × 4 metric)._
