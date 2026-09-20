# Day 8 — RAG Pipeline

## Mục tiêu

Mỗi nhóm xây dựng một chatbot RAG trả lời câu hỏi từ bộ tài liệu do nhóm thu thập. Sản phẩm phải có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Nhóm tự chọn bài toán và thu thập dữ liệu phù hợp; repo không cung cấp dữ liệu mẫu.

## Bản triển khai của nhóm

- **Chủ đề:** Dịch vụ đại học — học vụ, ký túc xá, thư viện và công tác sinh viên.
- **Corpus:** 5 PDF quy định/sổ tay và 5 trang công khai từ CTU, HUIT, HCMUS,
  USSH và CTUET; dữ liệu chuẩn hóa nằm trong `data/standardized/`.
- **Retrieval:** cosine dense + BM25, gộp đúng một lần bằng RRF; fallback
  page/section index không dùng vector.
- **Generation:** hỗ trợ OpenAI, Gemini, Anthropic; khi không có API key sẽ dùng
  extractive baseline có citation để demo offline và đánh giá tái lập được.
- **Evaluation:** 17 golden Q&A và A/B dense-only với hybrid + RRF trong
  `group_project/evaluation/RESULT.md`.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Trên Windows nên dùng Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Mặc định `EMBEDDING_PROVIDER=hashing` nên pipeline chạy hoàn toàn offline. Có thể
đổi sang `sentence_transformers` và khai báo model trong `.env` khi cần embedding
ngữ nghĩa chất lượng cao hơn.

Điền API key cần dùng trong `.env`; không commit file này.

```bash
# 1. Thu thập và chuẩn hoá
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# A/B evaluation và cập nhật RESULT.md
python -m src.evaluate

# 3. Chạy sản phẩm
streamlit run app.py
```

Demo nhanh một câu trong domain và một câu ngoài domain:

```powershell
python -c "from src.task10_generation import generate_with_citation as g; print(g('Mức phạt trả tài liệu thư viện trễ hạn là bao nhiêu?')); print(g('Dự báo thời tiết trên sao Hỏa hôm nay'))"
```

## Lộ trình 3 giờ

| Mốc                  | Thời gian | Kết quả cần có                           |
| -------------------- | --------: | ---------------------------------------- |
| 0. Setup             |   10 phút | Môi trường và `.env` sẵn sàng            |
| 1. Data              |   25 phút | ≥3 legal, ≥5 news, Markdown đã chuẩn hoá |
| 2. Index & search    |   30 phút | ChromaDB, dense search và BM25 chạy được |
| 3. Fusion & fallback |   25 phút | RRF và fallback tuân thủ contract        |
| 4. Generation & UI   |   30 phút | Chatbot trả lời có citation              |
| 5. Evaluation        |   30 phút | 15+ Q&A, 4 metric, A/B comparison        |
| 6. Demo & handoff    |   30 phút | Test, report, demo và push repository    |

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```
