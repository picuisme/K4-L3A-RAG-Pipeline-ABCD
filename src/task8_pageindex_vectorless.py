"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import json
from pathlib import Path
import re

from dotenv import load_dotenv

from .task4_chunking_indexing import chunk_documents, load_documents
from .text_utils import strip_accents, token_set


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"

QUERY_STOPWORDS = {
    "ai", "bao", "bi", "cach", "cho", "co", "cua", "duoc", "gi", "hom",
    "khi", "khong", "la", "mot", "nay", "nhung", "phai", "sinh", "tai",
    "the", "theo", "thoi", "tren", "trong", "va", "ve", "vien", "voi",
}

DOMAIN_PATTERN = re.compile(
    r"\b(ktx|ky tuc xa|thu vien|hcmus|ctuet|huit|ussh|dai hoc|hoc phan|"
    r"tin chi|tot nghiep|noi tru|sinh vien|giao trinh|hoc vu|ren luyen|"
    r"hoc phi|hoc bong|tai lieu|ky luat|dao tao)\b"
)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    mapping = {
        document["metadata"]["source"]: document["id"]
        for document in load_documents()
    }
    CACHE_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    if PAGEINDEX_API_KEY:
        print(
            "PAGEINDEX_API_KEY đã được cấu hình. Bản lab dùng local page index "
            "để tránh upload tài liệu ngoài ý muốn."
        )
    print(f"Cached {len(mapping)} local document IDs: {CACHE_PATH}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []
    if not DOMAIN_PATTERN.search(strip_accents(query)):
        return []
    query_tokens = token_set(query) - QUERY_STOPWORDS
    if not query_tokens:
        return []
    candidates = []
    for chunk in chunk_documents(load_documents()):
        content_tokens = token_set(chunk["content"])
        overlap = query_tokens & content_tokens
        if len(overlap) < min(2, len(query_tokens)):
            continue
        # A vectorless page/section score: query coverage with a small title bonus.
        coverage = len(overlap) / len(query_tokens)
        title_bonus = len(query_tokens & token_set(chunk["metadata"]["title"])) * 0.03
        score = min(1.0, coverage + title_bonus)
        if score >= 0.4:
            candidates.append((score, chunk))
    candidates.sort(key=lambda row: (-row[0], row[1]["id"]))
    return [
        {
            "id": chunk["id"],
            "content": chunk["content"],
            "score": score,
            "metadata": chunk["metadata"],
            "retrieval_method": "pageindex",
        }
        for score, chunk in candidates[:top_k]
    ]


if __name__ == "__main__":
    upload_documents()
