"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from .contracts import validate_search_results

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> None:
    """Upload tài liệu PDF và lưu document IDs để tái sử dụng."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        print("[PageIndex] Bỏ qua upload: PAGEINDEX_API_KEY chưa được cấu hình.")
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    cache = _load_cache()

    if not LANDING_LEGAL_DIR.exists():
        return

    for pdf_path in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        filename = pdf_path.name
        if filename in cache:
            continue

        try:
            print(f"[PageIndex] Uploading {filename}...")
            res = client.submit_document(file_path=str(pdf_path))
            doc_id = res.get("doc_id") or res.get("id") or res.get("document_id")
            if doc_id:
                cache[filename] = doc_id
                print(f"[PageIndex] Uploaded {filename} -> doc_id: {doc_id}")
        except Exception as e:
            print(f"[PageIndex] Lỗi khi upload {filename}: {e}")

    _save_cache(cache)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult từ PageIndex vectorless search."""
    if top_k <= 0 or not query.strip():
        return []

    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        return []

    cache = _load_cache()
    if not cache:
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=api_key)
    results: list[dict] = []
    seen_ids = set()

    for filename, doc_id in cache.items():
        try:
            query_res = client.submit_query(doc_id=doc_id, query=query)
            # Chờ hoặc đọc kết quả truy vấn
            nodes = query_res.get("nodes", []) or query_res.get("results", []) or []
            if isinstance(query_res.get("answer"), str) and not nodes:
                nodes = [{"text": query_res["answer"], "score": 0.9}]

            for rank, node in enumerate(nodes):
                content = node.get("text") or node.get("content") or ""
                if not content.strip():
                    continue

                chunk_id = f"pageindex::{doc_id}::{rank}"
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                score = float(node.get("score", max(0.01, 1.0 - (rank / (top_k + 1)))))
                results.append({
                    "id": chunk_id,
                    "content": content,
                    "score": score,
                    "metadata": {
                        "source": filename,
                        "title": filename.replace(".pdf", "").replace("-", " ").title(),
                        "doc_type": "legal",
                        "url": None,
                        "chunk_index": rank,
                    },
                    "retrieval_method": "pageindex",
                })
        except Exception as e:
            print(f"[PageIndex] Query thất bại trên doc {doc_id}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    final_results = results[:top_k]
    if final_results:
        validate_search_results(final_results, top_k=top_k, expected_method="pageindex")
    return final_results


if __name__ == "__main__":
    upload_documents()
