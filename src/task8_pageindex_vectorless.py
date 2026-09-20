"""
Task 8 — PageIndex vectorless fallback.

PageIndex là dịch vụ ngoài: mọi lỗi (thiếu key, timeout, rate limit, API đổi
schema) đều phải được bọc lại để Task 9 rơi về hybrid thay vì làm crash UI.
Vì vậy hàm ở đây chỉ raise RuntimeError có thông điệp rõ ràng; phần try/except
nằm ở retrieve().

Chạy upload một lần:
    python -m src.task8_pageindex_vectorless
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

RETRIEVAL_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình trong .env")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict[str, str]:
    if DOC_ID_CACHE.exists():
        try:
            return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    DOC_ID_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upload_documents() -> dict[str, str]:
    """Upload PDF gốc và lưu mapping source -> document ID để không upload lại."""
    client = _client()
    cache = _load_cache()

    for path in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        if path.name in cache:
            print(f"Skip (đã upload): {path.name} -> {cache[path.name]}")
            continue
        try:
            response = client.submit_document(file_path=str(path))
            doc_id = response.get("doc_id") or response.get("id")
            if not doc_id:
                raise RuntimeError(f"response không có doc_id: {response}")
            cache[path.name] = doc_id
            print(f"Uploaded: {path.name} -> {doc_id}")
        except Exception as error:  # noqa: BLE001
            print(f"Failed: {path.name} — {error}")

    _save_cache(cache)
    return cache


def _extract_nodes(payload: dict) -> list[dict]:
    """Lấy danh sách node từ nhiều hình dạng response khác nhau của API."""
    for key in ("retrieved_nodes", "nodes", "results", "sources"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return value
    result = payload.get("result") or payload.get("retrieval") or {}
    if isinstance(result, dict):
        for key in ("retrieved_nodes", "nodes", "results", "sources"):
            value = result.get(key)
            if isinstance(value, list) and value:
                return value
    return []


def _node_text(node: dict) -> str:
    for key in ("text", "content", "node_text", "chunk", "summary"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult (score giảm dần theo rank nếu API không trả score)."""
    import time

    if top_k <= 0 or not query.strip():
        return []

    client = _client()
    cache = _load_cache()
    if not cache:
        raise RuntimeError("Chưa upload tài liệu lên PageIndex (chạy task8 một lần)")

    results: list[dict] = []
    for source_name, doc_id in cache.items():
        try:
            submitted = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            payload = submitted
            waited = 0
            while retrieval_id and not _extract_nodes(payload):
                if waited >= RETRIEVAL_TIMEOUT_SECONDS:
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
                waited += POLL_INTERVAL_SECONDS
                payload = client.get_retrieval(retrieval_id)
                if str(payload.get("status", "")).lower() in {"failed", "error"}:
                    break

            for node in _extract_nodes(payload):
                text = _node_text(node)
                if not text:
                    continue
                results.append(
                    {
                        "id": f"pageindex::{doc_id}::{node.get('node_id', len(results))}",
                        "content": text,
                        "_raw_score": float(node.get("relevance", 0.0) or 0.0),
                        "metadata": {
                            "source": source_name,
                            "title": node.get("title") or source_name,
                            "doc_type": "legal",
                            "url": None,
                            "chunk_index": int(node.get("page_index", 0) or 0),
                        },
                    }
                )
        except Exception as error:  # noqa: BLE001
            print(f"PageIndex lỗi với {source_name}: {error}")

    if not results:
        return []

    results.sort(key=lambda item: item["_raw_score"], reverse=True)
    output: list[dict] = []
    seen: set[str] = set()
    for rank, item in enumerate(results[:top_k], 1):
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        item.pop("_raw_score", None)
        item["score"] = 1.0 / rank  # score giảm dần theo rank khi API không trả score
        item["retrieval_method"] = "pageindex"
        output.append(item)
    return output


if __name__ == "__main__":
    upload_documents()
