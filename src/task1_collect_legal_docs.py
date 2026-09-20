"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path
from urllib.parse import urlparse

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Add a public URL here only when the corresponding document is not already
# supplied manually.  Keeping this mapping explicit makes provenance auditable.
LEGAL_DOCUMENT_URLS: dict[str, str] = {}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Download configured public documents without overwriting local files."""
    setup_directory()
    for filename, url in LEGAL_DOCUMENT_URLS.items():
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".doc", ".docx"}:
            raise ValueError(f"Unsupported legal document type: {filename}")
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError(f"Invalid public URL for {filename}")

        destination = DATA_DIR / filename
        if destination.exists() and destination.stat().st_size > 1024:
            print(f"Exists: {destination}")
            continue

        response = requests.get(
            url,
            timeout=45,
            headers={"User-Agent": "UniversityServicesRAG/1.0"},
        )
        response.raise_for_status()
        destination.write_bytes(response.content)
        print(f"Downloaded: {destination}")

    documents = [
        path
        for path in DATA_DIR.iterdir()
        if path.suffix.lower() in {".pdf", ".doc", ".docx"}
        and path.stat().st_size > 1024
    ]
    if len(documents) < 3:
        raise RuntimeError(
            "Cần ít nhất 3 PDF/DOCX trong data/landing/legal; "
            "hãy thêm file thủ công hoặc khai báo LEGAL_DOCUMENT_URLS."
        )
    print(f"Validated {len(documents)} legal documents")


if __name__ == "__main__":
    setup_directory()
    download_documents()
