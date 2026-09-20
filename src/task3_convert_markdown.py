"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown hoặc PyMuPDF/pdfplumber để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.
"""

import json
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _extract_text_from_document(path: Path) -> str:
    """Trích xuất text/markdown từ tài liệu (PDF/DOCX) với fallback linh hoạt."""
    # Thử MarkItDown trước
    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
        result = converter.convert(str(path))
        if result and getattr(result, "text_content", None) and result.text_content.strip():
            return result.text_content.strip()
    except Exception:
        pass

    # Fallback sang PyMuPDF (fitz)
    try:
        import fitz

        doc = fitz.open(path)
        pages_text = []
        for page in doc:
            text = page.get_text()
            if text and text.strip():
                pages_text.append(text.strip())
        if pages_text:
            return "\n\n".join(pages_text)
    except Exception:
        pass

    # Fallback sang pdfplumber
    try:
        import pdfplumber

        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text.strip())
        if pages_text:
            return "\n\n".join(pages_text)
    except Exception:
        pass

    raise RuntimeError(f"Không thể trích xuất nội dung từ: {path}")


def convert_legal_docs() -> None:
    """Convert PDF/DOCX từ landing/legal vào standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".pdf", ".doc", ".docx"}
    files = sorted(
        path for path in legal_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in supported_exts
    )

    for path in files:
        text_content = _extract_text_from_document(path)
        # Bổ sung header tiêu đề nếu chưa có
        title = path.stem.replace("-", " ").replace("_", " ").title()
        if not text_content.startswith("#"):
            content = f"# {title}\n\n**Source:** {path.name}\n\n---\n\n{text_content}"
        else:
            content = text_content

        target_file = output_dir / f"{path.stem}.md"
        target_file.write_text(content, encoding="utf-8")
        print(f"Legal standardized: {target_file.name} ({len(content)} chars)")


def convert_news_articles() -> None:
    """Convert JSON từ landing/news vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in news_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        title = data.get("title", path.stem).strip()
        url = data.get("url", "").strip()
        date_crawled = data.get("date_crawled", "").strip()
        body = data.get("content_markdown", "").strip()

        header = (
            f"# {title}\n\n"
            f"**Source:** {url}\n\n"
            f"**Crawled:** {date_crawled}\n\n---\n\n"
        )
        content = header + body
        target_file = output_dir / f"{path.stem}.md"
        target_file.write_text(content, encoding="utf-8")
        print(f"News standardized: {target_file.name} ({len(content)} chars)")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing sang standardized markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
