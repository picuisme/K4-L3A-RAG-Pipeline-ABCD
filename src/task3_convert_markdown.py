"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path
import json
import re


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert legal PDF/DOC/DOCX files and add auditable metadata headers."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        text = _convert_document(path)
        if len(text.strip()) < 200:
            raise ValueError(f"Không trích xuất đủ nội dung từ {path.name}")
        title = _title_from_text(text, path.stem.replace("-", " ").title())
        header = (
            f"# {title}\n\n"
            f"**Source:** {path.name}\n\n"
            "**Type:** legal\n\n---\n\n"
        )
        destination = output_dir / f"{path.stem}.md"
        destination.write_text(header + text.strip() + "\n", encoding="utf-8")
        print(f"Converted: {destination}")


def convert_news_articles() -> None:
    """Normalize crawled JSON while retaining provenance in the Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"url", "title", "date_crawled", "content_markdown"}
        if not required <= data.keys():
            raise ValueError(f"{path.name} thiếu metadata bắt buộc")
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**URL:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n"
            "**Type:** news\n\n---\n\n"
        )
        destination = output_dir / f"{path.stem}.md"
        destination.write_text(
            header + data["content_markdown"].strip() + "\n", encoding="utf-8"
        )
        print(f"Converted: {destination}")


def _convert_document(path: Path) -> str:
    try:
        from markitdown import MarkItDown

        return MarkItDown().convert(str(path)).text_content
    except (ImportError, ModuleNotFoundError):
        if path.suffix.lower() != ".pdf":
            raise RuntimeError("Cần cài markitdown để chuyển DOC/DOCX")
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)


def _title_from_text(text: str, fallback: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    meaningful = [line for line in lines if 8 <= len(line) <= 180]
    return meaningful[0] if meaningful else fallback


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
