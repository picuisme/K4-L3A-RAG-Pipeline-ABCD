"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

landing/ giữ file gốc để tra cứu khi convert sai; standardized/ là thứ duy nhất
Task 4 đọc. Mỗi file .md mang một front matter cố định để metadata nguồn
(title, source, doc_type, url) đi xuyên suốt pipeline:

    ---
    title: ...
    source: ...
    doc_type: legal | news
    institution: tên trường ban hành (rỗng với bài báo đa nguồn)
    url: ...
    ---

`institution` quan trọng vì corpus trải trên nhiều trường và các văn bản mâu
thuẫn nhau; thiếu nó thì câu trả lời "được mượn 6 cuốn" không biết của trường nào.

Chạy:
    python -m src.task3_convert_markdown
"""

from __future__ import annotations

import json
import re
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

MIN_CHARS = 200
LEGAL_SUFFIXES = {".pdf", ".doc", ".docx"}


def _escape(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().replace('"', "'")


def build_front_matter(
    title: str,
    source: str,
    doc_type: str,
    url: str | None,
    institution: str = "",
) -> str:
    return (
        "---\n"
        f'title: "{_escape(title)}"\n'
        f'source: "{_escape(source)}"\n'
        f'doc_type: "{doc_type}"\n'
        f'institution: "{_escape(institution)}"\n'
        f'url: "{_escape(url) if url else ""}"\n'
        "---\n\n"
    )


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Tách front matter và body. Dùng lại ở Task 4."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")


def strip_web_boilerplate(text: str) -> str:
    """Bỏ menu, danh sách link và dòng lặp trong Markdown crawl từ web.

    Crawl4AI trả về cả navigation, "tin liên quan" và footer. Những dòng đó
    chiếm phần lớn độ dài bài báo, sinh ra hàng trăm chunk nhiễu cạnh tranh
    trực tiếp với văn bản quy định ở cả dense lẫn BM25. Chỉ áp dụng cho news —
    file legal convert từ PDF không có link.
    """
    cleaned_lines: list[str] = []
    seen: set[str] = set()

    for raw_line in text.split("\n"):
        line = _IMAGE_RE.sub("", raw_line)

        links = _LINK_RE.findall(line)
        if links:
            without_links = _LINK_RE.sub("", line).strip()
            link_text = "".join(label for label, _ in links)
            # Dòng gồm từ 2 link trở lên mà phần chữ ngoài link quá ngắn thì
            # gần như chắc chắn là menu hoặc danh sách "bài liên quan".
            if len(links) >= 2 and len(without_links) < max(
                30, len(link_text) // 3
            ):
                continue
            line = _LINK_RE.sub(r"\1", line)

        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if len(stripped.strip("*-_|# ")) < 3:
            continue

        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines.append(line.rstrip())

    return _clean("\n".join(cleaned_lines))


def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong landing/legal sang standardized/legal."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    converted = 0
    for path in sorted(legal_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LEGAL_SUFFIXES:
            continue
        target = output_dir / f"{path.stem}.md"
        try:
            result = converter.convert(str(path))
            body = _clean(result.text_content)
            if len(body) < MIN_CHARS:
                print(
                    f"Bỏ qua {path.name}: chỉ trích được {len(body)} ký tự "
                    "(có thể là PDF scan, cần đổi nguồn khác)"
                )
                continue
            from .task1_collect_legal_docs import INSTITUTION_BY_FILE, SOURCES

            institution = INSTITUTION_BY_FILE.get(path.name, "")
            entry = SOURCES.get(path.name, {})
            title = entry.get("description") or (
                getattr(result, "title", None)
                or path.stem.replace("-", " ").replace("_", " ").strip()
            )
            if institution:
                title = f"{title} — {institution}"
            target.write_text(
                build_front_matter(
                    title, path.name, "legal", entry.get("url"), institution
                )
                + body,
                encoding="utf-8",
            )
            print(f"Saved: legal/{target.name} ({len(body)} ký tự) — {institution or 'n/a'}")
            converted += 1
        except Exception as error:  # noqa: BLE001
            print(f"Failed: {path.name} — {error}")
    return converted


def convert_news_articles() -> int:
    """Convert JSON trong landing/news sang standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for path in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = _clean(data.get("content_markdown", ""))
            body = strip_web_boilerplate(raw)
            if raw and len(body) < len(raw):
                print(
                    f"  {path.name}: bỏ boilerplate {len(raw)} → {len(body)} ký tự"
                )
            if len(body) < MIN_CHARS:
                print(f"Bỏ qua {path.name}: nội dung {len(body)} ký tự")
                continue
            header = build_front_matter(
                data.get("title", path.stem),
                path.name,
                "news",
                data.get("url"),
                institution="",  # bài báo thường nói về nhiều trường
            )
            crawled = data.get("date_crawled", "")
            header += f"# {_escape(data.get('title', path.stem))}\n\n"
            header += f"**Source:** {data.get('url', '')}\n\n"
            header += f"**Crawled:** {crawled}\n\n---\n\n"
            (output_dir / f"{path.stem}.md").write_text(
                header + body, encoding="utf-8"
            )
            print(f"Saved: news/{path.stem}.md ({len(body)} ký tự)")
            converted += 1
        except Exception as error:  # noqa: BLE001
            print(f"Failed: {path.name} — {error}")
    return converted


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"\nSaved Markdown to: {OUTPUT_DIR} (legal={legal}, news={news})")
    if legal < 3 or news < 5:
        raise SystemExit(
            "Chưa đạt tối thiểu 3 legal + 5 news. Kiểm tra lại Task 1/Task 2."
        )


if __name__ == "__main__":
    convert_all()
