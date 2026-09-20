"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tansinhvien.ctu.edu.vn/sinh-hoat/huong-dan-tan-sinh-vien-dang-ky-o-ky-tuc-xa",
    "https://ssc.ctu.edu.vn/thong-bao/225-tbhk32526.html",
    "https://ttktx.huit.edu.vn/tin-tuc-hoat-dong/qui-che-luu-tru-ktx",
    "https://ttktx.huit.edu.vn/gioi-thieu/gioi-thieu-trung-tam-ky-tuc-xa-sinh-vien",
    "https://dsa.ctu.edu.vn/thong-bao/nhung-cau-hoi-thuong-gap.html",
]


class _ArticleParser(HTMLParser):
    """Small dependency-free HTML-to-text parser for public university pages."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip() or "Không có tiêu đề"

    @property
    def markdown(self) -> str:
        text = " ".join(self.text_parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n\n", text)
        return text.strip()


async def crawl_article(url: str) -> dict:
    """Fetch one public page and return the required serializable schema."""

    def _fetch() -> dict:
        response = requests.get(
            url,
            timeout=45,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; UniversityServicesRAG/1.0; "
                    "+https://github.com/)"
                )
            },
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        parser = _ArticleParser()
        parser.feed(response.text)
        content = parser.markdown
        if len(content) < 200:
            raise ValueError("Trang không có đủ nội dung văn bản để lập chỉ mục")
        return {
            "url": url,
            "title": parser.title,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }

    return await asyncio.to_thread(_fetch)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = await asyncio.gather(
        *(crawl_article(url) for url in ARTICLE_URLS),
        return_exceptions=True,
    )
    saved = 0
    for index, (url, outcome) in enumerate(zip(ARTICLE_URLS, outcomes), 1):
        try:
            if isinstance(outcome, Exception):
                raise outcome
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(outcome, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved += 1
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")
    if saved < 5:
        raise RuntimeError(f"Chỉ crawl thành công {saved}/5 bài; kiểm tra URL hoặc mạng")


if __name__ == "__main__":
    asyncio.run(crawl_all())
