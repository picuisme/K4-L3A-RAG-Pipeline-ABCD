"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Mỗi bài lưu thành một JSON trong data/landing/news/ với đủ 4 field bắt buộc:
url, title, date_crawled, content_markdown.

Chiến lược crawl: Crawl4AI (headless chromium) là đường chính. Nếu Crawl4AI
không cài được browser hoặc trang trả về rỗng, fallback sang requests +
MarkItDown để pipeline không đứng vì một dependency.

Cài browser trước khi chạy:
    python -m playwright install chromium

Chạy:
    python -m src.task2_crawl_news
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

MIN_CONTENT_CHARS = 400

ARTICLE_URLS = [
    # Ký túc xá
    "https://tuoitre.vn/10-500-cho-o-cho-tan-sinh-vien-tai-ky-tuc-xa-dai-hoc-quoc-gia-tp-hcm-phi-ra-sao-20240807091824962.htm",
    "https://cafef.vn/muc-phi-ky-tuc-xa-cac-truong-dai-hoc-2025-cao-nhat-hon-3-trieu-dong-thang-188250806170622821.chn",
    # Đăng ký học phần
    "https://xaydungchinhsach.chinhphu.vn/cach-dang-ki-tin-chi-hoc-phan-hieu-qua-cho-tan-sinh-vien-119240809084631415.htm",
    "https://huflit.edu.vn/vi/tin-tuc/cach-dang-ky-hoc-phan/",
    # Thư viện
    "https://thanhnien.vn/nhung-thu-vien-dai-hoc-sang-chanh-khien-sinh-vien-me-man-ngoi-li-ca-ngay-1851510758.htm",
    "https://svvn.tienphong.vn/thu-vien-quoc-gia-khong-gian-hoc-tap-va-lam-viec-ly-tuong-cua-sinh-vien-post1773289.tpo",
    "https://nlv.gov.vn/nghiep-vu-thu-vien/mo-hinh-khong-gian-hoc-tap-o-cac-thu-vien-dai-hoc.html",
]


def slugify(url: str) -> str:
    """Tên file ổn định theo URL để chạy lại không tạo file trùng."""
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"\.(html?|chn|tpo|aspx)$", "", tail)
    tail = re.sub(r"[^a-zA-Z0-9]+", "-", tail).strip("-").lower()
    return (tail or "article")[:80]


def clean_markdown(text: str) -> str:
    """Bỏ dòng trống thừa và ký tự điều khiển."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def crawl_article(url: str) -> dict:
    """Crawl một URL và trả về dict đúng 4 field bắt buộc."""
    title, markdown = "", ""

    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            raw = getattr(result, "markdown", "") or ""
            markdown = clean_markdown(str(raw))
            metadata = getattr(result, "metadata", None) or {}
            title = (metadata.get("title") or "").strip()
    except Exception as error:  # noqa: BLE001 - fallback thay vì làm hỏng cả batch
        print(f"  Crawl4AI lỗi ({error}); dùng fallback requests + MarkItDown")

    if len(markdown) < MIN_CONTENT_CHARS:
        title, markdown = _fallback_fetch(url, fallback_title=title)

    if len(markdown) < MIN_CONTENT_CHARS:
        raise ValueError(f"nội dung quá ngắn ({len(markdown)} ký tự)")

    return {
        "url": url,
        "title": title or slugify(url).replace("-", " ").title(),
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "content_markdown": markdown,
    }


def _fallback_fetch(url: str, fallback_title: str = "") -> tuple[str, str]:
    """requests tải HTML, MarkItDown chuyển sang Markdown."""
    import tempfile

    import requests
    from markitdown import MarkItDown

    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    html = response.text

    title = fallback_title
    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", encoding="utf-8", delete=False
    ) as handle:
        handle.write(html)
        temp_path = handle.name
    try:
        converted = MarkItDown().convert(temp_path)
        return title, clean_markdown(converted.text_content)
    finally:
        Path(temp_path).unlink(missing_ok=True)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved, failed = 0, []
    for index, url in enumerate(ARTICLE_URLS, 1):
        output = DATA_DIR / f"article_{index:02d}_{slugify(url)}.json"
        if output.exists() and output.stat().st_size > MIN_CONTENT_CHARS:
            print(f"Skip (đã có): {output.name}")
            saved += 1
            continue
        try:
            print(f"Crawling: {url}")
            article = await crawl_article(url)
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output.name} ({len(article['content_markdown'])} ký tự)")
            saved += 1
        except Exception as error:  # noqa: BLE001
            failed.append((url, str(error)))
            print(f"Failed: {url} — {error}")

    print(f"\nSaved {saved}, failed {len(failed)}")
    if saved < 5:
        raise SystemExit(
            "Chưa đủ 5 bài viết. Thêm hoặc thay URL trong ARTICLE_URLS "
            "(site chặn crawler thì đổi nguồn khác) rồi chạy lại."
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())
