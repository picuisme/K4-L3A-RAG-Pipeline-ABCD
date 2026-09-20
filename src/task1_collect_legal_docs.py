"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Đề tài nhóm: **Dịch vụ đại học** — ký túc xá, thư viện, đăng ký học phần.

Nguồn chọn theo tiêu chí: văn bản công khai của cơ sở giáo dục, có số quyết
định hoặc đơn vị ban hành rõ ràng, URL kiểm chứng được, và PDF có lớp text
(không phải bản scan) để MarkItDown convert được ở Task 3.

Chạy:
    python -m src.task1_collect_legal_docs
"""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Corpus cố ý trải trên nhiều trường: mỗi trường chỉ công khai được vài mảng
# dịch vụ dưới dạng PDF có lớp text. Hệ quả là các văn bản MÂU THUẪN nhau
# (số sách được mượn, giờ đóng cổng… khác nhau theo trường), nên `institution`
# là metadata bắt buộc: nó được ghi vào front matter ở Task 3, đi vào metadata
# của từng chunk ở Task 4, được prepend vào nội dung chunk để dense/BM25 phân
# biệt được, và hiện trong context ở Task 10 để câu trả lời luôn nêu rõ trường.
SOURCES: dict[str, dict[str, str]] = {
    "noi-quy-ky-tuc-xa-dh-can-tho.pdf": {
        "url": "https://dsa.ctu.edu.vn/images/upload/vbanply/KTX%20sinh%20vien/"
        "Noi%20quy%20KTX%20nam%202016.pdf",
        "institution": "Trường Đại học Cần Thơ",
        "description": "Nội quy công tác nội trú tại Ký túc xá",
    },
    "noi-quy-thu-vien-ussh-vnuhcm.pdf": {
        "url": "https://hcmussh.edu.vn/static/document/11717.pdf",
        "institution": "Trường ĐH Khoa học Xã hội và Nhân văn, ĐHQG-HCM",
        "description": "Nội quy Thư viện",
    },
    "quy-che-dao-tao-dai-hoc-hcmus.pdf": {
        "url": "https://www.ctda.hcmus.edu.vn/wp-content/uploads/2023/03/"
        "Quy-che-dao-tao-2021.pdf",
        "institution": "Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM",
        "description": "Quy chế đào tạo trình độ đại học (QĐ 1175/QĐ-KHTN)",
    },
    "noi-quy-ky-tuc-xa-huit.pdf": {
        "url": "https://kcnck.huit.edu.vn/app_web/images/documents/n00ct/"
        "noi-quy-quy-che-ky-tuc-xa.pdf",
        "institution": "Trường ĐH Công nghiệp Thực phẩm TP.HCM",
        "description": "Nội quy - Quy chế lưu trú tại Ký túc xá",
    },
    "so-tay-sinh-vien-ctuet.pdf": {
        "url": "https://khoacntt.ctuet.edu.vn/wp-content/uploads/2020/07/"
        "So-tay-sinh-vien-K7_2019_Ban-in.pdf",
        "institution": "Trường ĐH Kỹ thuật - Công nghệ Cần Thơ",
        "description": "Sổ tay sinh viên / Quy định đào tạo theo tín chỉ",
    },
}

# filename -> tên trường, dùng lại ở Task 3.
INSTITUTION_BY_FILE: dict[str, str] = {
    filename: entry["institution"] for filename, entry in SOURCES.items()
}

MIN_BYTES = 1024


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải các PDF/DOCX từ nguồn công khai vào data/landing/legal/.

    Chạy lại sẽ bỏ qua file đã tải hợp lệ nên không tạo dữ liệu trùng.
    Site chặn crawler (WAF/captcha) thì đổi nguồn khác trong SOURCES,
    không tìm cách vượt qua.
    """
    import requests

    setup_directory()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    downloaded, skipped, failed = 0, 0, []
    for filename, entry in SOURCES.items():
        url, description = entry["url"], entry["description"]
        target = DATA_DIR / filename
        if target.exists() and target.stat().st_size > MIN_BYTES:
            print(f"Skip (đã có): {filename}")
            skipped += 1
            continue
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
            if len(content) <= MIN_BYTES:
                raise ValueError(f"file quá nhỏ ({len(content)} bytes)")
            target.write_bytes(content)
            print(f"Saved: {filename} ({len(content) // 1024} KB) — {description}")
            downloaded += 1
        except Exception as error:  # noqa: BLE001 - báo lỗi từng nguồn, không dừng cả batch
            failed.append((filename, url, str(error)))
            print(f"Failed: {filename} — {error}")

    print(f"\nDownloaded {downloaded}, skipped {skipped}, failed {len(failed)}")
    valid = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in {".pdf", ".doc", ".docx"}
        and path.stat().st_size > MIN_BYTES
    ]
    print(f"Tổng tài liệu hợp lệ trong data/landing/legal: {len(valid)}")
    if len(valid) < 3:
        raise SystemExit(
            "Chưa đủ 3 tài liệu chính sách. Hãy thay URL trong SOURCES bằng "
            "nguồn công khai khác rồi chạy lại."
        )


if __name__ == "__main__":
    download_documents()
