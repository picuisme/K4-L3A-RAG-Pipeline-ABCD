"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from .contracts import validate_document


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
load_dotenv()

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hashing").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hashing-vi-1024")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

DOCUMENT_TITLES = {
    "legal/noi-quy-ky-tuc-xa-dh-can-tho.md": "Nội quy Ký túc xá Trường Đại học Cần Thơ",
    "legal/noi-quy-ky-tuc-xa-huit.md": "Quy chế lưu trú Ký túc xá HUIT",
    "legal/noi-quy-thu-vien-ussh-vnuhcm.md": "Nội quy Thư viện USSH - ĐHQG-HCM",
    "legal/quy-che-dao-tao-dai-hoc-hcmus.md": "Quy chế đào tạo đại học HCMUS",
    "legal/so-tay-sinh-vien-ctuet.md": "Sổ tay sinh viên CTUET",
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed with one shared provider; hashing is the reproducible default."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "sentence_transformers":
        model = _sentence_transformer()
        return model.encode(texts, normalize_embeddings=True).tolist()
    if EMBEDDING_PROVIDER != "hashing":
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER}")

    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=EMBEDDING_DIM,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
        lowercase=True,
    )
    return vectorizer.transform(texts).toarray().astype(float).tolist()


@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


class LocalVectorCollection:
    """Minimal persistent cosine store used when Chroma is unavailable."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / "local_vectors.json"
        directory.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def upsert(self, *, ids, documents, embeddings, metadatas) -> None:
        records = self._read()
        for item_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            records[item_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }
        self.path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )

    def query(self, *, query_embeddings, n_results, include=None) -> dict:
        import numpy as np

        records = self._read()
        if not records or n_results <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        query = np.asarray(query_embeddings[0], dtype=float)
        scored = []
        for item_id, record in records.items():
            vector = np.asarray(record["embedding"], dtype=float)
            denominator = float(np.linalg.norm(query) * np.linalg.norm(vector))
            similarity = float(np.dot(query, vector) / denominator) if denominator else 0.0
            scored.append((similarity, item_id, record))
        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen = scored[: min(n_results, len(scored))]
        return {
            "ids": [[row[1] for row in chosen]],
            "documents": [[row[2]["document"] for row in chosen]],
            "metadatas": [[row[2]["metadata"] for row in chosen]],
            "distances": [[1.0 - row[0] for row in chosen]],
        }


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except (ImportError, ModuleNotFoundError):
        return LocalVectorCollection(CHROMA_DIR)


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        url_match = re.search(r"^\*\*URL:\*\*\s+(.+)$", content, re.MULTILINE)
        source_match = re.search(r"^\*\*Source:\*\*\s+(.+)$", content, re.MULTILINE)
        relative = path.relative_to(STANDARDIZED_DIR).as_posix()
        document = {
            "id": relative,
            "content": content,
            "metadata": {
                "source": source_match.group(1).strip() if source_match else path.name,
                "title": DOCUMENT_TITLES.get(
                    relative,
                    title_match.group(1).strip() if title_match else path.stem,
                ),
                "doc_type": "legal" if "legal" in path.parts else "news",
                "url": url_match.group(1).strip() if url_match else None,
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    chunks = []
    for document in documents:
        validate_document(document)
        parts = _split_text(document["content"])
        for index, text in enumerate(parts):
            stable = hashlib.sha1(
                f"{document['id']}:{index}:{text}".encode("utf-8")
            ).hexdigest()[:12]
            chunk = {
                "id": f"{document['id']}::chunk-{index:04d}-{stable}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def _split_text(text: str) -> list[str]:
    """Recursive paragraph-aware splitter with deterministic overlap."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph]
        if len(paragraph) > CHUNK_SIZE:
            piece_size = CHUNK_SIZE - CHUNK_OVERLAP
            pieces = [
                paragraph[start : start + piece_size]
                for start in range(0, len(paragraph), piece_size)
            ]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
                continue
            if current:
                chunks.append(current)
                overlap = current[-CHUNK_OVERLAP:].lstrip()
                current = f"{overlap}\n\n{piece}".strip()
                if len(current) > CHUNK_SIZE:
                    current = piece
            else:
                current = piece
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned an unexpected vector count")
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        raise ValueError("Không có chunk để lập chỉ mục")
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
