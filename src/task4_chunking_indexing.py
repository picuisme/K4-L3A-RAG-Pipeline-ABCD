"""
Task 4 — Chunking, embedding và indexing.

Quy tắc giữ xuyên suốt pipeline:
    - id chunk = "<đường dẫn tương đối>::chunk-<index>" → ổn định, chạy lại
      upsert đè đúng chỗ, không sinh bản ghi trùng.
    - metadata nguồn (source/title/doc_type/url) đi kèm từ Task 3 tới Task 10.
    - embed_texts() là hàm embedding DUY NHẤT; Task 5 import lại chính nó nên
      query và corpus luôn cùng model và cùng dimension.

Chạy:
    python -m src.task4_chunking_indexing
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .task3_convert_markdown import parse_front_matter


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
# 500 ký tự ~ 1 điều khoản/1 đoạn văn bản quy định: đủ ngữ cảnh để trả lời,
# đủ nhỏ để top-5 chunk không nuốt hết context window.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0") or 0)

COLLECTION_NAME = "rag_documents"
EMBED_BATCH_SIZE = 64

_sentence_transformer = None
_collection = None


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer

        model_name = EMBEDDING_MODEL or "BAAI/bge-m3"
        _sentence_transformer = SentenceTransformer(model_name)
    return _sentence_transformer.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = EMBEDDING_MODEL or "text-embedding-3-small"
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [text.replace("\n", " ") for text in texts[start : start + EMBED_BATCH_SIZE]]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = EMBEDDING_MODEL or "text-embedding-004"
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.models.embed_content(model=model, contents=batch)
        vectors.extend(list(item.values) for item in response.embeddings)
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Hàm embedding duy nhất của pipeline (Task 4 và Task 5 dùng chung)."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "sentence_transformers":
        return _embed_sentence_transformers(texts)
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    if EMBEDDING_PROVIDER == "gemini":
        return _embed_gemini(texts)
    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# --------------------------------------------------------------------------- #
# Load / chunk / index
# --------------------------------------------------------------------------- #
def load_documents() -> list[dict]:
    """Đọc Markdown đã chuẩn hóa và trả về danh sách Document."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        content = body.strip() or raw.strip()
        if not content:
            continue
        doc_type = meta.get("doc_type") or (
            "legal" if "legal" in path.parts else "news"
        )
        url = meta.get("url") or None
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": meta.get("source") or path.name,
                    "title": meta.get("title") or path.stem,
                    "doc_type": doc_type,
                    "institution": meta.get("institution", ""),
                    "url": url,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định và chunk_index liên tục.

    Corpus trải trên nhiều trường và các văn bản mâu thuẫn nhau, nên tên trường
    được prepend vào đầu mỗi chunk: dense và BM25 mới phân biệt được
    "6 cuốn ở USSH" với quy định của trường khác, thay vì trả về chunk đúng chủ
    đề nhưng sai trường. Ngân sách ký tự được trừ đi độ dài prefix để tổng độ
    dài chunk vẫn nằm trong CHUNK_SIZE.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitters: dict[int, object] = {}

    def splitter_for(budget: int):
        if budget not in splitters:
            splitters[budget] = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=min(CHUNK_OVERLAP, max(budget // 10, 0)),
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return splitters[budget]

    chunks: list[dict] = []
    for document in documents:
        institution = (document["metadata"].get("institution") or "").strip()
        prefix = f"[{institution}] " if institution else ""
        budget = max(CHUNK_SIZE - len(prefix), 100)

        index = 0
        for text in splitter_for(budget).split_text(document["content"]):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": f"{prefix}{text}",
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
            index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field khác."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _storable(metadata: dict) -> dict:
    """Chroma không nhận None trong metadata — chuyển sang chuỗi rỗng."""
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB (id ổn định nên chạy lại không trùng)."""
    if not chunks:
        print("Không có chunk nào để index.")
        return
    collection = get_collection()
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_storable(chunk["metadata"]) for chunk in batch],
        )


def load_corpus_from_store() -> list[dict]:
    """Đọc lại toàn bộ chunk đã index — Task 6 dùng chung corpus này."""
    collection = get_collection()
    payload = collection.get(include=["documents", "metadatas"])
    corpus: list[dict] = []
    for item_id, content, metadata in zip(
        payload.get("ids", []),
        payload.get("documents", []),
        payload.get("metadatas", []),
    ):
        metadata = dict(metadata or {})
        metadata.setdefault("chunk_index", 0)
        corpus.append({"id": item_id, "content": content, "metadata": metadata})
    corpus.sort(key=lambda item: item["id"])
    return corpus


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {STANDARDIZED_DIR}")
    if not documents:
        raise SystemExit("Chưa có Markdown chuẩn hóa. Chạy Task 3 trước.")

    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embedded_chunks = embed_chunks(chunks)
    dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
    print(f"Embedded with {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL} (dim={dim})")

    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks → {CHROMA_DIR}")
    print(f"Collection size: {get_collection().count()}")


if __name__ == "__main__":
    run_pipeline()
