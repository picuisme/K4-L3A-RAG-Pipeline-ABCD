"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn (RecursiveCharacterTextSplitter).
    3. Embed chunks bằng một provider duy nhất (hỗ trợ sentence_transformers, openai, gemini).
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from .contracts import validate_document

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Cấu hình Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Cấu hình Embedding
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_st_model = None


def get_embedding_model():
    """Lazy load SentenceTransformer model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embedding cho danh sách văn bản theo EMBEDDING_PROVIDER trong .env."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).strip().lower()

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip()
        if "text-embedding" not in model_name:
            model_name = "text-embedding-3-small"

        all_vectors: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model_name)
            all_vectors.extend([item.embedding for item in response.data])
        return all_vectors

    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-004").strip()
        all_vectors = []
        for text in texts:
            res = client.models.embed_content(model=model_name, contents=text)
            all_vectors.append(res.embedding.values)
        return all_vectors

    else:
        # Mặc định: sentence_transformers
        model = get_embedding_model()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document theo contract."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not path.is_file() or path.name.startswith("."):
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        # Trích xuất url nguồn nếu có
        url = None
        for line in content.splitlines()[:10]:
            if line.startswith("**Source:**"):
                candidate = line.replace("**Source:**", "").strip()
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    url = candidate
                break

        # Trích xuất tiêu đề nếu có
        title = path.stem.replace("-", " ").replace("_", " ").title()
        for line in content.splitlines()[:5]:
            if line.startswith("# "):
                extracted = line[2:].strip()
                if extracted:
                    title = extracted
                break

        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        document = {
            "id": doc_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        }
        validate_document(document)
        documents.append(document)

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        splits = [s.strip() for s in splitter.split_text(document["content"]) if s.strip()]
        for index, text in enumerate(splits):
            chunk_metadata = dict(document["metadata"])
            chunk_metadata["chunk_index"] = index

            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": chunk_metadata,
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()

    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        clean_metadatas = []
        for chunk in batch:
            meta = dict(chunk["metadata"])
            clean_meta: dict[str, Any] = {}
            for k, v in meta.items():
                clean_meta[k] = "" if v is None else v
            clean_metadatas.append(clean_meta)

        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=clean_metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("1. Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("2. Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"3. Embedding chunks (provider: {os.getenv('EMBEDDING_PROVIDER', EMBEDDING_PROVIDER)})...")
    embedded_chunks = embed_chunks(chunks)
    print(f"Generated embeddings for {len(embedded_chunks)} chunks.")

    print("4. Indexing into ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks into ChromaDB at {CHROMA_DIR}.")


if __name__ == "__main__":
    run_pipeline()
