"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve
from .text_utils import token_set


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation dạng [S1], [S2]. Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    ordered: list[dict | None] = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            ordered[left] = chunk
            left += 1
        else:
            ordered[right] = chunk
            right -= 1
    return [chunk for chunk in ordered if chunk is not None]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[S{index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']} | Chunk: {metadata['chunk_index']}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER.lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=LLM_MODEL or "gpt-5-mini",
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.output_text.strip()
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=LLM_MODEL or "gemini-2.5-flash",
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return (response.text or "").strip()
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic

        response = Anthropic().messages.create(
            model=LLM_MODEL or "claude-3-5-haiku-latest",
            max_tokens=700,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    raise RuntimeError("Không có API key cho LLM provider đã chọn")


def extractive_answer(query: str, chunks: list[dict], max_sentences: int = 4) -> str:
    """Grounded offline answer used for local demos and deterministic evaluation."""
    query_tokens = token_set(query)
    candidates: list[tuple[float, int, str]] = []
    for source_index, chunk in enumerate(chunks, 1):
        normalized_content = re.sub(r"\s*\n\s*", " ", chunk["content"])
        sentences = re.split(r"(?<=[.!?])\s+", normalized_content)
        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip(" -")
            if not 30 <= len(sentence) <= 420:
                continue
            tokens = token_set(sentence)
            overlap = query_tokens & tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1) + len(overlap) / max(len(tokens), 1)
            candidates.append((score, source_index, sentence))
    candidates.sort(key=lambda item: -item[0])
    selected: list[str] = []
    seen: set[str] = set()
    for _, source_index, sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(f"{sentence} [S{source_index}]")
        if len(selected) >= max_sentences:
            break
    return " ".join(selected) if selected else SAFE_REFUSAL


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        answer = extractive_answer(query, reordered)
    if not answer or answer == SAFE_REFUSAL:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    # Return sources in the same order used by [S1], [S2], ... citations.
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0]["retrieval_method"],
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
