"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env (OpenAI, Gemini, Anthropic) với extractive fallback offline.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re
from typing import Literal
from dotenv import load_dotenv

from .contracts import validate_generation_result
from .task9_retrieval_pipeline import retrieve
from .text_utils import token_set

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SYSTEM_PROMPT = """Bạn là trợ lý giải đáp thắc mắc dựa trên tài liệu quy chế và thông tin đại học.
Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên thông tin có trong Context. Tuyệt đối không suy diễn hay đưa thông tin ngoài tài liệu.
2. Mỗi luận điểm hoặc thông tin nêu ra phải kèm citation đối chiếu với tài liệu dạng [S1], [S2]...
3. Nếu Context không có thông tin hoặc không đủ để trả lời câu hỏi, bạn PHẢI từ chối rõ ràng bằng câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
"""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context để giảm lost-in-the-middle."""
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
    """Tạo context có title và source label cho LLM."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "unknown")
        chunk_idx = metadata.get("chunk_index", 0)
        parts.append(
            f"[S{index} | Title: {title} | Source: {source} | Chunk: {chunk_idx}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình .env."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY chưa được thiết lập")
        client = OpenAI(api_key=api_key)
        target_model = model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY chưa được thiết lập")
        client = genai.Client(api_key=api_key)
        target_model = model or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=target_model,
            contents=user_message,
            config={
                "system_instruction": system_prompt,
                "temperature": TEMPERATURE,
            },
        )
        return response.text or ""

    elif provider == "anthropic":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY chưa được thiết lập")
        client = anthropic.Anthropic(api_key=api_key)
        target_model = model or "claude-3-5-haiku-20241022"
        response = client.messages.create(
            model=target_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
        )
        return response.content[0].text or ""

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


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
    """Trả về GenerationResult gồm câu trả lời và nguồn trích dẫn."""
    if top_k <= 0 or not query.strip():
        result = {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }
        validate_generation_result(result)
        return result

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        result = {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }
        validate_generation_result(result)
        return result

    retrieval_method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source: Literal["hybrid", "pageindex", "none"] = (
        "pageindex" if retrieval_method == "pageindex" else "hybrid"
    )

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Dưới đây là các đoạn thông tin (Context):\n{context}\n\nCâu hỏi: {query}\n\nHãy trả lời câu hỏi trên dựa CHỈ trên Context và trích dẫn nguồn chi tiết dạng [S1], [S2]...:"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer.strip():
            answer = extractive_answer(query, reordered)
    except Exception as exc:
        print(f"[Generation] LLM fallback sang extractive: {exc}")
        answer = extractive_answer(query, reordered)

    if not answer or answer == SAFE_REFUSAL:
        answer = SAFE_REFUSAL

    result = {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }
    validate_generation_result(result)
    return result


if __name__ == "__main__":
    print(generate_with_citation("Giờ giấc mở cửa phòng tự học KTX ĐHQG-HCM thế nào?"))
