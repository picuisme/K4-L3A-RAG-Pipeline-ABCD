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
from typing import Literal
from dotenv import load_dotenv

from .contracts import validate_generation_result
from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SYSTEM_PROMPT = """Bạn là trợ lý giải đáp thắc mắc dựa trên tài liệu quy chế và thông tin đại học.
Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên thông tin có trong Context. Tuyệt đối không suy diễn hay đưa thông tin ngoài tài liệu.
2. Mỗi luận điểm hoặc thông tin nêu ra phải kèm citation đối chiếu với tài liệu (ví dụ: [Source: <tên file>] hoặc [Document <số>]).
3. Nếu Context không có thông tin hoặc không đủ để trả lời câu hỏi, bạn PHẢI từ chối rõ ràng bằng câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context để giảm lost-in-the-middle."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label cho LLM."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "unknown")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình .env."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
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


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult gồm câu trả lời và nguồn trích dẫn."""
    if top_k <= 0 or not query.strip():
        result = {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
        validate_generation_result(result)
        return result

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        result = {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
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
    user_message = f"Dưới đây là các đoạn thông tin (Context):\n{context}\n\nCâu hỏi: {query}\n\nHãy trả lời câu hỏi trên dựa CHỈ trên Context và trích dẫn nguồn chi tiết:"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer.strip():
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as exc:
        print(f"[Generation] LLM call error: {exc}")
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    result = {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }
    validate_generation_result(result)
    return result


if __name__ == "__main__":
    print(generate_with_citation("Giờ giấc mở cửa phòng tự học KTX ĐHQG-HCM thế nào?"))
