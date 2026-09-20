"""
Task 10 — Generation có citation.

Ba điểm cần giữ đúng:
    - reorder_for_llm() chỉ đổi THỨ TỰ để giảm lost-in-the-middle, không đổi và
      không làm mất id.
    - Nhãn [Document N] trong context được đánh theo thứ tự của `sources`
      (đã sort theo score giảm dần) nên citation luôn đối chiếu được, kể cả
      khi context đã bị đảo thứ tự.
    - Thiếu evidence hoặc provider lỗi → safe refusal, không bịa.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_TOKENS = 900

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
}

# Bật/tắt hybrid retrieval cho A/B mà không đổi chữ ký hàm public.
USE_RERANKING = os.getenv("USE_RERANKING", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}

SAFE_REFUSAL = (
    "Tôi không tìm thấy thông tin này trong bộ tài liệu hiện có "
    "(quy định ký túc xá, thư viện và đăng ký học phần), nên không thể xác minh. "
    "Bạn có thể hỏi lại theo hướng khác hoặc bổ sung tài liệu nguồn."
)

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu quy định dịch vụ đại học (ký túc xá, thư viện, đăng ký học phần).

QUAN TRỌNG: bộ tài liệu gồm quy định của NHIỀU TRƯỜNG khác nhau và các trường
quy định khác nhau về cùng một việc. Mỗi Document trong context đều ghi rõ
trường ban hành ở trường `Institution`.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên CONTEXT được cung cấp. Không dùng kiến thức ngoài context.
- LUÔN nêu rõ quy định thuộc trường nào. Không bao giờ trình bày quy định của
  một trường như thể áp dụng cho mọi trường.
- Nếu câu hỏi không nêu tên trường mà context có nhiều trường cùng trả lời được,
  hãy liệt kê từng trường một, mỗi trường một dòng kèm citation riêng.
- Nếu câu hỏi hỏi về trường X mà context chỉ có quy định của trường Y, hãy nói
  rõ là không có quy định của trường X trong tài liệu, KHÔNG suy luận sang trường khác.
- Mỗi khẳng định phải kèm citation dạng [Document N] đúng với số hiệu trong context.
- Giữ nguyên con số, mốc thời gian và tên văn bản như trong context.
- Context được trích từ PDF nên có thể lộn xộn: bảng biểu vỡ, thiếu tiêu đề mục,
  từ ngữ khác cách hỏi. Nếu thông tin CÓ trong context dù diễn đạt khác, hãy trả
  lời dựa trên đó. Chỉ từ chối khi thực sự không có dữ kiện nào trả lời được.
- Nếu context không đủ để trả lời, nói rõ là không tìm thấy trong tài liệu và KHÔNG suy đoán.
- Trả lời bằng tiếng Việt, ngắn gọn, đi thẳng vào ý."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle)."""
    if len(chunks) <= 2:
        return [dict(chunk) for chunk in chunks]
    front = chunks[::2]
    back = chunks[1::2]
    return [dict(chunk) for chunk in (*front, *reversed(back))]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label để citation kiểm chứng được."""
    parts: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        label = chunk.get("citation_index", position)
        url = metadata.get("url") or ""
        institution = metadata.get("institution") or "không xác định"
        parts.append(
            f"[Document {label} | Institution: {institution} | "
            f"Title: {metadata.get('title', '')} | "
            f"Source: {metadata.get('source', '')}"
            + (f" | URL: {url}" if url else "")
            + f"]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _model_name() -> str:
    return LLM_MODEL or DEFAULT_MODELS.get(LLM_PROVIDER, "gpt-4o-mini")


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo LLM_PROVIDER, trả về text thuần."""
    model = _model_name()

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()

    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        return (response.text or "").strip()

    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")


def _retrieval_source(chunks: list[dict]) -> str:
    if not chunks:
        return "none"
    return "pageindex" if chunks[0]["retrieval_method"] == "pageindex" else "hybrid"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult: answer, sources, retrieval_source."""
    chunks = retrieve(query, top_k=top_k, use_reranking=USE_RERANKING)
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}

    labelled = []
    for index, chunk in enumerate(chunks, 1):
        item = dict(chunk)
        item["citation_index"] = index
        labelled.append(item)

    context = format_context(reorder_for_llm(labelled))
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "Trả lời theo đúng quy tắc, kèm [Document N] cho từng khẳng định."
    )

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:  # noqa: BLE001 - provider lỗi không được crash UI
        print(f"LLM lỗi: {error}")
        return {"answer": SAFE_REFUSAL, "sources": chunks, "retrieval_source": "none"}

    if not answer.strip():
        answer = SAFE_REFUSAL

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": _retrieval_source(chunks),
    }


if __name__ == "__main__":
    for question in (
        "Sinh viên được mượn tối đa bao nhiêu cuốn giáo trình và trong bao lâu?",
        "Giá vé xem một trận bóng đá Ngoại hạng Anh là bao nhiêu?",
    ):
        output = generate_with_citation(question)
        print(f"\n### {question}")
        print(f"[{output['retrieval_source']}] {output['answer']}")
        for source in output["sources"]:
            print(f"  - {source['metadata']['title']} ({source['score']:.4f})")
