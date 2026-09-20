"""Chatbot RAG — Dịch vụ đại học (ký túc xá, thư viện, đăng ký học phần)."""

import time

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot — Dịch vụ đại học",
    page_icon="🎓",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict], retrieval_source: str, elapsed: float) -> None:
    """Hiển thị nguồn, retrieval method và score để đối chiếu citation."""
    if not sources:
        st.info(f"Không có nguồn nào được truy xuất · {elapsed:.2f}s")
        return
    with st.expander(
        f"Nguồn đã dùng ({len(sources)}) · retrieval_source = "
        f"`{retrieval_source}` · {elapsed:.2f}s",
        expanded=True,
    ):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            url = metadata.get("url")
            title = metadata.get("title", "—")
            institution = metadata.get("institution") or "—"
            header = f"**[Document {index}]** {title}"
            if url:
                header += f" — [link]({url})"
            st.markdown(header)
            st.caption(f"🏛 **{institution}**")
            st.caption(
                f"source: `{metadata.get('source', '—')}` · "
                f"doc_type: `{metadata.get('doc_type', '—')}` · "
                f"chunk: `{metadata.get('chunk_index', 0)}` · "
                f"method: `{source.get('retrieval_method', '—')}` · "
                f"score: `{source.get('score', 0):.4f}`"
            )
            st.markdown(
                f"> {source.get('content', '')[:400].replace(chr(10), ' ')}..."
            )
            if index < len(sources):
                st.divider()


with st.sidebar:
    st.title("🎓 RAG Chatbot")
    st.caption(
        "Hỏi đáp quy định **ký túc xá, thư viện và đăng ký học phần** "
        "dựa trên tài liệu nhóm tự thu thập."
    )
    st.warning(
        "Corpus gồm quy định của **nhiều trường**, và các trường quy định "
        "khác nhau. Nên nêu tên trường trong câu hỏi; nếu không, bot sẽ liệt kê "
        "từng trường riêng.",
        icon="🏛",
    )
    top_k = st.slider("Số chunks (top_k)", 3, 10, 5)
    use_hybrid = st.toggle(
        "Hybrid retrieval (dense + BM25 + RRF)",
        value=True,
        help="Tắt để so sánh với Config A — dense-only.",
    )
    st.divider()
    st.caption("Câu hỏi ngoài phạm vi tài liệu sẽ nhận safe refusal, không bịa.")
    if st.button("Xoá hội thoại"):
        st.session_state.messages = []
        st.rerun()

st.title("Chatbot quy định dịch vụ đại học")
st.caption(
    "Nhập câu hỏi về ký túc xá, thư viện hoặc đăng ký học phần. "
    "Mỗi câu trả lời kèm [Document N] đối chiếu được với danh sách nguồn bên dưới."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
                message.get("elapsed", 0.0),
            )

query = st.chat_input("Ví dụ: Sinh viên được mượn tối đa bao nhiêu cuốn giáo trình?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            started = time.perf_counter()
            try:
                import src.task10_generation as generation

                generation.USE_RERANKING = use_hybrid
                result = generation.generate_with_citation(query, top_k=top_k)
            except Exception as error:  # noqa: BLE001 - không để UI chết
                result = {
                    "answer": f"Pipeline gặp lỗi: `{error}`. "
                    "Kiểm tra `.env` và đã chạy `python -m src.task4_chunking_indexing` chưa.",
                    "sources": [],
                    "retrieval_source": "none",
                }
            elapsed = time.perf_counter() - started

        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"], elapsed)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "elapsed": elapsed,
        }
    )
