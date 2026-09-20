"""
Streamlit Chatbot Demo cho RAG Pipeline Tra cứu Quy chế & Dịch vụ Sinh viên.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve, SCORE_THRESHOLD

st.set_page_config(
    page_title="Trợ lý RAG - Quy chế & Dịch vụ Sinh viên",
    page_icon="🎓",
    layout="wide",
)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar cấu hình
with st.sidebar:
    st.title("⚙️ Cấu hình RAG")
    st.markdown(
        """
        **Đề tài:** Hệ thống tra cứu Quy chế đào tạo, Nội quy KTX, Thư viện & Dịch vụ sinh viên đại học.
        """
    )
    st.divider()

    top_k = st.slider("Số lượng Chunks trích xuất (top_k)", min_value=1, max_value=10, value=5)
    use_rerank = st.toggle("Sử dụng Hybrid Search + RRF (Task 7)", value=True)
    score_thresh = st.slider(
        "Ngưỡng Dense Score Fallback (Task 9)",
        min_value=0.1,
        max_value=0.8,
        value=float(SCORE_THRESHOLD),
        step=0.05,
    )

    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Kho dữ liệu: CTUET, HCMUS, USSH, HUIT, KTX ĐHQG-HCM...")

# Tiêu đề chính
st.title("🎓 Trợ lý Tra cứu Quy chế & Dịch vụ Sinh viên")
st.caption("Hỏi đáp có trích dẫn nguồn kiểm chứng (Grounding & Verification) dựa trên tài liệu thực tế")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Xem {len(msg['sources'])} nguồn tài liệu trích dẫn"):
                for idx, src in enumerate(msg["sources"], start=1):
                    meta = src.get("metadata", {})
                    title = meta.get("title", "Tài liệu")
                    source_file = meta.get("source", "N/A")
                    method = src.get("retrieval_method", "hybrid")
                    score = src.get("score", 0.0)
                    url = meta.get("url")

                    st.markdown(
                        f"**{idx}. {title}** (`{source_file}`)  \n"
                        f"- **Phương thức:** `{method}` | **Điểm số:** `{score:.4f}`"
                    )
                    if url:
                        st.markdown(f"- **Link:** [{url}]({url})")
                    st.text(src.get("content", "")[:300] + ("..." if len(src.get("content", "")) > 300 else ""))
                    st.divider()

# Nhận câu hỏi từ người dùng
query = st.chat_input("Đặt câu hỏi về quy chế đào tạo, học phí, đăng ký tín chỉ, ký túc xá...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu và tổng hợp câu trả lời..."):
            try:
                # Nếu người dùng tắt rerank thì retrieve với use_reranking=False
                if not use_rerank:
                    from src.task10_generation import call_llm, format_context, reorder_for_llm, SYSTEM_PROMPT

                    chunks = retrieve(query, top_k=top_k, score_threshold=score_thresh, use_reranking=False)
                    if not chunks:
                        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
                        sources = []
                    else:
                        reordered = reorder_for_llm(chunks)
                        context = format_context(reordered)
                        prompt = f"Dưới đây là các đoạn thông tin (Context):\n{context}\n\nCâu hỏi: {query}\n\nHãy trả lời câu hỏi trên dựa CHỈ trên Context và trích dẫn nguồn chi tiết:"
                        answer = call_llm(SYSTEM_PROMPT, prompt)
                        sources = chunks
                else:
                    gen_result = generate_with_citation(query, top_k=top_k)
                    answer = gen_result["answer"]
                    sources = gen_result["sources"]

            except Exception as e:
                answer = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
                sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Xem {len(sources)} nguồn tài liệu trích dẫn"):
                    for idx, src in enumerate(sources, start=1):
                        meta = src.get("metadata", {})
                        title = meta.get("title", "Tài liệu")
                        source_file = meta.get("source", "N/A")
                        method = src.get("retrieval_method", "hybrid")
                        score = src.get("score", 0.0)
                        url = meta.get("url")

                        st.markdown(
                            f"**{idx}. {title}** (`{source_file}`)  \n"
                            f"- **Phương thức:** `{method}` | **Điểm số:** `{score:.4f}`"
                        )
                        if url:
                            st.markdown(f"- **Link:** [{url}]({url})")
                        st.text(src.get("content", "")[:300] + ("..." if len(src.get("content", "")) > 300 else ""))
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
