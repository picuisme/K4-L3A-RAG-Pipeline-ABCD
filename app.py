"""
Streamlit chat UI for the University Services RAG assistant.
Giao diện Tra cứu Hồ sơ & Dịch vụ Sinh viên (Dossier Theme).
"""

from __future__ import annotations

import html
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve, SCORE_THRESHOLD

st.set_page_config(
    page_title="Hồ sơ Sinh viên — RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --navy: #102A43;
  --cobalt: #2456A6;
  --paper: #F5F7FA;
  --white: #FFFFFF;
  --seal: #D64545;
  --ink: #172B4D;
  --muted: #627D98;
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] {
  background: var(--navy);
  border-right: 4px solid var(--cobalt);
}
[data-testid="stSidebar"] * { color: #EAF2FF !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
  color: #89B4FF !important;
}
.block-container { max-width: 1080px; padding-top: 1.5rem; }
.dossier-header {
  background: var(--white);
  border-top: 5px solid var(--cobalt);
  border-bottom: 1px solid #BCCCDC;
  padding: 1.4rem 1.6rem 1.1rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 24px rgba(16, 42, 67, .07);
}
.dossier-kicker {
  color: var(--seal);
  font: 700 .75rem/1.2 Consolas, monospace;
  letter-spacing: .13em;
  text-transform: uppercase;
}
.dossier-title {
  color: var(--navy);
  font: 700 clamp(1.8rem, 3.5vw, 3rem)/1.1 Georgia, serif;
  margin: .35rem 0 .55rem;
}
.dossier-subtitle { color: var(--muted); max-width: 760px; margin: 0; font-size: .95rem; }
.status-strip {
  display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1rem;
  font: 600 .72rem/1.2 Consolas, monospace; color: var(--navy);
}
.status-strip span { border: 1px solid #BCCCDC; padding: .3rem .55rem; background: #F0F4F8; }
[data-testid="stChatMessage"] {
  background: var(--white);
  border: 1px solid #D9E2EC;
  border-radius: 4px;
  box-shadow: 0 3px 12px rgba(16, 42, 67, .04);
  margin-bottom: .8rem;
}
.source-card {
  background: #FBFCFE;
  border-left: 3px solid var(--seal);
  padding: .75rem .9rem;
  margin: .55rem 0;
  border-radius: 0 4px 4px 0;
  border-top: 1px solid #E2E8F0;
  border-right: 1px solid #E2E8F0;
  border-bottom: 1px solid #E2E8F0;
}
.source-label {
  font: 700 .70rem/1.2 Consolas, monospace;
  color: var(--seal); letter-spacing: .08em;
}
.source-title { color: var(--navy); font-weight: 700; margin: .2rem 0; font-size: .95rem; }
.source-meta { color: var(--muted); font-size: .78rem; }
.source-excerpt { color: var(--ink); font-size: .86rem; margin-top: .35rem; line-height: 1.45; }
.stTextInput input:focus, .stChatInput textarea:focus {
  border-color: var(--cobalt) !important;
  box-shadow: 0 0 0 2px rgba(36, 86, 166, .18) !important;
}
@media (max-width: 700px) {
  .block-container { padding: 1rem .8rem 5rem; }
  .dossier-header { padding: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    """Hiển thị danh sách nguồn trích dẫn dưới dạng source cards đẹp mắt."""
    if not sources:
        return
    st.caption(
        f"📋 **Sổ nguồn đối chiếu:** {len(sources)} trích đoạn · Phương thức: `{retrieval_source.upper()}`"
    )
    with st.expander(f"📚 Xem chi tiết {len(sources)} nguồn trích dẫn [S1] - [S{len(sources)}]", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            excerpt = " ".join(source.get("content", "").split())[:320]
            source_name = metadata.get("source", "Không rõ nguồn")
            source_url = metadata.get("url")
            method = source.get("retrieval_method", "hybrid").upper()
            score = source.get("score", 0.0)
            chunk_idx = metadata.get("chunk_index", 0)

            if source_url:
                source_line = f'<a href="{html.escape(source_url)}" target="_blank" style="color: #2456A6; text-decoration: underline;">{html.escape(source_name)}</a>'
            else:
                source_line = html.escape(source_name)

            st.markdown(
                f"""
<div class="source-card">
  <div class="source-label">[S{index}] · {html.escape(method)} · Score: {score:.4f}</div>
  <div class="source-title">{html.escape(metadata.get('title', 'Tài liệu'))}</div>
  <div class="source-meta">File: {source_line} · Đoạn #{chunk_idx}</div>
  <div class="source-excerpt">{html.escape(excerpt)}…</div>
</div>
""",
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar cấu hình
with st.sidebar:
    st.markdown("## ⚙️ HỒ SƠ SINH VIÊN")
    st.caption("Tra cứu quy chế · Ký túc xá · Thư viện · Học vụ")
    st.divider()

    top_k = st.slider("Số nguồn đối chiếu (top_k)", min_value=1, max_value=10, value=5)
    use_rerank = st.toggle("Sử dụng Hybrid Search + RRF (Task 7)", value=True)
    score_thresh = st.slider(
        "Ngưỡng Fallback Score (Task 9)",
        min_value=0.1,
        max_value=0.8,
        value=float(SCORE_THRESHOLD),
        step=0.05,
    )

    st.divider()
    st.markdown(
        "**💡 Cách hỏi hiệu quả**\n\n"
        "Nêu rõ tên trường và nội dung quy chế, ví dụ:\n"
        "- *Ký túc xá Đại học Cần Thơ quy định giờ giấc ra sao?*\n"
        "- *Mức phạt trả trễ tài liệu thư viện USSH là bao nhiêu?*\n"
        "- *Đối tượng ưu tiên số 1 khi xét ở KTX HUIT gồm những ai?*"
    )
    st.divider()

    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Câu trả lời đối chiếu từ corpus nội bộ và trích dẫn mã [S1], [S2]…")

# Header chính
st.markdown(
    """
<section class="dossier-header">
  <div class="dossier-kicker">Hệ thống Tra cứu Quy định Đại học · Nhóm ABCD</div>
  <h1 class="dossier-title">Hỏi đúng hồ sơ.<br>Nhận đúng căn cứ.</h1>
  <p class="dossier-subtitle">Trợ lý tra cứu thông tin quy chế đào tạo, nội quy ký túc xá và dịch vụ sinh viên từ tài liệu thực tế, đảm bảo câu trả lời luôn có bằng chứng trích dẫn nguồn kiểm chứng.</p>
  <div class="status-strip">
    <span>05 VĂN BẢN QUY CHẾ</span>
    <span>07 BÀI VIẾT DỊCH VỤ</span>
    <span>HYBRID RETRIEVAL + RRF</span>
    <span>GROUNDED CITATIONS</span>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"], message.get("retrieval_source", "hybrid"))

# Chat input
query = st.chat_input("Nhập câu hỏi về học vụ, KTX hoặc thư viện…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang đối chiếu hồ sơ và tổng hợp câu trả lời…"):
            try:
                if not use_rerank:
                    from src.task10_generation import call_llm, extractive_answer, format_context, reorder_for_llm, SYSTEM_PROMPT, SAFE_REFUSAL

                    chunks = retrieve(query, top_k=top_k, score_threshold=score_thresh, use_reranking=False)
                    if not chunks:
                        answer = SAFE_REFUSAL
                        sources = []
                        retrieval_source = "none"
                    else:
                        reordered = reorder_for_llm(chunks)
                        context = format_context(reordered)
                        prompt = f"Dưới đây là các đoạn thông tin (Context):\n{context}\n\nCâu hỏi: {query}\n\nHãy trả lời câu hỏi trên dựa CHỈ trên Context và trích dẫn nguồn chi tiết dạng [S1], [S2]...:"
                        try:
                            answer = call_llm(SYSTEM_PROMPT, prompt)
                            if not answer.strip():
                                answer = extractive_answer(query, reordered)
                        except Exception:
                            answer = extractive_answer(query, reordered)
                        sources = reordered
                        retrieval_source = "dense"
                else:
                    gen_result = generate_with_citation(query, top_k=top_k)
                    answer = gen_result["answer"]
                    sources = gen_result["sources"]
                    retrieval_source = gen_result.get("retrieval_source", "hybrid")

            except Exception as error:
                answer = f"Đã xảy ra lỗi: {error}"
                sources = []
                retrieval_source = "none"

        st.markdown(answer)
        if sources:
            render_sources(sources, retrieval_source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
