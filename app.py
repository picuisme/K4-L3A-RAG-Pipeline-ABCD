"""Streamlit chat UI for the University Services RAG assistant."""

from __future__ import annotations

import html

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="Hồ sơ Sinh viên",
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
.block-container { max-width: 1080px; padding-top: 2rem; }
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
  font: 700 .72rem/1.2 Consolas, monospace;
  letter-spacing: .13em;
  text-transform: uppercase;
}
.dossier-title {
  color: var(--navy);
  font: 700 clamp(2rem, 4vw, 3.35rem)/1.02 Georgia, serif;
  margin: .35rem 0 .55rem;
}
.dossier-subtitle { color: var(--muted); max-width: 720px; margin: 0; }
.status-strip {
  display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1rem;
  font: 600 .72rem/1.2 Consolas, monospace; color: var(--navy);
}
.status-strip span { border: 1px solid #BCCCDC; padding: .3rem .55rem; }
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
}
.source-label {
  font: 700 .69rem/1.2 Consolas, monospace;
  color: var(--seal); letter-spacing: .08em;
}
.source-title { color: var(--navy); font-weight: 700; margin: .2rem 0; }
.source-meta { color: var(--muted); font-size: .78rem; }
.source-excerpt { color: var(--ink); font-size: .86rem; margin-top: .35rem; }
.stTextInput input:focus, .stChatInput textarea:focus {
  border-color: var(--cobalt) !important;
  box-shadow: 0 0 0 2px rgba(36, 86, 166, .18) !important;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 700px) {
  .block-container { padding: 1rem .8rem 5rem; }
  .dossier-header { padding: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    if not sources:
        return
    st.caption(
        f"Sổ nguồn · {len(sources)} trích đoạn · phương thức {retrieval_source.upper()}"
    )
    for index, source in enumerate(sources, 1):
        metadata = source["metadata"]
        excerpt = " ".join(source["content"].split())[:280]
        source_name = metadata.get("source", "Không rõ nguồn")
        source_url = metadata.get("url")
        if source_url:
            source_line = (
                f'<a href="{html.escape(source_url)}" target="_blank">'
                f"{html.escape(source_name)}</a>"
            )
        else:
            source_line = html.escape(source_name)
        st.markdown(
            f"""
<div class="source-card">
  <div class="source-label">S{index} · {html.escape(source['retrieval_method'].upper())}
    · {source['score']:.4f}</div>
  <div class="source-title">{html.escape(metadata['title'])}</div>
  <div class="source-meta">{source_line} · đoạn {metadata['chunk_index']}</div>
  <div class="source-excerpt">{html.escape(excerpt)}…</div>
</div>
""",
            unsafe_allow_html=True,
        )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## HỒ SƠ SINH VIÊN")
    st.caption("Tra cứu quy chế · Ký túc xá · Thư viện · Học vụ")
    st.divider()
    top_k = st.slider("Số nguồn đối chiếu", 3, 10, 5)
    st.markdown(
        "**Cách hỏi hiệu quả**\n\n"
        "Nêu rõ trường và dịch vụ, ví dụ: *Sinh viên HCMUS cần điều kiện gì để "
        "được xét tốt nghiệp?*"
    )
    st.divider()
    st.caption("Câu trả lời chỉ dựa trên corpus và luôn kèm mã nguồn [S1], [S2]…")

st.markdown(
    """
<section class="dossier-header">
  <div class="dossier-kicker">Bộ tra cứu quy định đại học · Corpus nội bộ</div>
  <h1 class="dossier-title">Hỏi đúng hồ sơ.<br>Nhận đúng căn cứ.</h1>
  <p class="dossier-subtitle">Tra cứu quy định dành cho sinh viên từ tài liệu gốc,
  với từng câu trả lời được nối lại tới đoạn nguồn đã sử dụng.</p>
  <div class="status-strip"><span>05 VĂN BẢN</span><span>05 TRANG CÔNG KHAI</span><span>HYBRID + RRF</span></div>
</section>
""",
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []), message.get("retrieval_source", "none")
            )

query = st.chat_input("Nhập câu hỏi về học vụ, KTX hoặc thư viện…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang đối chiếu hồ sơ nguồn…"):
            try:
                result = generate_with_citation(query, top_k=top_k)
            except Exception as error:
                result = {
                    "answer": (
                        "Chưa thể mở chỉ mục. Hãy chạy `python -m "
                        "src.task4_chunking_indexing` rồi thử lại."
                    ),
                    "sources": [],
                    "retrieval_source": "none",
                }
                st.error(f"Chi tiết kỹ thuật: {error}")
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
