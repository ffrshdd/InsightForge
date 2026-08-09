import os
import datetime
import shutil
from pathlib import Path

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.ai_helper import generate_summary, gemini_connected
from utils.analytics import get_document_stats, get_file_type_chart_data, get_total_chunks
from utils.document_loader import extract_text
from utils.embeddings import load_vectorstore, save_vectorstore

load_dotenv()

st.set_page_config(
    page_title="InsightForge",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

USER_NAME = "Farshad"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

body { font-family: 'Google Sans', 'Segoe UI', sans-serif; }

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

.stButton > button {
    width: 100%;
    border-radius: 20px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    background: #4F46E5;
    color: white;
    transition: 0.2s ease;
    box-shadow: 0 2px 8px rgba(79,70,229,0.3);
}

.stButton > button:hover {
    background: #6366F1;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(79,70,229,0.4);
}

.m-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: 0.2s ease;
}

.m-card:hover {
    border-color: #4F46E5;
    box-shadow: 0 6px 24px rgba(79,70,229,0.15);
    transform: translateY(-1px);
}

.m-card-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.m-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}

.m-card-sub {
    font-size: 0.78rem;
    color: #6e7681;
    margin-top: 6px;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0;
    line-height: 1.1;
}

.app-subtitle {
    color: #8b949e;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}

.greeting {
    font-size: 1rem;
    color: #8b949e;
    margin-bottom: 1.5rem;
}

.chat-user {
    background: #1f2937;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e6edf3;
    font-size: 0.92rem;
    max-width: 85%;
    margin-left: auto;
    border: 1px solid #374151;
}

.chat-ai {
    background: #161b22;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e6edf3;
    font-size: 0.92rem;
    max-width: 90%;
    border: 1px solid #30363d;
}

.chat-label-user {
    font-size: 0.72rem;
    color: #6e7681;
    text-align: right;
    margin-bottom: 2px;
}

.chat-label-ai {
    font-size: 0.72rem;
    color: #4F46E5;
    margin-bottom: 2px;
    font-weight: 600;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.2rem 0;
}

.source-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

.source-filename {
    font-size: 0.8rem;
    color: #4F46E5;
    font-weight: 600;
    margin-bottom: 4px;
}

.source-chunk {
    font-size: 0.76rem;
    color: #8b949e;
    line-height: 1.5;
}

.empty-state {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    color: #8b949e;
}

.empty-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 10px;
}

.empty-body {
    font-size: 0.9rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


def get_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12:
        return f"Good Morning, {USER_NAME} 👋"
    elif hour < 17:
        return f"Good Afternoon, {USER_NAME} 👋"
    else:
        return f"Good Evening, {USER_NAME} 👋"


def get_file_size_str(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{round(size/1024, 1)} KB"
    else:
        return f"{round(size/(1024*1024), 2)} MB"


def rebuild_vectorstore():
    data_files = list(Path("data").glob("*.*"))
    if not data_files:
        vs_path = Path("vectorstore")
        if vs_path.exists():
            shutil.rmtree(vs_path)
            vs_path.mkdir()
        return

    all_chunks = []
    all_metadatas = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    for file_path in data_files:
        try:
            class FakeFile:
                def __init__(self, p):
                    self.name = p.name
                    self._bytes = p.read_bytes()
                def getvalue(self):
                    return self._bytes

            text = extract_text(FakeFile(file_path), file_path)
            if text.strip():
                chunks = splitter.split_text(text)
                all_chunks.extend(chunks)
                all_metadatas.extend([{"filename": file_path.name}] * len(chunks))
        except Exception:
            pass

    if all_chunks:
        save_vectorstore(all_chunks, metadatas=all_metadatas)


def export_txt(question, answer, sources):
    lines = [
        "InsightForge Export",
        "=" * 40,
        f"Question: {question}",
        "",
        "Answer:",
        answer,
        "",
        "Sources:",
    ]
    for src in sources:
        lines.append(f"- [{src['filename']}] {src['chunk'][:150]}...")
    return "\n".join(lines)


def export_md(question, answer, sources):
    lines = [
        "# InsightForge Export",
        "",
        f"**Question:** {question}",
        "",
        "## Answer",
        "",
        answer,
        "",
        "## Sources",
        "",
    ]
    for src in sources:
        lines.append(f"**📄 {src['filename']}**")
        lines.append(f"> {src['chunk'][:150]}...")
        lines.append("")
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    if gemini_connected():
        st.success("✅ Gemini AI Connected")
    else:
        st.error("❌ GEMINI_API_KEY not found")

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD, JSON, CSV",
        type=["pdf", "docx", "txt", "md", "json", "csv"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Process & Index"):
        if uploaded_files:
            with st.spinner("Processing..."):
                all_chunks = []
                all_metadatas = []
                splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

                for file in uploaded_files:
                    save_path = Path("data") / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    try:
                        text = extract_text(file, save_path)
                        if text.strip():
                            chunks = splitter.split_text(text)
                            all_chunks.extend(chunks)
                            all_metadatas.extend(
                                [{"filename": file.name}] * len(chunks)
                            )
                    except Exception as e:
                        st.warning(f"Skipped {file.name}: {e}")

                if all_chunks:
                    save_vectorstore(all_chunks, metadatas=all_metadatas)
                    st.success(f"✅ {len(uploaded_files)} file(s) indexed!")
                else:
                    st.error("No text found in uploaded files.")
        else:
            st.error("Upload files first.")

    st.markdown("---")
    st.markdown("### 📂 Documents")
    data_files = list(Path("data").glob("*.*"))
    if data_files:
        for f in data_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 **{f.name}**")
                st.caption(get_file_size_str(f))
            with col2:
                if st.button("🗑", key=f"del_{f.name}"):
                    f.unlink()
                    rebuild_vectorstore()
                    st.rerun()
    else:
        st.caption("No documents yet.")

    st.markdown("---")
    st.markdown("### Workspace")
    tab = st.radio("", ["🏠 Home", "🔍 Smart Search", "📊 Analytics"])


# ── Home ─────────────────────────────────────────────────
if tab == "🏠 Home":
    st.markdown('<div class="app-title">🔨 InsightForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Personal AI Knowledge Forge • Built by Farshad S, Tamil Nadu 🔥</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="greeting">{get_greeting()}</div>', unsafe_allow_html=True)

    stats = get_document_stats()
    index_status = get_total_chunks()

    if stats["documents"] == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">👋 Welcome to InsightForge</div>
            <div class="empty-body">
                Upload your first document to build your personal knowledge base.<br><br>
                <strong>Supported formats:</strong><br>
                PDF &bull; DOCX &bull; TXT &bull; Markdown &bull; JSON &bull; CSV<br><br>
                <strong>You can ask questions like:</strong><br>
                &bull; Summarize my resume<br>
                &bull; Explain this research paper<br>
                &bull; Find my internship experience<br>
                &bull; What are the key points in this document?
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">📚 Documents</div>
                <div class="m-card-value">{stats['documents']}</div>
                <div class="m-card-sub">in knowledge base</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">💾 Storage</div>
                <div class="m-card-value">{stats['storage_mb']}</div>
                <div class="m-card-sub">MB used</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">🧠 Index</div>
                <div class="m-card-value" style="font-size:1.2rem;padding-top:6px;">{index_status}</div>
                <div class="m-card-sub">vector store</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            ai_status = "Ready" if gemini_connected() else "No Key"
            ai_color = "#4ade80" if gemini_connected() else "#f87171"
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">🤖 AI</div>
                <div class="m-card-value" style="font-size:1.2rem;padding-top:6px;color:{ai_color};">{ai_status}</div>
                <div class="m-card-sub">Gemini 2.5 Flash</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">🕒 Recent Upload</div>', unsafe_allow_html=True)
            st.info(stats["recent_upload"])
        with col_b:
            st.markdown('<div class="section-title">📄 File Types</div>', unsafe_allow_html=True)
            for ftype, count in stats["file_types"].items():
                st.markdown(f"**{ftype}** — {count} file(s)")

        if st.session_state.search_history:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔍 Recent Searches</div>', unsafe_allow_html=True)
            for q in st.session_state.search_history[-5:][::-1]:
                st.caption(f"• {q}")


# ── Smart Search ──────────────────────────────────────────
elif tab == "🔍 Smart Search":
    st.markdown('<div class="app-title">🔍 Smart Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Ask anything about your documents</div>', unsafe_allow_html=True)

    query = st.text_input("Your question", placeholder="e.g. Summarize my resume")

    if st.button("🔎 Ask InsightForge"):
        if query:
            with st.spinner("📄 Reading documents..."):
                vectorstore = load_vectorstore()
            if vectorstore:
                with st.spinner("🧠 Creating context..."):
                    docs = vectorstore.similarity_search(query, k=5)
                    context = "\n\n".join([doc.page_content for doc in docs])
                with st.spinner("✨ Generating answer..."):
                    answer = generate_summary(context, query)

                sources = []
                for doc in docs:
                    filename = doc.metadata.get("filename", "Unknown")
                    sources.append({
                        "filename": filename,
                        "chunk": doc.page_content,
                    })

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": sources,
                })

                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
            else:
                st.warning("Upload and index documents first.")
        else:
            st.warning("Enter a question.")

    if st.session_state.chat_history:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💬 Conversation</div>', unsafe_allow_html=True)

        for item in st.session_state.chat_history[::-1]:
            st.markdown('<div class="chat-label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-user">{item["question"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-label-ai">🔨 InsightForge</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-ai">{item["answer"]}</div>', unsafe_allow_html=True)

            with st.expander("📄 Sources"):
                for src in item["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-filename">📄 {src['filename']}</div>
                        <div class="source-chunk">{src['chunk'][:200]}...</div>
                    </div>""", unsafe_allow_html=True)

            col_txt, col_md = st.columns(2)
            with col_txt:
                st.download_button(
                    "⬇ Download TXT",
                    data=export_txt(item["question"], item["answer"], item["sources"]),
                    file_name="insightforge_answer.txt",
                    mime="text/plain",
                    key=f"txt_{item['question'][:20]}",
                )
            with col_md:
                st.download_button(
                    "⬇ Download Markdown",
                    data=export_md(item["question"], item["answer"], item["sources"]),
                    file_name="insightforge_answer.md",
                    mime="text/markdown",
                    key=f"md_{item['question'][:20]}",
                )

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if st.button("🗑 Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ── Analytics ─────────────────────────────────────────────
elif tab == "📊 Analytics":
    st.markdown('<div class="app-title">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your knowledge base at a glance</div>', unsafe_allow_html=True)

    stats = get_document_stats()
    index_status = get_total_chunks()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="m-card">
            <div class="m-card-label">📚 Total Documents</div>
            <div class="m-card-value">{stats['documents']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="m-card">
            <div class="m-card-label">💾 Total Storage</div>
            <div class="m-card-value">{stats['storage_mb']}</div>
            <div class="m-card-sub">MB</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="m-card">
            <div class="m-card-label">🧠 Index Status</div>
            <div class="m-card-value" style="font-size:1.1rem;padding-top:8px;">{index_status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    labels, values = get_file_type_chart_data()
    if labels:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(
                names=labels,
                values=values,
                title="File Types",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Purples_r,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_bar = px.bar(
                x=labels,
                y=values,
                title="Document Count by Type",
                labels={"x": "Type", "y": "Count"},
                color_discrete_sequence=["#4F46E5"],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Details</div>', unsafe_allow_html=True)

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">🕒 Most Recent</div>
                <div class="m-card-value" style="font-size:1rem;padding-top:6px;">{stats['recent_upload']}</div>
            </div>""", unsafe_allow_html=True)
        with col_y:
            most_common = max(stats["file_types"], key=stats["file_types"].get) if stats["file_types"] else "—"
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-label">📄 Most Common Type</div>
                <div class="m-card-value" style="font-size:1rem;padding-top:6px;">{most_common}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Upload documents to see analytics.")


st.markdown("---")
st.markdown("**InsightForge v1.3 Stable** • Made in Tamil Nadu 🔥")