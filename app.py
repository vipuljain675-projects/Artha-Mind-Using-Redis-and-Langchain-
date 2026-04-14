"""
app.py — ArthaMind: AI Financial Report Analyst
Main Streamlit application with premium dark UI
"""

import os
import time
import json
import streamlit as st
from dotenv import load_dotenv

import uuid
from redis import Redis
from rq import Queue

from ingest import load_vector_store
from chain import build_agent_executor, generate_summary
from utils import (
    format_kpi_value,
    SAMPLE_QUESTIONS,
    get_model_options,
    make_kpi_gauge,
    make_bar_chart,
    highlight_numbers,
)

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArthaMind — AI Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Reset ── */
:root {
    --bg-primary: #020c12;
    --bg-card: #0a1628;
    --bg-surface: #0f1f35;
    --bg-hover: #152840;
    --accent-green: #10b981;
    --accent-green-bright: #34d399;
    --accent-gold: #f59e0b;
    --accent-blue: #3b82f6;
    --accent-red: #ef4444;
    --text-primary: #f0f9ff;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: #1e3a5f;
    --border-bright: #2d5a8e;
    --glow-green: 0 0 20px rgba(16,185,129,0.25);
    --glow-gold: 0 0 20px rgba(245,158,11,0.2);
}

* { font-family: 'Inter', sans-serif !important; }

/* ── App Background ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(16,185,129,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(59,130,246,0.04) 0%, transparent 50%) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050f1e 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }

/* ── Hero Header ── */
.arthamind-hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2744 50%, #0a1628 100%);
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(16,185,129,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
.arthamind-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(16,185,129,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.arthamind-hero::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, #10b981, #3b82f6, transparent);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0f9ff 30%, #10b981 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin: 0 0 20px 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
    margin-right: 8px;
    margin-bottom: 8px;
}
.badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    margin: 20px 0;
}
.kpi-card {
    background: linear-gradient(145deg, #0a1628, #0d1f3a);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease forwards;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
    opacity: 0.6;
}
.kpi-card:hover {
    border-color: rgba(16,185,129,0.4);
    box-shadow: var(--glow-green);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.2;
}
.kpi-delta {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 6px;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}
.kpi-delta.positive {
    color: #34d399;
    background: rgba(16, 185, 129, 0.12);
}
.kpi-delta.negative {
    color: #f87171;
    background: rgba(239, 68, 68, 0.12);
}
.kpi-delta.neutral {
    color: var(--text-secondary);
    background: rgba(148, 163, 184, 0.1);
}

/* ── Chat Interface ── */
.chat-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    margin: 20px 0 40px 0;
    max-height: 480px;
    overflow-y: auto;
    scroll-behavior: smooth;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

.message-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
    animation: slideInRight 0.3s ease;
}
.message-ai {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    animation: slideInLeft 0.3s ease;
}
.bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}
.bubble-ai {
    background: linear-gradient(135deg, #0a1f35, #0f2a45);
    border: 1px solid var(--border-bright);
    color: var(--text-primary);
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.avatar-ai {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #064e3b, #10b981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-right: 10px;
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(16,185,129,0.3);
}
.avatar-user {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-left: 10px;
    flex-shrink: 0;
}
.typing-indicator {
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
}
.typing-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: typing 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.icon-green { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.25); }
.icon-blue { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.25); }
.icon-gold { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.25); }

/* ── Quick Question Pills ── */
.q-pill {
    display: inline-block;
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #6ee7b7;
    cursor: pointer;
    margin: 4px;
    transition: all 0.2s ease;
}
.q-pill:hover {
    background: rgba(16,185,129,0.15);
    border-color: rgba(16,185,129,0.5);
    color: #34d399;
}

/* ── Executive Summary Content ── */
.summary-container {
    padding: 10px 0;
}
[data-testid="stMarkdownContainer"] h2 {
    color: var(--text-primary) !important;
    font-size: 1.15rem !important;
    margin-top: 28px !important;
    margin-bottom: 12px !important;
    border-bottom: 1px dashed var(--border);
    padding-bottom: 8px;
}
[data-testid="stMarkdownContainer"] ul {
    background: rgba(16,185,129,0.03);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 20px 20px 20px 40px !important;
}
[data-testid="stMarkdownContainer"] li {
    margin-bottom: 10px;
    color: var(--text-secondary);
    line-height: 1.6;
}
[data-testid="stMarkdownContainer"] strong {
    color: #f0f9ff !important;
}

/* ── Upload Zone ── */
.upload-zone {
    border: 2px dashed var(--border-bright);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    color: var(--text-secondary);
    background: rgba(16,185,129,0.02);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: var(--accent-green);
    background: rgba(16,185,129,0.05);
}

/* ── Streamlit Override ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    padding: 10px 16px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #065f46, #10b981) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16,185,129,0.4) !important;
}
.stSelectbox select, [data-baseweb="select"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="stMetricLabel"] p { color: var(--text-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* ── Status Banner ── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 18px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 12px 0;
}
.status-success {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    color: #34d399;
}
.status-warning {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fbbf24;
}
.status-error {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    color: #f87171;
}

/* ── Source Docs Expander ── */
.source-chip {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #93c5fd;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(16,185,129,0.03) !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-green) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 8px !important;
    padding: 0 !important;
    border-bottom: 2px solid var(--border) !important;
    margin-bottom: 15px !important;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-surface) !important;
    color: white !important;
    border-bottom: 2px solid var(--accent-green) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 10px !important; 
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],         # List of {role, content}
        "vectorstore": None,
        "agent_executor": None,
        "kpis": None,
        "doc_meta": None,
        "summary": None,
        "processing": False,
        "job_id": None,
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "selected_model": "llama-3.1-8b-instant",
        "active_tab": "chat",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 16px 0;">
        <div style="font-size:2.5rem;margin-bottom:8px;">📊</div>
        <div style="font-size:1.1rem;font-weight:700;color:#f0f9ff;">ArthaMind</div>
        <div style="font-size:0.75rem;color:#475569;margin-top:4px;">AI Financial Analyst</div>
    </div>
    <hr style="border-color:#1e3a5f;margin:0 0 20px 0;">
    """, unsafe_allow_html=True)

    # API Key
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Groq API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        label="",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        key="api_input",
        label_visibility="collapsed",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        st.markdown('<div class="status-banner status-success">✅ API key connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-warning">⚠️ Enter your Groq API key</div>', unsafe_allow_html=True)
        st.markdown(
            '<a href="https://console.groq.com" target="_blank" style="color:#10b981;font-size:0.8rem;">→ Get free key at console.groq.com</a>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">AI Model</div>', unsafe_allow_html=True)
    model_opts = get_model_options()
    chosen_model_label = st.selectbox(
        label="",
        options=list(model_opts.keys()),
        key="model_select",
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model_opts[chosen_model_label]

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload PDF
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Upload Report</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="",
        type=["pdf"],
        key="pdf_upload",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload annual reports, earnings PDFs, 10-K/10-Q filings",
    )

    # Status / Polling Block
    if st.session_state.job_id:
        try:
            r = Redis(host='localhost', port=6379)
            q = Queue(connection=r)
            job = q.fetch_job(st.session_state.job_id)

            if job and not job.is_finished and not job.is_failed:
                st.markdown(f'<div class="status-banner status-warning">⏳ Processing {job.kwargs.get("filename")} in background...</div>', unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()
            elif job and job.is_failed:
                st.markdown('<div class="status-banner status-error">❌ Background processing failed.</div>', unsafe_allow_html=True)
                st.session_state.job_id = None
            elif job and job.is_finished:
                # Job is done! Load results into session
                result = job.result
                if result.get("status") == "success":
                    st.session_state.doc_meta = {
                        "filenames": result["filenames"],
                        "total_pages": result["total_pages"],
                        "total_chunks": result["total_chunks"]
                    }
                    st.session_state.kpis = result["kpis"]
                    st.session_state.vectorstore = load_vector_store("financial_index")
                    
                    # Init Chain
                    st.session_state.agent_executor = build_agent_executor(
                        st.session_state.vectorstore,
                        st.session_state.api_key,
                        model=st.session_state.selected_model,
                    )
                    st.session_state.chat_history = []
                    st.session_state.summary = None
                    st.session_state.job_id = None
                    
                    st.markdown('<div class="status-banner status-success">✅ Report processed successfully!</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="status-banner status-error">❌ {result.get("error")}</div>', unsafe_allow_html=True)
                    st.session_state.job_id = None
        except Exception as e:
             st.markdown(f'<div class="status-banner status-error">⚠️ Redis Connection Error: {str(e)}</div>', unsafe_allow_html=True)
             st.session_state.job_id = None

    # Process button
    if uploaded_files and len(uploaded_files) > 0 and st.session_state.api_key and not st.session_state.job_id:
        if st.button("🚀 Analyze Async", use_container_width=True):
            with st.spinner("Queueing…"):
                # Save uploaded files to temp disk
                tmp_dir = os.path.join(os.path.dirname(__file__), "reports")
                os.makedirs(tmp_dir, exist_ok=True)
                file_payloads = []
                for f in uploaded_files:
                    tmp_path = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.pdf")
                    with open(tmp_path, "wb") as w:
                        w.write(f.read())
                    file_payloads.append({"file_path": tmp_path, "filename": f.name})
                    
                # Enqueue the job
                r = Redis(host='localhost', port=6379)
                q = Queue(connection=r)
                from worker import async_ingest_and_extract
                job = q.enqueue(
                    async_ingest_and_extract,
                    files=file_payloads,
                    api_key=st.session_state.api_key,
                    job_timeout='10m'
                )
                
                # Store job ID in session state so UI loop picks it up
                st.session_state.job_id = job.id
            st.rerun()
    elif uploaded_files and not st.session_state.api_key:
        st.markdown('<div class="status-banner status-warning">⚠️ Add API key first</div>', unsafe_allow_html=True)

    # Doc info
    if st.session_state.doc_meta:
        meta = st.session_state.doc_meta
        filenames_display = "<br>".join(meta['filenames'])
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);border-radius:12px;padding:14px;margin-top:16px;">
            <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;margin-bottom:8px;">Active Documents</div>
            <div style="font-size:0.82rem;color:#94a3b8;word-break:break-all;">{filenames_display}</div>
            <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
                <span style="background:rgba(16,185,129,0.1);border-radius:6px;padding:2px 8px;font-size:0.72rem;color:#34d399;">
                    📄 {meta['total_pages']} pages
                </span>
                <span style="background:rgba(59,130,246,0.1);border-radius:6px;padding:2px 8px;font-size:0.72rem;color:#93c5fd;">
                    🧩 {meta['total_chunks']} chunks
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.agent_executor = build_agent_executor(
                st.session_state.vectorstore,
                st.session_state.api_key,
                model=st.session_state.selected_model,
            )
            st.rerun()

    # Footer
    st.markdown("""
    <div style="margin-top:40px;text-align:center;padding:10px 0;">
        <div style="font-size:0.72rem;color:#334155;">Powered by LangChain + Groq + Llama 3.1</div>
        <div style="font-size:0.68rem;color:#1e3a5f;margin-top:2px;">Built with ❤️ for financial intelligence</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="arthamind-hero">
    <div class="hero-title">📊 ArthaMind</div>
    <div class="hero-subtitle">AI-Powered Financial Report Analyst — Instant insights from any financial document</div>
    <span class="hero-badge"><span class="badge-dot"></span>Llama 3 Powered</span>
    <span class="hero-badge" style="color:#93c5fd;background:rgba(59,130,246,0.1);border-color:rgba(59,130,246,0.25);">
        <span class="badge-dot" style="background:#3b82f6;"></span>RAG Architecture
    </span>
    <span class="hero-badge" style="color:#fbbf24;background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.25);">
        <span class="badge-dot" style="background:#f59e0b;"></span>Real-time Analysis
    </span>
</div>
""", unsafe_allow_html=True)

# No document uploaded state
if not st.session_state.vectorstore:
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown("""
        <div class="kpi-card" style="animation-delay:0.1s;">
            <div style="font-size:2rem;margin-bottom:12px;">📄</div>
            <div style="font-size:1rem;font-weight:700;color:#f0f9ff;margin-bottom:8px;">Step 1 — Upload PDF</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.5;">Upload any financial report — annual reports, earnings releases, 10-K/10-Q filings</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="kpi-card" style="animation-delay:0.2s;">
            <div style="font-size:2rem;margin-bottom:12px;">🔑</div>
            <div style="font-size:1rem;font-weight:700;color:#f0f9ff;margin-bottom:8px;">Step 2 — Add API Key</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.5;">Get your free Groq API key at console.groq.com — runs Llama 3 at 500+ tokens/sec</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="kpi-card" style="animation-delay:0.3s;">
            <div style="font-size:2rem;margin-bottom:12px;">💬</div>
            <div style="font-size:1rem;font-weight:700;color:#f0f9ff;margin-bottom:8px;">Step 3 — Ask Anything</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.5;">Chat about revenue, margins, risks, debt, growth outlook — get analyst-grade answers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:40px;padding:40px;border:1px dashed #1e3a5f;border-radius:20px;background:rgba(16,185,129,0.02);">
        <div style="font-size:3rem;margin-bottom:12px;">📊</div>
        <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-bottom:8px;">No report loaded yet</div>
        <div style="font-size:0.85rem;color:#334155;">Upload a PDF report from the sidebar to begin analysis</div>
        <div style="margin-top:16px;font-size:0.78rem;color:#1e3a5f;">
            Supports: Annual Reports • Earnings Releases • 10-K • 10-Q • Investor Presentations
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Document loaded — show full UI
    
    # ── KPI Dashboard ──────────────────────────────────────────────────────
    if st.session_state.kpis:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon icon-green">📈</div>
            <h2>Key Financial Metrics</h2>
        </div>
        """, unsafe_allow_html=True)

        # Multi-document KPI dropdown logic
        if isinstance(st.session_state.kpis, dict) and not "company_name" in st.session_state.kpis:
            # It's a dict mapping filenames to KPIs
            available_files = list(st.session_state.kpis.keys())
            if available_files:
                selected_file_for_kpi = st.selectbox("Select document KPIs to view:", available_files, key="kpi_selector")
                kpis = st.session_state.kpis[selected_file_for_kpi]
            else:
                kpis = {}
        else:
            kpis = st.session_state.kpis

        if kpis and "error" not in kpis:
            company = kpis.get("company_name", "Company")
            period = kpis.get("report_period", "")

            if company:
                st.markdown(f"""
                <div style="font-size:1.4rem;font-weight:700;color:#f0f9ff;margin-bottom:4px;">{company}</div>
                <div style="font-size:0.85rem;color:#475569;margin-bottom:20px;">{period}</div>
                """, unsafe_allow_html=True)

            # KPI Cards
            kpi_items = [
                ("💰 Gross Revenue", kpis.get("revenue"), kpis.get("revenue_growth")),
                ("📊 Net Income", kpis.get("net_income"), kpis.get("net_margin")),
                ("💼 EBITDA", kpis.get("ebitda"), None),
                ("📈 EPS", kpis.get("eps"), None),
                ("💧 Cash Flow", kpis.get("operating_cash_flow"), None),
                ("🏦 Total Assets", kpis.get("total_assets"), None),
                ("📉 Total Debt", kpis.get("total_debt"), kpis.get("debt_to_equity")),
                ("💹 ROE", kpis.get("roe"), None),
            ]

            kpi_html = '<div class="kpi-grid">'
            for label, value, delta in kpi_items:
                val_str = format_kpi_value(value)
                delta_html = ""
                if delta and delta != "N/A" and delta != "null":
                    is_pos = "+" in str(delta) or (not str(delta).startswith("-"))
                    cls = "positive" if is_pos else "negative"
                    delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'

                kpi_html += f"""<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val_str}</div>{delta_html}</div>"""

            kpi_html += "</div>"
            st.markdown(kpi_html, unsafe_allow_html=True)

            # Highlights & Risks row
            high = kpis.get("key_highlight")
            risk = kpis.get("risk_flag")
            if high or risk:
                c1, c2 = st.columns(2)
                if high:
                    with c1:
                        st.markdown(f"""
                        <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.2);
                                    border-radius:14px;padding:18px;height:100%;">
                            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                                        letter-spacing:0.1em;color:#10b981;margin-bottom:8px;">✅ KEY HIGHLIGHT</div>
                            <div style="font-size:0.88rem;color:#94a3b8;line-height:1.6;">{high}</div>
                        </div>
                        """, unsafe_allow_html=True)
                if risk:
                    with c2:
                        st.markdown(f"""
                        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.18);
                                    border-radius:14px;padding:18px;height:100%;">
                            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                                        letter-spacing:0.1em;color:#ef4444;margin-bottom:8px;">⚠️ RISK TO WATCH</div>
                            <div style="font-size:0.88rem;color:#94a3b8;line-height:1.6;">{risk}</div>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: Chat | Summary ───────────────────────────────────────────────
    tab_chat, tab_summary = st.tabs(["💬  Analyst Chat", "📋  Executive Summary"])

    # ── CHAT TAB ──────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon icon-blue">💬</div>
            <h2>Chat with the Report</h2>
        </div>
        """, unsafe_allow_html=True)

        # Quick question pills
        if not st.session_state.chat_history:
            st.markdown('<div style="margin-bottom:12px;font-size:0.8rem;color:#475569;font-weight:600;">💡 Try asking:</div>', unsafe_allow_html=True)
            pills_html = ""
            for q in SAMPLE_QUESTIONS[:6]:
                pills_html += f'<span class="q-pill">{q}</span>'
            st.markdown(f'<div>{pills_html}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # Chat messages
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container" id="chat-box">'
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="message-user">
                        <div class="bubble-user">{msg['content']}</div>
                        <div class="avatar-user">👤</div>
                    </div>"""
                else:
                    content = msg["content"].replace("\n", "<br>")
                    chat_html += f"""
                    <div class="message-ai">
                        <div class="avatar-ai">🤖</div>
                        <div class="bubble-ai">{content}</div>
                    </div>"""
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        # Input row
        col_input, col_btn = st.columns([5, 1])
        
        # Use a key stored in session state to clear input after send
        if "input_key" not in st.session_state:
            st.session_state.input_key = 0
            
        with col_input:
            user_question = st.text_input(
                label="",
                placeholder="Ask about revenue, profit margins, risks, strategy",
                key=f"user_question_{st.session_state.input_key}",
                label_visibility="collapsed",
            )
        with col_btn:
            send_btn = st.button("Send", use_container_width=True)

        # Only trigger on explicit button click with non-empty input
        if send_btn and user_question and user_question.strip() and st.session_state.agent_executor:
            question = user_question.strip()
            # Rotate the key to clear the input box
            st.session_state.input_key += 1
            
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Agent is analyzing and searching..."):
                try:
                    result = st.session_state.agent_executor.invoke({"input": question})
                    answer = result.get("output", "I couldn't find a response.")
                except Exception as e:
                    answer = f"Error: {str(e)}"

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        # Sample question buttons
        if not st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.8rem;color:#334155;margin-bottom:8px;">Click to ask:</div>', unsafe_allow_html=True)
            q_cols = st.columns(2)
            for i, q in enumerate(SAMPLE_QUESTIONS[:4]):
                with q_cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        with st.spinner("Agent is analyzing and searching..."):
                            try:
                                result = st.session_state.agent_executor.invoke({"input": q})
                                answer = result.get("output", "No response.")
                            except Exception as e:
                                answer = f"Error: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()

    # ── SUMMARY TAB ───────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon icon-gold">📋</div>
            <h2>Executive Summary</h2>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.summary:
            st.markdown("""
            <div style="text-align:center;padding:32px;color:#334155;">
                <div style="font-size:2rem;margin-bottom:10px;">🔍</div>
                <div style="font-size:0.9rem;">Click below to generate a comprehensive executive summary</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📝 Generate Executive Summary", use_container_width=True):
                with st.spinner("Drafting executive summary…"):
                    # Ensure company name is safely fetched from multi-document dict
                    active_company = "the company"
                    if isinstance(st.session_state.kpis, dict):
                        if "company_name" in st.session_state.kpis:
                            active_company = st.session_state.kpis.get("company_name", "the company")
                        elif list(st.session_state.kpis.values()):
                            active_company = list(st.session_state.kpis.values())[0].get("company_name", "the company")

                    summary = generate_summary(
                        st.session_state.vectorstore,
                        st.session_state.api_key,
                        company_name=active_company
                    )
                    st.session_state.summary = summary
                st.rerun()
        else:
            with st.container():
                st.markdown('<div class="summary-container">', unsafe_allow_html=True)
                st.markdown(st.session_state.summary)
                st.markdown('</div>', unsafe_allow_html=True)
                
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Regenerate Summary", use_container_width=True):
                    st.session_state.summary = None
                    st.rerun()
            with c2:
                st.download_button(
                    label="⬇️ Download Summary",
                    data=st.session_state.summary,
                    file_name=f"summary_{st.session_state.doc_meta['filenames'][0].replace('.pdf','')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
