"""
app.py — ArthaMind: AI Financial Report Analyst
Main Streamlit application with premium dark UI
"""

import html
import os
import time
import json
import traceback
import streamlit as st
from dotenv import load_dotenv

import uuid

# Redis/RQ are optional — app works without them (sync fallback)
try:
    from redis import Redis
    from rq import Queue
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from ingest import extract_raw_text, load_vector_store, build_peer_vectorstore
from chain import (
    build_agent_executor,
    extract_kpis_with_llm,
    generate_summary,
    normalize_answer_provider,
    resolve_agent_model,
    run_agent_turn,
    run_peer_comparison,
    fetch_commodity_live_price,
    COMMODITY_SEARCH_QUERIES,
)
from utils import (
    format_kpi_value,
    SAMPLE_QUESTIONS,
    get_answer_provider_options,
    get_default_model,
    get_model_options,
    make_kpi_gauge,
    make_bar_chart,
    highlight_numbers,
)

load_dotenv()

def _secret(key: str, default: str = "") -> str:
    """Read from os.getenv (for local .env and Render deployment)."""
    return os.getenv(key, default)

LIVE_SEARCH_OPTIONS = {
    "🌐 OpenAI Web Search": "openai",
    "🔎 Gemini Google Search": "gemini",
    "🪶 DuckDuckGo Fallback": "duckduckgo",
}

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

/* ── Sidebar toggle button — always visible ── */
[data-testid="collapsedControl"] {
    background: rgba(16,185,129,0.15) !important;
    border: 1px solid rgba(16,185,129,0.4) !important;
    border-radius: 0 8px 8px 0 !important;
    color: #10b981 !important;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #10b981 !important;
    color: #10b981 !important;
}
/* Override any hide rules on the sidebar collapse button */
section[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

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
    default_answer_provider = os.getenv("ANSWER_PROVIDER")
    if not default_answer_provider:
        if os.getenv("GEMINI_API_KEY"):
            default_answer_provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            default_answer_provider = "openai"
        else:
            default_answer_provider = "groq"
    default_answer_provider = normalize_answer_provider(default_answer_provider)

    default_live_provider = os.getenv("LIVE_SEARCH_PROVIDER")
    if not default_live_provider:
        if os.getenv("OPENAI_API_KEY"):
            default_live_provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            default_live_provider = "gemini"
        else:
            default_live_provider = "duckduckgo"

    defaults = {
        "chat_history": [],         # List of {role, content}
        "vectorstore": None,
        "agent_executor": None,
        "kpis": None,
        "doc_meta": None,
        "summary": None,
        "processing": False,
        "job_id": None,
        "api_key": _secret("GROQ_API_KEY"),
        "openai_api_key": _secret("OPENAI_API_KEY"),
        "gemini_api_key": _secret("GEMINI_API_KEY"),
        "gemini_web_api_key": _secret("GEMINI_WEB_API_KEY"),
        "answer_provider": default_answer_provider,
        "selected_model": get_default_model(default_answer_provider),
        "live_search_provider": default_live_provider,
        "openai_web_model": _secret("OPENAI_WEB_SEARCH_MODEL", "gpt-4.1-mini"),
        "gemini_web_model": _secret("GEMINI_WEB_SEARCH_MODEL", "gemini-2.5-flash"),
        "active_tab": "chat",
        "active_file": None,
        "input_key": 0,
        "agent_requested_model": None,
        "agent_active_file": None,
        "agent_effective_model": None,
        "agent_notice": None,
        "agent_answer_provider": None,
        "agent_live_search_provider": None,
        "agent_openai_api_key": None,
        "agent_gemini_api_key": None,
        "kpi_notice": None,
        # Peer Compare
        "peer_vectorstores": {},        # {filename: FAISS}
        "peer_kpis": {},                # {filename: kpi_dict}
        "peer_chat_history": [],
        "peer_chat_mode": "primary",    # "primary" | "all"
        # Commodity Radar
        "commodity_alerts": [],         # list of alert dicts
        "commodity_prices": {},         # {commodity: {price, raw_text, error}}
        "commodity_last_checked": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def is_valid_kpi_payload(payload) -> bool:
    return isinstance(payload, dict) and bool(payload) and "error" not in payload


def gemini_web_key() -> str:
    """Return the dedicated Gemini live-search key if set, else fall back to the main key."""
    return st.session_state.get("gemini_web_api_key") or st.session_state.get("gemini_api_key", "")



def merge_kpi_results(previous_kpis, new_kpis):
    """Preserve the last good KPI snapshot if a re-run extraction fails."""
    if not isinstance(new_kpis, dict):
        return new_kpis, None

    previous_map = previous_kpis if isinstance(previous_kpis, dict) else {}
    merged = {}
    notices = []

    for filename, payload in new_kpis.items():
        previous_payload = previous_map.get(filename)
        if is_valid_kpi_payload(payload):
            merged[filename] = payload
        elif is_valid_kpi_payload(previous_payload):
            merged[filename] = previous_payload
            notices.append(
                f"KPI extraction failed for {filename}, so ArthaMind kept the previous successful KPI snapshot."
            )
        else:
            merged[filename] = payload
            notices.append(
                f"KPI extraction failed for {filename}. Chat and summary can still use the report, but KPI cards may be unavailable."
            )

    return merged, " ".join(notices) if notices else None


def recover_missing_kpis(kpi_map):
    """Try a local KPI recovery pass when the worker returns an error payload."""
    if not isinstance(kpi_map, dict):
        return kpi_map, None

    recovered = dict(kpi_map)
    notices = []

    for filename, payload in kpi_map.items():
        if is_valid_kpi_payload(payload):
            continue

        local_path = os.path.join(os.path.dirname(__file__), "reports", filename)
        if not os.path.exists(local_path) or not st.session_state.api_key:
            continue

        try:
            with open(local_path, "rb") as report_file:
                raw_text = extract_raw_text(report_file.read(), filename, max_pages=10)
            recovered_payload = extract_kpis_with_llm(raw_text, st.session_state.api_key)
            if is_valid_kpi_payload(recovered_payload):
                recovered[filename] = recovered_payload
                notices.append(f"Recovered KPI cards locally for {filename}.")
        except Exception as exc:
            notices.append(f"Local KPI recovery failed for {filename}: {exc}")

    return recovered, " ".join(notices) if notices else None


def sync_agent_executor(force: bool = False) -> None:
    """Keep the tool-calling chat agent aligned with the selected model and document."""
    if not st.session_state.vectorstore:
        return

    if st.session_state.answer_provider != "groq":
        st.session_state.agent_executor = None
        st.session_state.agent_requested_model = st.session_state.selected_model
        st.session_state.agent_active_file = st.session_state.get("active_file")
        st.session_state.agent_effective_model = st.session_state.selected_model
        st.session_state.agent_answer_provider = st.session_state.answer_provider
        st.session_state.agent_notice = (
            f"Analyst Chat is powered by `{st.session_state.selected_model}` via "
            f"{st.session_state.answer_provider.capitalize()}."
        )
        return

    if not st.session_state.api_key:
        return

    active_filename = st.session_state.get("active_file")
    current_agent = st.session_state.get("agent_executor")
    needs_rebuild = force or current_agent is None

    if current_agent is not None and not needs_rebuild:
        needs_rebuild = any((
            st.session_state.get("agent_answer_provider") != st.session_state.answer_provider,
            st.session_state.get("agent_requested_model") != st.session_state.selected_model,
            st.session_state.get("agent_active_file") != active_filename,
            st.session_state.get("agent_live_search_provider") != st.session_state.live_search_provider,
            st.session_state.get("agent_openai_api_key") != st.session_state.openai_api_key,
            st.session_state.get("agent_gemini_api_key") != st.session_state.gemini_api_key,
        ))

    if needs_rebuild:
        effective_model, notice = resolve_agent_model(st.session_state.selected_model)
        st.session_state.agent_executor = build_agent_executor(
            st.session_state.vectorstore,
            st.session_state.api_key,
            model=st.session_state.selected_model,
            active_filename=active_filename,
            live_search_provider=st.session_state.live_search_provider,
            openai_api_key=st.session_state.openai_api_key,
            gemini_api_key=gemini_web_key(),
            openai_web_model=st.session_state.openai_web_model,
            gemini_web_model=st.session_state.gemini_web_model,
        )
        st.session_state.agent_answer_provider = st.session_state.answer_provider
        st.session_state.agent_requested_model = st.session_state.selected_model
        st.session_state.agent_active_file = active_filename
        st.session_state.agent_effective_model = effective_model
        st.session_state.agent_notice = notice
        st.session_state.agent_live_search_provider = st.session_state.live_search_provider
        st.session_state.agent_openai_api_key = st.session_state.openai_api_key
        st.session_state.agent_gemini_api_key = st.session_state.gemini_api_key


def submit_chat_question(question: str) -> None:
    """Run one chat turn with graceful model fallback handling."""
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Agent is analyzing and searching..."):
        try:
            sync_agent_executor()
            result = run_agent_turn(
                agent_executor=st.session_state.agent_executor,
                question=question,
                vectorstore=st.session_state.vectorstore,
                api_key=st.session_state.api_key,
                requested_model=st.session_state.selected_model,
                answer_provider=st.session_state.answer_provider,
                active_filename=st.session_state.get("active_file"),
                live_search_provider=st.session_state.live_search_provider,
                openai_api_key=st.session_state.openai_api_key,
                gemini_api_key=gemini_web_key(),
                openai_web_model=st.session_state.openai_web_model,
                gemini_web_model=st.session_state.gemini_web_model,
                chat_history=st.session_state.chat_history[:-1],
            )
            st.session_state.agent_executor = result["agent_executor"]
            st.session_state.agent_answer_provider = st.session_state.answer_provider
            st.session_state.agent_requested_model = st.session_state.selected_model
            st.session_state.agent_active_file = st.session_state.get("active_file")
            st.session_state.agent_effective_model = result.get("effective_model")
            st.session_state.agent_notice = result.get("notice")
            st.session_state.agent_live_search_provider = st.session_state.live_search_provider
            st.session_state.agent_openai_api_key = st.session_state.openai_api_key
            st.session_state.agent_gemini_api_key = st.session_state.gemini_api_key
            answer = result["answer"]
        except Exception as exc:
            traceback.print_exc()
            error_str = str(exc)
            if "503" in error_str or "high demand" in error_str.lower():
                answer = (
                    "**Gemini is currently experiencing high demand (503).**\n\n"
                    "I've attempted several retries, but the service is still unavailable. "
                    "I've switched ArthaMind to use our fallback logic, but for the most stable experience right now, "
                    "please consider switching the **Answer Engine** to **Groq** or **OpenAI** in the sidebar."
                )
            elif "401" in error_str or "invalid_api_key" in error_str.lower():
                answer = "It looks like the API key provided is invalid or expired. Please check your settings in the sidebar."
            else:
                answer = (
                    "I hit an unexpected issue while analyzing the report. "
                    "This can sometimes happen during high traffic or if the document structure is highly atypical. "
                    "Please try your question again, or switch the **Answer Engine/Model** in the sidebar."
                )


    st.session_state.chat_history.append({"role": "assistant", "content": answer})


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

    # KPI / Ingestion Key
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Groq API Key (KPI Cards)</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        label="Groq API Key (KPI Cards)",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        key="api_input",
        label_visibility="collapsed",
    )
    st.session_state.api_key = api_key_input

    if st.session_state.api_key:
        st.markdown('<div class="status-banner status-success">✅ Groq KPI key connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-warning">⚠️ Add Groq key for KPI extraction and Analyze Async</div>', unsafe_allow_html=True)
        st.markdown(
            '<a href="https://console.groq.com" target="_blank" style="color:#10b981;font-size:0.8rem;">→ Get free key at console.groq.com</a>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Answer Engine</div>', unsafe_allow_html=True)
    provider_opts = get_answer_provider_options()
    provider_labels = list(provider_opts.keys())
    if st.session_state.answer_provider not in provider_opts.values():
        st.session_state.answer_provider = "gemini"
    current_provider_label = next(
        label for label, provider_name in provider_opts.items()
        if provider_name == st.session_state.answer_provider
    )
    chosen_provider_label = st.selectbox(
        label="Answer Engine",
        options=provider_labels,
        index=provider_labels.index(current_provider_label),
        label_visibility="collapsed",
    )
    st.session_state.answer_provider = provider_opts[chosen_provider_label]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Live Web Search</div>', unsafe_allow_html=True)
    live_search_labels = list(LIVE_SEARCH_OPTIONS.keys())
    if st.session_state.live_search_provider not in LIVE_SEARCH_OPTIONS.values():
        st.session_state.live_search_provider = "duckduckgo"
    current_live_label = next(
        label for label, provider in LIVE_SEARCH_OPTIONS.items()
        if provider == st.session_state.live_search_provider
    )
    chosen_live_label = st.selectbox(
        label="Live Web Search Provider",
        options=live_search_labels,
        index=live_search_labels.index(current_live_label),
        label_visibility="collapsed",
    )
    st.session_state.live_search_provider = LIVE_SEARCH_OPTIONS[chosen_live_label]

    openai_key_input = st.text_input(
        label="OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="OpenAI API key (optional)",
        key="openai_api_input",
        label_visibility="collapsed",
    )
    st.session_state.openai_api_key = openai_key_input

    gemini_key_input = st.text_input(
        label="Gemini API Key (Answer Engine)",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="Gemini key for answer generation",
        key="gemini_api_input",
        label_visibility="collapsed",
    )
    st.session_state.gemini_api_key = gemini_key_input

    gemini_web_key_input = st.text_input(
        label="Gemini API Key (Live Search)",
        value=st.session_state.gemini_web_api_key,
        type="password",
        placeholder="2nd Gemini key for live search (diff account)",
        key="gemini_web_api_input",
        label_visibility="collapsed",
    )
    st.session_state.gemini_web_api_key = gemini_web_key_input

    if st.session_state.live_search_provider == "openai":
        if st.session_state.openai_api_key:
            st.markdown('<div class="status-banner status-success">✅ OpenAI live search ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-banner status-warning">⚠️ OpenAI selected, but API key is missing. Search will fall back to DuckDuckGo.</div>', unsafe_allow_html=True)
    elif st.session_state.live_search_provider == "gemini":
        web_key = st.session_state.gemini_web_api_key or st.session_state.gemini_api_key
        if web_key:
            using_secondary = bool(st.session_state.gemini_web_api_key)
            label = "✅ Gemini live search ready" + (" (secondary key)" if using_secondary else "")
            st.markdown(f'<div class="status-banner status-success">{label}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-banner status-warning">⚠️ Gemini selected, but API key is missing. Search will fall back to DuckDuckGo.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-banner status-warning">🌐 Using DuckDuckGo fallback for best-effort live search</div>', unsafe_allow_html=True)

    st.caption("Answer Engine powers Analyst Chat and Executive Summary. Live Web Search powers the current-world context layer.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Answer Model</div>', unsafe_allow_html=True)
    model_opts = get_model_options(st.session_state.answer_provider)
    model_labels = list(model_opts.keys())
    if st.session_state.selected_model not in model_opts.values():
        st.session_state.selected_model = get_default_model(st.session_state.answer_provider)
    current_model_label = next(
        label for label, model_name in model_opts.items()
        if model_name == st.session_state.selected_model
    )
    chosen_model_label = st.selectbox(
        label="AI Model",
        options=model_labels,
        index=model_labels.index(current_model_label),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model_opts[chosen_model_label]
    if st.session_state.answer_provider == "groq" and st.session_state.selected_model == "llama-3.3-70b-versatile":
        st.caption("Analyst Chat auto-uses Llama 3.1 8B for stable tool calls. Executive Summary still uses 70B.")
    elif st.session_state.answer_provider == "gemini":
        st.caption("Analyst Chat and Executive Summary are answered by Gemini. Groq is only used for KPI extraction.")
    elif st.session_state.answer_provider == "openai":
        st.caption("Analyst Chat and Executive Summary are answered by OpenAI. Groq is only used for KPI extraction.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload PDF — Primary
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Upload Report</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Upload Financial Report",
        type=["pdf"],
        key="pdf_upload",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload annual reports, earnings PDFs, 10-K/10-Q filings",
    )

    # ── Competitor Reports ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">⚔️ Peer Compare — Upload Competitors</div>', unsafe_allow_html=True)
    st.caption("Upload 1–2 competitor reports for head-to-head benchmarking")

    comp1_file = st.file_uploader(
        label="Competitor 1",
        type=["pdf"],
        key="comp1_upload",
        label_visibility="visible",
        help="Competitor report 1",
    )
    comp2_file = st.file_uploader(
        label="Competitor 2",
        type=["pdf"],
        key="comp2_upload",
        label_visibility="visible",
        help="Competitor report 2 (optional)",
    )

    # Process competitor reports when uploaded
    if comp1_file or comp2_file:
        if st.button("⚔️ Load Competitors", use_container_width=True):
            with st.spinner("Building competitor indexes…"):
                new_peers = {}
                new_peer_kpis = {}
                for cf in [comp1_file, comp2_file]:
                    if cf is None:
                        continue
                    try:
                        fbytes = cf.getvalue()
                        vs = build_peer_vectorstore(fbytes, cf.name)
                        new_peers[cf.name] = vs
                        # Quick KPI extraction
                        if st.session_state.api_key:
                            raw = extract_raw_text(fbytes, cf.name, max_pages=10)
                            kpi_result = extract_kpis_with_llm(raw, st.session_state.api_key)
                            new_peer_kpis[cf.name] = kpi_result
                    except Exception as e:
                        st.error(f"Failed to load {cf.name}: {e}")
                st.session_state.peer_vectorstores = new_peers
                st.session_state.peer_kpis = new_peer_kpis
                st.session_state.peer_chat_history = []
            st.success(f"✅ {len(new_peers)} competitor(s) loaded!")
            st.rerun()

    if st.session_state.peer_vectorstores:
        peer_names = list(st.session_state.peer_vectorstores.keys())
        st.markdown(
            f'<div class="status-banner status-success">⚔️ Competitors loaded: ' +
            ", ".join([f"<b>{n}</b>" for n in peer_names]) +
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear Competitors", use_container_width=True):
            st.session_state.peer_vectorstores = {}
            st.session_state.peer_kpis = {}
            st.session_state.peer_chat_history = []
            st.rerun()

    # Status / Polling Block (Redis async path)
    if st.session_state.job_id and _REDIS_AVAILABLE:
        try:
            r = Redis(host=_secret("REDIS_HOST", "localhost"), port=int(_secret("REDIS_PORT", "6379")),
                      password=_secret("REDIS_PASSWORD") or None)
            q = Queue(connection=r)
            job = q.fetch_job(st.session_state.job_id)

            if job and not job.is_finished and not job.is_failed:
                st.markdown(f'<div class="status-banner status-warning">⏳ Processing in background...</div>', unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()
            elif job and job.is_failed:
                st.markdown('<div class="status-banner status-error">❌ Background processing failed.</div>', unsafe_allow_html=True)
                st.session_state.job_id = None
            elif job and job.is_finished:
                result = job.result
                if result.get("status") == "success":
                    st.session_state.doc_meta = {
                        "filenames": result["filenames"],
                        "total_pages": result["total_pages"],
                        "total_chunks": result["total_chunks"]
                    }
                    merged_kpis, kpi_notice = merge_kpi_results(
                        st.session_state.get("kpis"), result["kpis"],
                    )
                    merged_kpis, recovery_notice = recover_missing_kpis(merged_kpis)
                    st.session_state.kpis = merged_kpis
                    st.session_state.kpi_notice = " ".join(
                        [msg for msg in (kpi_notice, recovery_notice) if msg]
                    ) or None
                    st.session_state.vectorstore = load_vector_store("financial_index")
                    first_file = result["filenames"][0] if result["filenames"] else None
                    st.session_state.active_file = first_file
                    sync_agent_executor(force=True)
                    st.session_state.chat_history = []
                    st.session_state.summary = None
                    st.session_state.job_id = None
                    st.markdown('<div class="status-banner status-success">✅ Report processed!</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="status-banner status-error">❌ {result.get("error")}</div>', unsafe_allow_html=True)
                    st.session_state.job_id = None
        except Exception as e:
            st.markdown(f'<div class="status-banner status-warning">Redis unavailable — switching to sync mode.</div>', unsafe_allow_html=True)
            st.session_state.job_id = None

    # Process button — async (Redis) or sync fallback
    if uploaded_files and len(uploaded_files) > 0 and st.session_state.api_key and not st.session_state.job_id:
        btn_label = "🚀 Analyze Async" if _REDIS_AVAILABLE else "🚀 Analyze"
        if st.button(btn_label, use_container_width=True):
            # Try Redis async first; fall back to inline sync
            redis_ok = False
            if _REDIS_AVAILABLE:
                try:
                    r = Redis(
                        host=_secret("REDIS_HOST", "localhost"),
                        port=int(_secret("REDIS_PORT", "6379")),
                        password=_secret("REDIS_PASSWORD") or None,
                        socket_connect_timeout=2,
                    )
                    r.ping()
                    redis_ok = True
                except Exception:
                    pass

            if redis_ok:
                with st.spinner("Queueing…"):
                    tmp_dir = os.path.join(os.path.dirname(__file__), "reports")
                    os.makedirs(tmp_dir, exist_ok=True)
                    file_payloads = []
                    for f in uploaded_files:
                        persistent_path = os.path.join(tmp_dir, f.name)
                        with open(persistent_path, "wb") as pf:
                            pf.write(f.getbuffer())
                        tmp_path = os.path.join(tmp_dir, f"{str(uuid.uuid4())}.pdf")
                        with open(tmp_path, "wb") as w:
                            w.write(f.getbuffer())
                        file_payloads.append({"file_path": tmp_path, "filename": f.name})
                    q = Queue(connection=r)
                    from worker import async_ingest_and_extract
                    job = q.enqueue(async_ingest_and_extract, files=file_payloads,
                                    api_key=st.session_state.api_key, job_timeout='10m')
                    st.session_state.job_id = job.id
                st.rerun()
            else:
                # Sync fallback — runs ingest inline (works on Streamlit Cloud)
                with st.spinner("Processing report… this may take 30-60 seconds"):
                    try:
                        from worker import async_ingest_and_extract
                        import tempfile, os as _os
                        tmp_dir = tempfile.mkdtemp()
                        file_payloads = []
                        for f in uploaded_files:
                            tmp_path = _os.path.join(tmp_dir, f.name)
                            with open(tmp_path, "wb") as w:
                                w.write(f.getbuffer())
                            file_payloads.append({"file_path": tmp_path, "filename": f.name})
                        result = async_ingest_and_extract(files=file_payloads, api_key=st.session_state.api_key)
                        if result.get("status") == "success":
                            st.session_state.doc_meta = {
                                "filenames": result["filenames"],
                                "total_pages": result["total_pages"],
                                "total_chunks": result["total_chunks"],
                            }
                            merged_kpis, kpi_notice = merge_kpi_results(
                                st.session_state.get("kpis"), result["kpis"]
                            )
                            merged_kpis, recovery_notice = recover_missing_kpis(merged_kpis)
                            st.session_state.kpis = merged_kpis
                            st.session_state.kpi_notice = " ".join(
                                [m for m in (kpi_notice, recovery_notice) if m]
                            ) or None
                            st.session_state.vectorstore = load_vector_store("financial_index")
                            first_file = result["filenames"][0] if result["filenames"] else None
                            st.session_state.active_file = first_file
                            sync_agent_executor(force=True)
                            st.session_state.chat_history = []
                            st.session_state.summary = None
                            st.markdown('<div class="status-banner status-success">✅ Report processed!</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Processing failed: {result.get('error')}")
                    except Exception as exc:
                        st.error(f"Processing error: {exc}")
                st.rerun()
    elif uploaded_files and not st.session_state.api_key:
        st.markdown('<div class="status-banner status-warning">⚠️ Add the Groq KPI key first to build KPI cards and analyze the report.</div>', unsafe_allow_html=True)

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
            sync_agent_executor(force=True)
            st.rerun()

    # Footer
    st.markdown("""
    <div style="margin-top:40px;text-align:center;padding:10px 0;">
        <div style="font-size:0.72rem;color:#334155;">Powered by LangChain + Gemini/OpenAI answer engines + Groq KPI extraction</div>
        <div style="font-size:0.68rem;color:#1e3a5f;margin-top:2px;">Built with ❤️ for financial intelligence</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="arthamind-hero">
    <div class="hero-title">📊 ArthaMind</div>
    <div class="hero-subtitle">AI-Powered Financial Report Analyst — Instant insights from any financial document</div>
    <span class="hero-badge"><span class="badge-dot"></span>Multi-LLM Powered</span>
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
            <div style="font-size:0.85rem;color:#64748b;line-height:1.5;">Choose Gemini, OpenAI, or Groq for answers. Groq is still used for KPI extraction from uploaded reports.</div>
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
    sync_agent_executor()
    
    # ── KPI Dashboard ──────────────────────────────────────────────────────
    if st.session_state.kpis:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon icon-green">📈</div>
            <h2>Key Financial Metrics</h2>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("kpi_notice"):
            st.markdown(
                f'<div class="status-banner status-warning">⚠️ {html.escape(st.session_state.kpi_notice)}</div>',
                unsafe_allow_html=True,
            )

        # Multi-document KPI dropdown logic
        if isinstance(st.session_state.kpis, dict) and not "company_name" in st.session_state.kpis:
            # It's a dict mapping filenames to KPIs
            available_files = list(st.session_state.kpis.keys())
            if available_files:
                selected_file_for_kpi = st.selectbox("Select document KPIs to view:", available_files, key="kpi_selector")
                st.session_state.active_file = selected_file_for_kpi
                sync_agent_executor()
                kpis = st.session_state.kpis[selected_file_for_kpi]
            else:
                kpis = {}
        else:
            kpis = st.session_state.kpis
            if st.session_state.agent_executor is None and st.session_state.vectorstore:
                sync_agent_executor()

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
        elif isinstance(kpis, dict) and "error" in kpis:
            st.markdown(
                '<div class="status-banner status-warning">⚠️ KPI extraction failed for this document. The report is still loaded, so chat and executive summary should continue to work.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: Chat | Summary | Peer Compare ────────────────────────────────
    has_peers = bool(st.session_state.peer_vectorstores)
    if has_peers:
        tab_chat, tab_summary, tab_peer = st.tabs(["💬  Analyst Chat", "📋  Executive Summary", "⚔️  Peer Compare"])
    else:
        tab_chat, tab_summary = st.tabs(["💬  Analyst Chat", "📋  Executive Summary"])
        tab_peer = None

    # ── CHAT TAB ──────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon icon-blue">💬</div>
            <h2>Chat with the Report</h2>
        </div>
        """, unsafe_allow_html=True)

        agent_notice = st.session_state.get("agent_notice")
        if agent_notice:
            st.markdown(
                f'<div class="status-banner status-warning">💡 {agent_notice}</div>',
                unsafe_allow_html=True,
            )

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

        with col_input:
            user_question = st.text_input(
                label="Ask a Question About the Report",
                placeholder="Ask about revenue, risks, outlook, or what-if scenarios",
                key=f"user_question_{st.session_state.input_key}",
                label_visibility="collapsed",
            )
        with col_btn:
            send_btn = st.button("Send", use_container_width=True)

        # Only trigger on explicit button click with non-empty input
        if send_btn and user_question and user_question.strip() and st.session_state.vectorstore:
            question = user_question.strip()
            # Rotate the key to clear the input box
            st.session_state.input_key += 1
            submit_chat_question(question)
            st.rerun()

        # Sample question buttons
        if not st.session_state.chat_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.8rem;color:#334155;margin-bottom:8px;">Click to ask:</div>', unsafe_allow_html=True)
            q_cols = st.columns(2)
            for i, q in enumerate(SAMPLE_QUESTIONS[:4]):
                with q_cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        submit_chat_question(q)
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

                    try:
                        summary = generate_summary(
                            st.session_state.vectorstore,
                            st.session_state.api_key,
                            company_name=active_company,
                            answer_provider=st.session_state.answer_provider,
                            model=st.session_state.selected_model,
                            openai_api_key=st.session_state.openai_api_key,
                            gemini_api_key=st.session_state.gemini_api_key,
                        )
                        st.session_state.summary = summary
                    except Exception as exc:
                        error_str = str(exc)
                        if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                            st.error(
                                "**Gemini quota exceeded (HTTP 429).**\n\n"
                                "Your free-tier Gemini API key has hit its daily limit. "
                                "Please switch the **Answer Engine** in the sidebar to **Groq** "
                                "and try generating the summary again."
                            )
                        elif "503" in error_str or "unavailable" in error_str.lower():
                            st.error(
                                "**Answer Engine temporarily unavailable (503).**\n\n"
                                "Please switch to **Groq** in the sidebar or try again in a moment."
                            )
                        else:
                            st.error(f"Summary generation failed: {error_str[:300]}")
                        st.stop()
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

    # ── PEER COMPARE TAB ────────────────────────────────────────────────────
    if tab_peer is not None:
        with tab_peer:
            st.markdown("""
            <div class="section-header">
                <div class="section-icon" style="background:rgba(139,92,246,0.15);color:#a78bfa;">⚔️</div>
                <h2>Peer Benchmarking</h2>
            </div>
            """, unsafe_allow_html=True)

            # ── KPI Comparison Table ──────────────────────────────────────
            all_kpis = {}
            primary_kpis = st.session_state.kpis or {}
            if isinstance(primary_kpis, dict) and "company_name" not in primary_kpis:
                for fn, kd in primary_kpis.items():
                    if isinstance(kd, dict) and "error" not in kd:
                        all_kpis[kd.get("company_name", fn)] = kd
                        break
            elif isinstance(primary_kpis, dict) and "company_name" in primary_kpis:
                all_kpis[primary_kpis.get("company_name", "Primary")] = primary_kpis

            for fn, kd in st.session_state.peer_kpis.items():
                if isinstance(kd, dict) and "error" not in kd:
                    all_kpis[kd.get("company_name", fn.replace(".pdf", ""))] = kd

            if len(all_kpis) >= 2:
                kpi_rows = [
                    ("💰 Revenue",       "revenue"),
                    ("📊 EBITDA",         "ebitda"),
                    ("📈 ROE",            "roe"),
                    ("💧 Operating CF",   "operating_cash_flow"),
                    ("📉 Total Debt",     "total_debt"),
                    ("📐 D/E Ratio",      "debt_to_equity"),
                    ("🎯 EPS",            "eps"),
                    ("📦 Revenue Growth", "revenue_growth"),
                ]
                companies_list = list(all_kpis.keys())
                th_cells = "".join([f'<th style="padding:10px 14px;text-align:right;color:#94a3b8;font-size:0.8rem;">{c}</th>' for c in companies_list])
                table_html = f"""
                <div style="overflow-x:auto;margin:16px 0;">
                <table style="width:100%;border-collapse:collapse;background:rgba(15,23,42,0.6);border-radius:12px;overflow:hidden;">
                <thead><tr style="border-bottom:1px solid #1e3a5f;">
                    <th style="padding:10px 14px;text-align:left;color:#64748b;font-size:0.78rem;text-transform:uppercase;">Metric</th>
                    {th_cells}
                </tr></thead><tbody>"""

                for label, key in kpi_rows:
                    vals = [format_kpi_value(all_kpis[c].get(key)) for c in companies_list]
                    numeric_vals = []
                    for v in vals:
                        try:
                            numeric_vals.append(float(str(v).replace("%","").replace("x","").replace(",","").replace("₹","").replace("Rs.","").split()[0]))
                        except Exception:
                            numeric_vals.append(None)
                    best_idx = None
                    if any(v is not None for v in numeric_vals):
                        valid = [(i, v) for i, v in enumerate(numeric_vals) if v is not None]
                        if key in ("total_debt", "debt_to_equity"):
                            best_idx = min(valid, key=lambda x: x[1])[0]
                        else:
                            best_idx = max(valid, key=lambda x: x[1])[0]
                    td_cells = ""
                    for i, v in enumerate(vals):
                        color = "#10b981" if i == best_idx else "#cbd5e1"
                        weight = "700" if i == best_idx else "400"
                        td_cells += f'<td style="padding:10px 14px;text-align:right;color:{color};font-weight:{weight};font-size:0.85rem;">{v}</td>'
                    table_html += f'<tr style="border-bottom:1px solid rgba(30,58,95,0.5);"><td style="padding:10px 14px;color:#94a3b8;font-size:0.82rem;">{label}</td>{td_cells}</tr>'

                table_html += "</tbody></table></div>"
                st.markdown(table_html, unsafe_allow_html=True)
                st.caption("🟢 Green = Best performer on that metric")
            else:
                st.info("Load at least 1 competitor report from the sidebar to enable benchmarking. KPIs for both companies will be compared here.")

            st.markdown("---")

            # ── Peer Chat ────────────────────────────────────────────────
            st.markdown("""<div style="font-size:1rem;font-weight:600;color:#f0f9ff;margin-bottom:12px;">🤖 Ask Cross-Company Questions</div>""", unsafe_allow_html=True)

            combined_vs = {}
            if st.session_state.vectorstore:
                primary_name = list(all_kpis.keys())[0] if all_kpis else "Primary"
                combined_vs[primary_name] = st.session_state.vectorstore
            combined_vs.update({
                (st.session_state.peer_kpis.get(fn, {}).get("company_name") or fn.replace(".pdf", "")): vs
                for fn, vs in st.session_state.peer_vectorstores.items()
            })

            peer_sample_qs = [
                "Which company has the best EBITDA margin and why?",
                "Compare debt profiles — who is financially stronger?",
                "Which company is most exposed to commodity price risk?",
                "Who has better revenue growth momentum?",
            ]

            if not st.session_state.peer_chat_history:
                sq_cols = st.columns(2)
                for i, q in enumerate(peer_sample_qs):
                    with sq_cols[i % 2]:
                        if st.button(q, key=f"peerq_{i}", use_container_width=True):
                            st.session_state.peer_chat_history.append({"role": "user", "content": q})
                            with st.spinner("Comparing across all reports..."):
                                try:
                                    ans = run_peer_comparison(
                                        question=q, vectorstores=combined_vs,
                                        groq_api_key=st.session_state.api_key,
                                        answer_provider=st.session_state.answer_provider,
                                        answer_model=st.session_state.selected_model,
                                        openai_api_key=st.session_state.openai_api_key,
                                        gemini_api_key=gemini_web_key(),
                                        live_search_provider=st.session_state.live_search_provider,
                                        chat_history=[],
                                    )
                                except Exception as e:
                                    ans = f"Peer comparison failed: {e}"
                            st.session_state.peer_chat_history.append({"role": "assistant", "content": ans})
                            st.rerun()

            if st.session_state.peer_chat_history:
                peer_chat_html = '<div class="chat-container">'
                for msg in st.session_state.peer_chat_history:
                    if msg["role"] == "user":
                        peer_chat_html += f'<div class="message-user"><div class="bubble-user">{msg["content"]}</div><div class="avatar-user">👤</div></div>'
                    else:
                        content = msg["content"].replace("\n", "<br>")
                        peer_chat_html += f'<div class="message-ai"><div class="avatar-ai">🤖</div><div class="bubble-ai">{content}</div></div>'
                peer_chat_html += "</div>"
                st.markdown(peer_chat_html, unsafe_allow_html=True)
                if st.button("🗑️ Clear Peer Chat", key="clear_peer_chat"):
                    st.session_state.peer_chat_history = []
                    st.rerun()

            pc1, pc2 = st.columns([5, 1])
            with pc1:
                peer_q = st.text_input("Peer question", placeholder="e.g. Which company has lower debt risk going into FY27?", key="peer_q_input", label_visibility="collapsed")
            with pc2:
                peer_send = st.button("Ask", key="peer_send_btn", use_container_width=True)

            if peer_send and peer_q.strip() and combined_vs:
                st.session_state.peer_chat_history.append({"role": "user", "content": peer_q.strip()})
                with st.spinner("Comparing across all reports..."):
                    try:
                        ans = run_peer_comparison(
                            question=peer_q.strip(), vectorstores=combined_vs,
                            groq_api_key=st.session_state.api_key,
                            answer_provider=st.session_state.answer_provider,
                            answer_model=st.session_state.selected_model,
                            openai_api_key=st.session_state.openai_api_key,
                            gemini_api_key=gemini_web_key(),
                            live_search_provider=st.session_state.live_search_provider,
                            chat_history=st.session_state.peer_chat_history[:-1],
                        )
                    except Exception as e:
                        ans = f"Peer comparison failed: {str(e)}"
                st.session_state.peer_chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

# ── COMMODITY RADAR ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <div class="section-icon" style="background:rgba(245,158,11,0.15);color:#fbbf24;">⚡</div>
    <h2>Commodity Radar</h2>
</div>
""", unsafe_allow_html=True)

alert_unit_map = {
    "Lithium Carbonate": "$/tonne",
    "Crude Oil (WTI)": "$/barrel",
    "Natural Gas": "$/MMBtu",
    "Copper": "$/tonne",
    "INR/USD": "₹/USD",
    "Coal": "$/tonne",
}

rad_col1, rad_col2 = st.columns([1, 2])

with rad_col1:
    st.markdown('<div style="font-size:0.85rem;font-weight:600;color:#f0f9ff;margin-bottom:10px;">🔔 Set New Alert</div>', unsafe_allow_html=True)
    commodity_opts = list(COMMODITY_SEARCH_QUERIES.keys())
    alert_commodity = st.selectbox("Commodity", commodity_opts, key="alert_commodity")
    alert_direction = st.selectbox("Alert when price goes", ["Above", "Below"], key="alert_direction")
    alert_threshold = st.number_input("Threshold", min_value=0.0, value=28000.0, step=100.0, key="alert_threshold")
    unit = alert_unit_map.get(alert_commodity, "")
    st.caption(f"Unit: {unit}")
    if st.button("+ Add Alert", key="add_alert_btn", use_container_width=True):
        st.session_state.commodity_alerts.append({
            "commodity": alert_commodity, "direction": alert_direction,
            "threshold": alert_threshold, "unit": unit, "triggered": False,
        })
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Live Prices", key="refresh_commodities", use_container_width=True):
        with st.spinner("Fetching live prices..."):
            prices = {}
            for comm in commodity_opts:
                prices[comm] = fetch_commodity_live_price(
                    comm,
                    live_search_provider=st.session_state.live_search_provider,
                    openai_api_key=st.session_state.openai_api_key,
                    gemini_api_key=gemini_web_key(),
                )
        st.session_state.commodity_prices = prices
        st.session_state.commodity_last_checked = time.strftime("%H:%M:%S")
        st.rerun()

with rad_col2:
    st.markdown('<div style="font-size:0.85rem;font-weight:600;color:#f0f9ff;margin-bottom:10px;">📡 Live Tracker</div>', unsafe_allow_html=True)
    if st.session_state.commodity_last_checked:
        st.caption(f"Last checked: {st.session_state.commodity_last_checked}")
    else:
        st.caption("Click 'Refresh Live Prices' to fetch current market data.")

    prices = st.session_state.commodity_prices
    if prices:
        for comm, pdata in prices.items():
            price = pdata.get("price")
            unit = alert_unit_map.get(comm, "")
            alert_flag = ""
            for alert in st.session_state.commodity_alerts:
                if alert["commodity"] == comm and price is not None:
                    trig = (alert["direction"] == "Above" and price > alert["threshold"]) or \
                           (alert["direction"] == "Below" and price < alert["threshold"])
                    if trig:
                        alert_flag = " 🚨"
                        alert["triggered"] = True
            if price:
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.7);border:1px solid {"#ef4444" if alert_flag else "#1e3a5f"};border-radius:10px;padding:10px 14px;margin-bottom:8px;">'
                    f'<span style="color:#fbbf24;font-weight:600;font-size:0.85rem;">{comm}</span>{alert_flag}'
                    f'<span style="float:right;color:#f0f9ff;font-weight:700;">{unit} {price:,.1f}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.4);border:1px solid #1e2a3f;border-radius:10px;padding:8px 14px;margin-bottom:6px;color:#475569;font-size:0.8rem;">{comm} — price unavailable</div>',
                    unsafe_allow_html=True
                )

    if st.session_state.commodity_alerts:
        st.markdown('<div style="font-size:0.85rem;font-weight:600;color:#f0f9ff;margin:16px 0 8px 0;">🔔 Active Alerts</div>', unsafe_allow_html=True)
        to_remove = []
        for i, alert in enumerate(st.session_state.commodity_alerts):
            triggered = alert.get("triggered", False)
            bc = "#ef4444" if triggered else "#1e3a5f"
            bg = "rgba(239,68,68,0.08)" if triggered else "rgba(15,23,42,0.7)"
            status = "🚨 TRIGGERED" if triggered else f"{alert['direction']} {alert['unit']}{alert['threshold']:,.0f}"
            impact = ""
            if triggered and "lithium" in alert["commodity"].lower():
                impact = "<br><span style='color:#fbbf24;font-size:0.75rem;'>💡 Per sensitivity matrix: ~Rs.640 Cr EBITDA impact per $10K/tonne</span>"
            st.markdown(
                f'<div style="background:{bg};border:1px solid {bc};border-radius:10px;padding:10px 14px;margin-bottom:8px;">'
                f'<b style="color:#94a3b8;">{alert["commodity"]}</b>'
                f'<span style="float:right;color:#fbbf24;font-size:0.78rem;">{status}</span>{impact}</div>',
                unsafe_allow_html=True
            )
            al1, al2, al3 = st.columns([3, 1, 1])
            with al2:
                if triggered:
                    if st.button("📊 Analyze", key=f"analyze_{i}", use_container_width=True):
                        p = st.session_state.commodity_prices.get(alert["commodity"], {}).get("price", "unknown")
                        auto_q = f"{alert['commodity']} has crossed {alert['unit']}{alert['threshold']:,.0f} — current price ~{p}. What is the exact EBITDA impact and is FY27 guidance still achievable?"
                        st.session_state.chat_history.append({"role": "user", "content": auto_q})
                        st.rerun()
            with al3:
                if st.button("❌", key=f"rm_{i}", help="Remove alert"):
                    to_remove.append(i)
        for idx in reversed(to_remove):
            st.session_state.commodity_alerts.pop(idx)
        if to_remove:
            st.rerun()

