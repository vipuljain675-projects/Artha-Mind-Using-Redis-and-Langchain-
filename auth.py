"""
auth.py - ArthaMind Hybrid Authentication System
Features:
  - Stunning premium login UI (full-page, not boxed)
  - Email/Password signup/login with bcrypt
  - Google OAuth2 login
  - Persistent sessions via token file (survives refresh + browser close)
"""

import streamlit as st
import sqlite3
import bcrypt
import os
import json
import uuid
import time
from typing import Optional
from pathlib import Path
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "users.db"
SESSION_FILE = ".arthamind_sessions.json"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days in seconds

# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            auth_provider TEXT NOT NULL,
            name TEXT
        )
    """)
    # Migrate old DBs that may be missing the name column
    try:
        conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
        conn.commit()
    except Exception:
        pass  # Column already exists
    conn.close()

def get_user(email: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, email, password_hash, auth_provider, name FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "email": row[1], "password_hash": row[2], "auth_provider": row[3], "name": row[4]}
    return None

def create_user(email: str, password: str = None, provider: str = "local", name: str = None):
    conn = sqlite3.connect(DB_PATH)
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() if password else None
        display_name = name or email.split("@")[0].capitalize()
        conn.execute("INSERT INTO users (email, password_hash, auth_provider, name) VALUES (?,?,?,?)",
                     (email, hashed, provider, display_name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ─────────────────────────────────────────────────────────────────────────────
# Session persistence (file-based tokens — works across browser restarts)
# ─────────────────────────────────────────────────────────────────────────────
def _load_sessions() -> dict:
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_sessions(sessions: dict):
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f)

def create_session_token(email: str, name: str, provider: str) -> str:
    token = str(uuid.uuid4())
    sessions = _load_sessions()
    sessions[token] = {"email": email, "name": name, "provider": provider, "created": time.time()}
    _save_sessions(sessions)
    return token

def resolve_session_token(token: str) -> Optional[dict]:
    sessions = _load_sessions()
    entry = sessions.get(token)
    if not entry:
        return None
    if time.time() - entry["created"] > SESSION_MAX_AGE:
        sessions.pop(token, None)
        _save_sessions(sessions)
        return None
    return entry

def delete_session_token(token: str):
    sessions = _load_sessions()
    sessions.pop(token, None)
    _save_sessions(sessions)

# ─────────────────────────────────────────────────────────────────────────────
# Google OAuth
# ─────────────────────────────────────────────────────────────────────────────
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

def _google_creds():
    return (
        os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_SECRET"),
        os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/"),
    )

def get_google_auth_url():
    client_id, _, redirect_uri = _google_creds()
    if not client_id:
        return None
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ])
    auth_url, state = google.authorization_url(_GOOGLE_AUTH_URL, access_type="offline", prompt="select_account")
    st.session_state["_oauth_state"] = state
    return auth_url

def handle_google_callback():
    client_id, client_secret, redirect_uri = _google_creds()
    if "code" not in st.query_params or not client_id:
        return
    try:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=st.session_state.get("_oauth_state"))
        full_url = f"{redirect_uri}?code={st.query_params['code']}"
        if "state" in st.query_params:
            full_url += f"&state={st.query_params['state']}"
        google.fetch_token(_GOOGLE_TOKEN_URL, client_secret=client_secret, authorization_response=full_url)
        profile = google.get("https://www.googleapis.com/oauth2/v1/userinfo").json()
        email = profile.get("email")
        name = profile.get("name", email)
        user = get_user(email)
        if not user:
            create_user(email, provider="google", name=name)
        _login_success(email, name, "google")
        st.query_params.clear()
    except Exception as e:
        st.query_params.clear()
        st.session_state["_auth_error"] = f"Google login failed: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _login_success(email: str, name: str, provider: str):
    token = create_session_token(email, name, provider)
    _persist_token_to_disk(token)
    st.session_state["authenticated"] = True
    st.session_state["user_email"] = email
    st.session_state["user_name"] = name
    st.session_state["user_provider"] = provider
    st.session_state["_session_token"] = token
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def check_auth() -> bool:
    """Returns True if authenticated. Restores session from file token if needed."""
    init_db()
    handle_google_callback()

    if st.session_state.get("authenticated"):
        return True

    # Try restoring from persisted file token
    token = st.session_state.get("_session_token")
    if token:
        entry = resolve_session_token(token)
        if entry:
            st.session_state["authenticated"] = True
            st.session_state["user_email"] = entry["email"]
            st.session_state["user_name"] = entry["name"]
            st.session_state["user_provider"] = entry["provider"]
            return True

    # Try reading the token that was stored in the URL query param workaround
    # (Streamlit-safe cookie alternative: write token to a temp hidden file keyed by browser)
    _try_restore_from_disk()
    return st.session_state.get("authenticated", False)

def _try_restore_from_disk():
    """Read last saved token from a marker file created on login."""
    marker = Path(".arthamind_last_token")
    if marker.exists():
        token = marker.read_text().strip()
        entry = resolve_session_token(token)
        if entry:
            st.session_state["authenticated"] = True
            st.session_state["user_email"] = entry["email"]
            st.session_state["user_name"] = entry["name"]
            st.session_state["user_provider"] = entry["provider"]
            st.session_state["_session_token"] = token

def _persist_token_to_disk(token: str):
    Path(".arthamind_last_token").write_text(token)

def logout():
    token = st.session_state.get("_session_token")
    if token:
        delete_session_token(token)
    marker = Path(".arthamind_last_token")
    if marker.exists():
        marker.unlink()
    for k in ["authenticated", "user_email", "user_name", "user_provider", "_session_token", "_oauth_state"]:
        st.session_state.pop(k, None)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Login UI — Premium full-page design
# ─────────────────────────────────────────────────────────────────────────────
def login_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Kill all default Streamlit chrome ── */
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"] { display: none !important; }

    /* ── Full page dark background ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #020c12 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stAppViewContainer"] > section {
        padding: 0 !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100vw !important;
    }

    /* ── Two-column layout ── */
    .auth-page {
        display: flex;
        min-height: 100vh;
        width: 100%;
    }
    .auth-left {
        flex: 1;
        background: linear-gradient(135deg, #020c12 0%, #0a1e35 60%, #061726 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 80px 60px;
        position: relative;
        overflow: hidden;
    }
    .auth-left::before {
        content: '';
        position: absolute;
        top: -30%;
        left: -20%;
        width: 70%;
        height: 80%;
        background: radial-gradient(ellipse, rgba(16,185,129,0.08) 0%, transparent 65%);
        pointer-events: none;
    }
    .auth-left::after {
        content: '';
        position: absolute;
        bottom: -20%;
        right: -10%;
        width: 50%;
        height: 60%;
        background: radial-gradient(ellipse, rgba(59,130,246,0.06) 0%, transparent 65%);
        pointer-events: none;
    }
    .auth-right {
        width: 480px;
        min-width: 480px;
        background: #050f1e;
        border-left: 1px solid rgba(16,185,129,0.12);
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 60px 50px;
    }
    .brand-logo {
        font-size: 3.5rem;
        margin-bottom: 24px;
        position: relative;
        z-index: 1;
    }
    .brand-name {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f0f9ff 30%, #10b981 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 12px 0;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    .brand-tagline {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 400;
        margin: 0 0 48px 0;
        line-height: 1.6;
        position: relative;
        z-index: 1;
        max-width: 380px;
    }
    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 18px;
        position: relative;
        z-index: 1;
    }
    .feature-item {
        display: flex;
        align-items: center;
        gap: 14px;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .feature-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .auth-panel-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f0f9ff;
        margin: 0 0 6px 0;
    }
    .auth-panel-subtitle {
        font-size: 0.9rem;
        color: #475569;
        margin: 0 0 32px 0;
    }
    .google-signin-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: white;
        color: #1f2937;
        border-radius: 10px;
        padding: 13px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none !important;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-family: 'Inter', sans-serif;
        margin-bottom: 24px;
    }
    .google-signin-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        color: #1f2937;
    }
    .divider-line {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 24px 0;
    }
    .divider-line::before, .divider-line::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.08);
    }
    .divider-line span {
        color: #334155;
        font-size: 0.8rem;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ── Form styling ── */
    div[data-testid="stTextInput"] input {
        background: #0a1628 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #f0f9ff !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(16,185,129,0.5) !important;
        box-shadow: 0 0 0 3px rgba(16,185,129,0.08) !important;
    }
    div[data-testid="stTextInput"] label {
        color: #64748b !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 13px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16,185,129,0.35) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #0a1628 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 24px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        border: none !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #10b981 !important;
        color: white !important;
    }
    .auth-error {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        color: #f87171;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }
    .auth-success {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        color: #34d399;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Left branding panel ──
    client_id, client_secret, _ = _google_creds()
    google_btn_html = ""
    if client_id and client_secret:
        auth_url = get_google_auth_url()
        if auth_url:
            google_btn_html = f"""
            <a href="{auth_url}" target="_self" class="google-signin-btn">
                <svg width="18" height="18" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
                Continue with Google
            </a>
            """

    left_col, right_col = st.columns([1.2, 0.85])

    with left_col:
        st.markdown("""
        <div class="auth-left" style="padding: 80px 60px; min-height: 100vh;">
            <div class="brand-logo">📊</div>
            <div class="brand-name">ArthaMind</div>
            <div class="brand-tagline">
                AI-powered financial intelligence platform. Analyze reports, compare peers, and surface insights instantly.
            </div>
            <div class="feature-list">
                <div class="feature-item">
                    <div class="feature-icon">🧠</div>
                    <span>Multi-LLM RAG engine with Groq & Gemini</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📈</div>
                    <span>Instant KPI extraction from PDF reports</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">⚡</div>
                    <span>Peer comparison across multiple companies</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🌐</div>
                    <span>Real-time commodity radar & price alerts</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div style="padding: 0 10px;">
            <div class="auth-panel-title">Welcome back</div>
            <div class="auth-panel-subtitle">Sign in to continue to ArthaMind</div>
        </div>
        """, unsafe_allow_html=True)

        if google_btn_html:
            st.markdown(google_btn_html, unsafe_allow_html=True)

        st.markdown("""<div class="divider-line"><span>or continue with email</span></div>""", unsafe_allow_html=True)

        # Show any persistent auth errors
        if st.session_state.get("_auth_error"):
            st.markdown(f"<div class='auth-error'>⚠️ {st.session_state['_auth_error']}</div>", unsafe_allow_html=True)
            st.session_state.pop("_auth_error", None)
        if st.session_state.get("_auth_success"):
            st.markdown(f"<div class='auth-success'>✅ {st.session_state['_auth_success']}</div>", unsafe_allow_html=True)
            st.session_state.pop("_auth_success", None)

        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                login_btn = st.form_submit_button("Sign In", use_container_width=True)
                if login_btn:
                    if not email or not password:
                        st.session_state["_auth_error"] = "Please fill in all fields."
                        st.rerun()
                    user = get_user(email)
                    if user and user["auth_provider"] == "local" and verify_password(password, user["password_hash"]):
                        token = create_session_token(email, user["name"] or email.split("@")[0].capitalize(), "local")
                        _persist_token_to_disk(token)
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = email
                        st.session_state["user_name"] = user["name"] or email.split("@")[0].capitalize()
                        st.session_state["user_provider"] = "local"
                        st.session_state["_session_token"] = token
                        st.rerun()
                    elif user and user["auth_provider"] == "google":
                        st.session_state["_auth_error"] = "This account uses Google Sign-In. Click 'Continue with Google' above."
                        st.rerun()
                    else:
                        st.session_state["_auth_error"] = "Invalid email or password."
                        st.rerun()

        with tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                new_name = st.text_input("Full name", placeholder="Vipul Jain")
                new_email = st.text_input("Email address", placeholder="you@example.com")
                new_password = st.text_input("Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm password", type="password", placeholder="••••••••")
                signup_btn = st.form_submit_button("Create Account", use_container_width=True)
                if signup_btn:
                    if not new_email or not new_password or not confirm_password:
                        st.session_state["_auth_error"] = "Please fill in all fields."
                        st.rerun()
                    elif new_password != confirm_password:
                        st.session_state["_auth_error"] = "Passwords do not match."
                        st.rerun()
                    elif len(new_password) < 6:
                        st.session_state["_auth_error"] = "Password must be at least 6 characters."
                        st.rerun()
                    else:
                        ok = create_user(new_email, new_password, "local", new_name)
                        if ok:
                            # Auto-login after signup
                            token = create_session_token(new_email, new_name or new_email.split("@")[0].capitalize(), "local")
                            _persist_token_to_disk(token)
                            st.session_state["authenticated"] = True
                            st.session_state["user_email"] = new_email
                            st.session_state["user_name"] = new_name or new_email.split("@")[0].capitalize()
                            st.session_state["user_provider"] = "local"
                            st.session_state["_session_token"] = token
                            st.rerun()
                        else:
                            st.session_state["_auth_error"] = "An account with this email already exists."
                            st.rerun()
