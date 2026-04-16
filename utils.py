"""
utils.py — Helper utilities for the Financial Analyst AI
"""

import re
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import random
import functools



# ── KPI Formatting ──────────────────────────────────────────────────────────────

def format_kpi_value(value) -> str:
    """Return a display-safe string for KPI values."""
    if value is None or value == "null":
        return "N/A"
    return str(value)


def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2, jitter=True):
    """
    Retry decorator with exponential backoff.
    Immediately raises without retrying on HTTP 429 (quota exhausted).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 429 = quota exhausted — retrying won't help, raise immediately
                    error_str = str(e)
                    if "429" in error_str or "quota" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                        raise e

                    if retries == max_retries:
                        raise e

                    wait_time = delay
                    if jitter:
                        wait_time += random.uniform(0, 1)

                    time.sleep(wait_time)
                    retries += 1
                    delay *= backoff_factor
        return wrapper
    return decorator


def kpi_delta_color(value: str) -> str:
    """Return green/red based on positive/negative growth string."""
    if not value or value == "N/A":
        return "normal"
    if "+" in value or (value.replace(".", "").replace("%", "").lstrip("-").isdigit() and not value.startswith("-")):
        return "normal"   # Streamlit uses 'normal' for green delta
    return "inverse"


# ── Chart Builders ────────────────────────────────────────────────────────────

def make_kpi_gauge(label: str, value_str: str, max_val: float = 100) -> go.Figure:
    """Create a gauge chart for margin/ratio metrics."""
    try:
        num = float(re.sub(r"[^0-9.\-]", "", value_str))
    except Exception:
        num = 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=num,
        title={"text": label, "font": {"size": 14, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 22, "color": "#f0fdf4"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#334155"},
            "bar": {"color": "#22c55e"},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "#1e293b"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#0f2a1a"},
                {"range": [max_val * 0.66, max_val], "color": "#14532d"},
            ],
            "threshold": {
                "line": {"color": "#86efac", "width": 3},
                "thickness": 0.8,
                "value": num,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
    )
    return fig


def make_sparkline(data: list, label: str, color: str = "#22c55e") -> go.Figure:
    """Mini area chart — useful for trend visualization."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode="lines+markers",
        line=dict(color=color, width=2.5, shape="spline"),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor=f"rgba(34,197,94,0.12)",
        name=label,
    ))
    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_bar_chart(labels: list, values: list, title: str, color: str = "#22c55e") -> go.Figure:
    """Horizontal bar chart for comparative metrics."""
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#166534"], [1, "#22c55e"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f0fdf4", size=14)),
        height=220,
        margin=dict(l=10, r=60, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
    )
    return fig


# ── Answer Post-Processing ────────────────────────────────────────────────────

def highlight_numbers(text: str) -> str:
    """
    Wrap financial numbers in <span> tags for markdown highlighting.
    Works with Streamlit's unsafe_allow_html.
    """
    # Match: $1.2B, ₹5,000 Cr, 18.3%, -3.2%, +45%
    pattern = r"([\$₹€£]?\s*[\d,]+\.?\d*\s*(?:B|M|K|Cr|Bn|Mn|Lakh|%)?|\+[\d.]+%?|-[\d.]+%?)"
    highlighted = re.sub(
        pattern,
        r'<span style="color:#22c55e;font-weight:600">\1</span>',
        text,
    )
    return highlighted


def clean_json_response(raw: str) -> dict:
    """Safely parse JSON from LLM response, stripping markdown fences."""
    try:
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue
        return json.loads(raw)
    except Exception:
        return {}


# ── Sample Questions ──────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "📊 What was the total revenue and how did it grow compared to last year?",
    "💰 What is the net profit margin? Is it improving or declining?",
    "⚠️ What are the main risk factors mentioned in the report?",
    "📈 Summarize the company's growth strategy going forward.",
    "🏦 What is the company's debt situation and liquidity position?",
    "🌍 If crude prices rise or the rupee weakens, what happens to the business?",
    "🎯 What were the biggest achievements this financial year?",
    "📉 Are there any warning signs or areas of concern for investors?",
    "💼 What does management say about the outlook for next year?",
    "🗞️ What current world developments could matter most for this business?",
]


def get_answer_provider_options() -> dict:
    """Available answer engines for analyst chat and executive summary."""
    return {
        "🔎 Gemini": "gemini",
        "🌐 OpenAI": "openai",
        "⚡ Groq": "groq",
    }


def get_model_options(provider: str = "groq") -> dict:
    """Available models grouped by answer provider."""
    normalized = (provider or "groq").lower()
    if normalized == "gemini":
        return {
            "🔎 Gemini 2.5 Flash (Fast)": "gemini-2.5-flash",
            "🧠 Gemini 2.5 Pro (Deep Analysis)": "gemini-2.5-pro",
        }
    if normalized == "openai":
        return {
            "🌐 GPT-4.1 Mini (Balanced)": "gpt-4.1-mini",
            "🧠 GPT-4.1 (Higher Quality)": "gpt-4.1",
        }
    return {
        "⚡ Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
        "🧠 Llama 3.3 70B (Premium Summary)": "llama-3.3-70b-versatile",
    }


def get_default_model(provider: str = "groq") -> str:
    """Return the default model for a provider."""
    return next(iter(get_model_options(provider).values()))
