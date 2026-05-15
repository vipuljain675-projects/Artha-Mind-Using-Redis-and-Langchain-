"""
report_generator.py — AI-Powered Financial Report Generator for ArthaMind
Generates a professional PDF projection report for the next fiscal year
using KPI data from the uploaded report and LLM-generated narrative.
"""

import os
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_groq import ChatGroq
from currency_utils import (
    DEFAULT_DISPLAY_CURRENCY,
    convert_currency_mentions,
    detect_currency_code,
    format_source_value_in_currency,
    get_currency_meta,
    infer_unit_multiplier,
    normalize_currency_code,
    parse_money_value,
)
from chain import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
    generate_text_with_provider,
    normalize_answer_provider,
)


REPORT_PIPELINE_VERSION = "2026-05-05-grounded-v7"
REPORT_GROQ_MODEL = "llama-3.3-70b-versatile"
REPORT_GROQ_MAX_TOKENS = 6144

UNICODE_FONT_FAMILY = "ArthaSans"
UNICODE_FONT_PATHS = {
    "": [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    "B": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    "I": [
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "/Library/Fonts/Arial Italic.ttf",
    ],
    "BI": [
        "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
        "/Library/Fonts/Arial Bold Italic.ttf",
    ],
}

REPORT_CONTEXT_QUERIES = [
    ("Company Overview", "company overview business lines core products headquarters management commentary"),
    ("Financial Baseline", "financial performance revenue EBITDA PAT cash flow debt ROE gross margin guidance"),
    ("Segments", "segment wise performance business segment revenue EBITDA margin verticals"),
    ("Programs", "programme milestones order book contracts delivery timeline certification export launch"),
    ("Risks", "risk management framework sensitivity matrix sanctions supply chain imported components fx"),
    ("Outlook", "management outlook guidance capex domestic content localization strategic priorities"),
]

REPORTS_DIR = Path(__file__).parent / "reports"


def _find_font_path(style: str) -> str:
    for candidate in UNICODE_FONT_PATHS.get(style, []):
        if os.path.exists(candidate):
            return candidate
    return ""


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_json_payload(raw: str) -> dict:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _default_model_for_provider(provider: str) -> str:
    normalized = normalize_answer_provider(provider)
    if normalized == "openai":
        return DEFAULT_OPENAI_MODEL
    if normalized == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_GROQ_MODEL


def _report_groq_model(requested_model: str = "") -> str:
    if requested_model and requested_model != DEFAULT_GROQ_MODEL:
        return requested_model
    return REPORT_GROQ_MODEL


def _generate_report_text_via_groq(
    prompt: str,
    api_key: str,
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = REPORT_GROQ_MAX_TOKENS,
) -> str:
    llm = ChatGroq(
        api_key=api_key,
        model_name=_report_groq_model(model),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = llm.invoke(prompt)
    content = getattr(response, "content", "")
    if isinstance(content, list):
        return "\n".join(str(part) for part in content).strip()
    return str(content or "").strip()


def _generate_report_text_with_provider(
    provider: str,
    prompt: str,
    requested_model: str,
    groq_api_key: str = "",
    openai_api_key: str = "",
    gemini_api_key: str = "",
) -> str:
    normalized = normalize_answer_provider(provider)
    if normalized == "groq":
        if not groq_api_key:
            raise RuntimeError("Groq answer engine is selected but GROQ_API_KEY is missing.")
        return _generate_report_text_via_groq(
            prompt=prompt,
            api_key=groq_api_key,
            model=requested_model,
            temperature=0.2,
        )

    return generate_text_with_provider(
        provider=normalized,
        prompt=prompt,
        model=requested_model,
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        temperature=0.2,
    )


def _repair_projection_json(prompt: str, malformed_output: str, api_key: str) -> str:
    repair_prompt = f"""You are a JSON repair engine.
The following model response was intended to be valid JSON for a financial projection report, but it was malformed or truncated.
Return ONLY valid JSON. Do not add markdown fences or commentary.
If the prior output was truncated, complete the missing fields consistently using the original instructions.

Original generation prompt:
{prompt[-16000:]}

Malformed or partial output:
{malformed_output[-12000:]}
"""
    return _generate_report_text_via_groq(
        prompt=repair_prompt,
        api_key=api_key,
        model=REPORT_GROQ_MODEL,
        temperature=0.1,
        max_tokens=REPORT_GROQ_MAX_TOKENS,
    )


def _format_kpi_snapshot(kpis: dict) -> str:
    if not isinstance(kpis, dict):
        return "{}"
    return json.dumps(kpis, indent=2, ensure_ascii=False)


def collect_report_context(vectorstore=None, active_filename: str = "", per_query_limit: int = 1200) -> str:
    """Retrieve grounded context from the uploaded report for the projection writer."""
    if vectorstore is None:
        return ""

    snippets = []
    seen = set()

    for label, query in REPORT_CONTEXT_QUERIES:
        try:
            docs = vectorstore.similarity_search(query, k=3)
        except Exception:
            continue

        filtered_docs = []
        for doc in docs:
            source_file = (doc.metadata or {}).get("source_file")
            if active_filename and source_file and source_file != active_filename:
                continue
            filtered_docs.append(doc)

        for doc in filtered_docs:
            text = _clean_ws(getattr(doc, "page_content", ""))
            if not text:
                continue
            key = text[:240]
            if key in seen:
                continue
            seen.add(key)
            page = (doc.metadata or {}).get("page")
            page_label = f"page {page + 1}" if isinstance(page, int) else "page n/a"
            snippets.append(f"[{label} | {page_label}] {text[:per_query_limit]}")
            break

    return "\n\n".join(snippets[:8])


def _load_report_text(active_filename: str = "") -> str:
    if not active_filename:
        return ""
    report_path = REPORTS_DIR / active_filename
    if not report_path.exists():
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(report_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                filtered_lines = []
                for raw_line in text.splitlines():
                    line = _clean_ws(raw_line)
                    if not line:
                        continue
                    if re.match(r"Page\s+\d+\s+\|", line):
                        continue
                    if "Annual Report" in line and "CONFIDENTIAL" in line:
                        continue
                    filtered_lines.append(line)
                pages.append("\n".join(filtered_lines))
        return "\n".join(pages)
    except Exception:
        return ""


def _extract_section(text: str, heading_number: int) -> str:
    pattern = re.compile(
        rf"\n{heading_number}\.\s+[^\n]+\n(.*?)(?=\n{heading_number + 1}\.\s+[^\n]+\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search("\n" + (text or ""))
    return match.group(1).strip() if match else ""


def _parse_numeric_token(token: str) -> Optional[float]:
    if not token:
        return None
    cleaned = token.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_amount_values(text: str) -> list[float]:
    if not text:
        return []
    return [
        float(match.replace(",", ""))
        for match in re.findall(r"\d[\d,]*\.?\d*", text)
    ]


def _detect_unit_system(
    revenue_text: str,
    report_text: str = "",
    preferred_currency_code: str = "",
) -> dict:
    sample = f"{revenue_text}\n{report_text[:2000]}"
    parsed = parse_money_value(
        revenue_text,
        fallback_currency=detect_currency_code(sample, DEFAULT_DISPLAY_CURRENCY),
        metric_key="revenue",
    )
    source_currency_code = (
        parsed.get("currency_code")
        if parsed
        else detect_currency_code(sample, DEFAULT_DISPLAY_CURRENCY) or DEFAULT_DISPLAY_CURRENCY
    )
    source_unit_multiplier = (
        parsed.get("unit_multiplier")
        if parsed
        else infer_unit_multiplier(sample or "million")
    )
    target_currency_code = normalize_currency_code(preferred_currency_code or source_currency_code)
    target_meta = get_currency_meta(target_currency_code)
    return {
        "source_currency_code": source_currency_code,
        "source_unit_multiplier": source_unit_multiplier,
        "target_currency_code": target_currency_code,
        "display_unit": target_meta["display_unit"],
        "currency_symbol": target_meta["prefix"],
    }


def _format_amount(value: Optional[float], unit_system: dict) -> str:
    if value is None:
        return "Not disclosed"
    return format_source_value_in_currency(
        value,
        source_currency_code=unit_system["source_currency_code"],
        source_unit_multiplier=unit_system["source_unit_multiplier"],
        target_currency_code=unit_system["target_currency_code"],
    )


def _format_change_pct(value: Optional[float]) -> str:
    if value is None:
        return "Not disclosed"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _format_bps(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "Not disclosed"
    bps = round(delta_pct * 100)
    sign = "+" if bps >= 0 else ""
    return f"{sign}{bps} bps"


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _convert_payload_currency_mentions(value, target_currency_code: str, fallback_source_currency: str = ""):
    if isinstance(value, str):
        return convert_currency_mentions(
            value,
            target_currency_code=target_currency_code,
            fallback_source_currency=fallback_source_currency,
        )
    if isinstance(value, list):
        return [
            _convert_payload_currency_mentions(item, target_currency_code, fallback_source_currency)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _convert_payload_currency_mentions(item, target_currency_code, fallback_source_currency)
            for key, item in value.items()
        }
    return value


def _extract_explicit_fy_label(user_instructions: str) -> str:
    match = re.search(r"FY\s*(\d{4})-(\d{2})", user_instructions or "", re.IGNORECASE)
    if not match:
        return ""
    start_year = int(match.group(1))
    end_suffix = int(match.group(2))
    end_year = 2000 + end_suffix if end_suffix < 100 else end_suffix
    if end_year < start_year:
        end_year += 100
    return f"FY{start_year}-{str(end_year)[-2:]}"


def _derive_target_fy_label(report_period: str, user_instructions: str = "") -> str:
    explicit = _extract_explicit_fy_label(user_instructions)
    if explicit:
        return explicit

    match = re.search(r"FY\s*(\d{4})(?:\s*-\s*(\d{2,4}))?", report_period or "", re.IGNORECASE)
    if not match:
        current_year = datetime.now().year
        return f"FY{current_year}-{str(current_year + 1)[-2:]}"

    start_year = int(match.group(1))
    end_part = match.group(2)
    if end_part:
        end_year = int(end_part)
        if end_year < 100:
            end_year = (start_year // 100) * 100 + end_year
            if end_year < start_year:
                end_year += 100
        next_start = start_year + 1
        next_end = end_year + 1
        return f"FY{next_start}-{str(next_end)[-2:]}"
    next_year = start_year + 1
    return f"FY{next_year}"


def _parse_user_targets(user_instructions: str) -> dict:
    text = user_instructions or ""
    rev_match = re.search(r"(\d+(?:\.\d+)?)%\s+revenue growth", text, re.IGNORECASE)
    margin_match = re.search(r"EBITDA margin.*?to\s*(\d+(?:\.\d+)?)%", text, re.IGNORECASE)
    margin_from_match = re.search(r"EBITDA margin\s+from\s*(\d+(?:\.\d+)?)%", text, re.IGNORECASE)
    sourcing_match = re.search(
        r"(\d+(?:\.\d+)?)%\s+domestic or non-Western sourcing.*?by\s*(Q\d)",
        text,
        re.IGNORECASE,
    )
    return {
        "revenue_growth_pct": float(rev_match.group(1)) if rev_match else None,
        "ebitda_margin_target_pct": float(margin_match.group(1)) if margin_match else None,
        "ebitda_margin_current_pct": float(margin_from_match.group(1)) if margin_from_match else None,
        "mentions_su57e": "su-57e" in text.lower(),
        "mentions_su30mki_mlu": "su-30mki" in text.lower() and "mlu" in text.lower(),
        "mentions_mumt": "mum-t" in text.lower(),
        "mentions_rd": "r&d" in text.lower() or "research" in text.lower(),
        "sourcing_target_pct": float(sourcing_match.group(1)) if sourcing_match else None,
        "sourcing_target_timeline": sourcing_match.group(2).upper() if sourcing_match else "",
    }


def _collect_numbered_items_after_label(text: str, labels: tuple[str, ...], limit: int = 8) -> list[str]:
    if not text:
        return []

    items: list[str] = []
    started = False
    stop_line_pattern = re.compile(
        r"^(?:Key Highlight|Risk Flag|Operational Footprint|Financial Baseline|Segment(?:s)?|"
        r"Revenue(?: Growth)? Guidance|EBITDA Growth|Operating Cash Flow|Capex|Strategic Priorities|"
        r"Management Outlook|Scenario|Risk Management)\b",
        re.IGNORECASE,
    )
    heading_pattern = re.compile(r"^[A-Z][A-Z0-9 /,&()'._-]{8,}$")

    for raw_line in text.splitlines():
        line = _clean_ws(raw_line)
        if not line:
            continue

        if not started:
            if any(label.lower() in line.lower() for label in labels):
                started = True
            continue

        numbered = re.match(r"^(\d+)\.\s*(.+)", line)
        if numbered:
            item_text = _clean_ws(numbered.group(2))
            if items and (
                stop_line_pattern.match(item_text)
                or heading_pattern.match(item_text)
            ):
                break
            items.append(item_text)
            if len(items) >= limit:
                break
            continue

        if not items:
            continue

        if stop_line_pattern.match(line):
            break
        if line.endswith(":") and len(line) <= 48:
            break
        if heading_pattern.match(line):
            break

        items[-1] = _clean_ws(f"{items[-1]} {line}")

    deduped: list[str] = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:limit]


def _parse_core_business_lines(report_text: str) -> list[str]:
    section = _extract_section(report_text, 2)
    section_items = _collect_numbered_items_after_label(
        section,
        ("Core Products:", "Core business lines:"),
        limit=8,
    )

    report_items = _collect_numbered_items_after_label(
        report_text,
        ("Core Products:", "Core business lines:"),
        limit=8,
    )
    if len(report_items) > len(section_items):
        return report_items
    if section_items:
        return section_items
    if report_items:
        return report_items

    if not section:
        return []
    block_match = re.search(
        r"Core (?:Products|business lines):\s*(.*?)(?:Key Highlight:|Risk Flag:)",
        section,
        re.IGNORECASE | re.DOTALL,
    )
    block = block_match.group(1) if block_match else section
    lines = re.findall(r"\n\d+\.\s*(.+)", "\n" + block)
    return [_clean_ws(line) for line in lines[:8]]


def _parse_segments(report_text: str) -> list[dict]:
    section = _extract_section(report_text, 6)
    if not section:
        return []
    pattern = re.compile(
        r"(?P<segment>[^\n:]+):\s*\nRevenue:\s*(?P<revenue>[^\n|]+?)\s*\|\s*EBITDA:\s*(?P<ebitda>[^\n|]+?)\s*\|\s*EBITDA Margin:\s*(?P<margin>[^\n]+)",
        re.IGNORECASE,
    )
    segments = []
    for match in pattern.finditer(section):
        revenue_text = _clean_ws(match.group("revenue"))
        ebitda_text = _clean_ws(match.group("ebitda"))
        margin_text = _clean_ws(match.group("margin"))
        revenue_values = _parse_amount_values(revenue_text)
        ebitda_values = _parse_amount_values(ebitda_text)
        segments.append(
            {
                "segment": _clean_ws(match.group("segment")),
                "revenue_text": revenue_text,
                "revenue_value": revenue_values[0] if revenue_values else None,
                "ebitda_text": ebitda_text,
                "ebitda_value": ebitda_values[0] if ebitda_values else None,
                "margin_text": margin_text,
                "margin_value": _parse_amount_values(margin_text)[0] if _parse_amount_values(margin_text) else None,
            }
        )
    return segments


def _parse_risks(report_text: str) -> list[dict]:
    section = _extract_section(report_text, 7)
    if not section:
        return []
    pattern = re.compile(
        r"\n\d+\.\s*(?P<risk>.+?)\s*\((?P<severity>High|Medium|Low|Medium-High)\):\s*(?P<body>.*?)(?=\n\d+\.\s+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    risks = []
    for match in pattern.finditer("\n" + section):
        body = _clean_ws(match.group("body"))
        exposure = ""
        exposure_match = re.search(
            r"([^.]*?(?:USD|Rs\.|approximately|~)[^.]*\.)|"
            r"(Long-term revenue risk of [^.]*\.)|"
            r"([^.]*\d+%[^.]*\.)",
            body,
            re.IGNORECASE,
        )
        if exposure_match:
            exposure = _clean_ws(exposure_match.group(0))
        risks.append(
            {
                "risk": _clean_ws(match.group("risk")),
                "severity": _clean_ws(match.group("severity")).title(),
                "body": body,
                "financial_exposure": exposure,
            }
        )
    return risks


def _parse_scenarios(report_text: str) -> list[dict]:
    section = _extract_section(report_text, 8)
    if not section:
        return []
    pattern = re.compile(
        r"Scenario\s+(?P<label>[A-Z])\s*-\s*(?P<scenario>.*?):\s*\n(?P<body>.*?)(?=\nScenario\s+[A-Z]\s*-|\n9\.\s+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    scenarios = []
    for match in pattern.finditer("\n" + section):
        body = _clean_ws(match.group("body"))
        revenue_match = re.search(r"Revenue (?:down|upside|risk of)?\s*([A-Z$0-9,\-\s.+]+?)(?:;|$)", body, re.IGNORECASE)
        ebitda_match = re.search(r"EBITDA (?:down|upside|impact|margin compressed)?\s*([A-Z$0-9,\-\s.+%]+?)(?:;|$)", body, re.IGNORECASE)
        scenarios.append(
            {
                "scenario": _clean_ws(match.group("scenario")),
                "body": body,
                "revenue_impact": _clean_ws(revenue_match.group(0)) if revenue_match else "",
                "ebitda_impact": _clean_ws(ebitda_match.group(0)) if ebitda_match else "",
            }
        )
    return scenarios


def _parse_guidance(report_text: str) -> dict:
    section = _extract_section(report_text, 9)
    if not section:
        return {}
    data = {}
    patterns = {
        "revenue_growth_guidance": r"Revenue Growth Guidance:\s*([^\n]+)",
        "ebitda_growth_guidance": r"EBITDA Growth:\s*([^\n]+)",
        "operating_cash_flow_guidance": r"Operating Cash Flow:\s*([^\n]+)",
        "capex_guidance": r"Capex:\s*([^\n]+)",
        "net_debt_ebitda_guidance": r"Net Debt/EBITDA:\s*([^\n]+)",
        "regional_share_guidance": r"(?:India Revenue Share|Domestic Content Index):\s*([^\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, section, re.IGNORECASE)
        if match:
            data[key] = _clean_ws(match.group(1))
    data["body"] = _clean_ws(section)
    return data


def _parse_report_specific_metrics(report_text: str) -> dict:
    metrics = {}
    margin_match = re.search(r"EBITDA Margin:\s*([0-9.]+%)", report_text, re.IGNORECASE)
    rd_match = re.search(r"R&D as % of Revenue:\s*([0-9.]+%)\s*\(([^)]+)\)", report_text, re.IGNORECASE)
    order_book_match = re.search(r"Order Book \(Total\):\s*([^\n]+)", report_text, re.IGNORECASE)
    if margin_match:
        metrics["ebitda_margin"] = margin_match.group(1)
    if rd_match:
        metrics["rd_pct"] = rd_match.group(1)
        metrics["rd_amount"] = _clean_ws(rd_match.group(2))
    if order_book_match:
        metrics["order_book"] = _clean_ws(order_book_match.group(1))
    return metrics


def _build_source_validated_baseline_snapshot(kpis: dict, report_metrics: dict, report_text: str) -> list[dict]:
    snapshot: list[dict] = []

    def add_item(label: str, value: Optional[str]) -> None:
        cleaned = _clean_ws(str(value)) if value not in (None, "", "null") else ""
        if cleaned:
            snapshot.append({"label": label, "value": cleaned})

    add_item("Revenue", kpis.get("revenue"))
    add_item("Revenue Growth", kpis.get("revenue_growth"))
    add_item("EBITDA", kpis.get("ebitda"))
    add_item("EBITDA Margin", report_metrics.get("ebitda_margin") or kpis.get("ebitda_margin"))
    add_item("Net Income / PAT", kpis.get("net_income"))
    add_item("Operating Cash Flow", kpis.get("operating_cash_flow"))
    add_item("Total Debt", kpis.get("total_debt"))
    add_item("ROE", kpis.get("roe"))

    if report_metrics.get("rd_pct") and report_metrics.get("rd_amount"):
        add_item("R&D Intensity", f"{report_metrics['rd_pct']} ({report_metrics['rd_amount']})")
    elif report_metrics.get("rd_pct"):
        add_item("R&D Intensity", report_metrics["rd_pct"])

    add_item("Order Book", report_metrics.get("order_book"))

    gross_margin_match = re.search(r"Gross Margin:\s*([0-9.]+%)", report_text, re.IGNORECASE)
    if gross_margin_match:
        add_item("Gross Margin", gross_margin_match.group(1))

    add_item("Key Highlight", kpis.get("key_highlight"))
    add_item("Risk Flag", kpis.get("risk_flag"))
    return snapshot


def _first_amount(text: str) -> Optional[float]:
    values = _parse_amount_values(text)
    return values[0] if values else None


def _range_midpoint(text: str) -> Optional[float]:
    values = _parse_amount_values(text)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return sum(values[:2]) / 2


def _project_segment_revenues(segments: list[dict], target_revenue: float, user_targets: dict) -> list[dict]:
    if not segments or target_revenue is None:
        return []

    preferences = []
    for segment in segments:
        name = segment.get("segment", "").lower()
        if "fighter" in name:
            pref = 16.5 if user_targets.get("mentions_su57e") else 12.0
            programs = "Su-57E export pathway, Su-30 family deliveries"
        elif "mro" in name or "upgrade" in name:
            pref = 15.0 if user_targets.get("mentions_su30mki_mlu") else 10.0
            programs = "Su-30MKI Phase-2 MLU mobilisation and fleet sustainment"
        elif "advanced" in name:
            pref = 18.0 if user_targets.get("mentions_mumt") else 12.0
            programs = "Su-57E development support, next-generation systems R&D"
        elif "spares" in name or "logistics" in name:
            pref = 7.0
            programs = "Spares support, aftermarket readiness"
        else:
            pref = 6.0
            programs = segment.get("segment", "Core program support")
        preferences.append((segment, pref, programs))

    raw_total = sum(
        (item[0].get("revenue_value") or 0) * (1 + item[1] / 100)
        for item in preferences
    )
    scale = target_revenue / raw_total if raw_total else 1.0

    projected = []
    for segment, pref_growth, programs in preferences:
        base_revenue = segment.get("revenue_value")
        if base_revenue is None:
            continue
        projected_value = base_revenue * (1 + pref_growth / 100) * scale
        actual_growth = ((projected_value / base_revenue) - 1) * 100 if base_revenue else None
        margin_value = segment.get("margin_value")
        projected.append(
            {
                "segment": segment.get("segment", "Segment"),
                "projected_revenue_value": projected_value,
                "growth_pct": actual_growth,
                "margin_text": f"Current baseline {segment.get('margin_text')}" if margin_value is not None else "Not separately guided",
                "key_programs": programs,
            }
        )
    return projected


def _build_grounded_projection_payload(
    kpis: dict,
    company_name: str,
    user_instructions: str,
    report_text: str,
    fy_label: str,
    preferred_currency_code: str = "",
) -> dict:
    user_targets = _parse_user_targets(user_instructions)
    unit_system = _detect_unit_system(
        kpis.get("revenue", ""),
        report_text,
        preferred_currency_code=preferred_currency_code,
    )
    report_metrics = _parse_report_specific_metrics(report_text)
    guidance = _parse_guidance(report_text)
    segments = _parse_segments(report_text)
    risks = _parse_risks(report_text)
    scenarios = _parse_scenarios(report_text)
    business_lines = _parse_core_business_lines(report_text)

    current_revenue = _first_amount(kpis.get("revenue", ""))
    current_ebitda = _first_amount(kpis.get("ebitda", ""))
    current_pat = _first_amount(kpis.get("net_income", ""))
    current_ocf = _first_amount(kpis.get("operating_cash_flow", ""))
    current_debt = _first_amount(kpis.get("total_debt", ""))
    current_ebitda_margin = _first_amount(report_metrics.get("ebitda_margin", "")) or (
        _safe_div(current_ebitda, current_revenue) * 100 if _safe_div(current_ebitda, current_revenue) is not None else None
    )
    current_rd_pct = _first_amount(report_metrics.get("rd_pct", ""))
    current_rd_amount = _first_amount(report_metrics.get("rd_amount", ""))

    revenue_growth_target = user_targets.get("revenue_growth_pct")
    if revenue_growth_target is None:
        guidance_values = _parse_amount_values(guidance.get("revenue_growth_guidance", ""))
        revenue_growth_target = sum(guidance_values[:2]) / 2 if len(guidance_values) >= 2 else 10.0

    ebitda_margin_target = user_targets.get("ebitda_margin_target_pct")
    if ebitda_margin_target is None and current_ebitda_margin is not None:
        ebitda_margin_target = current_ebitda_margin + 0.8

    projected_revenue = current_revenue * (1 + revenue_growth_target / 100) if current_revenue is not None else None
    projected_ebitda = (
        projected_revenue * ebitda_margin_target / 100
        if projected_revenue is not None and ebitda_margin_target is not None
        else None
    )

    pat_conversion = _safe_div(current_pat, current_ebitda)
    projected_pat = projected_ebitda * pat_conversion if projected_ebitda is not None and pat_conversion is not None else None

    ocf_conversion = _safe_div(current_ocf, current_ebitda)
    projected_ocf = projected_ebitda * ocf_conversion if projected_ebitda is not None and ocf_conversion is not None else None

    rd_pct_target = current_rd_pct
    if current_rd_pct is not None and (user_targets.get("mentions_rd") or user_targets.get("mentions_mumt")):
        rd_pct_target = round(current_rd_pct + 0.8, 1)
    projected_rd_amount = (
        projected_revenue * rd_pct_target / 100
        if projected_revenue is not None and rd_pct_target is not None
        else None
    )

    capex_midpoint = _range_midpoint(guidance.get("capex_guidance", ""))
    ocf_guidance_midpoint = _range_midpoint(guidance.get("operating_cash_flow_guidance", ""))
    net_debt_ebitda_target = guidance.get("net_debt_ebitda_guidance", "")
    flat_debt_leverage = _safe_div(current_debt, projected_ebitda)

    projected_segments = _project_segment_revenues(segments, projected_revenue, user_targets)
    baseline_snapshot = _build_source_validated_baseline_snapshot(kpis, report_metrics, report_text)

    mandate_targets = []
    if revenue_growth_target is not None:
        mandate_targets.append(f"Deliver approximately {revenue_growth_target:.1f}% revenue growth in {fy_label}.")
    if ebitda_margin_target is not None:
        mandate_targets.append(f"Lift EBITDA margin to about {ebitda_margin_target:.1f}% in {fy_label}.")
    if user_targets.get("mentions_su57e"):
        mandate_targets.append("Convert the Su-57E export opportunity into firm commercial progress.")
    if user_targets.get("mentions_su30mki_mlu"):
        mandate_targets.append("Execute the Su-30MKI Phase-2 MLU programme on schedule; full completion is not supported by the source-report FY timeline.")
    if user_targets.get("mentions_mumt"):
        mandate_targets.append("Increase next-generation R&D intensity, including MUM-T capability workstreams, as a target-case assumption.")
    if user_targets.get("sourcing_target_pct") is not None:
        mandate_targets.append(
            f"Move critical microelectronics sourcing toward {user_targets['sourcing_target_pct']:.0f}% domestic or non-Western coverage by {user_targets.get('sourcing_target_timeline') or 'the stated deadline'}."
        )

    assumption_flags = []
    if guidance.get("revenue_growth_guidance") and revenue_growth_target is not None:
        assumption_flags.append(
            f"Source management guidance for the next fiscal year is {guidance['revenue_growth_guidance']}; this report models a {revenue_growth_target:.1f}% target case."
        )
    if user_targets.get("mentions_su30mki_mlu"):
        assumption_flags.append(
            "The FY2024-25 source report says Su-30MKI Phase-2 MLU work is expected to begin in Q1 of the next fiscal year and contribute over three years, so full completion within one year is not source-supported."
        )
    if user_targets.get("mentions_mumt"):
        assumption_flags.append(
            "MUM-T capability expansion is treated as a management target-case assumption because it is not explicitly described in the FY2024-25 source report."
        )
    if user_targets.get("sourcing_target_pct") is not None:
        assumption_flags.append(
            "The source report flags sanctions-related microelectronics dependence but does not disclose a current percentage mix, so the sourcing target is modeled qualitatively rather than from a disclosed baseline."
        )

    executive_summary = (
        f"{company_name} enters {fy_label} with a reported FY base of {_format_amount(current_revenue, unit_system)}, "
        f"EBITDA of {_format_amount(current_ebitda, unit_system)}, and EBITDA margin of {report_metrics.get('ebitda_margin', 'not disclosed')}. "
        f"This projection models a management target case rather than a pure carry-forward of source guidance: revenue is set at {_format_amount(projected_revenue, unit_system)} "
        f"({_format_change_pct(revenue_growth_target)}) and EBITDA margin at approximately {ebitda_margin_target:.1f}% based on the user's stated mandate. "
        f"The key upside driver is the Su-57E export pathway, while the key execution engine is the Su-30MKI Phase-2 MLU programme for India. "
        f"The main constraint remains sanctions-linked sourcing risk in avionics and microelectronics, which the report itself flags as a high-severity issue. "
        f"Source management guidance for the next fiscal year is {guidance.get('revenue_growth_guidance', 'not disclosed')}, so this target case should be read as a stretch scenario above baseline guidance where applicable. "
        f"Where the user mandate goes beyond the source report, those items are disclosed explicitly as assumptions rather than treated as reported facts."
    )

    company_overview_text = (
        f"{company_name} is a combat-aircraft and aerospace systems manufacturer with a reported order book of {report_metrics.get('order_book', 'not disclosed')} "
        f"and a material India-linked export and MRO base. The FY2024-25 report highlights the Su-57E export variant, the Su-30MKI Phase-2 MLU contract, and persistent sanctions-related sourcing constraints as the main strategic themes."
    )

    strategic_context_text = (
        f"The grounded base case in the source report points to {guidance.get('revenue_growth_guidance', 'undisclosed next-year growth guidance')} and "
        f"{guidance.get('ebitda_growth_guidance', 'undisclosed EBITDA growth guidance')}. This target-case report intentionally layers a more aggressive management mandate on top of that base, "
        f"so the right interpretation is not 'reported guidance' but 'board-level stretch case with disclosed assumptions'."
    )

    projection_rows = [
        {
            "metric": "Revenue",
            "current": _format_amount(current_revenue, unit_system),
            "projected": _format_amount(projected_revenue, unit_system),
            "change": _format_change_pct(revenue_growth_target),
            "commentary": (
                f"The revenue target is modeled directly from the user's {revenue_growth_target:.1f}% growth ask on the FY2024-25 base. "
                f"This is above the source report's next-year revenue guidance of {guidance.get('revenue_growth_guidance', 'not disclosed')}."
            ),
        },
        {
            "metric": "EBITDA",
            "current": _format_amount(current_ebitda, unit_system),
            "projected": _format_amount(projected_ebitda, unit_system),
            "change": _format_change_pct(((projected_ebitda / current_ebitda) - 1) * 100 if current_ebitda and projected_ebitda else None),
            "commentary": (
                f"Projected EBITDA is calculated from the target-case revenue base and an EBITDA margin of {ebitda_margin_target:.1f}%. "
                "This is a target-case arithmetic output, not a separately reported management figure."
            ),
        },
        {
            "metric": "EBITDA Margin",
            "current": report_metrics.get("ebitda_margin", "Not disclosed"),
            "projected": f"{ebitda_margin_target:.1f}%",
            "change": _format_bps(ebitda_margin_target - current_ebitda_margin if current_ebitda_margin is not None else None),
            "commentary": (
                "The margin improvement assumption reflects the user's explicit target and is directionally consistent with reducing titanium-cost drag and supply-chain inefficiency. "
                "The source report itself does not guide to 18.5%, so this should be read as a target case."
            ),
        },
        {
            "metric": "PAT",
            "current": _format_amount(current_pat, unit_system),
            "projected": _format_amount(projected_pat, unit_system),
            "change": _format_change_pct(((projected_pat / current_pat) - 1) * 100 if current_pat and projected_pat else None),
            "commentary": (
                "PAT is projected by holding the current PAT-to-EBITDA conversion broadly constant, which is the most conservative grounded method available from the source data."
            ),
        },
        {
            "metric": "Operating Cash Flow",
            "current": _format_amount(current_ocf, unit_system),
            "projected": _format_amount(projected_ocf, unit_system),
            "change": _format_change_pct(((projected_ocf / current_ocf) - 1) * 100 if current_ocf and projected_ocf else None),
            "commentary": (
                f"Current OCF-to-EBITDA conversion is carried forward into the target case. Source management guidance for next-year OCF is {guidance.get('operating_cash_flow_guidance', 'not disclosed')}, "
                "so any value above that range should be interpreted as stretch-case output contingent on the full margin target being achieved."
            ),
        },
    ]

    if current_rd_amount is not None and projected_rd_amount is not None and rd_pct_target is not None:
        projection_rows.append(
            {
                "metric": "R&D Spend",
                "current": _format_amount(current_rd_amount, unit_system),
                "projected": _format_amount(projected_rd_amount, unit_system),
                "change": f"{rd_pct_target:.1f}% of revenue",
                "commentary": (
                    f"Current disclosed R&D intensity is {report_metrics.get('rd_pct', 'not disclosed')}. "
                    f"The target case raises this to {rd_pct_target:.1f}% of revenue to reflect the user's instruction to emphasise next-generation R&D."
                ),
            }
        )

    if capex_midpoint is not None:
        projection_rows.append(
            {
                "metric": "Capex",
                "current": "Not disclosed in FY2024-25 source excerpt",
                "projected": _format_amount(capex_midpoint, unit_system),
                "change": "Source-guidance midpoint",
                "commentary": (
                    f"The source report guides next-year capex at {guidance.get('capex_guidance', 'not disclosed')}; the midpoint is used here rather than an invented target-case number."
                ),
            }
        )

    if user_targets.get("sourcing_target_pct") is not None:
        projection_rows.append(
            {
                "metric": "Critical Microelectronics Sourcing",
                "current": "Sanctions-affected dependence flagged; no current % disclosed",
                "projected": f"{user_targets['sourcing_target_pct']:.0f}% domestic or non-Western sourcing by {user_targets.get('sourcing_target_timeline') or 'target deadline'}",
                "change": "Strategic de-risking milestone",
                "commentary": (
                    "This is included as a target-case milestone because the source report flags sanctions-related microelectronics dependence but does not provide a quantified baseline mix."
                ),
            }
        )

    revenue_margin_bridge = [
        {
            "driver": "Su-57E export pathway",
            "impact": "Revenue",
            "detail": (
                "The source report explicitly identifies the Su-57E export variant as a major next-year catalyst and source Scenario B quantifies a first export-order upside at USD 960 Mn of revenue. "
                "The target-case revenue growth assumption relies heavily on at least partial execution against that pathway."
            ),
        },
        {
            "driver": "Su-30MKI Phase-2 MLU execution",
            "impact": "Revenue / Cash Flow",
            "detail": (
                "The source report says formal work is expected to begin in Q1 of the next fiscal year and contribute USD 320-380 Mn over three years. "
                "That supports mobilisation and revenue recognition, but not full one-year completion."
            ),
        },
        {
            "driver": "Titanium and alloy cost discipline",
            "impact": "Margin",
            "detail": (
                "The source risk framework states that a USD 200/tonne rise in titanium reduces EBITDA by approximately USD 38 Mn. "
                "Any path to 18.5% EBITDA margin therefore depends on offsetting or avoiding that raw-material headwind."
            ),
        },
        {
            "driver": "R&D and sourcing de-risking",
            "impact": "Long-term Margin / Resilience",
            "detail": (
                "Higher R&D and non-Western/domestic substitution do not create instant margin uplift, but they reduce future execution fragility and support export sustainability under sanctions pressure."
            ),
        },
    ]

    segment_projection_items = []
    for item in projected_segments:
        segment_projection_items.append(
            {
                "segment": item["segment"],
                "projected_revenue": _format_amount(item["projected_revenue_value"], unit_system),
                "growth": _format_change_pct(item["growth_pct"]),
                "margin": item["margin_text"],
                "key_programs": item["key_programs"],
                "outlook": (
                    "Projected revenue is scaled so all segment totals reconcile back to the report-wide target-case revenue number. "
                    "Segment margin uplift is not separately guided in the source report, so the margin field is shown as the current baseline rather than an invented point forecast."
                ),
            }
        )

    milestone_items = []
    if user_targets.get("mentions_su57e"):
        milestone_items.append(
            {
                "program": "Su-57E export variant",
                "milestone": "First international commercial contract / order conversion",
                "timeline": fy_label,
                "business_impact": "The source report identifies the first export order as a major revenue catalyst and Scenario B quantifies USD 960 Mn of potential upside.",
                "execution_priority": "This is source-supported and central to any target case above baseline guidance.",
            }
        )
    if user_targets.get("mentions_su30mki_mlu"):
        milestone_items.append(
            {
                "program": "Su-30MKI Phase-2 MLU",
                "milestone": "Programme start, customer mobilisation, and milestone execution",
                "timeline": f"Q1 {fy_label}",
                "business_impact": "The source report says formal work begins in Q1 of the next fiscal year and contributes USD 320-380 Mn over three years.",
                "execution_priority": "Grounded milestone is mobilisation, not full completion within one year.",
            }
        )
    if user_targets.get("sourcing_target_pct") is not None:
        milestone_items.append(
            {
                "program": "Critical microelectronics sourcing",
                "milestone": "Domestic / non-Western substitution plan",
                "timeline": user_targets.get("sourcing_target_timeline") or fy_label,
                "business_impact": "This addresses the high-severity sanctions and export-control risk identified in the source report.",
                "execution_priority": "This is a target-case assumption rather than a source-disclosed milestone.",
            }
        )
    if user_targets.get("mentions_mumt"):
        milestone_items.append(
            {
                "program": "Next-generation MUM-T R&D",
                "milestone": "Incremental R&D allocation and capability workstream launch",
                "timeline": fy_label,
                "business_impact": "This supports longer-term product relevance but is not directly quantified in the FY2024-25 source report.",
                "execution_priority": "Treat as a management assumption, not a reported program milestone.",
            }
        )

    risk_items = []
    for risk in risks[:5]:
        body = risk.get("body", "")
        mitigation = (
            "Use source-guided monitoring and conservative scenario planning. "
            "Where the user mandate directly addresses the risk, that mitigation is included as a target-case response rather than as a reported fact."
        )
        if "sanctions" in risk["risk"].lower():
            mitigation = "Prioritise non-Western and domestic substitution for critical electronics, expand qualified suppliers, and avoid relying on unsupported Western inputs."
        elif "titanium" in risk["risk"].lower() or "aluminium" in risk["risk"].lower():
            mitigation = "Tighten raw-material procurement, lock in supply where possible, and treat commodity inflation as the main margin swing factor."
        elif "india" in risk["risk"].lower():
            mitigation = "Preserve India programme execution quality while broadening export conversion, because the source report already quantifies concentration risk."
        risk_items.append(
            {
                "risk": risk["risk"],
                "severity": risk["severity"],
                "financial_exposure": risk.get("financial_exposure") or body,
                "mitigation": mitigation,
                "watchpoint": body,
            }
        )

    scenario_matrix = [
        {
            "scenario": "Management Target Case",
            "revenue_impact": _format_amount(projected_revenue, unit_system),
            "ebitda_impact": f"EBITDA margin target of {ebitda_margin_target:.1f}%",
            "probability": "Medium",
            "implication": (
                f"This is the user-requested stretch case. It sits above the source report's next-year revenue guidance of {guidance.get('revenue_growth_guidance', 'not disclosed')} and therefore requires clean execution."
            ),
        }
    ]
    for scenario in scenarios[:4]:
        label = scenario["scenario"]
        probability = "Medium"
        if "export order" in label.lower():
            probability = "Medium"
        elif "india reduces" in label.lower() or "cost" in label.lower():
            probability = "Medium"
        scenario_matrix.append(
            {
                "scenario": label,
                "revenue_impact": scenario.get("revenue_impact") or "See source scenario detail",
                "ebitda_impact": scenario.get("ebitda_impact") or "See source scenario detail",
                "probability": probability,
                "implication": scenario.get("body", ""),
            }
        )

    priorities = [
        "Keep the full report in one unit system and one fiscal-year frame.",
        "Convert the Su-57E opportunity into firm commercial progress rather than implied future upside.",
        "Execute Su-30MKI Phase-2 MLU mobilisation on the source-reported timeline.",
        "Treat sanctions-linked microelectronics dependence as the primary strategic de-risking agenda.",
    ]
    if user_targets.get("mentions_mumt"):
        priorities.append("Raise next-generation R&D intensity transparently, with the increase disclosed as a management target-case assumption.")

    management_outlook = {
        "guidance_headline": (
            f"Source management guidance for the next fiscal year is {guidance.get('revenue_growth_guidance', 'not disclosed')} with EBITDA growth of {guidance.get('ebitda_growth_guidance', 'not disclosed')}. "
            f"This report instead models a target case of {_format_amount(projected_revenue, unit_system)} revenue and about {ebitda_margin_target:.1f}% EBITDA margin, which is more ambitious than the reported baseline."
        ),
        "board_watch_items": [
            "Whether Su-57E export conversion actually materialises into firm contracts.",
            "Whether Su-30MKI Phase-2 MLU starts on the source-reported Q1 timeline.",
            "Whether titanium and alloy inflation erodes the margin-improvement target.",
            "Whether sanctions-driven substitution of microelectronics progresses fast enough to de-risk supply.",
        ],
        "closing_commentary": (
            "The corrected reading of the source report is that a stronger next fiscal year is plausible, but some user-requested outcomes are stretch assumptions rather than reported management guidance. "
            "This report therefore prioritises internal consistency, source-anchored sensitivities, and explicit disclosure of unsupported assumptions over overly aggressive narrative certainty."
        ),
    }

    payload = {
        "executive_summary": executive_summary,
        "company_overview": {
            "current_position": company_overview_text,
            "business_lines": business_lines,
            "strategic_context": strategic_context_text,
        },
        "management_mandate": {
            "summary": (
                f"For {fy_label}, management's requested target case is to deliver {revenue_growth_target:.1f}% revenue growth, move EBITDA margin toward {ebitda_margin_target:.1f}%, "
                "advance the Su-57E and Su-30MKI program agenda, raise next-generation R&D emphasis, and reduce sanctions-linked sourcing risk. "
                "Where the source report does not fully support the requested timeline or baseline, that gap is disclosed explicitly."
            ),
            "targets": mandate_targets,
        },
        "baseline_snapshot": baseline_snapshot,
        "projection_scorecard": projection_rows,
        "revenue_margin_bridge": revenue_margin_bridge,
        "segment_projections": segment_projection_items,
        "program_milestones": milestone_items,
        "balance_sheet_cashflow": {
            "operating_cash_flow": (
                f"{_format_amount(projected_ocf, unit_system)} target-case output; source guidance {guidance.get('operating_cash_flow_guidance', 'not disclosed')}"
            ),
            "capex": _format_amount(capex_midpoint, unit_system) if capex_midpoint is not None else guidance.get("capex_guidance", "Not disclosed"),
            "net_debt": (
                f"{net_debt_ebitda_target}; flat-debt sensitivity implies about {flat_debt_leverage:.2f}x"
                if flat_debt_leverage is not None and net_debt_ebitda_target
                else net_debt_ebitda_target or "Not disclosed"
            ),
            "domestic_content": (
                f"{user_targets['sourcing_target_pct']:.0f}% critical microelectronics from domestic/non-Western sources by {user_targets.get('sourcing_target_timeline')}"
                if user_targets.get("sourcing_target_pct") is not None
                else "No quantified sourcing target disclosed"
            ),
            "commentary": (
                f"Source next-year capex guidance is {guidance.get('capex_guidance', 'not disclosed')} and source leverage guidance is {guidance.get('net_debt_ebitda_guidance', 'not disclosed')}. "
                "This section keeps those guideposts intact and only layers target-case arithmetic on OCF where it can be reconciled back to the EBITDA assumption."
            ),
        },
        "risks": risk_items,
        "scenario_matrix": scenario_matrix,
        "strategic_priorities": priorities,
        "management_outlook": management_outlook,
        "assumption_flags": assumption_flags,
        "_generation_notice": (
            "Grounded target-case projection built from the uploaded source report, extracted KPI baseline, and user mandate. "
            "Unsupported asks are disclosed as assumptions rather than treated as reported facts."
        ),
    }

    return _convert_payload_currency_mentions(
        payload,
        target_currency_code=unit_system["target_currency_code"],
        fallback_source_currency=unit_system["source_currency_code"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Projection Engine
# ─────────────────────────────────────────────────────────────────────────────
def generate_projections_with_llm(
    kpis: dict,
    company_name: str,
    user_instructions: str,
    answer_provider: str,
    api_key: str,
    answer_model: str = "",
    openai_key: str = "",
    gemini_key: str = "",
    vectorstore=None,
    active_filename: str = "",
    preferred_currency_code: str = "",
) -> dict:
    """Generate a detailed board-style projection report payload."""
    fy_label = _derive_target_fy_label(kpis.get("report_period", ""), user_instructions)
    normalized_provider = normalize_answer_provider(answer_provider)
    selected_model = answer_model or _default_model_for_provider(normalized_provider)
    source_context = collect_report_context(vectorstore=vectorstore, active_filename=active_filename)
    report_text = _load_report_text(active_filename)

    if report_text:
        return _build_grounded_projection_payload(
            kpis=kpis,
            company_name=company_name,
            user_instructions=user_instructions,
            report_text=report_text,
            fy_label=fy_label,
            preferred_currency_code=preferred_currency_code,
        )
    full_prompt = ""

    system_prompt = f"""You are a top-tier equity research analyst and defence-sector strategy advisor.
Your task is to write a board-ready annual projection report for {company_name} for {fy_label}.

Use ONLY the supplied KPI snapshot, retrieved source excerpts from the uploaded report, and the user's management mandate.
You may extrapolate future projections, but they must stay internally consistent and grounded in the supplied facts.
Do not produce placeholders such as "N/A" or "not available" unless the source truly gives you zero basis.
When exact future amounts are uncertain, provide a management-style projected estimate and explain the logic in commentary.
Present monetary outputs in {normalize_currency_code(preferred_currency_code or DEFAULT_DISPLAY_CURRENCY)} unless the user explicitly requests another currency.

Return ONLY valid JSON in exactly this structure:
{{
  "executive_summary": "6-8 sentence high-conviction summary for the next fiscal year",
  "company_overview": {{
    "current_position": "1 paragraph on where the company stands today",
    "business_lines": ["line 1", "line 2", "line 3", "line 4"],
    "strategic_context": "1 paragraph linking current base to next-year ambition"
  }},
  "management_mandate": {{
    "summary": "1 paragraph summarising the user's ask in management language",
    "targets": ["target 1", "target 2", "target 3", "target 4", "target 5"]
  }},
  "projection_scorecard": [
    {{"metric": "Revenue", "current": "current-year value", "projected": "next-year projection", "change": "+X%", "commentary": "2-3 sentence commentary"}},
    {{"metric": "EBITDA", "current": "current-year value", "projected": "next-year projection", "change": "+X%", "commentary": "2-3 sentence commentary"}},
    {{"metric": "EBITDA Margin", "current": "current-year margin", "projected": "next-year margin", "change": "+X bps", "commentary": "2-3 sentence commentary"}},
    {{"metric": "PAT", "current": "current-year value", "projected": "next-year projection", "change": "+X%", "commentary": "2-3 sentence commentary"}},
    {{"metric": "Operating Cash Flow", "current": "current-year value", "projected": "next-year projection", "change": "+X%", "commentary": "2-3 sentence commentary"}},
    {{"metric": "Capex", "current": "current-year value", "projected": "next-year projection", "change": "+/-X%", "commentary": "2-3 sentence commentary"}},
    {{"metric": "R&D Spend", "current": "current-year value", "projected": "next-year projection", "change": "X% of revenue", "commentary": "2-3 sentence commentary"}},
    {{"metric": "Domestic Content / Localisation", "current": "current-year value", "projected": "next-year projection", "change": "+X pts", "commentary": "2-3 sentence commentary"}}
  ],
  "revenue_margin_bridge": [
    {{"driver": "driver 1", "impact": "Revenue/Margin/Cash Flow", "detail": "2-3 sentences"}},
    {{"driver": "driver 2", "impact": "Revenue/Margin/Cash Flow", "detail": "2-3 sentences"}},
    {{"driver": "driver 3", "impact": "Revenue/Margin/Cash Flow", "detail": "2-3 sentences"}},
    {{"driver": "driver 4", "impact": "Revenue/Margin/Cash Flow", "detail": "2-3 sentences"}}
  ],
  "segment_projections": [
    {{"segment": "Segment name", "projected_revenue": "target-currency amount", "growth": "+X%", "margin": "X%", "key_programs": "named programs or product lines", "outlook": "2-3 sentences"}},
    {{"segment": "Segment name", "projected_revenue": "target-currency amount", "growth": "+X%", "margin": "X%", "key_programs": "named programs or product lines", "outlook": "2-3 sentences"}},
    {{"segment": "Segment name", "projected_revenue": "target-currency amount", "growth": "+X%", "margin": "X%", "key_programs": "named programs or product lines", "outlook": "2-3 sentences"}}
  ],
  "program_milestones": [
    {{"program": "Program name", "milestone": "milestone", "timeline": "Qx FY / Hx FY", "business_impact": "1-2 sentences", "execution_priority": "1 sentence"}},
    {{"program": "Program name", "milestone": "milestone", "timeline": "Qx FY / Hx FY", "business_impact": "1-2 sentences", "execution_priority": "1 sentence"}},
    {{"program": "Program name", "milestone": "milestone", "timeline": "Qx FY / Hx FY", "business_impact": "1-2 sentences", "execution_priority": "1 sentence"}}
  ],
  "balance_sheet_cashflow": {{
    "operating_cash_flow": "projected OCF",
    "capex": "projected capex",
    "net_debt": "projected net debt or leverage",
    "domestic_content": "projected localisation or sourcing outcome",
    "commentary": "1 paragraph on funding, leverage, capex, and working capital"
  }},
  "risks": [
    {{"risk": "Risk name", "severity": "High/Medium/Low", "financial_exposure": "quantified or directional exposure", "mitigation": "2 sentences", "watchpoint": "leading indicator to monitor"}},
    {{"risk": "Risk name", "severity": "High/Medium/Low", "financial_exposure": "quantified or directional exposure", "mitigation": "2 sentences", "watchpoint": "leading indicator to monitor"}},
    {{"risk": "Risk name", "severity": "High/Medium/Low", "financial_exposure": "quantified or directional exposure", "mitigation": "2 sentences", "watchpoint": "leading indicator to monitor"}},
    {{"risk": "Risk name", "severity": "High/Medium/Low", "financial_exposure": "quantified or directional exposure", "mitigation": "2 sentences", "watchpoint": "leading indicator to monitor"}}
  ],
  "scenario_matrix": [
    {{"scenario": "Base Case", "revenue_impact": "impact", "ebitda_impact": "impact", "probability": "High/Medium/Low", "implication": "2 sentences"}},
    {{"scenario": "Upside Case", "revenue_impact": "impact", "ebitda_impact": "impact", "probability": "High/Medium/Low", "implication": "2 sentences"}},
    {{"scenario": "Downside Case", "revenue_impact": "impact", "ebitda_impact": "impact", "probability": "High/Medium/Low", "implication": "2 sentences"}},
    {{"scenario": "Cost Shock / Supply Shock", "revenue_impact": "impact", "ebitda_impact": "impact", "probability": "High/Medium/Low", "implication": "2 sentences"}}
  ],
  "strategic_priorities": ["priority 1", "priority 2", "priority 3", "priority 4"],
  "management_outlook": {{
    "guidance_headline": "1 paragraph with the overall FY call",
    "board_watch_items": ["item 1", "item 2", "item 3", "item 4"],
    "closing_commentary": "1 closing paragraph"
  }}
}}

Writing rules:
- Make it read like a professional annual-report projection section, not a chatbot answer.
- Use exact named programmes, segments, and risks when they appear in the source context.
- Keep every commentary field concrete, quantitative, and businesslike.
- Use the requested display currency consistently across the report output.
- Ensure internal consistency between growth rates, margins, and narrative.
"""

    user_prompt = f"""Company: {company_name}
Projection Year: {fy_label}
Answer Engine: {normalized_provider}

Current KPI Snapshot:
{_format_kpi_snapshot(kpis)}

Retrieved Source Context From Uploaded Report:
{source_context or "No additional source excerpts were available from the uploaded report."}

Management Instructions:
{user_instructions}

Generate the JSON now."""

    raw_response = ""
    primary_error = None
    notices = []
    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        raw_response = _generate_report_text_with_provider(
            provider=normalized_provider,
            prompt=full_prompt,
            requested_model=selected_model,
            groq_api_key=api_key,
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
        )
    except Exception as exc:
        primary_error = exc
        if normalized_provider != "groq" and api_key:
            raw_response = _generate_report_text_via_groq(
                prompt=full_prompt,
                api_key=api_key,
                model=REPORT_GROQ_MODEL,
                temperature=0.2,
            )
            notices.append(
                f"Primary provider {normalized_provider} failed and ArthaMind used Groq fallback for this report."
            )
        else:
            raise RuntimeError(f"Projection generation failed via {normalized_provider}: {exc}") from exc

    try:
        payload = _extract_json_payload(raw_response)
    except Exception as exc:
        if api_key:
            try:
                repaired_response = _repair_projection_json(full_prompt, raw_response, api_key)
                payload = _extract_json_payload(repaired_response)
                notices.append("Projection JSON required a Groq repair pass before rendering.")
            except Exception as repair_exc:
                error_prefix = (
                    f"Primary provider failed ({primary_error}) and Groq fallback returned malformed JSON."
                    if primary_error and normalized_provider != "groq"
                    else f"{normalized_provider.capitalize()} returned malformed JSON."
                )
                raise RuntimeError(
                    f"{error_prefix} Raw excerpt: {raw_response[:400]}"
                ) from repair_exc
        else:
            error_prefix = (
                f"Primary provider failed ({primary_error}) and Groq fallback returned non-JSON output."
                if primary_error and normalized_provider != "groq"
                else f"{normalized_provider.capitalize()} returned non-JSON output."
            )
            raise RuntimeError(f"{error_prefix} Raw excerpt: {raw_response[:400]}") from exc

    if notices:
        payload["_generation_notice"] = " ".join(notices)
    fallback_source_currency = detect_currency_code(kpis.get("revenue", ""), DEFAULT_DISPLAY_CURRENCY)
    return _convert_payload_currency_mentions(
        payload,
        target_currency_code=normalize_currency_code(preferred_currency_code or DEFAULT_DISPLAY_CURRENCY),
        fallback_source_currency=fallback_source_currency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PDF Builder
# ─────────────────────────────────────────────────────────────────────────────
class ArthaMindReport(FPDF):
    def __init__(self, company_name: str, fy_label: str):
        super().__init__()
        self.company_name = company_name
        self.fy_label = fy_label
        self.base_font_family = "Helvetica"
        self.unicode_font_enabled = self._register_fonts()
        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _register_fonts(self) -> bool:
        regular = _find_font_path("")
        if not regular:
            return False

        try:
            self.add_font(UNICODE_FONT_FAMILY, style="", fname=regular)
            self.add_font(UNICODE_FONT_FAMILY, style="B", fname=_find_font_path("B") or regular)
            self.add_font(UNICODE_FONT_FAMILY, style="I", fname=_find_font_path("I") or regular)
            self.add_font(UNICODE_FONT_FAMILY, style="BI", fname=_find_font_path("BI") or _find_font_path("B") or regular)
            self.base_font_family = UNICODE_FONT_FAMILY
            return True
        except Exception:
            self.base_font_family = "Helvetica"
            return False

    def _hex(self, hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        replacements = {
            '—': '-', '–': '-', '“': '"', '”': '"', 
            '‘': "'", '’': "'", '…': '...', '\u00a0': ' ',
            '\u200b': '', '\ufeff': '', '\ufe0f': '',
            '₹': 'Rs. ', '€': 'EUR ', '£': 'GBP '
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if self.unicode_font_enabled:
            return text

        fallback_replacements = {'•': '-'}
        for old, new in fallback_replacements.items():
            text = text.replace(old, new)
        return text.encode('latin-1', 'replace').decode('latin-1')

    def section_title(self, text: str):
        text = self._clean(text)
        self.ln(6)
        r, g, b = self._hex("10b981")
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font(self.base_font_family, "B", 11)
        self.cell(0, 9, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(30, 30, 50)
        self.ln(3)

    def body_text(self, text: str, size: int = 10):
        text = self._clean(text)
        self.set_font(self.base_font_family, "", size)
        self.set_text_color(50, 50, 70)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def small_caps(self, text: str):
        text = self._clean(text)
        self.set_font(self.base_font_family, "B", 8)
        self.set_text_color(100, 115, 135)
        self.cell(0, 5, text.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def callout_box(self, title: str, body: str, fill_hex: str = "f8fafc", border_hex: str = "cbd5e1"):
        self.set_fill_color(*self._hex(fill_hex))
        self.set_draw_color(*self._hex(border_hex))
        self.set_text_color(30, 50, 80)
        self.set_font(self.base_font_family, "B", 10)
        self.multi_cell(0, 6, self._clean(title), border=1, fill=True)
        self.set_x(self.l_margin)
        self.set_font(self.base_font_family, "", 9)
        self.set_text_color(70, 85, 105)
        self.multi_cell(0, 6, self._clean(body), border="LRB", fill=True)
        self.ln(2)

    def divider(self):
        self.set_draw_color(220, 228, 236)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def metric_band(self, metric: str, current: str, projected: str, change: str):
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(203, 213, 225)
        self.set_font(self.base_font_family, "B", 10)
        self.set_text_color(15, 23, 42)
        self.cell(50, 8, self._clean(metric), border=1, fill=True)
        self.set_font(self.base_font_family, "", 9)
        self.set_text_color(71, 85, 105)
        self.cell(44, 8, self._clean(f"Current: {current}"), border=1, fill=True)
        self.set_text_color(*self._hex("059669"))
        self.cell(52, 8, self._clean(f"Projected: {projected}"), border=1, fill=True)
        self.set_text_color(51, 65, 85)
        self.cell(0, 8, self._clean(f"Change: {change}"), border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def labelled_paragraph(self, label: str, body: str):
        self.set_x(self.l_margin)
        self.set_font(self.base_font_family, "B", 9)
        self.set_text_color(30, 50, 80)
        self.cell(0, 6, self._clean(label), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font(self.base_font_family, "", 9)
        self.set_text_color(80, 90, 110)
        self.multi_cell(0, 6, self._clean(body), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def kpi_row(self, label: str, value: str, note: str = ""):
        label, value, note = self._clean(label), self._clean(value), self._clean(note)
        self.set_font(self.base_font_family, "B", 10)
        self.set_text_color(30, 30, 50)
        self.cell(65, 8, label, border="B")
        self.set_font(self.base_font_family, "B", 10)
        r, g, b = self._hex("059669")
        self.set_text_color(r, g, b)
        self.cell(50, 8, value, border="B")
        self.set_font(self.base_font_family, "", 9)
        self.set_text_color(100, 100, 120)
        self.cell(0, 8, note, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullet(self, text: str, color_hex: str = "334155"):
        text = self._clean(text)
        r, g, b = self._hex(color_hex)
        self.set_text_color(r, g, b)
        self.set_font(self.base_font_family, "", 10)
        self.set_x(self.l_margin)
        self.cell(6, 7, "•" if self.unicode_font_enabled else "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        text_width = self.w - self.r_margin - self.get_x()
        self.multi_cell(text_width, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def header(self):
        # Top accent bar
        r, g, b = self._hex("10b981")
        self.set_fill_color(r, g, b)
        self.rect(0, 0, 210, 3, "F")

        self.set_y(8)
        # Company name left
        self.set_font(self.base_font_family, "B", 13)
        r, g, b = self._hex("0a1628")
        self.set_text_color(r, g, b)
        self.cell(120, 8, self._clean(self.company_name))
        # Report label right
        self.set_font(self.base_font_family, "", 9)
        self.set_text_color(100, 100, 120)
        self.cell(0, 8, self._clean(f"ArthaMind | {self.fy_label} Projection Report"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        # Thin separator line
        self.set_draw_color(200, 210, 220)
        self.line(10, 18, 200, 18)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.base_font_family, "I", 8)
        self.set_text_color(150, 150, 170)
        footer_text = self._clean(
            f"Generated by ArthaMind AI  •  {datetime.now().strftime('%d %b %Y')}  •  Page {self.page_no()}"
        )
        self.cell(0, 10,
                  footer_text,
                  align="C")


# ─────────────────────────────────────────────────────────────────────────────
# Projection Normalisation Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _scorecard_rows(projections: dict) -> list[dict]:
    rows = projections.get("projection_scorecard")
    if isinstance(rows, list) and rows:
        return rows

    legacy_rows = []
    mapping = [
        ("Revenue", projections.get("revenue_projection", {}), "growth"),
        ("EBITDA", projections.get("ebitda_projection", {}), "margin"),
        ("PAT", projections.get("pat_projection", {}), "growth"),
        ("R&D Spend", projections.get("rd_spend_projection", {}), "pct_of_revenue"),
    ]
    for metric, payload, delta_key in mapping:
        if not isinstance(payload, dict) or not payload:
            continue
        legacy_rows.append(
            {
                "metric": metric,
                "current": "Current FY base",
                "projected": payload.get("value", "Not stated"),
                "change": payload.get(delta_key, ""),
                "commentary": payload.get("rationale", ""),
            }
        )
    return legacy_rows


def _headline_metrics(rows: list[dict], limit: int = 3) -> list[dict]:
    preferred = {"Revenue", "EBITDA", "PAT", "Operating Cash Flow", "R&D Spend"}
    selected = [row for row in rows if row.get("metric") in preferred]
    if len(selected) < limit:
        for row in rows:
            if row not in selected:
                selected.append(row)
            if len(selected) >= limit:
                break
    return selected[:limit]

# ─────────────────────────────────────────────────────────────────────────────
# Main PDF Generation Function
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(company_name: str, fy_label: str, projections: dict,
                         kpis: dict, user_instructions: str) -> bytes:
    """Assemble and return the PDF as bytes."""

    pdf = ArthaMindReport(company_name, fy_label)
    projection_rows = _scorecard_rows(projections)
    headline_rows = _headline_metrics(projection_rows)
    company_overview = projections.get("company_overview", {}) if isinstance(projections.get("company_overview"), dict) else {}
    management_mandate = projections.get("management_mandate", {}) if isinstance(projections.get("management_mandate"), dict) else {}
    balance_sheet_cashflow = projections.get("balance_sheet_cashflow", {}) if isinstance(projections.get("balance_sheet_cashflow"), dict) else {}
    management_outlook = projections.get("management_outlook", {}) if isinstance(projections.get("management_outlook"), dict) else {}
    mandate_targets = _ensure_list(management_mandate.get("targets") or projections.get("key_targets"))
    business_lines = _ensure_list(company_overview.get("business_lines"))
    validated_baseline = _ensure_list(projections.get("baseline_snapshot"))
    bridge_items = _ensure_list(projections.get("revenue_margin_bridge"))
    segment_items = _ensure_list(projections.get("segment_projections"))
    milestone_items = _ensure_list(projections.get("program_milestones"))
    risk_items = _ensure_list(projections.get("risks"))
    scenario_items = _ensure_list(projections.get("scenario_matrix"))
    priorities = _ensure_list(projections.get("strategic_priorities"))
    board_watch_items = _ensure_list(management_outlook.get("board_watch_items"))
    assumption_flags = _ensure_list(projections.get("assumption_flags"))

    # ── COVER ──────────────────────────────────────────────────────────────
    pdf.set_y(34)
    r, g, b = pdf._hex("0a1628")
    pdf.set_text_color(r, g, b)
    pdf.set_font(pdf.base_font_family, "B", 26)
    pdf.cell(0, 12, pdf._clean(company_name), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font(pdf.base_font_family, "", 14)
    r, g, b = pdf._hex("10b981")
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, pdf._clean(f"Annual Projection Report - {fy_label}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(4)
    pdf.set_font(pdf.base_font_family, "I", 10)
    pdf.set_text_color(120, 130, 150)
    pdf.cell(0, 8, pdf._clean(f"AI-Generated by ArthaMind  |  {datetime.now().strftime('%d %B %Y')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font(pdf.base_font_family, "", 8)
    pdf.set_text_color(140, 150, 165)
    pdf.cell(0, 6, pdf._clean(f"Projection Engine: {REPORT_PIPELINE_VERSION}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.ln(5)
    r, g, b = pdf._hex("10b981")
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.8)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(200, 210, 220)

    pdf.ln(8)
    pdf.callout_box(
        "Management Mandate",
        management_mandate.get("summary") or user_instructions,
        fill_hex="eefbf3",
        border_hex="a7f3d0",
    )

    if headline_rows:
        pdf.small_caps("Headline Projection Metrics")
        for row in headline_rows:
            pdf.metric_band(
                row.get("metric", "Metric"),
                row.get("current", "Current FY"),
                row.get("projected", "Projected"),
                row.get("change", "n/a"),
            )
        pdf.ln(2)

    pdf.set_fill_color(240, 249, 255)
    pdf.set_draw_color(180, 220, 210)
    pdf.set_font(pdf.base_font_family, "I", 8)
    pdf.set_text_color(80, 100, 120)
    pdf.multi_cell(
        0,
        5,
        "DISCLAIMER: This report is an AI-generated forward-looking projection based on historical data, "
        "retrieved report context, and stated management assumptions. It does not constitute financial advice. "
        "Actual results may differ materially from projections.",
        border=1,
        fill=True,
    )

    # ── PAGE 2: Narrative Core ───────────────────────────────────────────────
    pdf.add_page()

    pdf.section_title("1.  Executive Summary")
    pdf.body_text(projections.get("executive_summary", ""))

    pdf.section_title("2.  Company & Strategic Context")
    if kpis.get("report_period"):
        pdf.labelled_paragraph("Reporting Period", kpis.get("report_period"))
    if company_overview.get("current_position"):
        pdf.labelled_paragraph("Current Position", company_overview.get("current_position"))
    if business_lines:
        pdf.small_caps("Core Products & Business Lines")
        for line in business_lines:
            pdf.bullet(str(line), "334155")
        pdf.ln(1)
    if company_overview.get("strategic_context"):
        pdf.labelled_paragraph("Strategic Context", company_overview.get("strategic_context"))

    pdf.section_title("3.  Management Mandate For The Year")
    pdf.body_text(management_mandate.get("summary") or user_instructions)
    for target in mandate_targets:
        pdf.bullet(str(target), "1e3a5f")

    populated_baseline = []
    for item in validated_baseline:
        if not isinstance(item, dict):
            continue
        label = _clean_ws(str(item.get("label", "")))
        value = _clean_ws(str(item.get("value", "")))
        if label and value:
            populated_baseline.append((label, value))

    if not populated_baseline:
        legacy_baseline = [
            ("Revenue", kpis.get("revenue")),
            ("Revenue Growth", kpis.get("revenue_growth")),
            ("EBITDA", kpis.get("ebitda")),
            ("Net Income / PAT", kpis.get("net_income")),
            ("Operating Cash Flow", kpis.get("operating_cash_flow")),
            ("Total Debt", kpis.get("total_debt")),
            ("ROE", kpis.get("roe")),
            ("Key Highlight", kpis.get("key_highlight")),
            ("Risk Flag", kpis.get("risk_flag")),
        ]
        populated_baseline = [(label, value) for label, value in legacy_baseline if value and value != "null"]

    if populated_baseline:
        pdf.section_title("4.  Current-Year Baseline Snapshot")
        for label, value in populated_baseline:
            pdf.bullet(f"{label}: {value}", "475569")

    pdf.add_page()
    pdf.section_title("5.  FY Projection Scorecard")
    for row in projection_rows:
        pdf.metric_band(
            row.get("metric", "Metric"),
            row.get("current", "Current FY"),
            row.get("projected", "Projected"),
            row.get("change", "n/a"),
        )
        commentary = row.get("commentary")
        if commentary:
            pdf.set_font(pdf.base_font_family, "", 9)
            pdf.set_text_color(80, 90, 110)
            pdf.multi_cell(0, 6, pdf._clean(commentary), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

    if bridge_items:
        pdf.section_title("6.  Revenue & Margin Build")
        for item in bridge_items:
            if not isinstance(item, dict):
                continue
            heading = item.get("driver", "Driver")
            impact = item.get("impact", "")
            title = f"{heading} | {impact}" if impact else heading
            pdf.labelled_paragraph(title, item.get("detail", ""))

    if segment_items:
        pdf.section_title("7.  Segment Outlook")
        for seg in segment_items:
            if not isinstance(seg, dict):
                continue
            title = (
                f"{seg.get('segment', 'Segment')} | Revenue: {seg.get('projected_revenue', 'n/a')} | "
                f"Growth: {seg.get('growth', 'n/a')} | Margin: {seg.get('margin', 'n/a')}"
            )
            body_parts = []
            if seg.get("key_programs"):
                body_parts.append(f"Key programs: {seg.get('key_programs')}.")
            if seg.get("outlook"):
                body_parts.append(seg.get("outlook"))
            pdf.callout_box(title, " ".join(body_parts), fill_hex="f8fafc", border_hex="dbeafe")

    pdf.add_page()
    if milestone_items:
        pdf.section_title("8.  Program Milestones & Execution Plan")
        for milestone in milestone_items:
            if not isinstance(milestone, dict):
                continue
            title = (
                f"{milestone.get('program', 'Program')} | {milestone.get('milestone', 'Milestone')} | "
                f"{milestone.get('timeline', 'Timeline TBD')}"
            )
            detail = " ".join(
                part for part in [
                    milestone.get("business_impact", ""),
                    milestone.get("execution_priority", ""),
                ] if part
            )
            pdf.callout_box(title, detail or "Execution details were not provided.", fill_hex="fefce8", border_hex="fde68a")

    pdf.section_title("9.  Balance Sheet & Cash Flow Outlook")
    balance_items = [
        ("Operating Cash Flow", balance_sheet_cashflow.get("operating_cash_flow")),
        ("Capex", balance_sheet_cashflow.get("capex")),
        ("Net Debt / Leverage", balance_sheet_cashflow.get("net_debt")),
        ("Domestic Content / Localisation", balance_sheet_cashflow.get("domestic_content")),
    ]
    for label, value in balance_items:
        if value:
            pdf.bullet(f"{label}: {value}", "334155")
    if balance_sheet_cashflow.get("commentary"):
        pdf.body_text(balance_sheet_cashflow.get("commentary"))

    if risk_items:
        pdf.section_title("10.  Risk Management Framework")
        severity_colors = {"High": "ef4444", "Medium": "f59e0b", "Low": "10b981"}
        for risk in risk_items:
            if not isinstance(risk, dict):
                continue
            sev = risk.get("severity", "Medium")
            title = f"[{sev}] {risk.get('risk', 'Risk')}"
            body = " ".join(
                part for part in [
                    f"Exposure: {risk.get('financial_exposure')}" if risk.get("financial_exposure") else "",
                    f"Mitigation: {risk.get('mitigation')}" if risk.get("mitigation") else "",
                    f"Watchpoint: {risk.get('watchpoint')}" if risk.get("watchpoint") else "",
                ] if part
            )
            pdf.callout_box(title, body, fill_hex="fff7ed", border_hex=severity_colors.get(sev, "cbd5e1"))

    if scenario_items:
        pdf.section_title("11.  Scenario Sensitivity Matrix")
        for scenario in scenario_items:
            if not isinstance(scenario, dict):
                continue
            title = (
                f"{scenario.get('scenario', 'Scenario')} | Revenue Impact: {scenario.get('revenue_impact', 'n/a')} | "
                f"EBITDA Impact: {scenario.get('ebitda_impact', 'n/a')} | Probability: {scenario.get('probability', 'n/a')}"
            )
            pdf.callout_box(title, scenario.get("implication", ""), fill_hex="eff6ff", border_hex="bfdbfe")

    if assumption_flags:
        pdf.section_title("12.  Assumption Register")
        for item in assumption_flags:
            pdf.bullet(str(item), "7c2d12")

    pdf.section_title("13.  Strategic Priorities & Management Outlook")
    for priority in priorities:
        pdf.bullet(str(priority), "0f766e")
    if management_outlook.get("guidance_headline"):
        pdf.labelled_paragraph("Guidance Headline", management_outlook.get("guidance_headline"))
    if board_watch_items:
        pdf.small_caps("Board Watch Items")
        for item in board_watch_items:
            pdf.bullet(str(item), "334155")
    if management_outlook.get("closing_commentary"):
        pdf.labelled_paragraph("Closing Commentary", management_outlook.get("closing_commentary"))
    if projections.get("_generation_notice"):
        pdf.callout_box("Generation Note", projections.get("_generation_notice"), fill_hex="f8fafc", border_hex="cbd5e1")

    # ── Return as bytes ─────────────────────────────────────────────────────
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def build_report(
    company_name: str,
    kpis: dict,
    user_instructions: str,
    answer_provider: str,
    api_key: str,
    answer_model: str = "",
    openai_key: str = "",
    gemini_key: str = "",
    vectorstore=None,
    active_filename: str = "",
    preferred_currency_code: str = "",
) -> tuple[bytes, dict]:
    """
    Full pipeline: LLM projection → PDF bytes.
    Returns (pdf_bytes, projections_dict)
    """
    fy_label = _derive_target_fy_label(kpis.get("report_period", ""), user_instructions)

    projections = generate_projections_with_llm(
        kpis=kpis,
        company_name=company_name,
        user_instructions=user_instructions,
        answer_provider=answer_provider,
        api_key=api_key,
        answer_model=answer_model,
        openai_key=openai_key,
        gemini_key=gemini_key,
        vectorstore=vectorstore,
        active_filename=active_filename,
        preferred_currency_code=preferred_currency_code,
    )

    pdf_bytes = generate_pdf_report(company_name, fy_label, projections, kpis, user_instructions)
    return pdf_bytes, projections
