"""
currency_utils.py — Shared currency parsing, conversion, and display helpers
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_DISPLAY_CURRENCY = "USD"

_TEN_MILLION = 10_000_000
_ONE_MILLION = 1_000_000

CURRENCY_METADATA = {
    "USD": {
        "name": "US Dollar",
        "prefix": "$",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "EUR": {
        "name": "Euro",
        "prefix": "€",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "GBP": {
        "name": "British Pound",
        "prefix": "£",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "INR": {
        "name": "Indian Rupee",
        "prefix": "Rs.",
        "space_after_prefix": True,
        "display_unit": "Crore",
        "display_divisor": _TEN_MILLION,
    },
    "JPY": {
        "name": "Japanese Yen",
        "prefix": "¥",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "CNY": {
        "name": "Chinese Yuan",
        "prefix": "CNY",
        "space_after_prefix": True,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "CAD": {
        "name": "Canadian Dollar",
        "prefix": "C$",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "AUD": {
        "name": "Australian Dollar",
        "prefix": "A$",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "CHF": {
        "name": "Swiss Franc",
        "prefix": "CHF",
        "space_after_prefix": True,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "AED": {
        "name": "UAE Dirham",
        "prefix": "AED",
        "space_after_prefix": True,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "SGD": {
        "name": "Singapore Dollar",
        "prefix": "S$",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
    "HKD": {
        "name": "Hong Kong Dollar",
        "prefix": "HK$",
        "space_after_prefix": False,
        "display_unit": "million",
        "display_divisor": _ONE_MILLION,
    },
}

# Reference FX fallback assumptions.
FALLBACK_USD_PER_CURRENCY_UNIT = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "INR": 0.012,
    "JPY": 0.0067,
    "CNY": 0.138,
    "CAD": 0.73,
    "AUD": 0.66,
    "CHF": 1.10,
    "AED": 0.272,
    "SGD": 0.74,
    "HKD": 0.128,
}

FX_CACHE_TTL_SECONDS = 1800
FX_API_BASE_URL = "https://api.frankfurter.dev/v2/rates"
FX_API_SOURCE_NAME = "Frankfurter"
FX_CACHE = {
    "fetched_at": 0.0,
    "date": "",
    "rates": dict(FALLBACK_USD_PER_CURRENCY_UNIT),
    "source": "Fallback FX assumptions",
    "live": False,
    "error": None,
}

CURRENCY_ALIAS_PATTERNS = [
    (re.compile(r"\bHKD\b|HK\$", re.IGNORECASE), "HKD"),
    (re.compile(r"\bCAD\b|C\$", re.IGNORECASE), "CAD"),
    (re.compile(r"\bAUD\b|A\$", re.IGNORECASE), "AUD"),
    (re.compile(r"\bSGD\b|S\$", re.IGNORECASE), "SGD"),
    (re.compile(r"\bINR\b|\bRs\.?\b|₹", re.IGNORECASE), "INR"),
    (re.compile(r"\bAED\b", re.IGNORECASE), "AED"),
    (re.compile(r"\bCHF\b", re.IGNORECASE), "CHF"),
    (re.compile(r"\bCNY\b|\bRMB\b", re.IGNORECASE), "CNY"),
    (re.compile(r"\bJPY\b|¥", re.IGNORECASE), "JPY"),
    (re.compile(r"\bEUR\b|€", re.IGNORECASE), "EUR"),
    (re.compile(r"\bGBP\b|£", re.IGNORECASE), "GBP"),
    (re.compile(r"\bUSD\b|\$", re.IGNORECASE), "USD"),
]

MONEY_KPI_KEYS = {
    "revenue",
    "net_income",
    "ebitda",
    "operating_cash_flow",
    "total_assets",
    "total_debt",
    "eps",
}
PER_SHARE_KPI_KEYS = {"eps"}

_AMOUNT_PREFIX_PATTERN = r"(?:USD|EUR|GBP|INR|JPY|CNY|CAD|AUD|CHF|AED|SGD|HKD|Rs\.?|₹|\$|€|£|¥|C\$|A\$|S\$|HK\$)"
_AMOUNT_UNIT_PATTERN = r"(?:trillion|tn|billion|bn|million|mn|crore|cr|lakh|thousand|k)"
_MONETARY_TEXT_PATTERN = re.compile(
    rf"(?P<prefix>{_AMOUNT_PREFIX_PATTERN})\s*"
    rf"(?P<num1>\d[\d,]*\.?\d*)"
    rf"(?:\s*-\s*(?P<num2>\d[\d,]*\.?\d*))?"
    rf"(?:\s*(?P<unit>{_AMOUNT_UNIT_PATTERN}))?",
    re.IGNORECASE,
)


def normalize_currency_code(code: str) -> str:
    candidate = (code or "").upper().strip()
    if candidate in CURRENCY_METADATA:
        return candidate
    return DEFAULT_DISPLAY_CURRENCY


def get_currency_options() -> dict[str, str]:
    return {
        f"{meta['name']} ({code})": code
        for code, meta in CURRENCY_METADATA.items()
    }


def get_currency_meta(code: str) -> dict:
    return CURRENCY_METADATA[normalize_currency_code(code)]


def _build_fx_request_url(base_currency: str = "USD") -> str:
    quotes = ",".join(sorted(code for code in CURRENCY_METADATA if code != base_currency))
    return f"{FX_API_BASE_URL}?{urlencode({'base': base_currency, 'quotes': quotes})}"


def _parse_fx_response(payload) -> tuple[dict[str, float], str]:
    rates = {"USD": 1.0}
    fx_date = ""

    if isinstance(payload, list):
        for row in payload:
            quote = normalize_currency_code(row.get("quote", ""))
            if quote == "USD":
                continue
            try:
                quote_per_usd = float(row.get("rate"))
                if quote_per_usd:
                    rates[quote] = 1.0 / quote_per_usd
            except Exception:
                continue
            if not fx_date:
                fx_date = str(row.get("date", "")).strip()
    elif isinstance(payload, dict):
        fx_date = str(payload.get("date", "")).strip()
        raw_rates = payload.get("rates", {})
        if isinstance(raw_rates, dict):
            for quote, rate in raw_rates.items():
                quote_code = normalize_currency_code(quote)
                try:
                    quote_per_usd = float(rate)
                    if quote_per_usd:
                        rates[quote_code] = 1.0 / quote_per_usd
                except Exception:
                    continue

    for code, fallback_rate in FALLBACK_USD_PER_CURRENCY_UNIT.items():
        rates.setdefault(code, fallback_rate)

    return rates, fx_date


def refresh_exchange_rate_cache(force_refresh: bool = False) -> dict:
    now = time.time()
    if (
        not force_refresh
        and FX_CACHE["fetched_at"]
        and now - FX_CACHE["fetched_at"] < FX_CACHE_TTL_SECONDS
    ):
        return FX_CACHE

    request = Request(
        _build_fx_request_url(base_currency="USD"),
        headers={"User-Agent": "ArthaMind/1.0"},
    )

    previous_rates = dict(FX_CACHE.get("rates") or FALLBACK_USD_PER_CURRENCY_UNIT)
    previous_date = FX_CACHE.get("date", "")

    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rates, fx_date = _parse_fx_response(payload)
        FX_CACHE.update(
            {
                "fetched_at": now,
                "date": fx_date,
                "rates": rates,
                "source": FX_API_SOURCE_NAME,
                "live": True,
                "error": None,
            }
        )
    except Exception as exc:
        had_previous_live_rates = FX_CACHE.get("live") and previous_rates
        FX_CACHE.update(
            {
                "fetched_at": now,
                "date": previous_date,
                "rates": previous_rates,
                "source": f"{FX_API_SOURCE_NAME} cached" if had_previous_live_rates else "Fallback FX assumptions",
                "live": bool(had_previous_live_rates),
                "error": str(exc),
            }
        )

    return FX_CACHE


def get_exchange_rate_status() -> dict:
    status = refresh_exchange_rate_cache()
    return {
        "source": status.get("source", "Unknown"),
        "date": status.get("date", ""),
        "live": bool(status.get("live")),
        "error": status.get("error"),
    }


def detect_currency_code(text: str, fallback_currency: str = "") -> str:
    sample = str(text or "")
    for pattern, code in CURRENCY_ALIAS_PATTERNS:
        if pattern.search(sample):
            return code
    fallback = (fallback_currency or "").upper().strip()
    return fallback if fallback in CURRENCY_METADATA else ""


def _detect_unit_multiplier(text: str, per_share: bool = False) -> tuple[str, float]:
    if per_share:
        return ("unit", 1.0)

    sample = str(text or "").lower()
    if re.search(r"\btrillion\b|\btn\b", sample):
        return ("trillion", 1_000_000_000_000.0)
    if re.search(r"\bbillion\b|\bbn\b", sample):
        return ("billion", 1_000_000_000.0)
    if re.search(r"\bcrore\b|\bcr\b", sample):
        return ("crore", _TEN_MILLION)
    if re.search(r"\bmillion\b|\bmn\b", sample):
        return ("million", _ONE_MILLION)
    if re.search(r"\blakh\b", sample):
        return ("lakh", 100_000.0)
    if re.search(r"\bthousand\b|\bk\b", sample):
        return ("thousand", 1_000.0)
    return ("unit", 1.0)


def _parse_first_numeric(text: str) -> Optional[float]:
    match = re.search(r"\d[\d,]*\.?\d*", str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_money_value(value, fallback_currency: str = "", metric_key: str = "") -> Optional[dict]:
    if value is None or value == "null":
        return None

    text = str(value).strip()
    if not text:
        return None

    per_share = metric_key in PER_SHARE_KPI_KEYS
    if not per_share and ("%" in text or re.search(r"\b\d+(\.\d+)?x\b", text, re.IGNORECASE)):
        return None

    numeric_value = _parse_first_numeric(text)
    if numeric_value is None:
        return None

    source_currency = detect_currency_code(text, fallback_currency)
    if not source_currency:
        return None

    unit_label, unit_multiplier = _detect_unit_multiplier(text, per_share=per_share)
    absolute_amount = numeric_value * unit_multiplier

    return {
        "raw_text": text,
        "currency_code": source_currency,
        "numeric_value": numeric_value,
        "unit_label": unit_label,
        "unit_multiplier": unit_multiplier,
        "absolute_amount": absolute_amount,
        "per_share": per_share,
    }


def infer_unit_multiplier(text: str, per_share: bool = False) -> float:
    return _detect_unit_multiplier(text, per_share=per_share)[1]


def convert_absolute_amount(
    amount: Optional[float],
    source_currency_code: str,
    target_currency_code: str,
) -> Optional[float]:
    if amount is None:
        return None

    source = normalize_currency_code(source_currency_code)
    target = normalize_currency_code(target_currency_code)
    rates = refresh_exchange_rate_cache().get("rates", FALLBACK_USD_PER_CURRENCY_UNIT)

    if source == target:
        return amount

    usd_amount = amount * rates.get(source, FALLBACK_USD_PER_CURRENCY_UNIT[source])
    return usd_amount / rates.get(target, FALLBACK_USD_PER_CURRENCY_UNIT[target])


def _format_display_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:,.0f}"
    if abs_value >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def _format_prefix(prefix: str, space_after_prefix: bool) -> str:
    return f"{prefix} " if space_after_prefix else prefix


def format_absolute_amount(
    amount: Optional[float],
    target_currency_code: str,
    aggregate: bool = True,
) -> str:
    if amount is None:
        return "Not disclosed"

    meta = get_currency_meta(target_currency_code)
    prefix = _format_prefix(meta["prefix"], meta["space_after_prefix"])

    if not aggregate:
        return f"{prefix}{amount:,.2f}"

    display_value = amount / meta["display_divisor"]
    return f"{prefix}{_format_display_number(display_value)} {meta['display_unit']}"


def format_amount_range(
    amount_one: Optional[float],
    amount_two: Optional[float],
    target_currency_code: str,
    aggregate: bool = True,
) -> str:
    if amount_one is None:
        return "Not disclosed"
    if amount_two is None:
        return format_absolute_amount(amount_one, target_currency_code, aggregate=aggregate)

    meta = get_currency_meta(target_currency_code)
    prefix = _format_prefix(meta["prefix"], meta["space_after_prefix"])

    if not aggregate:
        return f"{prefix}{amount_one:,.2f}-{amount_two:,.2f}"

    value_one = amount_one / meta["display_divisor"]
    value_two = amount_two / meta["display_divisor"]
    return f"{prefix}{_format_display_number(value_one)}-{_format_display_number(value_two)} {meta['display_unit']}"


def format_source_value_in_currency(
    value: Optional[float],
    source_currency_code: str,
    source_unit_multiplier: float,
    target_currency_code: str,
) -> str:
    if value is None:
        return "Not disclosed"
    absolute_source_amount = value * source_unit_multiplier
    converted_amount = convert_absolute_amount(
        absolute_source_amount,
        source_currency_code,
        target_currency_code,
    )
    return format_absolute_amount(converted_amount, target_currency_code, aggregate=True)


def infer_kpi_currency_code(kpis: dict) -> str:
    if not isinstance(kpis, dict):
        return DEFAULT_DISPLAY_CURRENCY

    for key in ("revenue", "net_income", "ebitda", "operating_cash_flow", "total_assets", "total_debt", "eps"):
        code = detect_currency_code(kpis.get(key, ""))
        if code:
            return code
    return DEFAULT_DISPLAY_CURRENCY


def format_kpi_value(
    value,
    metric_key: str = "",
    display_currency: str = DEFAULT_DISPLAY_CURRENCY,
    source_currency_hint: str = "",
) -> str:
    if value is None or value == "null":
        return "N/A"

    metric = (metric_key or "").strip().lower()
    if metric and metric not in MONEY_KPI_KEYS:
        return str(value)

    parsed = parse_money_value(value, fallback_currency=source_currency_hint, metric_key=metric)
    if not parsed:
        return str(value)

    converted_amount = convert_absolute_amount(
        parsed["absolute_amount"],
        parsed["currency_code"],
        display_currency,
    )
    aggregate = not parsed["per_share"]
    return format_absolute_amount(converted_amount, display_currency, aggregate=aggregate)


def convert_currency_mentions(
    text: str,
    target_currency_code: str = DEFAULT_DISPLAY_CURRENCY,
    fallback_source_currency: str = "",
) -> str:
    if not text:
        return text

    target = normalize_currency_code(target_currency_code)

    def _replace(match: re.Match) -> str:
        prefix = match.group("prefix") or ""
        unit = (match.group("unit") or "").strip()
        source_currency = detect_currency_code(prefix, fallback_source_currency)
        if not source_currency:
            return match.group(0)

        try:
            number_one = float(match.group("num1").replace(",", ""))
        except Exception:
            return match.group(0)

        number_two = match.group("num2")
        unit_label, unit_multiplier = _detect_unit_multiplier(unit or match.group(0))
        if not unit and not number_two and abs(number_one) < _ONE_MILLION:
            aggregate = False
            amount_one = convert_absolute_amount(number_one, source_currency, target)
            return format_absolute_amount(amount_one, target, aggregate=aggregate)

        amount_one = convert_absolute_amount(number_one * unit_multiplier, source_currency, target)
        amount_two = None
        if number_two:
            amount_two = convert_absolute_amount(
                float(number_two.replace(",", "")) * unit_multiplier,
                source_currency,
                target,
            )
        return format_amount_range(amount_one, amount_two, target, aggregate=True)

    return _MONETARY_TEXT_PATTERN.sub(_replace, str(text))
