"""
live_search.py — Provider-backed live web search helpers for ArthaMind.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
import warnings
from utils import retry_with_backoff


DEFAULT_OPENAI_WEB_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_WEB_MODEL = "gemini-2.5-flash"
MAX_SUMMARY_CHARS = 2000
MAX_SOURCES = 5


def _compact_text(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _source_label(title: str, url: str) -> str:
    clean_title = (title or "Source").strip()
    clean_url = (url or "").strip()
    if not clean_url:
        return clean_title

    netloc = urllib.parse.urlparse(clean_url).netloc
    if "vertexaisearch.cloud.google.com" in netloc:
        return clean_title
    return f"{clean_title} ({netloc or clean_url})"


def _dedupe_sources(sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    unique = []
    for title, url in sources:
        normalized_url = (url or "").strip()
        if not normalized_url or normalized_url in seen:
            continue
        seen.add(normalized_url)
        unique.append((title or normalized_url, normalized_url))
    return unique


def _format_search_output(
    provider_label: str,
    summary: str,
    sources: list[tuple[str, str]],
    status: str = "success",
    note: str = "",
) -> str:
    clean_summary = _compact_text((summary or "").strip() or "No recent web results found.")
    clean_sources = _dedupe_sources(sources)

    parts = [
        f"LIVE_WEB_STATUS: {status}",
        f"LIVE_WEB_PROVIDER: {provider_label}",
    ]
    if note:
        parts.append(f"LIVE_WEB_NOTE: {note}")
    parts.extend([
        "LIVE_WEB_SUMMARY:",
        clean_summary,
    ])
    if clean_sources:
        parts.append("LIVE_WEB_SOURCES:")
        parts.extend(f"- {_source_label(title, url)}" for title, url in clean_sources[:MAX_SOURCES])
    return "\n".join(parts)


def _http_error_message(prefix: str, exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="ignore")
        short_message = ""
        try:
            payload = json.loads(body)
            error_block = payload.get("error") or {}
            code = error_block.get("code") or exc.code
            message = (error_block.get("message") or "").strip()
            short_message = f"{code}: {message}" if message else str(code)
        except Exception:
            short_message = body.strip()[:160]
        return f"{prefix} failed with HTTP {exc.code}: {short_message}"
    return f"{prefix} failed: {exc}"


def _collect_openai_text_and_sources(payload: dict) -> tuple[str, list[tuple[str, str]]]:
    texts = []
    sources = []

    for item in payload.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                text = (content.get("text") or "").strip()
                if text:
                    texts.append(text)
                for annotation in content.get("annotations", []):
                    if annotation.get("type") != "url_citation":
                        continue
                    sources.append((
                        annotation.get("title") or annotation.get("url") or "Source",
                        annotation.get("url") or "",
                    ))
        elif item_type == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources", []):
                sources.append((
                    source.get("title") or source.get("url") or "Source",
                    source.get("url") or "",
                ))

    summary = "\n\n".join(part for part in texts if part).strip()
    return summary, sources


@retry_with_backoff(max_retries=3, initial_delay=2)
def openai_web_search(
    query: str,
    api_key: str,
    model: str = DEFAULT_OPENAI_WEB_MODEL,
) -> str:
    prompt = (
        "Provide a detailed, data-heavy business update for the query below. "
        "Prioritize recent macro, regulatory, commodity prices (with figures), tariffs, industry shifts, and company-specific news. "
        "Include as many quantified details (percentages, dollar amounts, dates) as possible. "
        "Use 5-7 descriptive bullets, followed by a 2-sentence strategic implication analysis.\n\n"
        f"Query: {query}"
    )


    request_body = {
        "model": model,
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
        "instructions": (
            "You are a financial research assistant. Focus on current, material, source-backed developments."
        ),
        "input": prompt,
    }

    request = urllib.request.Request(
        url="https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(_http_error_message("OpenAI web search", exc)) from exc

    summary, sources = _collect_openai_text_and_sources(payload)
    return _format_search_output("OpenAI Web Search", summary, sources)


@retry_with_backoff(max_retries=3, initial_delay=2)
def gemini_web_search(
    query: str,
    api_key: str,
    model: str = DEFAULT_GEMINI_WEB_MODEL,
) -> str:
    prompt = (
        "Provide a detailed, data-heavy business update for the query below. "
        "Prioritize recent macro, regulatory, commodity prices (with figures), tariffs, industry shifts, and company-specific news. "
        "Include as many quantified details (percentages, dollar amounts, dates) as possible. "
        "Use 5-7 descriptive bullets, followed by a 2-sentence strategic implication analysis.\n\n"
        f"Query: {query}"
    )


    request_body = {
        "contents": [{
            "parts": [{"text": prompt}],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.1,
        },
    }
    encoded_key = urllib.parse.quote(api_key, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={encoded_key}"
    )
    request = urllib.request.Request(
        url=url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(_http_error_message("Gemini web search", exc)) from exc

    candidates = payload.get("candidates", [])
    if not candidates:
        return _format_search_output(
            "Gemini Google Search",
            "No recent web results found.",
            [],
            status="no_results",
        )

    first_candidate = candidates[0]
    text_parts = []
    for part in (first_candidate.get("content") or {}).get("parts", []):
        text = (part.get("text") or "").strip()
        if text:
            text_parts.append(text)

    grounding = first_candidate.get("groundingMetadata") or {}
    sources = []
    for chunk in grounding.get("groundingChunks", []):
        web = chunk.get("web") or {}
        sources.append((web.get("title") or web.get("uri") or "Source", web.get("uri") or ""))

    summary = "\n\n".join(text_parts).strip()
    return _format_search_output("Gemini Google Search", summary, sources)


def duckduckgo_web_search(query: str) -> str:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS # type: ignore
        except ImportError:
             return _format_search_output("DuckDuckGo", "Search library not installed.", [], status="error")


    results = list(DDGS().text(query, max_results=5))
    formatted = []
    sources = []
    for result in results:
        title = result.get("title", "Untitled result")
        body = result.get("body", "")
        href = result.get("href", "")
        formatted.append(f"- {title}: {body}")
        sources.append((title, href))
    if not results:
        return _format_search_output(
            "DuckDuckGo",
            "No recent web results found.",
            [],
            status="no_results",
        )

    summary = "\n".join(formatted)
    return _format_search_output("DuckDuckGo", summary, sources)


def _candidate_sequence(selected_provider: str) -> list[str]:
    normalized = (selected_provider or "duckduckgo").lower()
    if normalized == "openai":
        return ["openai", "gemini", "duckduckgo"]
    if normalized == "gemini":
        return ["gemini", "openai", "duckduckgo"]
    return ["duckduckgo"]


def _append_live_note(search_output: str, note: str) -> str:
    clean_note = (note or "").strip()
    if not clean_note:
        return search_output
    return f"{search_output}\nLIVE_WEB_NOTE: {clean_note}"


def run_live_web_search(
    query: str,
    provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_model: str = DEFAULT_GEMINI_WEB_MODEL,
) -> str:
    notices = []

    for candidate in _candidate_sequence(provider):
        if candidate == "openai":
            if not openai_api_key:
                notices.append("OpenAI live search skipped because OPENAI_API_KEY is missing.")
                continue
            try:
                result = openai_web_search(query, openai_api_key, model=openai_model)
                if notices:
                    return _append_live_note(result, "Earlier fallback notes: " + " | ".join(notices))
                return result
            except Exception as exc:
                notices.append(str(exc))
                continue

        if candidate == "gemini":
            if not gemini_api_key:
                notices.append("Gemini live search skipped because GEMINI_API_KEY is missing.")
                continue
            try:
                result = gemini_web_search(query, gemini_api_key, model=gemini_model)
                if notices:
                    return _append_live_note(result, "Earlier fallback notes: " + " | ".join(notices))
                return result
            except Exception as exc:
                notices.append(str(exc))
                continue

        if candidate == "duckduckgo":
            try:
                result = duckduckgo_web_search(query)
                if notices:
                    return _append_live_note(result, "Earlier fallback notes: " + " | ".join(notices))
                return result
            except Exception as exc:
                notices.append(str(exc))

    return _format_search_output(
        "Unavailable",
        "Current live web context could not be retrieved from any configured provider.",
        [],
        status="unavailable",
        note=" | ".join(notices),
    )
