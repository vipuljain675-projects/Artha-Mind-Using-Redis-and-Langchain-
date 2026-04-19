"""
chain.py — LangChain RAG Chain Setup
Builds the Conversational Retrieval Chain with Groq + Memory
"""

import json
import re
from typing import Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request
import concurrent.futures

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
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
    retry_with_backoff,
)
from langchain_community.vectorstores import FAISS


from live_search import (
    DEFAULT_GEMINI_WEB_MODEL,
    DEFAULT_OPENAI_WEB_MODEL,
    run_live_web_search,
)

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DECOMMISSIONED_GROQ_MODELS = {
    "mixtral-8x7b-32768",
    "gemma-7b-it",
}
UNSTABLE_TOOL_CALL_MODELS = {
    "llama-3.3-70b-versatile",
}
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "brief", "briefly", "by", "for",
    "from", "give", "how", "if", "in", "into", "is", "it", "main", "me", "of",
    "on", "or", "our", "show", "should", "tell", "that", "the", "their", "this",
    "to", "us", "was", "what", "when", "where", "which", "who", "why", "with",
}


def resolve_llm_model(model: str = DEFAULT_GROQ_MODEL) -> Tuple[str, Optional[str]]:
    """Fallback if the selected Groq model has been retired."""
    requested_model = model or DEFAULT_GROQ_MODEL
    if requested_model in DECOMMISSIONED_GROQ_MODELS:
        notice = (
            f"`{requested_model}` is no longer available on Groq, "
            f"so ArthaMind switched to `{DEFAULT_GROQ_MODEL}`."
        )
        return DEFAULT_GROQ_MODEL, notice
    return requested_model, None


def resolve_agent_model(model: str = DEFAULT_GROQ_MODEL) -> Tuple[str, Optional[str]]:
    """Use a stable tool-calling model for the analyst chat agent."""
    requested_model, notice = resolve_llm_model(model)
    if requested_model in UNSTABLE_TOOL_CALL_MODELS:
        return (
            DEFAULT_GROQ_MODEL,
            (
                f"Analyst Chat is using `{DEFAULT_GROQ_MODEL}` because "
                f"`{requested_model}` is currently unstable for Groq tool calling."
            ),
        )
    return requested_model, notice


def normalize_answer_provider(provider: str = "groq") -> str:
    normalized = (provider or "groq").lower()
    if normalized in {"gemini", "openai", "groq"}:
        return normalized
    return "groq"


def build_llm(api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.15) -> ChatGroq:
    """
    Initialize Groq LLM.
    - temperature=0.15: near-deterministic for financial accuracy
    - llama3-8b-8192: fast and capable; can switch to llama3-70b for depth
    """
    return ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=2048,
    )


def _openai_response_text(payload: dict) -> str:
    texts = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = (content.get("text") or "").strip()
                if text:
                    texts.append(text)
    if texts:
        return "\n\n".join(texts).strip()
    return (payload.get("output_text") or "").strip()


@retry_with_backoff(max_retries=3, initial_delay=2)
def _openai_generate_text(prompt: str, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> str:

    request_body = {
        "model": model,
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
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"OpenAI answer generation failed with HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI answer generation failed: {exc}") from exc

    text = _openai_response_text(payload)
    if text:
        return text
    raise RuntimeError("OpenAI answer generation returned no text.")


@retry_with_backoff(max_retries=3, initial_delay=2)
def _gemini_generate_text(prompt: str, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> str:

    request_body = {
        "contents": [{
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "temperature": 0.15,
        },
    }
    encoded_key = urllib.parse.quote(api_key, safe="")
    request = urllib.request.Request(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={encoded_key}"
        ),
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"Gemini answer generation failed with HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini answer generation failed: {exc}") from exc

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini answer generation returned no candidates.")

    text_parts = []
    for part in (candidates[0].get("content") or {}).get("parts", []):
        text = (part.get("text") or "").strip()
        if text:
            text_parts.append(text)
    if text_parts:
        return "\n\n".join(text_parts).strip()
    raise RuntimeError("Gemini answer generation returned no text.")


def generate_text_with_provider(
    provider: str,
    prompt: str,
    model: str,
    groq_api_key: str = "",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    temperature: float = 0.15,
) -> str:
    normalized_provider = normalize_answer_provider(provider)
    if normalized_provider == "openai":
        if not openai_api_key:
            raise RuntimeError("OpenAI answer engine is selected but OPENAI_API_KEY is missing.")
        return _openai_generate_text(prompt, openai_api_key, model=model or DEFAULT_OPENAI_MODEL)
    if normalized_provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Gemini answer engine is selected but GEMINI_API_KEY is missing.")
        return _gemini_generate_text(prompt, gemini_api_key, model=model or DEFAULT_GEMINI_MODEL)

    if not groq_api_key:
        raise RuntimeError("Groq answer engine is selected but GROQ_API_KEY is missing.")
    llm = build_llm(groq_api_key, model=model or DEFAULT_GROQ_MODEL, temperature=temperature)
    response = llm.invoke(prompt)
    return response.content.strip()


from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from prompts import AGENT_SYSTEM_PROMPT


@tool
def live_stock_price(ticker: str) -> str:
    """Fetch live stock price and key statistics for a given stock ticker (e.g. AAPL, MSFT, RELIANCE.NS)."""
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        return f"Live Data for {ticker}:\nPrice: {price}\nTrailing P/E: {pe}\nMarket Cap: {market_cap}"
    except Exception as e:
        return f"Error fetching stock data for {ticker}. Ensure ticker is correct."


def _unique_preserve_order(items: list) -> list:
    seen = set()
    unique_items = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item.strip())
    return unique_items


def _keyword_query(query: str, max_terms: int = 8) -> str:
    tokens = re.findall(r"[A-Za-z0-9%./-]+", query.lower())
    keywords = []
    for token in tokens:
        if len(token) <= 2 or token in QUERY_STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return " ".join(keywords[:max_terms])


def _expand_document_queries(query: str) -> list:
    lowered = query.lower()
    expanded = [query]

    keyword_version = _keyword_query(query)
    if keyword_version:
        expanded.append(keyword_version)

    if any(term in lowered for term in ("risk", "risks", "risk factor", "concern", "warning sign", "red flag")):
        expanded.extend([
            "risk management framework key risks risk factors",
            "commodity currency regulatory telecom cybersecurity execution risk",
        ])

    if any(term in lowered for term in ("outlook", "guidance", "future", "next year", "forecast", "management")):
        expanded.extend([
            "management discussion outlook guidance",
            "macro environment revenue EBITDA guidance capex debt",
        ])

    if any(term in lowered for term in ("segment", "business", "division", "vertical")):
        expanded.extend([
            "segment-wise financial performance revenue by business segment EBITDA by business segment",
        ])

    if any(term in lowered for term in ("revenue", "margin", "profit", "ebitda", "eps", "cash flow", "debt", "assets")):
        expanded.extend([
            "financial performance at a glance key consolidated financial metrics",
            "key financial ratios debt profile cash flow analysis",
        ])

    if any(term in lowered for term in (
        "scenario", "sensitivity", "matrix", "what if", "impact", "happen",
        "oil", "crude", "rupee", "tariff", "rate", "inflation", "demand",
        "achievable", "achievability", "calculation", "quantitative", "math",
    )):
        expanded.extend([
            "scenario sensitivity matrix FY2026-27 analysis",
            "risk management framework scenario sensitivity EBITDA margins cash flow",
            "management discussion outlook guidance macro environment sensitivity",
            "impact of commodity prices tariffs interest rates currency fluctuations",
        ])


    return _unique_preserve_order(expanded)


def _format_doc_for_prompt(doc) -> str:
    source = doc.metadata.get("source_file", "?")
    page = doc.metadata.get("page")
    if isinstance(page, int):
        page_label = str(page + 1)
    elif page is None:
        page_label = "?"
    else:
        page_label = str(page)
    content = re.sub(r"\s+", " ", doc.page_content).strip()
    return f"Source (File: {source}, Page: {page_label}): {content}"


def _focus_terms_for_query(query: str) -> list:
    lowered = query.lower()
    focus_terms = []

    if any(term in lowered for term in ("risk", "risks", "risk factor", "concern", "warning sign", "red flag")):
        focus_terms.extend([
            "risk",
            "risk management",
            "commodity",
            "currency",
            "regulatory",
            "telecom",
            "cybersecurity",
            "execution risk",
        ])

    if any(term in lowered for term in ("outlook", "guidance", "future", "forecast", "next year", "management")):
        focus_terms.extend([
            "outlook",
            "guidance",
            "management discussion",
            "macro environment",
        ])

    if any(term in lowered for term in ("segment", "business", "division", "vertical")):
        focus_terms.extend([
            "segment",
            "business segment",
        ])

    if any(term in lowered for term in (
        "scenario", "sensitivity", "matrix", "what if", "impact", "oil",
        "crude", "rupee", "tariff", "rate", "inflation", "demand",
        "achievable", "achievability", "calculation",
    )):
        focus_terms.extend([
            "scenario",
            "sensitivity",
            "matrix",
            "risk",
            "outlook",
            "impact",
            "sensitivity analysis",
            "guidance",
            "macro",
        ])


    return _unique_preserve_order(focus_terms)


def get_document_search_tool(vectorstore: FAISS, active_filename: str = None):
    @tool
    def financial_document_search(query: str) -> str:
        """Search the uploaded report and return the most relevant page-tagged excerpts. Use this for facts, risks, outlook, segment data, guidance, sensitivities, and management commentary."""
        search_kwargs = {"k": 8, "fetch_k": 32, "lambda_mult": 0.7}
        if active_filename:
            search_kwargs["filter"] = {"source_file": active_filename}

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

        collected_docs = []
        seen = set()
        focus_terms = _focus_terms_for_query(query)
        for expanded_query in _expand_document_queries(query):
            for doc in retriever.invoke(expanded_query):
                doc_text = doc.page_content.lower()
                if focus_terms and not any(term in doc_text for term in focus_terms):
                    continue
                doc_key = (
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                    doc.page_content.strip(),
                )
                if doc_key in seen:
                    continue
                seen.add(doc_key)
                collected_docs.append(doc)
                if len(collected_docs) >= 10:
                    break
            if len(collected_docs) >= 10:
                break

        if not collected_docs:
            return "No relevant information found in the uploaded documents. Try a more specific question."

        return "\n\n".join(_format_doc_for_prompt(doc) for doc in collected_docs)
    return financial_document_search


def get_web_search_tool(
    provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_model: str = DEFAULT_GEMINI_WEB_MODEL,
):
    @tool
    def web_search(query: str) -> str:
        """Search the live web for current macro, regulatory, geopolitical, commodity, tariff, rate, competitor, or company context newer than the uploaded report."""
        provider_query = query
        if (provider or "").lower() == "duckduckgo":
            fallback_query = _keyword_query(query, max_terms=8)
            provider_query = fallback_query or query
        return run_live_web_search(
            provider_query,
            provider=provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openai_model=openai_model,
            gemini_model=gemini_model,
        )

    return web_search


def _question_needs_live_context(question: str) -> bool:
    lowered = question.lower()
    trigger_terms = (
        "current", "latest", "today", "now", "world", "macro", "live", "as of",
        "lithium", "tariff", "interest rate", "rates", "geopolitical", "regulation",
        "commodity", "oil", "crude", "rupee", "fx", "inflation", "demand",
        "what if", "impact", "scenario", "happen", "calculation", "reach", "achieve",
    )

    return any(term in lowered for term in trigger_terms)


def _format_chat_history_for_prompt(chat_history: Optional[list]) -> str:
    if not chat_history:
        return "No prior conversation history."

    recent_messages = chat_history[-12:]  # Keep full 12-message context
    lines = []
    for msg in recent_messages:
        role = (msg.get("role") or "user").upper()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation history."



def _build_direct_answer(
    question: str,
    vectorstore: FAISS,
    groq_api_key: str,
    answer_provider: str = "groq",
    answer_model: str = DEFAULT_GROQ_MODEL,
    active_filename: str = None,
    live_search_provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_web_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_web_model: str = DEFAULT_GEMINI_WEB_MODEL,
    chat_history: Optional[list] = None,
) -> str:
    # ── Parallel doc search + web search for lower latency ──────────────────
    needs_live = _question_needs_live_context(question)
    doc_search_fn = get_document_search_tool(vectorstore, active_filename)

    def _fetch_docs():
        return doc_search_fn.invoke(question)

    def _fetch_web():
        if not needs_live:
            return "No live web context requested."
        return run_live_web_search(
            question,
            provider=live_search_provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openai_model=openai_web_model,
            gemini_model=gemini_web_model,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        doc_future = pool.submit(_fetch_docs)
        web_future = pool.submit(_fetch_web)
        document_context = doc_future.result()
        live_context = web_future.result()

    conversation_context = _format_chat_history_for_prompt(chat_history)

    # ── Detect question type for structured output rules ────────────────────
    lowered = question.lower()
    is_guidance_q = any(t in lowered for t in (
        "achievable", "achievability", "guidance", "target", "can they",
        "will they hit", "realistic", "credible",
    ))
    is_scenario_q = any(t in lowered for t in (
        "what if", "if ", "scenario", "sensitivity", "impact", "calculate",
        "rate rise", "tariff", "lithium rises", "rupee weakens",
    ))
    is_calculation_q = any(t in lowered for t in (
        "calculate", "how much", "quantify", "exact", "math", "compute",
    ))

    verdict_instruction = ""
    if is_guidance_q:
        verdict_instruction = """
GUIDANCE VERDICT (MANDATORY):
- State the exact guidance target in absolute numbers (e.g., "Rs.X Crore at 18% growth")
- List ALL known headwinds from the report sensitivity matrix with their Rs. value
- Subtract headwinds from the guidance target to show net achievable figure
- End with a bold VERDICT on one line: **VERDICT: Achievable / Achievable but at risk / Difficult under current conditions**
- Explain in 1 sentence WHY"""
    elif is_scenario_q or is_calculation_q:
        verdict_instruction = """
CALCULATION FORMAT (MANDATORY):
- Show your arithmetic step-by-step using the report's sensitivity figures
- Present as: Base figure → Change → Impact → New figure
- When using an assumed baseline (e.g., price at time of writing), state the assumption explicitly
- End with: **NET IMPACT: [quantified summary]**"""

    prompt = f"""You are ArthaMind, an elite senior financial analyst with deep expertise in Indian equity research.

You MUST use the full conversation history to maintain continuity — if a prior calculation (e.g., a baseline price, a sensitivity figure) was established, build on it instead of redoing it.

ANSWER FORMAT:
1. **Direct Answer** — One sharp paragraph that directly answers the question
2. **Report Data** — All relevant figures with exact page references
3. **Live Web Context** — Current real-world data that affects the analysis (if available)
4. **Business Impact** — Why this matters strategically for the company{verdict_instruction}

RULES:
- Every figure must be sourced: report (cite page) or live web (cite source)
- For multi-part questions, answer every part explicitly
- Never be vague — if uncertain about a baseline, state the assumption clearly and calculate anyway
- Format numbers clearly: Rs.640 Crore, $24,000/tonne, 18.3% margin
- Use **bold** for key metrics and verdicts

QUESTION:
{question}

FULL CONVERSATION HISTORY (use this for context continuity):
{conversation_context}

REPORT EXCERPTS:
{document_context}

LIVE WEB CONTEXT:
{live_context}
"""
    return generate_text_with_provider(
        provider=answer_provider,
        prompt=prompt,
        model=answer_model,
        groq_api_key=groq_api_key,

        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        temperature=0.1,
    )


def build_agent_executor(
    vectorstore: FAISS,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    active_filename: str = None,
    live_search_provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_web_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_web_model: str = DEFAULT_GEMINI_WEB_MODEL,
) -> AgentExecutor:
    """
    Build the Agent Executor with Tool Calling capabilities.
    active_filename: restricts FAISS search to chunks from this file only.
    """
    effective_model, _ = resolve_agent_model(model)
    llm = build_llm(api_key, effective_model)

    tools = [
        get_document_search_tool(vectorstore, active_filename),
        live_stock_price,
        get_web_search_tool(
            provider=live_search_provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openai_model=openai_web_model,
            gemini_model=gemini_web_model,
        ),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        k=12,  # Full 12-turn session memory
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True
    )

    return agent_executor


def _is_retryable_tool_error(error: Exception) -> bool:
    """Groq sometimes rejects tool-calling payloads for specific models."""
    message = str(error)
    return any(token in message for token in ("tool_use_failed", "model_decommissioned", "Failed to call a function"))


def run_agent_turn(
    agent_executor: Optional[AgentExecutor],
    question: str,
    vectorstore: FAISS,
    api_key: str,
    requested_model: str = DEFAULT_GROQ_MODEL,
    answer_provider: str = "groq",
    active_filename: str = None,
    live_search_provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_web_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_web_model: str = DEFAULT_GEMINI_WEB_MODEL,
    chat_history: Optional[list] = None,
) -> dict:
    """
    Invoke the chat agent and retry with the default model if Groq rejects tool use.
    Returns the answer plus the possibly refreshed executor.
    """
    normalized_provider = normalize_answer_provider(answer_provider)
    if normalized_provider != "groq":
        try:
            direct_answer = _build_direct_answer(
                question=question,
                vectorstore=vectorstore,
                groq_api_key=api_key,
                answer_provider=normalized_provider,
                answer_model=requested_model,
                active_filename=active_filename,
                live_search_provider=live_search_provider,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                openai_web_model=openai_web_model,
                gemini_web_model=gemini_web_model,
                chat_history=chat_history,
            )
        except Exception as exc:
            # Cross-provider fallback if primary (Gemini/OpenAI) fails completely
            if api_key: # We have a Groq key to fall back to
                try:
                    direct_answer = _build_direct_answer(
                        question=question,
                        vectorstore=vectorstore,
                        groq_api_key=api_key,
                        answer_provider="groq",
                        answer_model=DEFAULT_GROQ_MODEL,
                        active_filename=active_filename,
                        live_search_provider=live_search_provider,
                        openai_api_key=openai_api_key,
                        gemini_api_key=gemini_api_key,
                        openai_web_model=openai_web_model,
                        gemini_web_model=gemini_web_model,
                        chat_history=chat_history,
                    )
                    return {
                        "answer": direct_answer,
                        "agent_executor": None,
                        "notice": f"Primary Answer Engine ({normalized_provider.capitalize()}) hit an error, but ArthaMind successfully fell back to direct analysis via Groq.",
                        "effective_model": DEFAULT_GROQ_MODEL,
                    }
                except Exception:
                    pass
            raise exc

        return {
            "answer": direct_answer,
            "agent_executor": None,
            "notice": f"Analyst Chat is powered by `{requested_model}` via {normalized_provider.capitalize()}.",
            "effective_model": requested_model,
        }


    active_executor = agent_executor or build_agent_executor(
        vectorstore,
        api_key,
        model=requested_model,
        active_filename=active_filename,
        live_search_provider=live_search_provider,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        openai_web_model=openai_web_model,
        gemini_web_model=gemini_web_model,
    )

    try:
        result = active_executor.invoke({"input": question})
        return {
            "answer": result.get("output", "I couldn't find a response."),
            "agent_executor": active_executor,
            "notice": resolve_agent_model(requested_model)[1],
            "effective_model": resolve_agent_model(requested_model)[0],
        }
    except Exception as exc:
        current_model, _ = resolve_agent_model(requested_model)

        if current_model != DEFAULT_GROQ_MODEL and _is_retryable_tool_error(exc):
            fallback_executor = build_agent_executor(
                vectorstore,
                api_key,
                model=DEFAULT_GROQ_MODEL,
                active_filename=active_filename,
                live_search_provider=live_search_provider,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                openai_web_model=openai_web_model,
                gemini_web_model=gemini_web_model,
            )
            try:
                retry = fallback_executor.invoke({"input": question})
                retry_notice = (
                    f"Analyst Chat retried with `{DEFAULT_GROQ_MODEL}` after "
                    f"`{current_model}` hit a Groq tool-calling error."
                )
                return {
                    "answer": retry.get("output", "I couldn't find a response."),
                    "agent_executor": fallback_executor,
                    "notice": retry_notice,
                    "effective_model": DEFAULT_GROQ_MODEL,
                }
            except Exception:
                direct_answer = _build_direct_answer(
                    question=question,
                    vectorstore=vectorstore,
                    groq_api_key=api_key,
                    answer_provider="groq",
                    answer_model=DEFAULT_GROQ_MODEL,
                    active_filename=active_filename,
                    live_search_provider=live_search_provider,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    openai_web_model=openai_web_model,
                    gemini_web_model=gemini_web_model,
                    chat_history=chat_history,
                )
                return {
                    "answer": direct_answer,
                    "agent_executor": fallback_executor,
                    "notice": (
                        f"Analyst Chat switched to a direct analysis fallback after "
                        f"Groq tool-calling failed on `{current_model}`."
                    ),
                    "effective_model": DEFAULT_GROQ_MODEL,
                }

        direct_answer = _build_direct_answer(
            question=question,
            vectorstore=vectorstore,
            groq_api_key=api_key,
            answer_provider="groq",
            answer_model=DEFAULT_GROQ_MODEL,
            active_filename=active_filename,
            live_search_provider=live_search_provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openai_web_model=openai_web_model,
            gemini_web_model=gemini_web_model,
            chat_history=chat_history,
        )
        return {
            "answer": direct_answer,
            "agent_executor": active_executor,
            "notice": "Analyst Chat switched to a direct analysis fallback after a tool-calling issue.",
            "effective_model": DEFAULT_GROQ_MODEL,
        }


def _prepare_kpi_context(text: str, max_chars: int = 5500) -> str:
    """Build a compact, metric-dense context block for KPI extraction."""
    normalized = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    header_terms = (
        "company", "annual report", "quarter", "financial year", "fy", "overview"
    )
    metric_terms = (
        "revenue", "income", "profit", "pat", "ebitda", "margin", "eps",
        "cash flow", "debt", "assets", "equity", "roe", "roce", "dividend",
        "guidance", "highlight", "risk",
    )

    selected = []
    seen = set()

    def add_line(candidate: str) -> None:
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        selected.append(candidate)

    for line in lines[:20]:
        lowered = line.lower()
        if any(term in lowered for term in header_terms):
            add_line(line)

    for line in lines:
        lowered = line.lower()
        if any(term in lowered for term in metric_terms):
            add_line(line)

    if not selected:
        selected = lines[:40]

    compact = "\n".join(selected)
    if len(compact) < max_chars and lines:
        prefix = "\n".join(lines[:25])
        compact = prefix + "\n" + compact

    return compact[:max_chars]


def extract_kpis_with_llm(text: str, api_key: str) -> dict:
    """
    Use Groq LLM to extract KPIs from raw text.
    Returns parsed JSON dict of financial metrics.
    """
    import json
    from prompts import KPI_EXTRACTION_PROMPT

    llm = build_llm(api_key, model="llama-3.1-8b-instant", temperature=0.0)
    compact_text = _prepare_kpi_context(text)

    last_error = None
    last_raw = ""
    for max_chars in (5500, 4200, 3200, 2400):
        prompt = KPI_EXTRACTION_PROMPT.format(text=compact_text[:max_chars])
        response = None
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Clean markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            if content.endswith("```"):
                content = content[:-3]

            parsed = json.loads(content.strip())
            if parsed:
                return parsed
            last_raw = content.strip()
        except Exception as e:
            last_error = e
            if response is not None:
                last_raw = getattr(response, "content", "")
            if "Request too large" not in str(e) and "rate_limit_exceeded" not in str(e):
                break

    return {"error": str(last_error) if last_error else "KPI extraction returned no data.", "raw": last_raw}


def generate_summary(
    vectorstore: FAISS,
    api_key: str,
    company_name: str = "the company",
    answer_provider: str = "groq",
    model: str = DEFAULT_GROQ_MODEL,
    openai_api_key: str = "",
    gemini_api_key: str = "",
) -> str:
    """Generate a structured executive summary. Falls back to Groq if primary provider quota is exceeded."""
    from prompts import SUMMARY_PROMPT

    # Pull broad context from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    broad_docs = retriever.invoke("revenue profit income financial performance highlights risks outlook guidance")
    context = "\n\n".join([d.page_content for d in broad_docs])

    formatted_prompt = SUMMARY_PROMPT.format(company_name=company_name)
    prompt = f"""Based on this financial report content, provide an executive summary.

CONTENT:
{context[:4000]}

{formatted_prompt}"""

    normalized = normalize_answer_provider(answer_provider)
    effective_model = model
    if normalized == "groq":
        effective_model, _ = resolve_llm_model(model)

    try:
        return generate_text_with_provider(
            provider=normalized,
            prompt=prompt,
            model=effective_model,
            groq_api_key=api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            temperature=0.2,
        )
    except Exception as exc:
        error_str = str(exc)
        # If provider quota/key issue and Groq key available, fall back to Groq
        if normalized != "groq" and api_key and any(
            signal in error_str
            for signal in ("429", "quota", "503", "unavailable", "missing", "RESOURCE_EXHAUSTED")
        ):
            fallback_model, _ = resolve_llm_model(DEFAULT_GROQ_MODEL)
            return (
                f"⚠️ *{normalized.capitalize()} is currently unavailable (quota/rate limit). "
                f"Executive Summary generated via Groq fallback.*\n\n"
                + generate_text_with_provider(
                    provider="groq",
                    prompt=prompt,
                    model=fallback_model,
                    groq_api_key=api_key,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    temperature=0.2,
                )
            )
        raise exc


# ── Peer Comparison Engine ──────────────────────────────────────────────────

def run_peer_comparison(
    question: str,
    vectorstores: dict,          # {company_name: FAISS}
    groq_api_key: str,
    answer_provider: str = "groq",
    answer_model: str = DEFAULT_GROQ_MODEL,
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_web_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_web_model: str = DEFAULT_GEMINI_WEB_MODEL,
    live_search_provider: str = "duckduckgo",
    chat_history: Optional[list] = None,
) -> str:
    """
    Search all peer vectorstores in parallel and synthesise a cross-company answer.
    Uses smaller context per company to avoid LLM timeouts on 3-company comparisons.
    Falls back to Groq if primary provider times out or hits quota.
    """
    CHARS_PER_COMPANY = 900   # tight cap so total prompt stays manageable

    def _search_one(name_vs):
        name, vs = name_vs
        try:
            results = vs.similarity_search(question, k=3)
            context = " ".join([d.page_content for d in results])[:CHARS_PER_COMPANY]
            return f"=== {name} ===\n{context}"
        except Exception:
            return f"=== {name} ===\n[Search failed]"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(vectorstores)) as pool:
        contexts = list(pool.map(_search_one, vectorstores.items()))

    combined_context = "\n\n".join(contexts)
    conversation_context = _format_chat_history_for_prompt(chat_history)

    # Live context — only fetch if truly needed, keep short
    live_context = ""
    if _question_needs_live_context(question):
        try:
            raw = run_live_web_search(
                question,
                provider=live_search_provider,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                openai_model=openai_web_model,
                gemini_model=gemini_web_model,
            )
            live_context = raw[:400]
        except Exception:
            pass

    companies = ", ".join(vectorstores.keys())
    prompt = f"""You are ArthaMind, a senior equity analyst. Compare {len(vectorstores)} companies: {companies}.

QUESTION: {question}

REPORT DATA:
{combined_context}
{f"LIVE CONTEXT: {live_context}" if live_context else ""}
{f"PRIOR CHAT: {conversation_context}" if conversation_context.strip() else ""}

RULES:
- Use a markdown table for any numeric comparison
- Cite the company name for each figure
- End with a bold **PEER VERDICT:** (1-2 sentences, pick a winner and state why)
"""

    normalized = normalize_answer_provider(answer_provider)
    effective_model = answer_model
    if normalized == "groq":
        effective_model, _ = resolve_llm_model(answer_model)

    try:
        return generate_text_with_provider(
            provider=normalized,
            prompt=prompt,
            model=effective_model,
            groq_api_key=groq_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            temperature=0.1,
        )
    except Exception as exc:
        error_str = str(exc)
        if normalized != "groq" and groq_api_key and any(
            s in error_str for s in ("timed out", "timeout", "429", "503", "quota", "RESOURCE_EXHAUSTED")
        ):
            fallback_model, _ = resolve_llm_model(DEFAULT_GROQ_MODEL)
            return (
                "[Gemini timed out - Groq fallback used]\n\n"
                + generate_text_with_provider(
                    provider="groq",
                    prompt=prompt,
                    model=fallback_model,
                    groq_api_key=groq_api_key,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    temperature=0.1,
                )
            )
        raise exc


# ── Commodity Radar ──────────────────────────────────────────────────────────

COMMODITY_SEARCH_QUERIES = {
    "Lithium Carbonate": "current lithium carbonate spot price per tonne USD 2025 2026",
    "Crude Oil (WTI)":   "current WTI crude oil price per barrel USD today",
    "Natural Gas":       "current natural gas price per MMBtu USD today",
    "Copper":            "current copper price per tonne USD today LME",
    "INR/USD":           "current Indian rupee USD exchange rate today",
    "Coal":              "current thermal coal price per tonne USD today",
}


def fetch_commodity_live_price(
    commodity: str,
    live_search_provider: str = "duckduckgo",
    openai_api_key: str = "",
    gemini_api_key: str = "",
    openai_web_model: str = DEFAULT_OPENAI_WEB_MODEL,
    gemini_web_model: str = DEFAULT_GEMINI_WEB_MODEL,
) -> dict:
    """
    Fetch the live price of a commodity via web search.
    Returns {price: float, unit: str, direction: str, raw_text: str}
    """
    query = COMMODITY_SEARCH_QUERIES.get(commodity, f"current {commodity} price today USD")
    try:
        raw = run_live_web_search(
            query,
            provider=live_search_provider,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            openai_model=openai_web_model,
            gemini_model=gemini_web_model,
        )
        # Try to extract a number from the raw text
        import re as _re
        numbers = _re.findall(r'[\$₹]?\s*(\d[\d,]*\.?\d*)\s*(?:per\s+(?:tonne|barrel|MMBtu|ton))?', raw)
        price = None
        for n in numbers:
            candidate = float(n.replace(",", ""))
            # Filter out obviously wrong numbers (years, page numbers, etc.)
            if commodity == "INR/USD" and 75 < candidate < 100:
                price = candidate
                break
            elif "Oil" in commodity and 50 < candidate < 200:
                price = candidate
                break
            elif "Lithium" in commodity and 5000 < candidate < 100000:
                price = candidate
                break
            elif "Coal" in commodity and 50 < candidate < 500:
                price = candidate
                break
            elif "Copper" in commodity and 5000 < candidate < 20000:
                price = candidate
                break
            elif "Gas" in commodity and 0.5 < candidate < 30:
                price = candidate
                break
        return {
            "price": price,
            "raw_text": raw[:500],
            "error": None,
        }
    except Exception as exc:
        return {"price": None, "raw_text": "", "error": str(exc)}
