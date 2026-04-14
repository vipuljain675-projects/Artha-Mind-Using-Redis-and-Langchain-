"""
chain.py — LangChain RAG Chain Setup
Builds the Conversational Retrieval Chain with Groq + Memory
"""

import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import FAISS
from prompts import FINANCIAL_SYSTEM_PROMPT


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

@tool
def web_search(query: str) -> str:
    """Search the web for news or recent events to get unstructured current context."""
    from duckduckgo_search import DDGS
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Error searching the web: {e}"

def get_document_search_tool(vectorstore: FAISS):
    @tool
    def financial_document_search(query: str) -> str:
        """Search the uploaded financial reports (Annual reports, 10-K, earnings) for key metrics and commentary.
        CRITICAL: Pass short, targeted keywords (e.g. "EPS" or "Earnings Per Share") as the query rather than full sentences for best results!"""
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
        )
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the uploaded documents. Try searching with different keywords."
        return "\n\n".join([f"Source (File: {d.metadata.get('source_file', '?')}): {d.page_content}" for d in docs])
    return financial_document_search


def build_agent_executor(
    vectorstore: FAISS,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
) -> AgentExecutor:
    """
    Build the Agent Executor with Tool Calling capabilities.
    """
    llm = build_llm(api_key, model)

    tools = [
        get_document_search_tool(vectorstore),
        live_stock_price,
        web_search
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
        k=6,
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        max_iterations=7
    )

    return agent_executor


def extract_kpis_with_llm(text: str, api_key: str) -> dict:
    """
    Use Groq LLM to extract KPIs from raw text.
    Returns parsed JSON dict of financial metrics.
    """
    import json
    from prompts import KPI_EXTRACTION_PROMPT

    llm = build_llm(api_key, model="llama-3.1-8b-instant", temperature=0.0)

    prompt = KPI_EXTRACTION_PROMPT.format(text=text[:15000])

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

        return json.loads(content.strip())
    except Exception as e:
        return {"error": str(e), "raw": response.content if "response" in dir() else ""}


def generate_summary(vectorstore: FAISS, api_key: str, company_name: str = "the company") -> str:
    """Generate a structured executive summary of the entire report."""
    from prompts import SUMMARY_PROMPT

    llm = build_llm(api_key, temperature=0.2)

    # Pull broad context from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    broad_docs = retriever.get_relevant_documents(
        "revenue profit income financial performance highlights risks"
    )
    context = "\n\n".join([d.page_content for d in broad_docs])
    
    formatted_prompt = SUMMARY_PROMPT.format(company_name=company_name)

    prompt = f"""Based on this financial report content, provide an executive summary.

CONTENT:
{context[:4000]}

{formatted_prompt}"""

    response = llm.invoke(prompt)
    return response.content
