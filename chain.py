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


def build_qa_chain(
    vectorstore: FAISS,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    k_docs: int = 5,
) -> ConversationalRetrievalChain:
    """
    Build the full RAG Q&A chain with:
    - FAISS retriever (top-k semantic search)
    - ConversationBufferWindowMemory (last 6 exchanges)
    - Custom financial analyst system prompt
    """
    llm = build_llm(api_key, model)

    # Memory: keep last 6 Q&A pairs to maintain context without token overflow
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=6,
    )

    # Retriever: MMR (Max Marginal Relevance) avoids redundant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_docs, "fetch_k": 10, "lambda_mult": 0.7},
    )

    # Build system + human prompt
    system_prompt = SystemMessagePromptTemplate.from_template(FINANCIAL_SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(
        """Use the following financial report excerpts to answer the question.
        
Relevant Context:
{context}

Question: {question}

Provide a detailed, data-driven answer. Always reference specific numbers."""
    )

    qa_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    return chain


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


def generate_summary(vectorstore: FAISS, api_key: str) -> str:
    """Generate a structured executive summary of the entire report."""
    from prompts import SUMMARY_PROMPT

    llm = build_llm(api_key, temperature=0.2)

    # Pull broad context from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    broad_docs = retriever.get_relevant_documents(
        "revenue profit income financial performance highlights risks"
    )
    context = "\n\n".join([d.page_content for d in broad_docs])

    prompt = f"""Based on this financial report content, provide an executive summary.

CONTENT:
{context[:4000]}

{SUMMARY_PROMPT}"""

    response = llm.invoke(prompt)
    return response.content
