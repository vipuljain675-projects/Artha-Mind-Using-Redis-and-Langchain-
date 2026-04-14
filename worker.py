"""
worker.py — Background Job Worker for ArthaMind
Processes heavy PDF ingestion tasks asynchronously via Redis Queue (rq).
Runs as a separate process: `rq worker`
"""

import os
import multiprocessing
from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

from ingest import chunk_documents, build_vector_store
from langchain_community.document_loaders import PyPDFLoader
from chain import extract_kpis_with_llm


def async_ingest_and_extract(file_path: str, filename: str, api_key: str) -> dict:
    """
    Background job to process massive PDFs without blocking the UI.
    Steps:
    1. Load PDF from disk path
    2. Chunk into semantic pieces
    3. Build & save FAISS vector store to disk
    4. Call Groq LLM to extract financial KPIs
    5. Return structured results dict (stored in Redis by rq)
    """
    try:
        # 1. Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename

        # 2. Chunk & build vector DB
        chunks = chunk_documents(docs)
        build_vector_store(chunks, store_name="financial_index")

        # 3. Extract KPIs using first ~15 pages (covers balance sheet in longer reports)
        raw_text = "\n".join([d.page_content for d in docs[:15]])
        kpis = extract_kpis_with_llm(raw_text, api_key)

        # 4. Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "filename": filename,
            "total_pages": len(docs),
            "total_chunks": len(chunks),
            "kpis": kpis,
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    """
    Start the RQ Worker using SimpleWorker (no forking).
    This avoids the macOS ARM64 SIGABRT crash caused by PyTorch/FAISS
    being incompatible with multiprocessing fork.
    Run with: python worker.py
    """
    multiprocessing.set_start_method('spawn', force=True)
    redis_conn = Redis(host="localhost", port=6379)
    q = Queue(connection=redis_conn)
    w = SimpleWorker([q], connection=redis_conn)
    w.work()

