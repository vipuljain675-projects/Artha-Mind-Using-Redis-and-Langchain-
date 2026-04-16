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


def should_delete_temp_file(file_path: str, filename: str) -> bool:
    """
    Only delete transient worker copies.
    If the path basename matches the user-facing filename, treat it as the
    persistent report copy and keep it on disk.
    """
    return os.path.basename(file_path) != filename


def async_ingest_and_extract(files: list, api_key: str) -> dict:
    """
    Background job to process massive PDFs without blocking the UI.
    Accepts a list of files to support Multi-Document RAG.
    """
    try:
        all_chunks = []
        multi_kpis = {}
        total_pages = 0

        for file_info in files:
            file_path = file_info["file_path"]
            filename = file_info["filename"]

            # 1. Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            
            total_pages += len(docs)

            # 2. Chunk
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

            # 3. Extract KPIs for this specific document
            raw_text = "\n".join([d.page_content for d in docs[:15]])
            kpis = extract_kpis_with_llm(raw_text, api_key)
            multi_kpis[filename] = kpis

            # 4. Clean up temp file
            if os.path.exists(file_path) and should_delete_temp_file(file_path, filename):
                os.remove(file_path)

        # 5. Build/Append to vector DB using all chunks combined
        if all_chunks:
            build_vector_store(all_chunks, store_name="financial_index")

        return {
            "status": "success",
            "filenames": [f["filename"] for f in files],
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "kpis": multi_kpis,
        }

    except Exception as e:
        # Cleanup on failure
        for file_info in files:
            if (
                os.path.exists(file_info["file_path"])
                and should_delete_temp_file(file_info["file_path"], file_info["filename"])
            ):
                os.remove(file_info["file_path"])
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
