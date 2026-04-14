"""
ingest.py — PDF Ingestion & Vector Store Pipeline
Handles: PDF loading → Text extraction → Chunking → Embedding → FAISS storage
"""

import os
import json
import tempfile
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Initialize the embedding model (cached after first load)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf(file_bytes: bytes, filename: str) -> list:
    """
    Load a PDF from raw bytes and extract documents.
    Returns a list of LangChain Document objects.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        # Tag each page with the source filename
        for doc in docs:
            doc.metadata["source_file"] = filename
        return docs
    finally:
        os.unlink(tmp_path)


def chunk_documents(docs: list, chunk_size: int = 600, chunk_overlap: int = 80) -> list:
    """
    Split documents into overlapping chunks.
    - chunk_size=600: balanced for financial tables + prose
    - chunk_overlap=80: captures cross-boundary context (useful for multi-line tables)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vector_store(chunks: list, store_name: str = "financial_index") -> FAISS:
    """
    Build or append to a FAISS vector store from document chunks.
    Saves to disk for reuse across sessions.
    """
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    embeddings = get_embeddings()

    existing_store = load_vector_store(store_name)
    if existing_store:
        existing_store.add_documents(chunks)
        vectorstore = existing_store
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    save_path = str(VECTOR_STORE_DIR / store_name)
    vectorstore.save_local(save_path)
    return vectorstore


def load_vector_store(store_name: str = "financial_index") -> "FAISS":
    """
    Load an existing FAISS vector store from disk.
    Returns None if not found.
    """
    store_path = VECTOR_STORE_DIR / store_name
    if not store_path.exists():
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def process_pdf(file_bytes: bytes, filename: str, store_name: str = "financial_index") -> dict:
    """
    Full ingestion pipeline: PDF → Chunks → VectorStore.
    Returns metadata about the processed document.
    """
    docs = load_pdf(file_bytes, filename)
    chunks = chunk_documents(docs)
    vectorstore = build_vector_store(chunks, store_name)

    # Gather raw text for KPI extraction (first 4000 chars = usually cover page + highlights)
    raw_text = "\n".join([d.page_content for d in docs[:8]])

    return {
        "filename": filename,
        "total_pages": len(docs),
        "total_chunks": len(chunks),
        "vectorstore": vectorstore,
        "raw_text_sample": raw_text[:4000],
    }


def extract_raw_text(file_bytes: bytes, filename: str, max_pages: int = 10) -> str:
    """Extract the first N pages of text for quick KPI extraction."""
    docs = load_pdf(file_bytes, filename)
    return "\n\n".join([d.page_content for d in docs[:max_pages]])
