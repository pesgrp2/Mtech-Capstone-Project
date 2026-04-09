"""Optional one-shot FAISS build when BUILD_FAISS_INDEX=1."""

import os

import streamlit as st

from review_app.config import EMBEDDING_MODEL, FAISS_INDEX_DIR, PARQUET_PATH


def maybe_build_faiss_index() -> None:
    """
    If BUILD_FAISS_INDEX is set, build the index from parquet and stop the app.
    Run once: BUILD_FAISS_INDEX=1 streamlit run app.py
    """
    if os.environ.get("BUILD_FAISS_INDEX", "").strip() not in ("1", "true", "yes"):
        return

    import pandas as pd
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(PARQUET_PATH)
    _ = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embed = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    documents = [
        Document(page_content=row["embedding_text"], metadata={"asin": row["asin"]})
        for _, row in df.iterrows()
    ]
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embed)
    vectorstore.save_local(FAISS_INDEX_DIR)
    st.error(
        "FAISS index built. Stop the app, unset BUILD_FAISS_INDEX, and run again normally."
    )
    st.stop()
