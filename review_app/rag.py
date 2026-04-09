"""Embeddings, FAISS, LLM, QA chain, and per-product review summarization."""

import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from review_app.config import EMBEDDING_MODEL, FAISS_INDEX_DIR, OLLAMA_MODEL

RAG_QA_TEMPLATE = """
You are an AI assistant that analyzes product reviews.

Provide output strictly in the following format.

Summary:
Give a short summary of key product feedback.

Top Recommendations:
List the best products mentioned.

Review Sentiment:
Provide estimated sentiment percentage and common complaints.

Context:
{context}

Question:
{question}
"""

RAG_QA_PROMPT = PromptTemplate(
    template=RAG_QA_TEMPLATE.strip(),
    input_variables=["context", "question"],
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm():
    return Ollama(model=OLLAMA_MODEL)


@st.cache_resource
def build_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = load_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_QA_PROMPT},
        return_source_documents=True,
    )


def summarize_product_reviews_llm(llm, product_title: str, asin: str, review_blob: str) -> str:
    if not review_blob.strip():
        return "No review text found for this product in the dataset."
    prompt = (
        f"Product: {product_title}\nASIN: {asin}\n\n"
        "Below are customer review excerpts. Write a clear, concise summary for a shopper.\n"
        "Use short sections: Key positives, Common complaints, Overall verdict (2–3 sentences).\n"
        "Base everything only on the excerpts; do not invent facts.\n\n"
        f"Review excerpts:\n{review_blob}\n\nSummary:"
    )
    try:
        out = llm.invoke(prompt)
        if isinstance(out, str):
            return out
        if hasattr(out, "content"):
            return str(out.content)
        return str(out)
    except Exception as e:
        return f"Could not generate summary: {e}"
