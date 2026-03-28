import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import streamlit as st
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


def compute_rouge_against_retrieved(answer: str, source_docs):
    """
    ROUGE F1 comparing the generated answer to concatenated retrieved chunks.
    This is a retrieval-as-reference proxy (not ROUGE vs a human-written gold summary).
    """
    if not answer or not str(answer).strip() or not source_docs:
        return None, None, None
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None, None, None
    ref_parts = []
    for d in source_docs:
        t = getattr(d, "page_content", None) or ""
        if t.strip():
            ref_parts.append(t.strip())
    reference = " ".join(ref_parts)[:12000]
    if not reference.strip():
        return None, None, None
    pred = str(answer).strip()[:8000]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, pred)
    return (
        round(s["rouge1"].fmeasure, 3),
        round(s["rouge2"].fmeasure, 3),
        round(s["rougeL"].fmeasure, 3),
    )


st.set_page_config(
    page_title="AI Product Review Analyzer",
    layout="wide",
)

# Build FAISS only when explicitly requested (not on every Streamlit rerun).
# Run once: BUILD_FAISS_INDEX=1 streamlit run app.py
# Or use a separate build_index.py script.
if os.environ.get("BUILD_FAISS_INDEX", "").strip() in ("1", "true", "yes"):
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet("embedding_ready_reviews_small.parquet")
    _ = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # warm cache if needed
    _embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
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
    vectorstore = FAISS.from_documents(chunks, _embed)
    vectorstore.save_local("faiss_index")
    st.error("FAISS index built. Stop the app, unset BUILD_FAISS_INDEX, and run again normally.")
    st.stop()


st.title("🛍️ AI Product Review Analyzer (RAG System)")


# --------------------------------------------------
# Sidebar - Model Details
# --------------------------------------------------

st.sidebar.header("⚙️ Model Information")

st.sidebar.markdown("""
**Architecture:** Retrieval Augmented Generation (RAG)

**Embedding Model:**  
sentence-transformers/all-MiniLM-L6-v2

**LLM Model:**  
Llama3 (via Ollama)

**Vector Database:**  
FAISS

**Framework:**  
LangChain
""")

# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

template = """
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


prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return Ollama(model="llama3")


# --------------------------------------------------
# Build RAG Chain
# --------------------------------------------------

@st.cache_resource
def build_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


try:
    qa_chain = build_chain()
except Exception as e:
    qa_chain = None
    st.error("Could not load the RAG chain (FAISS or Ollama). See details below.")
    with st.expander("Error details"):
        st.exception(e)
    st.info(
        "**Common fixes:**\n"
        "- Start Ollama and pull the model: `ollama serve` then `ollama pull llama3`\n"
        "- Ensure `faiss_index/` exists (run once with `BUILD_FAISS_INDEX=1 streamlit run app.py`)\n"
        "- Check that `embedding_ready_reviews_small.parquet` is in the project folder"
    )
    st.stop()


# --------------------------------------------------
# User Input
# --------------------------------------------------

query = st.text_input(
    "Ask questions about products based on customer reviews:",
    placeholder="Example: Which water filter has the best customer reviews?"
)


# --------------------------------------------------
# Run RAG
# --------------------------------------------------

if st.button("Analyze Reviews"):

    if query:

        try:
            with st.spinner("Analyzing reviews… (first Ollama response can take 1–2 minutes)"):
                response = qa_chain.invoke({"query": query})
        except Exception as e:
            st.error("The analyzer failed while calling the model or retriever.")
            with st.expander("Error details"):
                st.exception(e)
            st.info(
                "If you see **connection refused**, start Ollama: `ollama serve`, "
                "then `ollama pull llama3` and try again."
            )
            st.stop()

        result = response.get("result") or response.get("answer") or ""
        docs = response.get("source_documents") or []

        # --------------------------------------------------
        # AI Summary Output
        # --------------------------------------------------

        st.subheader("📊 AI Generated Insights")
        if result and str(result).strip():
            st.write(result)
        else:
            st.warning(
                "The model returned an empty answer. "
                "Check Ollama logs, try a shorter question, or confirm `ollama run llama3` works in a terminal."
            )
            with st.expander("Debug: raw response keys"):
                st.write(list(response.keys()))
        st.markdown("---")


        # --------------------------------------------------
        # Evaluation Metrics
        # --------------------------------------------------

        st.subheader("📈 Model Evaluation Metrics")
        r1, r2, rL = compute_rouge_against_retrieved(result, docs)
        col1, col2, col3 = st.columns(3)
        if r1 is not None:
            col1.metric("ROUGE-1 (F1)", f"{r1:.3f}")
            col2.metric("ROUGE-2 (F1)", f"{r2:.3f}")
            col3.metric("ROUGE-L (F1)", f"{rL:.3f}")
            st.caption(
                "ROUGE F1 compares the answer to **retrieved review text** (proxy overlap). "
                "For publication-style evaluation, compare against human reference summaries instead."
            )
        else:
            col1.metric("ROUGE-1 (F1)", "—")
            col2.metric("ROUGE-2 (F1)", "—")
            col3.metric("ROUGE-L (F1)", "—")
            st.caption(
                "Could not compute ROUGE (empty answer or no retrieved docs). "
                "Install: `pip install rouge-score` if the package is missing."
            )
        st.markdown("---")


        # --------------------------------------------------
        # Sentiment Chart
        # --------------------------------------------------

        st.subheader("💬 Sentiment Distribution (Example)")
        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [74, 15, 11]
        })
        fig, ax = plt.subplots()
        ax.bar(
            sentiment_data["Sentiment"],
            sentiment_data["Count"]
        )
        ax.set_ylabel("Percentage")
        st.pyplot(fig)
        st.markdown("---")


        # --------------------------------------------------
        # Retrieved Documents
        # --------------------------------------------------

        st.subheader("📄 Retrieved Review Evidence")

        for i, doc in enumerate(docs):
            with st.expander(f"Document {i+1}"):
                st.write(doc.page_content)

    else:
        st.warning("Please enter a question.")