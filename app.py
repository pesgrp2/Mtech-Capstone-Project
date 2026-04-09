"""
Streamlit entry: wires RAG chain, analysis form, and results UI.
Implementation lives under `review_app/`.
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from review_app.build_index import maybe_build_faiss_index
from review_app.config import PARQUET_PATH
from review_app.data import load_asin_catalog, rank_products_by_query
from review_app.evaluation import compute_rouge_against_retrieved
from review_app.rag import build_chain, load_vectorstore
from review_app.ui import (
    inject_custom_css,
    product_review_summary_dialog,
    render_more_recommendations,
    render_recommendation_cards,
    render_sidebar_model_info,
    reset_summary_dialog_session_state,
)

st.set_page_config(
    page_title="AI Product Review Analyzer",
    layout="wide",
)

maybe_build_faiss_index()

st.title("🛍️ AI Product Review Analyzer (RAG System)")
inject_custom_css()
render_sidebar_model_info()


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
        f"- Check that `{PARQUET_PATH}` is in the project folder (or set `REVIEWS_PARQUET`)"
    )
    st.stop()


if st.session_state.get("active_query") and "analyze_query_input" not in st.session_state:
    st.session_state["analyze_query_input"] = st.session_state["active_query"]

st.markdown("##### Ask questions about products based on customer reviews")
with st.form("analyze_form", clear_on_submit=False):
    try:
        fc1, fc2 = st.columns([5, 1], vertical_alignment="center")
    except TypeError:
        fc1, fc2 = st.columns([5, 1])
    with fc1:
        q_input = st.text_input(
            "Question",
            label_visibility="collapsed",
            placeholder="Example: Which water filter has the best customer reviews?",
            key="analyze_query_input",
        )
    with fc2:
        submitted = st.form_submit_button(
            "Analyze Reviews",
            type="primary",
            use_container_width=True,
        )


class _DocContent:
    __slots__ = ("page_content",)

    def __init__(self, text):
        self.page_content = text


if submitted:
    q_clean = (q_input or "").strip()
    if not q_clean:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Analyzing reviews… (first Ollama response can take 1–2 minutes)"):
                response = qa_chain.invoke({"query": q_clean})
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
        src = response.get("source_documents") or []
        st.session_state["active_query"] = q_clean
        st.session_state["active_result"] = result
        st.session_state["active_doc_contents"] = [
            getattr(d, "page_content", str(d)) for d in src
        ]
        st.rerun()

if st.session_state.get("active_query"):
    q = st.session_state["active_query"]
    result = st.session_state.get("active_result") or ""
    doc_contents = st.session_state.get("active_doc_contents") or []
    docs = [_DocContent(t) for t in doc_contents]

    st.subheader("📊 AI Generated Insights")
    if result and str(result).strip():
        st.write(result)
    else:
        st.warning(
            "The model returned an empty answer. "
            "Check Ollama logs, try a shorter question, or confirm `ollama run llama3` works in a terminal."
        )
    st.markdown("---")

    st.subheader("⭐ Recommended products")
    vs = load_vectorstore()
    catalog = load_asin_catalog()
    all_recs = rank_products_by_query(vs, q)
    main_recs = [r for r in all_recs if r.get("in_main_cards")]
    more_recs = [r for r in all_recs if not r.get("in_main_cards")]
    render_recommendation_cards(main_recs, catalog)
    render_more_recommendations(more_recs, catalog)
    st.markdown("---")

    if st.button("Clear analysis"):
        for k in (
            "active_query",
            "active_result",
            "active_doc_contents",
            "summary_dialog_asin",
            "summary_dialog_title",
            "analyze_query_input",
        ):
            st.session_state.pop(k, None)
        st.rerun()

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

    st.subheader("💬 Sentiment Distribution (Example)")
    sentiment_data = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [74, 15, 11],
        }
    )
    fig, ax = plt.subplots()
    ax.bar(sentiment_data["Sentiment"], sentiment_data["Count"])
    ax.set_ylabel("Percentage")
    st.pyplot(fig)
    st.markdown("---")

    st.subheader("📄 Retrieved Review Evidence")
    for i, doc in enumerate(docs):
        with st.expander(f"Document {i+1}"):
            st.write(doc.page_content)


if st.session_state.get("summary_dialog_asin"):
    product_review_summary_dialog(
        st.session_state["summary_dialog_asin"],
        st.session_state.get("summary_dialog_title") or st.session_state["summary_dialog_asin"],
    )
    st.session_state.pop("summary_dialog_asin", None)
    st.session_state.pop("summary_dialog_title", None)
    reset_summary_dialog_session_state()
