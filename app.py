"""
Amazon Product Review Summarizer (RAG-based).
Production-oriented: sentiment breakdown, overall summary, fast & accurate.
Helps users decide quickly and reduce purchase time.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config (production-ready)
# -----------------------------
META_PATH = os.environ.get("META_CSV", "src/chinmay/Appliances_meta.csv")
REVIEWS_PATH = os.environ.get("REVIEWS_CSV", "src/chinmay/Appliances_reviews.csv")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
# Speed vs coverage: lower = faster, higher = more representative
MAX_REVIEWS_PER_PRODUCT = 50   # Max reviews to load & embed per product (sample for speed)
MAX_REVIEWS_FOR_SUMMARY = 8    # RAG: top-k reviews used for summary (fewer = faster)
MAX_SUMMARY_INPUT_CHARS = 4000 # Truncate for model stability
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 50
# Sentiment from star rating: 1-2 negative, 3 neutral, 4-5 positive
RATING_NEGATIVE = (1, 2)
RATING_NEUTRAL = (3,)
RATING_POSITIVE = (4, 5)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Amazon Review Summarizer", layout="wide")

st.title("📝 Amazon Product Review Summarizer (RAG-based)")
st.caption("Get sentiment breakdown and an overall summary so you can decide faster and save time.")

# -----------------------------
# Load Data (Cached)
# -----------------------------
@st.cache_data
def load_data():
    meta = pd.read_csv(META_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)
    return meta, reviews

try:
    meta_df, reviews_df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# Ensure rating column exists for sentiment
if "rating" not in reviews_df.columns:
    reviews_df["rating"] = 3  # fallback neutral

# -----------------------------
# Load Embedding Model (Cached) – lazy per product for speed
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")

embedder = load_embedder()

# -----------------------------
# Load Summarizer (Cached)
# -----------------------------
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    return tokenizer, model

tokenizer, model = load_summarizer()

# -----------------------------
# Helpers: Sentiment from rating
# -----------------------------
def sentiment_from_rating(rating):
    try:
        r = int(float(rating))
        if r in RATING_NEGATIVE:
            return "negative"
        if r in RATING_NEUTRAL:
            return "neutral"
        if r in RATING_POSITIVE:
            return "positive"
    except (ValueError, TypeError):
        pass
    return "neutral"

def compute_sentiment_counts(product_reviews_df):
    sentiments = product_reviews_df["rating"].apply(sentiment_from_rating)
    return sentiments.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)

# -----------------------------
# RAG: Get product reviews, then top-k by relevance for summary (within product only)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_product_reviews_and_embeddings(asin):
    subset = reviews_df[reviews_df["parent_asin"] == asin].copy()
    if subset.empty:
        return None
    # Cap to MAX_REVIEWS_PER_PRODUCT for speed (take first N for reproducibility)
    if len(subset) > MAX_REVIEWS_PER_PRODUCT:
        subset = subset.iloc[:MAX_REVIEWS_PER_PRODUCT]
    texts = subset["text"].fillna("").tolist()
    if not texts:
        return None
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    subset = subset.reset_index(drop=True)
    return {
        "df": subset,
        "texts": texts,
        "embeddings": embeddings,
    }

def get_rag_reviews_for_summary(product_data, product_title, top_k=MAX_REVIEWS_FOR_SUMMARY):
    """Within this product's reviews, pick top_k most relevant to product title (RAG)."""
    query_emb = embedder.encode([product_title], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, product_data["embeddings"])[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [product_data["texts"][i] for i in top_idx]

def summarize_reviews(review_texts):
    """Summarize a list of review strings; truncate to fit model."""
    joined = " ".join(review_texts)
    if len(joined) > MAX_SUMMARY_INPUT_CHARS:
        joined = joined[:MAX_SUMMARY_INPUT_CHARS].rsplit(" ", 1)[0] + "."
    inputs = tokenizer(
        joined,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=SUMMARY_MAX_LENGTH,
        min_length=SUMMARY_MIN_LENGTH,
        do_sample=False,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -----------------------------
# UI
# -----------------------------
product_options = meta_df["title"].dropna().unique().tolist()
selected_product = st.selectbox("Select a Product:", product_options, key="product_select")

if st.button("Generate Summary", type="primary"):

    product_row = meta_df[meta_df["title"] == selected_product]
    if product_row.empty:
        st.error("Product not found in metadata.")
        st.stop()

    asin = product_row.iloc[0]["parent_asin"]

    with st.spinner("Loading reviews and computing embeddings…"):
        product_data = get_product_reviews_and_embeddings(asin)

    if product_data is None:
        st.warning("No reviews found for this product.")
        st.stop()

    product_reviews_df = product_data["df"]
    n_reviews = len(product_reviews_df)

    # ---------- Sentiment (from rating) ----------
    counts = compute_sentiment_counts(product_reviews_df)
    n_pos = int(counts.get("positive", 0))
    n_neu = int(counts.get("neutral", 0))
    n_neg = int(counts.get("negative", 0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("😊 Positive", f"{n_pos} ({100*n_pos/max(n_reviews,1):.0f}%)")
    with col2:
        st.metric("😐 Neutral", f"{n_neu} ({100*n_neu/max(n_reviews,1):.0f}%)")
    with col3:
        st.metric("😞 Negative", f"{n_neg} ({100*n_neg/max(n_reviews,1):.0f}%)")

    total_for_product = len(reviews_df[reviews_df["parent_asin"] == asin])
    if total_for_product > MAX_REVIEWS_PER_PRODUCT:
        st.caption(f"Based on {n_reviews} review(s) (sampled from {total_for_product} total for speed) · Sentiment from star ratings.")
    else:
        st.caption(f"Based on {n_reviews} review(s) · Sentiment from star ratings (1–2: negative, 3: neutral, 4–5: positive).")

    # ---------- RAG: top-k reviews for summary ----------
    with st.spinner("Building summary from most relevant reviews…"):
        rag_texts = get_rag_reviews_for_summary(product_data, selected_product)
        summary = summarize_reviews(rag_texts)

    st.subheader("📋 Overall Summary")
    st.info(summary)
    st.caption(f"Summary from up to {MAX_REVIEWS_FOR_SUMMARY} most relevant reviews (RAG) for this product.")

    # ---------- Quick decision aid ----------
    st.subheader("🎯 Quick Takeaway")
    if n_neg > n_pos and n_reviews >= 5:
        st.warning("Reviews lean negative. Check the summary and consider alternatives if important.")
    elif n_pos > n_neg * 2 or (n_pos >= n_reviews // 2 and n_neg < n_reviews // 4):
        st.success("Reviews are mostly positive. Good candidate if it fits your needs.")
    else:
        st.info("Mixed feedback. Read the summary and focus on aspects that matter to you.")
