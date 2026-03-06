"""
Amazon Product Review Summarizer (RAG-based).
Amazon-style UI: browse products, open sentiment & summary in a popup.
"""
import os
import re
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
MAX_REVIEWS_PER_PRODUCT = 50
MAX_REVIEWS_FOR_SUMMARY = 8
MAX_SUMMARY_INPUT_CHARS = 4000
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 50
RATING_NEGATIVE = (1, 2)
RATING_NEUTRAL = (3,)
RATING_POSITIVE = (4, 5)
PRODUCTS_PER_ROW = 4
MAX_PRODUCTS_SHOWN = 24  # Show first N products (after filter)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Review Summarizer", layout="wide", initial_sidebar_state="collapsed")

# -----------------------------
# Amazon-style CSS
# -----------------------------
st.markdown("""
<style>
  /* Header bar */
  .amazon-header {
    background: linear-gradient(180deg, #131921 0%, #232f3e 100%);
    padding: 0.6rem 1rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-radius: 0 0 8px 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .amazon-header h1 {
    color: #febd69 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
  }
  .amazon-header p {
    color: #ccc !important;
    font-size: 0.85rem !important;
    margin: 0.2rem 0 0 0 !important;
  }
  /* Search bar */
  .stTextInput > div > div > input {
    border-radius: 4px;
    border: 2px solid #ff9900;
  }
  .stTextInput > div > div > input:focus {
    box-shadow: 0 0 0 2px rgba(255,153,0,0.3);
  }
  /* Product cards */
  .product-card {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    height: 100%;
    transition: box-shadow 0.2s, border-color 0.2s;
    display: flex;
    flex-direction: column;
  }
  .product-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    border-color: #ff9900;
  }
  .product-card img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    border-radius: 6px;
    background: #f8f8f8;
  }
  .product-card .title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #007185;
    line-height: 1.3;
    margin: 0.5rem 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 3.9em;
  }
  .product-card .title:hover { color: #c7511f; }
  .product-card .rating-row {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin-bottom: 0.3rem;
  }
  .product-card .stars { color: #ffa41c; font-size: 0.95rem; }
  .product-card .rating-count { color: #007185; font-size: 0.8rem; }
  .product-card .price { font-size: 1.1rem; font-weight: 700; color: #b12704; margin: 0.3rem 0; }
  .product-card .btn-summary {
    margin-top: auto;
    padding: 0.5rem 0.75rem;
    background: linear-gradient(180deg, #ffd479 0%, #ff9900 100%);
    color: #111 !important;
    border: 1px solid #e47911;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    width: 100%;
  }
  .product-card .btn-summary:hover {
    background: linear-gradient(180deg, #f5c76b 0%, #eb8c00 100%);
  }
  /* Modal / dialog styling */
  [data-testid="stModal"] {
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  }
  [data-testid="stModal"] h2 { color: #232f3e !important; }
</style>
""", unsafe_allow_html=True)

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

if "rating" not in reviews_df.columns:
    reviews_df["rating"] = 3

# -----------------------------
# Helpers: product image from meta
# -----------------------------
def get_first_image_url(images_str):
    if pd.isna(images_str) or not str(images_str).strip():
        return None
    s = str(images_str)
    m = re.search(r"'large':\s*'([^']+)'", s) or re.search(r'"large":\s*"([^"]+)"', s)
    if m:
        return m.group(1)
    m = re.search(r"https://[^\s'\")\]]+\.(?:jpg|jpeg|png|webp)", s, re.I)
    return m.group(0) if m else None

# -----------------------------
# Models (Cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    return tokenizer, model

embedder = load_embedder()
tokenizer, model = load_summarizer()

# -----------------------------
# Sentiment helpers
# -----------------------------
def sentiment_from_rating(rating):
    try:
        r = int(float(rating))
        if r in RATING_NEGATIVE: return "negative"
        if r in RATING_NEUTRAL: return "neutral"
        if r in RATING_POSITIVE: return "positive"
    except (ValueError, TypeError):
        pass
    return "neutral"

def compute_sentiment_counts(product_reviews_df):
    sentiments = product_reviews_df["rating"].apply(sentiment_from_rating)
    return sentiments.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)

# -----------------------------
# RAG & Summary
# -----------------------------
@st.cache_data(show_spinner=False)
def get_product_reviews_and_embeddings(asin):
    subset = reviews_df[reviews_df["parent_asin"] == asin].copy()
    if subset.empty:
        return None
    if len(subset) > MAX_REVIEWS_PER_PRODUCT:
        subset = subset.iloc[:MAX_REVIEWS_PER_PRODUCT]
    texts = subset["text"].fillna("").tolist()
    if not texts:
        return None
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    subset = subset.reset_index(drop=True)
    return {"df": subset, "texts": texts, "embeddings": embeddings}

def get_rag_reviews_for_summary(product_data, product_title, top_k=MAX_REVIEWS_FOR_SUMMARY):
    query_emb = embedder.encode([product_title], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, product_data["embeddings"])[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [product_data["texts"][i] for i in top_idx]

def summarize_reviews(review_texts):
    joined = " ".join(review_texts)
    if len(joined) > MAX_SUMMARY_INPUT_CHARS:
        joined = joined[:MAX_SUMMARY_INPUT_CHARS].rsplit(" ", 1)[0] + "."
    inputs = tokenizer(joined, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=SUMMARY_MAX_LENGTH,
        min_length=SUMMARY_MIN_LENGTH,
        do_sample=False,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -----------------------------
# Popup dialog: Sentiment + Summary
# -----------------------------
@st.dialog("Review summary & sentiment", width="large")
def show_summary_dialog(asin, product_title):
    with st.spinner("Loading reviews…"):
        product_data = get_product_reviews_and_embeddings(asin)
    if product_data is None:
        st.warning("No reviews found for this product.")
        return
    product_reviews_df = product_data["df"]
    n_reviews = len(product_reviews_df)
    counts = compute_sentiment_counts(product_reviews_df)
    n_pos = int(counts.get("positive", 0))
    n_neu = int(counts.get("neutral", 0))
    n_neg = int(counts.get("negative", 0))

    st.markdown(f"**{product_title[:80]}{'…' if len(product_title) > 80 else ''}**")
    st.caption(f"Based on {n_reviews} review(s)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("😊 Positive", f"{n_pos} ({100*n_pos/max(n_reviews,1):.0f}%)")
    with c2:
        st.metric("😐 Neutral", f"{n_neu} ({100*n_neu/max(n_reviews,1):.0f}%)")
    with c3:
        st.metric("😞 Negative", f"{n_neg} ({100*n_neg/max(n_reviews,1):.0f}%)")

    with st.spinner("Building summary…"):
        rag_texts = get_rag_reviews_for_summary(product_data, product_title)
        summary = summarize_reviews(rag_texts)

    st.subheader("Overall summary")
    st.info(summary)

    st.subheader("Quick takeaway")
    if n_neg > n_pos and n_reviews >= 5:
        st.warning("Reviews lean negative. Consider alternatives.")
    elif n_pos > n_neg * 2 or (n_pos >= n_reviews // 2 and n_neg < n_reviews // 4):
        st.success("Mostly positive. Good candidate if it fits your needs.")
    else:
        st.info("Mixed feedback. Read the summary and decide.")

    # Customer comments (collapsed by default)
    with st.expander("📝 Customer comments", expanded=False):
        rows = list(product_reviews_df.iterrows())
        for idx, (i, row) in enumerate(rows):
            text = str(row.get("text", "")).strip() or "(No text)"
            st.write(text)
            if idx < len(rows) - 1:
                st.divider()

# -----------------------------
# Session state for which product to show in popup
# -----------------------------
if "dialog_asin" not in st.session_state:
    st.session_state["dialog_asin"] = None
    st.session_state["dialog_title"] = None

# -----------------------------
# Header (Amazon-style)
# -----------------------------
st.markdown("""
<div class="amazon-header">
  <h1>📦 Review Summarizer</h1>
  <p>Select a product and open its review summary in one click.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Search filter
# -----------------------------
search = st.text_input("Search products by name", placeholder="Type to filter products…", key="search")
meta_filtered = meta_df.dropna(subset=["title"])
if search and search.strip():
    q = search.strip().lower()
    meta_filtered = meta_filtered[meta_filtered["title"].str.lower().str.contains(q, na=False)]
meta_filtered = meta_filtered.head(MAX_PRODUCTS_SHOWN)

# -----------------------------
# Product grid
# -----------------------------
if meta_filtered.empty:
    st.info("No products match your search. Try a different term.")
else:
    rows = [meta_filtered.iloc[i : i + PRODUCTS_PER_ROW] for i in range(0, len(meta_filtered), PRODUCTS_PER_ROW)]
    for row_df in rows:
        cols = st.columns(PRODUCTS_PER_ROW)
        for idx, (_, row) in enumerate(row_df.iterrows()):
            with cols[idx]:
                title = str(row.get("title", ""))
                asin = row.get("parent_asin", "")
                avg = row.get("average_rating", 0)
                try:
                    avg = float(avg) if pd.notna(avg) else 0
                except (ValueError, TypeError):
                    avg = 0
                num_ratings = row.get("rating_number", 0)
                try:
                    num_ratings = int(float(num_ratings)) if pd.notna(num_ratings) else 0
                except (ValueError, TypeError):
                    num_ratings = 0
                price = row.get("price", "")
                if pd.notna(price) and str(price).strip():
                    try:
                        p = float(price)
                        price_str = f"${p:.2f}" if p > 0 else ""
                    except (ValueError, TypeError):
                        price_str = str(price)[:20]
                else:
                    price_str = ""
                img_url = get_first_image_url(row.get("images"))

                if img_url:
                    img_html = '<img src="' + img_url + '" alt="Product"/>'
                else:
                    img_html = '<div style="height:120px;background:#f0f0f0;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#999;">No image</div>'
                price_html = ('<div class="price">' + price_str + "</div>") if price_str else ""
                title_display = title[:100] + ("…" if len(title) > 100 else "")
                title_attr = title[:200].replace('"', "&quot;")
                stars_html = "★" * int(round(avg)) + "☆" * (5 - int(round(avg)))

                card_html = (
                    '<div class="product-card">'
                    + img_html
                    + f'<div class="title" title="{title_attr}">{title_display}</div>'
                    + f'<div class="rating-row"><span class="stars">{stars_html}</span> <span class="rating-count">({num_ratings})</span></div>'
                    + price_html
                    + "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("See review summary", key=f"btn_{asin}_{hash(title) % 10**8}", type="primary"):
                    st.session_state["dialog_asin"] = asin
                    st.session_state["dialog_title"] = title
                    st.rerun()

# -----------------------------
# Open popup if a product was selected
# -----------------------------
if st.session_state.get("dialog_asin") and st.session_state.get("dialog_title"):
    show_summary_dialog(st.session_state["dialog_asin"], st.session_state["dialog_title"])
    # Clear after opening so next run doesn’t auto-open (user can click another product)
    st.session_state["dialog_asin"] = None
    st.session_state["dialog_title"] = None
