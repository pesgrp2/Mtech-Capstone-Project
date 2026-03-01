import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Amazon Review Summarizer", layout="wide")

st.title("📝 Amazon Product Review Summarizer (RAG-based)")

# -----------------------------
# Load Data (Cached)
# -----------------------------
@st.cache_data
def load_data():
    meta = pd.read_csv("src/chinmay/Appliances_meta.csv")
    reviews = pd.read_csv("src/chinmay/Appliances_reviews.csv")
    return meta, reviews

meta_df, reviews_df = load_data()

# -----------------------------
# Load Embedding Model (Cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedder = load_embedder()

# -----------------------------
# Compute Review Embeddings (Cached)
# -----------------------------
@st.cache_resource
def compute_embeddings(texts):
    return embedder.encode(texts, convert_to_numpy=True)

review_texts = reviews_df["text"].fillna("").tolist()
review_embeddings = compute_embeddings(review_texts)

# -----------------------------
# Load Lightweight Summarizer (Cached)
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # much smaller & safer
        device=-1
    )

summarizer = load_summarizer()

# -----------------------------
# UI Components
# -----------------------------
product_options = meta_df["title"].dropna().unique().tolist()
selected_product = st.selectbox("Select a Product Title:", product_options)

if st.button("Generate Summary"):

    product_row = meta_df[meta_df["title"] == selected_product]

    if not product_row.empty:

        asin = product_row.iloc[0]["parent_asin"]

        # Get product reviews
        product_reviews = reviews_df[
            reviews_df["parent_asin"] == asin
        ]["text"].fillna("").tolist()

        if product_reviews:

            # Encode query
            query_embedding = embedder.encode(
                [selected_product],
                convert_to_numpy=True
            )

            # Cosine similarity instead of FAISS
            similarities = cosine_similarity(
                query_embedding,
                review_embeddings
            )[0]

            top_indices = np.argsort(similarities)[-5:][::-1]
            top_reviews = [review_texts[i] for i in top_indices]

            # Join top reviews
            joined_reviews = " ".join(top_reviews)

            # Limit input length (important for stability)
            joined_reviews = joined_reviews[:1000]

            # Generate summary
            summary = summarizer(
                joined_reviews,
                max_length=120,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]

            st.subheader("📝 Review Summary")
            st.write(summary)

        else:
            st.warning("No reviews found for this product.")

    else:
        st.error("Product not found in metadata.")