import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Cache Models
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return embedder, summarizer

embedder, summarizer = load_models()

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    products_df = pd.read_excel("src/chinmay/Sample Meta.xlsx")
    reviews_df = pd.read_excel("src/chinmay/Sample Reviews.xlsx")
    return products_df, reviews_df

products_df, reviews_df = load_data()