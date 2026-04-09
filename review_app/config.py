"""Paths, model names, and UI / retrieval tuning constants."""

import os

PARQUET_PATH = os.environ.get("REVIEWS_PARQUET", "embedding_ready_reviews_small.parquet")
META_CSV_PATH = os.environ.get("META_CSV", "src/chinmay/Appliances_meta.csv")
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

RECOMMENDATION_TOP_K = 48
RECOMMENDATION_CARD_COUNT = 5
RECOMMENDATION_TOTAL_RANKED = 20
MAX_REVIEW_CHARS_FOR_SUMMARY = 8000
REC_CARD_IMAGE_HEIGHT = 240
