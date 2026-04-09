"""Review text aggregation and extractive snippets for cards."""

import re

import pandas as pd

from review_app.config import MAX_REVIEW_CHARS_FOR_SUMMARY
from review_app.data.loading import load_reviews_parquet_df


def collect_review_text_for_asin(asin: str) -> str:
    """Join review / embedding text for one ASIN (and matching parent_asin if present)."""
    df = load_reviews_parquet_df()
    if df is None or "asin" not in df.columns:
        return ""
    a = str(asin).strip()
    mask = df["asin"].astype(str) == a
    if "parent_asin" in df.columns:
        mask = mask | (df["parent_asin"].astype(str) == a)
    sub = df.loc[mask]
    if sub.empty:
        return ""
    text_col = None
    for c in ("embedding_text", "text", "review_text", "body"):
        if c in sub.columns:
            text_col = c
            break
    if not text_col:
        return ""
    parts = sub[text_col].dropna().astype(str).str.strip()
    parts = parts[parts != ""]
    blob = " ".join(parts.tolist())
    return blob[:MAX_REVIEW_CHARS_FOR_SUMMARY]


def extractive_card_summary(blob: str, max_len: int = 300) -> str:
    """Short shopper-facing preview from raw reviews (no LLM)."""
    if not blob or not str(blob).strip():
        return "No review text in dataset for this product."
    text = re.sub(r"\s+", " ", str(blob).strip())
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_dot = cut.rfind(".")
    if last_dot > max_len // 2:
        cut = cut[: last_dot + 1]
    else:
        sp = cut.rfind(" ")
        cut = (cut[:sp] if sp > 40 else cut) + "…"
    return cut
