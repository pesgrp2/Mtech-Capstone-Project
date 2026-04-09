"""Cached loads for parquet and meta CSV."""

import os

import pandas as pd
import streamlit as st

from review_app.config import META_CSV_PATH, PARQUET_PATH


@st.cache_data(show_spinner=False)
def load_meta_dataframe():
    if not os.path.isfile(META_CSV_PATH):
        return None
    try:
        return pd.read_csv(META_CSV_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_reviews_parquet_df():
    if not os.path.isfile(PARQUET_PATH):
        return None
    try:
        return pd.read_parquet(PARQUET_PATH)
    except Exception:
        return None
