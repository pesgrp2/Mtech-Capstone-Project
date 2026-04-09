"""ASIN catalog from parquet + optional meta CSV."""

import os

import pandas as pd
import streamlit as st

from review_app.config import META_CSV_PATH, PARQUET_PATH
from review_app.data.images import extract_first_product_image_url, sanitize_image_url


@st.cache_data(show_spinner=False)
def load_meta_product_extras():
    """parent_asin → {meta_title, image_url} from optional meta CSV."""
    if not os.path.isfile(META_CSV_PATH):
        return {}
    try:
        meta = pd.read_csv(META_CSV_PATH)
    except Exception:
        return {}
    if "parent_asin" not in meta.columns:
        return {}
    out = {}
    for _, row in meta.iterrows():
        pid = str(row["parent_asin"]).strip()
        if not pid:
            continue
        img = extract_first_product_image_url(row.get("images"))
        t = row.get("title")
        title = str(t).strip() if pd.notna(t) else ""
        out[pid] = {"meta_title": title, "image_url": img}
    return out


@st.cache_data(show_spinner=False)
def load_asin_catalog():
    """Map ASIN → {title, image_url} from parquet + optional meta CSV."""
    meta_extras = load_meta_product_extras()
    if not os.path.isfile(PARQUET_PATH):
        return {}
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception:
        return {}
    if "asin" not in df.columns:
        return {}
    catalog = {}
    title_col = None
    for c in ("title", "product_title", "product_name"):
        if c in df.columns:
            title_col = c
            break
    text_col = "embedding_text" if "embedding_text" in df.columns else None
    image_cols = [
        c
        for c in df.columns
        if c.lower()
        in (
            "image_url",
            "image",
            "main_image",
            "thumbnail",
            "img_url",
            "product_image",
        )
    ]
    for asin, g in df.groupby("asin"):
        a = str(asin)
        parent = None
        if "parent_asin" in df.columns and len(g):
            try:
                parent = str(g["parent_asin"].iloc[0]).strip()
            except Exception:
                parent = None

        if title_col:
            t = g[title_col].dropna().astype(str)
            title = t.iloc[0] if len(t) else a
        elif text_col:
            snippet = str(g[text_col].iloc[0]) if len(g) else a
            title = (snippet[:100] + "…") if len(snippet) > 100 else snippet
        else:
            title = a

        img_url = None
        for c in image_cols:
            try:
                v = g[c].dropna().iloc[0] if len(g[c].dropna()) else None
            except Exception:
                v = None
            if v is not None and str(v).strip():
                u = sanitize_image_url(str(v).strip()) or extract_first_product_image_url(v)
                if u:
                    img_url = sanitize_image_url(u) or u
                    break

        ex = meta_extras.get(a) or (meta_extras.get(parent) if parent else None)
        if ex:
            if ex.get("meta_title"):
                title = ex["meta_title"]
            if not img_url and ex.get("image_url"):
                img_url = sanitize_image_url(ex["image_url"]) or ex["image_url"]

        catalog[a] = {"title": title, "image_url": sanitize_image_url(img_url) if img_url else None}
    return catalog


def catalog_entry(catalog: dict, asin: str):
    raw = catalog.get(asin)
    if isinstance(raw, dict):
        return raw.get("title") or asin, raw.get("image_url")
    if raw is not None:
        return str(raw), None
    return asin, None
