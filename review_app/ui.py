"""Streamlit layout: theme CSS, recommendation cards, product summary dialog."""

import html as html_module
import re

import streamlit as st

from review_app.config import REC_CARD_IMAGE_HEIGHT
from review_app.data import (
    catalog_entry,
    collect_review_text_for_asin,
    extractive_card_summary,
    get_meta_details_for_asin,
    load_asin_catalog,
)
from review_app.rag import load_llm, summarize_product_reviews_llm

APP_CUSTOM_CSS = """
<style>
  .stApp {
    background: linear-gradient(165deg, #f4f6fb 0%, #e8ecf6 45%, #f0f2f8 100%);
    color-scheme: light;
  }
  [data-testid="stAppViewContainer"] > .main .block-container {
    background: transparent;
    padding-top: 1.25rem;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f4fa 100%) !important;
    border-right: 1px solid #d8dee9 !important;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
    color: #1e293b !important;
  }
  .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
    color: #0f172a !important;
  }
  .main .stMarkdown p, .main label, .stTextInput label {
    color: #334155 !important;
  }
  button[kind="primary"],
  .stFormSubmitButton button[kind="primary"] {
    background: linear-gradient(180deg, #ffe0b2 0%, #ffb74d 35%, #ff9800 100%) !important;
    border: 1px solid #e65100 !important;
    color: #1a1206 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  }
  button[kind="primary"]:hover,
  .stFormSubmitButton button[kind="primary"]:hover {
    background: linear-gradient(180deg, #ffcc80 0%, #ffa726 40%, #fb8c00 100%) !important;
    border-color: #bf360c !important;
    color: #000 !important;
  }
  button[kind="primary"]:focus-visible {
    box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.45) !important;
  }
  button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
  }
  button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
  }
  .stTextInput input {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
  }
  .stTextInput input:focus {
    border-color: #ff9800 !important;
    box-shadow: 0 0 0 1px rgba(255, 152, 0, 0.35);
  }
  [data-testid="stMetricValue"] {
    color: #0f172a !important;
  }
  [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
  }
  div[data-testid="stModal"] {
    background: #fafbfe;
  }
</style>
"""


DIALOG_WIDGET_KEY_PREFIX = "dlg_"
DIALOG_CONTEXT_ASIN_KEY = "_summary_dialog_product_asin"


def _clear_dialog_widget_session_state() -> None:
    """Remove Streamlit widget state for the product summary dialog (fresh UI next open)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(DIALOG_WIDGET_KEY_PREFIX):
            st.session_state.pop(k, None)


def _sync_dialog_product_context(asin: str) -> None:
    """
    When the user opens a different product, drop old widget values so sliders / expanders
    do not reuse the previous ASIN's state.
    """
    ctx = st.session_state.get(DIALOG_CONTEXT_ASIN_KEY)
    if ctx == asin:
        return
    _clear_dialog_widget_session_state()
    st.session_state[DIALOG_CONTEXT_ASIN_KEY] = asin


def reset_summary_dialog_session_state() -> None:
    """Call after the dialog closes so the next open always syncs cleanly."""
    st.session_state.pop(DIALOG_CONTEXT_ASIN_KEY, None)
    _clear_dialog_widget_session_state()


def inject_custom_css():
    st.markdown(APP_CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_model_info():
    st.sidebar.header("⚙️ Model Information")
    st.sidebar.markdown(
        """
**Architecture:** Retrieval Augmented Generation (RAG)

**Embedding Model:**  
sentence-transformers/all-MiniLM-L6-v2

**LLM Model:**  
Llama3 (via Ollama)

**Vector Database:**  
FAISS

**Framework:**  
LangChain
"""
    )


def recommendation_summary_button_column(asin: str, title: str, key_suffix: str):
    if st.button("Review summary", key=f"sum_card_{key_suffix}", use_container_width=True):
        st.session_state["summary_dialog_asin"] = asin
        st.session_state["summary_dialog_title"] = str(title)
        st.rerun()


def render_recommendation_cards(recommendations, catalog: dict):
    """Product cards with image, in-card review preview, match %, and full-summary button."""
    if not recommendations:
        st.info("No product matches to rank for this query.")
        return
    n = len(recommendations)
    cols = st.columns(n)
    h = REC_CARD_IMAGE_HEIGHT
    for i, rec in enumerate(recommendations):
        asin = rec["asin"]
        pct = rec["recommendation_pct"]
        title, img_url = catalog_entry(catalog, asin)
        safe_title = html_module.escape(str(title)[:160])
        safe_asin = html_module.escape(str(asin))
        safe_img_attr = html_module.escape(img_url) if img_url else ""
        review_blob = collect_review_text_for_asin(asin)
        preview = extractive_card_summary(review_blob)
        safe_preview = html_module.escape(preview)

        if img_url:
            img_block = (
                f'<img src="{safe_img_attr}" alt="" loading="lazy" '
                f'style="max-width:100%;max-height:{h}px;width:auto;height:auto;object-fit:contain;vertical-align:middle;" />'
            )
        else:
            img_block = (
                '<span style="color:#999;font-size:0.8rem;text-align:center;padding:0 8px;">'
                'No image<br/><span style="font-size:0.65rem;">Add images to parquet or set META_CSV</span></span>'
            )

        with cols[i]:
            st.markdown(
                f"""
<div style="
  border:1px solid #e47911;
  border-radius:12px;
  padding:0;
  background:linear-gradient(180deg,#fffaf5 0%,#ffffff 55%);
  min-height:640px;
  display:flex;
  flex-direction:column;
  box-shadow:0 4px 14px rgba(0,0,0,0.08);
  overflow:hidden;
">
  <div style="
    flex:0 0 auto;
    width:100%;
    height:{h}px;
    min-height:{h}px;
    background:#f4f4f4;
    display:flex;
    align-items:center;
    justify-content:center;
    border-bottom:1px solid #eee;
  ">
    {img_block}
  </div>
  <div style="padding:14px 12px 10px;flex:1;display:flex;flex-direction:column;">
    <div style="font-size:1.5rem;font-weight:800;color:#b12704;line-height:1.1;">{pct}%</div>
    <div style="font-size:0.72rem;color:#666;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px;">match share</div>
    <div style="font-size:0.88rem;font-weight:600;color:#007185;line-height:1.35;">{safe_title}</div>
    <div style="font-size:0.72rem;color:#888;margin-top:8px;margin-bottom:8px;">ASIN: {safe_asin}</div>
    <div style="
      margin-top:auto;
      padding:10px;
      background:linear-gradient(180deg,#fff8f0 0%,#ffeedd 100%);
      border:1px solid #ebae6f;
      border-radius:8px;
      border-left:4px solid #e47911;
      font-size:0.78rem;
      line-height:1.45;
      color:#3d2914;
    ">
      <div style="font-size:0.68rem;font-weight:700;color:#b45f06;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Review preview</div>
      <div style="color:#5c3d1e;">{safe_preview}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", f"{i}_{asin}")[:80]
            recommendation_summary_button_column(asin, title, safe_key)
    st.caption(
        "Cards show a **review preview** from your dataset (extractive snippet). "
        "**Full summary** opens the popup with product details and image gallery. "
        "Images from **META_CSV** when `parent_asin` matches."
    )


def render_more_recommendations(more_recs: list, catalog: dict):
    """Additional ranked products in an expander with summary buttons."""
    if not more_recs:
        return
    with st.expander(f"More relevant products ({len(more_recs)} more)", expanded=False):
        st.caption("Match % is share among the full ranked list (top picks for this question).")
        for rec in more_recs:
            asin = rec["asin"]
            title, _ = catalog_entry(catalog, asin)
            pct = rec.get("pct_in_top_pool", rec.get("recommendation_pct", 0))
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"**{pct}%** · {title[:120]}{'…' if len(str(title)) > 120 else ''}  \n`{asin}`"
                )
            with c2:
                if st.button("Summary", key=f"sum_more_{asin}", use_container_width=True):
                    st.session_state["summary_dialog_asin"] = asin
                    st.session_state["summary_dialog_title"] = str(title)
                    st.rerun()


@st.dialog("Product review summary", width="large")
def product_review_summary_dialog(asin: str, product_title: str):
    _sync_dialog_product_context(asin)

    meta = get_meta_details_for_asin(asin)
    display_title = (meta.get("title") or product_title or asin).strip()
    catalog = load_asin_catalog()
    _, fallback_img = catalog_entry(catalog, asin)
    image_urls = list(meta.get("image_urls") or [])
    if not image_urls and fallback_img:
        image_urls = [fallback_img]

    st.markdown("##### Product")
    st.markdown(f"**{display_title[:200]}{'…' if len(display_title) > 200 else ''}**")
    st.caption(f"ASIN `{asin}`")

    # Left: image only. Right: all metadata (no slider when there is exactly one image).
    try:
        col_img, col_detail = st.columns([0.36, 0.64], gap="large", vertical_alignment="top")
    except TypeError:
        col_img, col_detail = st.columns([0.36, 0.64], gap="large")

    # Keys must be unique per product (Streamlit-safe: letters, digits, underscore).
    asin_key = re.sub(r"[^a-zA-Z0-9]", "_", str(asin))
    dlg_key = f"{DIALOG_WIDGET_KEY_PREFIX}{asin_key}_"

    with col_img:
        try:
            img_wrap = st.container(border=True)
        except TypeError:
            img_wrap = st.container()
        with img_wrap:
            if image_urls:
                n = len(image_urls)
                if n == 1:
                    st.image(image_urls[0], use_container_width=True)
                else:
                    ix = st.slider(
                        "Photos",
                        min_value=1,
                        max_value=n,
                        value=1,
                        key=f"{dlg_key}gallery",
                        help="Choose which product photo to show",
                    )
                    st.image(image_urls[ix - 1], use_container_width=True)
                    st.caption(f"Photo {ix} of {n}")
            else:
                st.caption("No image")
                st.info("No product image in meta or catalog.")

    with col_detail:
        st.markdown("###### Details")
        r1, r2 = st.columns(2)
        r1.metric("Avg rating", str(meta.get("average_rating", "—")))
        r2.metric("# Ratings", str(meta.get("rating_number", "—")))
        st.markdown(
            "<div style='margin-top:0.5rem;color:#334155;font-size:0.95rem;line-height:1.75;'>"
            f"<p style='margin:0 0 0.35rem 0;'><b style='color:#0f172a;'>Price</b> · "
            f"{html_module.escape(str(meta.get('price', '—')))}</p>"
            f"<p style='margin:0 0 0.35rem 0;'><b style='color:#0f172a;'>Category</b> · "
            f"{html_module.escape(str(meta.get('main_category', '—')))}</p>"
            f"<p style='margin:0;'><b style='color:#0f172a;'>Store</b> · "
            f"{html_module.escape(str(meta.get('store', '—')))}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("##### Review summary")
    blob = collect_review_text_for_asin(asin)
    if not blob.strip():
        st.warning("No review text found for this ASIN in the parquet file.")
        return

    # Cache per ASIN so changing the photo slider does not re-invoke Ollama.
    # Key uses dlg_ prefix so it is cleared when switching products or closing the dialog.
    summary_key = f"{DIALOG_WIDGET_KEY_PREFIX}{asin_key}_llm_summary"
    summary = st.session_state.get(summary_key)
    if summary is None:
        with st.spinner("Summarizing reviews…"):
            try:
                llm = load_llm()
                summary = summarize_product_reviews_llm(llm, display_title, asin, blob)
            except Exception as e:
                summary = f"Error: {e}"
        st.session_state[summary_key] = summary

    st.markdown(summary)
    try:
        exp = st.expander("Sample of review text used", key=f"{dlg_key}review_sample")
    except TypeError:
        exp = st.expander("Sample of review text used")
    with exp:
        st.text(blob[:4000] + ("…" if len(blob) > 4000 else ""))
