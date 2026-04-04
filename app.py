import os
import re
from collections import defaultdict
import html as html_module
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import streamlit as st
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


def compute_rouge_against_retrieved(answer: str, source_docs):
    """
    ROUGE F1 comparing the generated answer to concatenated retrieved chunks.
    This is a retrieval-as-reference proxy (not ROUGE vs a human-written gold summary).
    """
    if not answer or not str(answer).strip() or not source_docs:
        return None, None, None
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None, None, None
    ref_parts = []
    for d in source_docs:
        t = getattr(d, "page_content", None) or ""
        if t.strip():
            ref_parts.append(t.strip())
    reference = " ".join(ref_parts)[:12000]
    if not reference.strip():
        return None, None, None
    pred = str(answer).strip()[:8000]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, pred)
    return (
        round(s["rouge1"].fmeasure, 3),
        round(s["rouge2"].fmeasure, 3),
        round(s["rougeL"].fmeasure, 3),
    )


PARQUET_PATH = os.environ.get("REVIEWS_PARQUET", "embedding_ready_reviews_small.parquet")
META_CSV_PATH = os.environ.get("META_CSV", "src/chinmay/Appliances_meta.csv")
RECOMMENDATION_TOP_K = 48  # chunks to score for product ranking (more coverage for "More")
RECOMMENDATION_CARD_COUNT = 5  # product cards in the main row
RECOMMENDATION_TOTAL_RANKED = 20  # total ranked products (5 cards + up to 15 in "More")
MAX_REVIEW_CHARS_FOR_SUMMARY = 8000  # cap parquet text sent to the LLM per product
# Image box inside each card (CSS px) — larger = sharper on retina / wide layout
REC_CARD_IMAGE_HEIGHT = 240


def extract_first_product_image_url(images_str):
    """Parse Amazon-style `images` cell; prefer hi_res, then large."""
    if images_str is None or (isinstance(images_str, float) and pd.isna(images_str)):
        return None
    s = str(images_str).strip()
    if not s:
        return None
    m = re.search(r"'hi_res':\s*'([^']+)'", s) or re.search(r'"hi_res":\s*"([^"]+)"', s)
    if m and m.group(1) and str(m.group(1)).lower() != "none":
        return m.group(1)
    m = re.search(r"'large':\s*'([^']+)'", s) or re.search(r'"large":\s*"([^"]+)"', s)
    if m:
        return m.group(1)
    m = re.search(r"https://m\.media-amazon\.com/images/I/[^\s'\"]+", s, re.I)
    return m.group(0).rstrip(".,);") if m else None


def sanitize_image_url(url):
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    if u.startswith("https://") or u.startswith("http://"):
        return u
    return None


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


def rank_products_by_query(vectorstore, query: str, k: int = RECOMMENDATION_TOP_K, max_products: int = RECOMMENDATION_TOTAL_RANKED):
    """
    Rank products by aggregated similarity of retrieved chunks (FAISS L2 distance).
    Returns list of dicts: asin, weight, pct_top_cards (share among first RECOMMENDATION_CARD_COUNT),
    pct_in_top_pool (share among all max_products ranked).
    """
    if not query or not str(query).strip():
        return []
    try:
        pairs = vectorstore.similarity_search_with_score(query.strip(), k=k)
    except Exception:
        return []
    asin_weights = defaultdict(float)
    for doc, dist in pairs:
        asin = (doc.metadata or {}).get("asin")
        if asin is None or str(asin).strip() == "":
            asin = "unknown"
        asin = str(asin)
        try:
            d = float(dist)
        except (TypeError, ValueError):
            d = 0.0
        w = 1.0 / (1.0 + d)
        asin_weights[asin] += w
    if not asin_weights:
        return []
    ranked = sorted(asin_weights.items(), key=lambda x: -x[1])[:max_products]
    total_pool = sum(w for _, w in ranked)
    n_top = min(RECOMMENDATION_CARD_COUNT, len(ranked))
    total_top = sum(w for _, w in ranked[:n_top]) if n_top else 0.0
    out = []
    for i, (asin, w) in enumerate(ranked):
        pct_pool = (100.0 * w / total_pool) if total_pool > 0 else 0.0
        if i < n_top and total_top > 0:
            pct_cards = round(100.0 * w / total_top, 1)
        else:
            pct_cards = round(pct_pool, 1)
        out.append(
            {
                "asin": asin,
                "weight": w,
                "recommendation_pct": pct_cards if i < n_top else round(pct_pool, 1),
                "pct_in_top_pool": round(pct_pool, 1),
                "in_main_cards": i < n_top,
            }
        )
    return out


@st.cache_data(show_spinner=False)
def load_reviews_parquet_df():
    if not os.path.isfile(PARQUET_PATH):
        return None
    try:
        return pd.read_parquet(PARQUET_PATH)
    except Exception:
        return None


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


@st.cache_data(show_spinner=False)
def load_meta_dataframe():
    if not os.path.isfile(META_CSV_PATH):
        return None
    try:
        return pd.read_csv(META_CSV_PATH)
    except Exception:
        return None


def extract_all_product_image_urls(images_str):
    """All hi_res (then large) image URLs from meta `images` cell, deduped."""
    if images_str is None or (isinstance(images_str, float) and pd.isna(images_str)):
        return []
    s = str(images_str)
    urls = []
    for m in re.finditer(r"'hi_res':\s*'([^']+)'", s):
        u = m.group(1).strip()
        if u and u.lower() != "none":
            urls.append(u)
    if not urls:
        for m in re.finditer(r'"hi_res":\s*"([^"]+)"', s):
            u = m.group(1).strip()
            if u and u.lower() != "none":
                urls.append(u)
    if not urls:
        for m in re.finditer(r"'large':\s*'([^']+)'", s):
            urls.append(m.group(1).strip())
    seen = set()
    out = []
    for u in urls:
        su = sanitize_image_url(u) or u
        if su.startswith("http") and su not in seen:
            seen.add(su)
            out.append(su)
    return out


def get_meta_details_for_asin(asin: str) -> dict:
    """Row from Appliances meta matched by parent_asin (or asin)."""
    dfm = load_meta_dataframe()
    if dfm is None or "parent_asin" not in dfm.columns:
        return {}
    a = str(asin).strip()
    row = dfm[dfm["parent_asin"].astype(str) == a]
    if row.empty and "asin" in dfm.columns:
        row = dfm[dfm["asin"].astype(str) == a]
    if row.empty:
        return {}
    r = row.iloc[0]
    imgs = extract_all_product_image_urls(r.get("images"))
    price = r.get("price", "")
    try:
        price_str = f"${float(price):.2f}" if pd.notna(price) and str(price).strip() else "—"
    except (ValueError, TypeError):
        price_str = str(price) if pd.notna(price) else "—"
    return {
        "title": str(r["title"]) if pd.notna(r.get("title")) else "",
        "price": price_str,
        "average_rating": r.get("average_rating", "—"),
        "rating_number": r.get("rating_number", "—"),
        "main_category": r.get("main_category", "—"),
        "store": str(r.get("store", "—")) if pd.notna(r.get("store")) else "—",
        "image_urls": imgs,
    }


def summarize_product_reviews_llm(llm, product_title: str, asin: str, review_blob: str) -> str:
    if not review_blob.strip():
        return "No review text found for this product in the dataset."
    prompt = (
        f"Product: {product_title}\nASIN: {asin}\n\n"
        "Below are customer review excerpts. Write a clear, concise summary for a shopper.\n"
        "Use short sections: Key positives, Common complaints, Overall verdict (2–3 sentences).\n"
        "Base everything only on the excerpts; do not invent facts.\n\n"
        f"Review excerpts:\n{review_blob}\n\nSummary:"
    )
    try:
        out = llm.invoke(prompt)
        if isinstance(out, str):
            return out
        if hasattr(out, "content"):
            return str(out.content)
        return str(out)
    except Exception as e:
        return f"Could not generate summary: {e}"


def _catalog_entry(catalog: dict, asin: str):
    raw = catalog.get(asin)
    if isinstance(raw, dict):
        return raw.get("title") or asin, raw.get("image_url")
    if raw is not None:
        return str(raw), None
    return asin, None


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
        title, img_url = _catalog_entry(catalog, asin)
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
                f'<span style="color:#999;font-size:0.8rem;text-align:center;padding:0 8px;">'
                f'No image<br/><span style="font-size:0.65rem;">Add images to parquet or set META_CSV</span></span>'
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
            title, _ = _catalog_entry(catalog, asin)
            pct = rec.get("pct_in_top_pool", rec.get("recommendation_pct", 0))
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{pct}%** · {title[:120]}{'…' if len(str(title)) > 120 else ''}  \n`{asin}`")
            with c2:
                if st.button("Summary", key=f"sum_more_{asin}", use_container_width=True):
                    st.session_state["summary_dialog_asin"] = asin
                    st.session_state["summary_dialog_title"] = str(title)
                    st.rerun()


def recommendation_summary_button_column(asin: str, title: str, key_suffix: str):
    if st.button("Review summary", key=f"sum_card_{key_suffix}", use_container_width=True):
        st.session_state["summary_dialog_asin"] = asin
        st.session_state["summary_dialog_title"] = str(title)
        st.rerun()


st.set_page_config(
    page_title="AI Product Review Analyzer",
    layout="wide",
)

# Build FAISS only when explicitly requested (not on every Streamlit rerun).
# Run once: BUILD_FAISS_INDEX=1 streamlit run app.py
# Or use a separate build_index.py script.
if os.environ.get("BUILD_FAISS_INDEX", "").strip() in ("1", "true", "yes"):
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet("embedding_ready_reviews_small.parquet")
    _ = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # warm cache if needed
    _embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    documents = [
        Document(page_content=row["embedding_text"], metadata={"asin": row["asin"]})
        for _, row in df.iterrows()
    ]
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, _embed)
    vectorstore.save_local("faiss_index")
    st.error("FAISS index built. Stop the app, unset BUILD_FAISS_INDEX, and run again normally.")
    st.stop()


st.title("🛍️ AI Product Review Analyzer (RAG System)")

# Light, warm theme + primary actions in orange (not “danger” red)
st.markdown(
    """
<style>
  /* App shell: soft light grey–blue (not black / harsh dark) */
  .stApp {
    background: linear-gradient(165deg, #f4f6fb 0%, #e8ecf6 45%, #f0f2f8 100%);
    color-scheme: light;
  }
  [data-testid="stAppViewContainer"] > .main .block-container {
    background: transparent;
    padding-top: 1.25rem;
  }
  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f4fa 100%) !important;
    border-right: 1px solid #d8dee9 !important;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
    color: #1e293b !important;
  }
  /* Headings & body text on main */
  .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
    color: #0f172a !important;
  }
  .main .stMarkdown p, .main label, .stTextInput label {
    color: #334155 !important;
  }
  /* Analyze Reviews = warm CTA orange (shopping / action, not error red) */
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
  /* Default (secondary) buttons */
  button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
  }
  button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
  }
  /* Search field */
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
  /* Metrics & expanders */
  [data-testid="stMetricValue"] {
    color: #0f172a !important;
  }
  [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
  }
  /* Dialog */
  div[data-testid="stModal"] {
    background: #fafbfe;
  }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar - Model Details
# --------------------------------------------------

st.sidebar.header("⚙️ Model Information")

st.sidebar.markdown("""
**Architecture:** Retrieval Augmented Generation (RAG)

**Embedding Model:**  
sentence-transformers/all-MiniLM-L6-v2

**LLM Model:**  
Llama3 (via Ollama)

**Vector Database:**  
FAISS

**Framework:**  
LangChain
""")

# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# --------------------------------------------------
# Prompt Template
# --------------------------------------------------

template = """
You are an AI assistant that analyzes product reviews.

Provide output strictly in the following format.

Summary:
Give a short summary of key product feedback.

Top Recommendations:
List the best products mentioned.

Review Sentiment:
Provide estimated sentiment percentage and common complaints.

Context:
{context}

Question:
{question}
"""


prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return Ollama(model="llama3")


@st.dialog("Product review summary", width="large")
def product_review_summary_dialog(asin: str, product_title: str):
    meta = get_meta_details_for_asin(asin)
    display_title = (meta.get("title") or product_title or asin).strip()
    catalog = load_asin_catalog()
    _, fallback_img = _catalog_entry(catalog, asin)
    image_urls = list(meta.get("image_urls") or [])
    if not image_urls and fallback_img:
        image_urls = [fallback_img]

    # Header + title (full width), then image left / ratings right, details under image
    st.markdown("##### Product")
    st.markdown(f"**{display_title[:200]}{'…' if len(display_title) > 200 else ''}**")
    st.caption(f"`{asin}`")

    try:
        col_img, col_right = st.columns([1, 1.05], gap="medium", vertical_alignment="top")
    except TypeError:
        col_img, col_right = st.columns([1, 1.05], gap="medium")

    with col_img:
        st.markdown("###### Product image")
        if image_urls:
            slug = re.sub(r"[^a-zA-Z0-9_]", "_", asin)[:48]
            n = len(image_urls)
            if n == 1:
                st.image(image_urls[0], use_container_width=True)
            else:
                ix = st.slider(
                    "Browse photos",
                    min_value=1,
                    max_value=n,
                    value=1,
                    key=f"dlg_gallery_{slug}",
                    help="Slide to view each product image",
                )
                st.image(image_urls[ix - 1], use_container_width=True)
                st.caption(f"Image {ix} of {n}")
        else:
            st.info("No product images found in meta or catalog.")

        st.markdown("###### Details")
        st.markdown(
            f"<div style='color:#444;font-size:0.9rem;line-height:1.65;'>"
            f"<b>Price</b> · {html_module.escape(str(meta.get('price', '—')))}<br/>"
            f"<b>Category</b> · {html_module.escape(str(meta.get('main_category', '—')))}<br/>"
            f"<b>Store</b> · {html_module.escape(str(meta.get('store', '—')))}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("###### Ratings")
        cma, cmb = st.columns(2)
        cma.metric("Avg rating", str(meta.get("average_rating", "—")))
        cmb.metric("# Ratings", str(meta.get("rating_number", "—")))

    st.divider()
    st.markdown("##### Review summary")
    blob = collect_review_text_for_asin(asin)
    if not blob.strip():
        st.warning("No review text found for this ASIN in the parquet file.")
        return
    with st.spinner("Summarizing reviews…"):
        try:
            llm = load_llm()
            summary = summarize_product_reviews_llm(llm, display_title, asin, blob)
        except Exception as e:
            summary = f"Error: {e}"
    st.markdown(summary)
    with st.expander("Sample of review text used"):
        st.text(blob[:4000] + ("…" if len(blob) > 4000 else ""))


# --------------------------------------------------
# Build RAG Chain
# --------------------------------------------------

@st.cache_resource
def build_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


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
        "- Check that `embedding_ready_reviews_small.parquet` is in the project folder"
    )
    st.stop()


# --------------------------------------------------
# User Input (form: Enter submits; button on the right)
# --------------------------------------------------

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


# --------------------------------------------------
# Run RAG (persist results so product buttons / dialog work on rerun)
# --------------------------------------------------

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
        # Do not set st.session_state["analyze_query_input"] here — the widget owns that key
        # after instantiation; the form already keeps the submitted text.
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

    # --------------------------------------------------
    # AI Summary Output
    # --------------------------------------------------

    st.subheader("📊 AI Generated Insights")
    if result and str(result).strip():
        st.write(result)
    else:
        st.warning(
            "The model returned an empty answer. "
            "Check Ollama logs, try a shorter question, or confirm `ollama run llama3` works in a terminal."
        )
    st.markdown("---")

    # --------------------------------------------------
    # Recommended products (retrieval-based cards + More)
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Sentiment Chart
    # --------------------------------------------------

    st.subheader("💬 Sentiment Distribution (Example)")
    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [74, 15, 11]
    })
    fig, ax = plt.subplots()
    ax.bar(
        sentiment_data["Sentiment"],
        sentiment_data["Count"]
    )
    ax.set_ylabel("Percentage")
    st.pyplot(fig)
    st.markdown("---")

    # --------------------------------------------------
    # Retrieved Documents
    # --------------------------------------------------

    st.subheader("📄 Retrieved Review Evidence")

    for i, doc in enumerate(docs):
        with st.expander(f"Document {i+1}"):
            st.write(doc.page_content)


# Open product summary popup (after Analyze flow / any rerun)
if st.session_state.get("summary_dialog_asin"):
    product_review_summary_dialog(
        st.session_state["summary_dialog_asin"],
        st.session_state.get("summary_dialog_title") or st.session_state["summary_dialog_asin"],
    )
    st.session_state.pop("summary_dialog_asin", None)
    st.session_state.pop("summary_dialog_title", None)