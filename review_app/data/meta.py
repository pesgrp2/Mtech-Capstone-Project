"""Product metadata row for dialogs (Appliances meta CSV)."""

import pandas as pd

from review_app.data.images import extract_all_product_image_urls
from review_app.data.loading import load_meta_dataframe


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
