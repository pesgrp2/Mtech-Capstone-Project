"""Data access: catalog, meta, reviews, ranking."""

from review_app.data.catalog import catalog_entry, load_asin_catalog, load_meta_product_extras
from review_app.data.meta import get_meta_details_for_asin
from review_app.data.ranking import rank_products_by_query
from review_app.data.reviews import collect_review_text_for_asin, extractive_card_summary

__all__ = [
    "catalog_entry",
    "load_asin_catalog",
    "load_meta_product_extras",
    "get_meta_details_for_asin",
    "rank_products_by_query",
    "collect_review_text_for_asin",
    "extractive_card_summary",
]
