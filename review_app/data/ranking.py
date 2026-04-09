"""Rank products from vector retrieval scores."""

from collections import defaultdict

from review_app.config import (
    RECOMMENDATION_CARD_COUNT,
    RECOMMENDATION_TOP_K,
    RECOMMENDATION_TOTAL_RANKED,
)


def rank_products_by_query(
    vectorstore,
    query: str,
    k: int = RECOMMENDATION_TOP_K,
    max_products: int = RECOMMENDATION_TOTAL_RANKED,
):
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
