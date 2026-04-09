"""Offline-style metrics (e.g. ROUGE vs retrieved text)."""


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
