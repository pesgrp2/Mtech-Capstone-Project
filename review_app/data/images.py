"""Parse Amazon-style image fields from meta / CSV cells."""

import re

import pandas as pd


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
