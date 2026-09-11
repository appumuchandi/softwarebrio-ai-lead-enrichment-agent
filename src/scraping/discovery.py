"""
URL discovery - vertical slice.

Prioritization per requirements:
1. Same-domain links discovered from homepage (actual links) - HIGHEST
2. Sitemap URLs filtered to relevant paths
3. Known paths as fallback/priorities (not blind requests - checked via discovery ordering)

For vertical slice, discovery is lightweight: fetch homepage, parse links, intersect with priorities.
Full sitemap parsing is implemented but optional in smoke mode.

Evidence-aware: caller must preserve returned URLs as source_pages.
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from src.config import get_settings

# Known paths treated as PRIORITIES / FALLBACK, not blind requests (we only include them if they look promising)
PRIORITY_PATHS = [
    "/",  # homepage always
    "/about",
    "/about-us",
    "/team",
    "/leadership",
    "/contact",
    "/contact-us",
    "/pricing",
    "/careers",
    "/company",
]

# Keywords to prioritize when filtering discovered links / sitemap
RELEVANT_KEYWORDS = {"about", "team", "leadership", "contact", "company", "pricing", "careers", "people", "culture"}


def _same_domain(url: str, domain: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        domain = domain.lower().strip().lstrip("www.")
        host = host.lstrip("www.")
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False


STATIC_EXTENSIONS = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".css", ".js", ".woff", ".woff2", ".mp4", ".mp3", ".avi", ".mov", ".xml", ".json", ".txt", ".csv")

def _is_static_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)

def _score_url(url: str) -> int:
    """Higher score = more relevant for enrichment."""
    path = urlparse(url).path.lower()
    if _is_static_asset(url):
        return -10  # deprioritize static assets
    score = 0
    for kw in RELEVANT_KEYWORDS:
        if kw in path:
            score += 10
    if path in ("/", ""):
        score += 5  # homepage is baseline
    if path in ("/about", "/about-us"):
        score += 8
    # Prefer shorter, cleaner URLs
    if len(path) < 20:
        score += 2
    return score


async def _fetch_sitemap_urls(domain: str, timeout: float = 6.0) -> list[str]:
    """Best-effort sitemap fetch, returns [] on any failure (fail-soft)."""
    candidates = [
        f"https://{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://www.{domain}/sitemap.xml",
    ]
    urls: list[str] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        for sitemap_url in candidates:
            try:
                resp = await client.get(sitemap_url)
                if resp.status_code != 200 or not resp.text:
                    continue
                text = resp.text
                # Quick check: is this XML?
                if "<url" not in text and "<sitemap" not in text:
                    continue
                # Parse
                try:
                    root = ET.fromstring(text)
                except ET.ParseError:
                    continue
                ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                # Handle sitemap index
                sitemaps = root.findall("ns:sitemap", ns) or root.findall("sitemap")
                if sitemaps:
                    # Fetch first 1 child sitemap (avoid explosion in vertical slice)
                    loc = sitemaps[0].find("ns:loc", ns)
                    if loc is None:
                        loc = sitemaps[0].find("loc")
                    if loc is not None and loc.text:
                        try:
                            sub = await client.get(loc.text.strip(), timeout=timeout)
                            if sub.status_code == 200:
                                try:
                                    sub_root = ET.fromstring(sub.text)
                                    for u in sub_root.findall("ns:url", ns) or sub_root.findall("url"):
                                        l = u.find("ns:loc", ns)
                                        if l is None:
                                            l = u.find("loc")
                                        if l is not None and l.text and _same_domain(l.text.strip(), domain):
                                            if not _is_static_asset(l.text.strip()):
                                                urls.append(l.text.strip())
                                except ET.ParseError:
                                    pass
                        except Exception:
                            pass
                else:
                    for u in root.findall("ns:url", ns) or root.findall("url"):
                        l = u.find("ns:loc", ns)
                        if l is None:
                            l = u.find("loc")
                        if l is not None and l.text and _same_domain(l.text.strip(), domain):
                            if not _is_static_asset(l.text.strip()):
                                urls.append(l.text.strip())
                if urls:
                    break
            except Exception:
                continue
    # Filter to relevant (exclude static)
    filtered = [u for u in urls if not _is_static_asset(u) and (any(kw in urlparse(u).path.lower() for kw in RELEVANT_KEYWORDS) or _score_url(u) >= 5)]
    # Dedupe preserve order
    seen = set()
    out = []
    for u in filtered:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:8]


async def _discover_links_from_homepage(domain: str, homepage_html: Optional[str], homepage_url: str) -> list[str]:
    """Parse <a href> from homepage HTML, return same-domain links prioritized."""
    if not homepage_html:
        return []
    # Regex href extraction (lightweight, no bs4 dependency for vertical slice)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', homepage_html, flags=re.I)
    links: list[str] = []
    for h in hrefs:
        if h.startswith("#") or h.startswith("mailto:") or h.startswith("tel:") or h.startswith("javascript:"):
            continue
        abs_url = urljoin(homepage_url, h).split("#")[0].rstrip("/")
        if not abs_url.startswith("http"):
            continue
        if not _same_domain(abs_url, domain):
            continue
        if _is_static_asset(abs_url):
            continue
        # Normalize
        parsed = urlparse(abs_url)
        # Drop query for dedup, keep path
        norm = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or f"{parsed.scheme}://{parsed.netloc}/"
        if norm not in links:
            links.append(norm)
    # Prioritize relevant
    scored = sorted(links, key=_score_url, reverse=True)
    # Keep top relevant + homepage always
    relevant = [u for u in scored if _score_url(u) >= 5]
    return relevant[:8]


async def discover_urls(domain: str, homepage_html: Optional[str] = None, homepage_url: Optional[str] = None) -> list[str]:
    """
    Discover URLs for a domain, prioritized.

    Order:
    1. Homepage same-domain links (if homepage_html provided)
    2. Sitemap filtered URLs
    3. Known priority paths as fallback (only if not already discovered)

    Returns 3-6 URLs, always includes https://{domain}/ as first if available.
    Fail-soft: never raises.
    """
    domain = domain.strip().lower().replace("https://", "").replace("http://", "").split("/")[0].lstrip("www.")
    homepage_url = homepage_url or f"https://{domain}/"

    discovered: list[str] = []
    seen: set[str] = set()

    def _add(u: str) -> None:
        nu = u.rstrip("/") or u
        # Normalize trailing slash for dedup but preserve https
        key = nu.lower()
        if key not in seen:
            seen.add(key)
            discovered.append(u)

    # 1. Homepage links (highest priority - actual discovered)
    if homepage_html:
        try:
            links = await _discover_links_from_homepage(domain, homepage_html, homepage_url)
            for l in links:
                _add(l)
        except Exception:
            pass

    # 2. Sitemap URLs (second priority)
    try:
        sitemap_urls = await _fetch_sitemap_urls(domain)
        for u in sitemap_urls:
            _add(u)
    except Exception:
        pass

    # Ensure homepage is first
    homepage_norm = f"https://{domain}/"
    if homepage_norm.rstrip("/") not in seen and f"https://{domain}" not in seen:
        discovered.insert(0, homepage_norm)
        seen.add(homepage_norm.rstrip("/").lower())
        seen.add(f"https://{domain}".lower())
    else:
        # Move homepage to front if already present
        for i, u in enumerate(discovered):
            if urlparse(u).netloc.lower().lstrip("www.") == domain and urlparse(u).path in ("/", ""):
                discovered.insert(0, discovered.pop(i))
                break
        if not discovered or urlparse(discovered[0]).path not in ("/", ""):
            discovered.insert(0, homepage_norm)

    # 3. Fallback: add priority paths that are not yet discovered, sorted by score
    # We do NOT blindly request every path; we return them as candidates for fetcher,
    # and fetcher will handle 404/timeout gracefully (fail-soft). Limit to top 2-3 missing.
    missing_priority = []
    for path in PRIORITY_PATHS:
        cand = f"https://{domain}{path}" if path != "/" else homepage_norm
        key = cand.rstrip("/").lower()
        if key not in seen and cand.rstrip("/") not in seen:
            missing_priority.append(cand)
    # Only add top 2-3 most relevant missing, not all
    missing_priority = sorted(missing_priority, key=_score_url, reverse=True)[:3]
    for m in missing_priority:
        _add(m)

    # Final: dedupe, limit to 6, sorted with homepage first then by score
    # Keep homepage fixed at 0, sort rest by score
    if len(discovered) > 1:
        rest = sorted(discovered[1:], key=_score_url, reverse=True)
        discovered = [discovered[0]] + rest
    # Limit
    discovered = discovered[:6]

    # Deduplicate final (preserve order)
    final = []
    seen_final = set()
    for u in discovered:
        k = u.lower().rstrip("/")
        if k not in seen_final:
            seen_final.add(k)
            final.append(u)
    return final
