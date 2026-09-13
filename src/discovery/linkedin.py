"""
DuckDuckGo-based external LinkedIn discovery for grounded leadership.

Bonus feature: AFTER deterministic evidence grounding, fill missing linkedin_url
for leadership members whose name is grounded but linkedin is null.

- Never invents URL: only returns linkedin.com/in/ URLs that appear in DuckDuckGo results.
- Verifies plausible association with exact name + company before assigning.
- Fail-soft: timeout / HTTP error / blocking / malformed -> None, never crashes pipeline.
- Bounded: max 5 searches per domain, deduplicate person, only grounded leadership.

Uses DuckDuckGo public web search endpoint (html.duckduckgo.com) via httpx.
No paid API, no API key.
Respects resilience: timeout returns None, not exception.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import unquote

import httpx

# Strict limit per domain (task requirement: e.g. 5)
MAX_LINKEDIN_SEARCHES_PER_DOMAIN: int = 5

# DuckDuckGo public HTML search – browser-accessible, no key, free
DUCKDUCKGO_HTML_URL: str = "https://html.duckduckgo.com/html/"

# Timeout for external search (bounded)
SEARCH_TIMEOUT_SECONDS: float = 10.0

# Prefer linkedin.com/in/ profile URLs
_LINKEDIN_IN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[^\s\"'<>\)\]]+",
    re.IGNORECASE,
)

# Broader pattern to capture bare www.linkedin.com/in/ and http variants (for DDG text/content)
_LINKEDIN_BARE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s\"'<>\)\]\?&]+",
    re.IGNORECASE,
)

# DuckDuckGo redirect parameter containing URL-encoded target
_UDDG_RE = re.compile(r"uddg=([^&\"'<>]+)", re.IGNORECASE)

# Generic fallback to detect any linkedin URL (used for malformed check)
_LINKEDIN_ANY_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/[^\s\"'<>\)\]]+",
    re.IGNORECASE,
)


def _clean_linkedin_url(raw: str) -> str:
    """Normalize LinkedIn URL: strip trailing punctuation, ensure https://www.linkedin.com/in/<profile>."""
    url = raw.strip().rstrip(".,;)]\"'")
    # Remove URL parameters/fragment that are tracking artifacts, keep path
    # e.g., https://linkedin.com/in/foo?trk=... -> https://www.linkedin.com/in/foo
    url = url.split("?")[0].split("#")[0].rstrip("/")
    # Handle protocol-relative //www.linkedin.com/...
    if url.startswith("//"):
        url = "https:" + url
    # Ensure scheme is https and host is www.linkedin.com
    if url.lower().startswith("http://"):
        url = "https://" + url[len("http://") :]
    elif url.lower().startswith("https://"):
        pass
    elif url.lower().startswith("www.linkedin.com/"):
        url = "https://" + url
    elif url.lower().startswith("linkedin.com/"):
        url = "https://www." + url
    # Normalize linkedin.com -> www.linkedin.com
    # After scheme handling, ensure host is www
    # e.g., https://linkedin.com/in/foo -> https://www.linkedin.com/in/foo
    if url.lower().startswith("https://linkedin.com/in/"):
        url = "https://www.linkedin.com/in/" + url[len("https://linkedin.com/in/") :]
    elif url.lower().startswith("https://linkedin.com"):
        # generic fallback for linkedin.com without www (just in case)
        idx = url.lower().find("linkedin.com")
        # Preserve path case after domain
        suffix = url[idx + len("linkedin.com") :]
        # Ensure suffix starts with /in/ etc.
        url = "https://www.linkedin.com" + suffix
    # Preserve case of path but domain normalized to lower; return as is
    return url


def _verify_candidate(
    context_text: str,
    linkedin_url: str,
    name: str,
    company: str,
) -> bool:
    """
    Verify result is plausibly associated with exact person and company.

    - Name tokens must appear in context or slug
    - At least one company token (or full company substring) must appear in context

    If verification is uncertain, return False (leave linkedin as null).
    """
    name = (name or "").strip()
    company = (company or "").strip()
    if not name:
        return False

    ctx_lower = (context_text or "").lower()
    url_lower = (linkedin_url or "").lower()
    name_lower = name.lower()

    # Tokenize name: split on non-alphanum, keep tokens >=2 chars
    name_tokens = [t for t in re.split(r"[^a-z0-9]+", name_lower) if len(t) >= 2]
    if not name_tokens:
        return False

    # Name association checks
    name_in_context = name_lower in ctx_lower
    tokens_in_context = all(tok in ctx_lower for tok in name_tokens)

    # Slug check: linkedin /in/<slug> often contains name
    slug = ""
    if "/in/" in url_lower:
        slug = url_lower.split("/in/", 1)[1].split("?")[0].split("#")[0].rstrip("/")
        slug_norm = re.sub(r"[^a-z0-9]+", " ", slug).strip()
    else:
        slug_norm = ""

    tokens_in_slug = False
    if slug_norm:
        tokens_in_slug = all(tok in slug_norm for tok in name_tokens)
        # Also allow slug without separator e.g. "abhinavasthana"
        if not tokens_in_slug:
            # Check concatenated tokens appear
            concatenated = "".join(name_tokens)
            tokens_in_slug = concatenated in slug.replace("-", "").replace("_", "")

    name_ok = name_in_context or tokens_in_context or tokens_in_slug
    if not name_ok:
        return False

    # Company association: require at least one company token in context
    # If company empty, fallback to domain token already? But spec says company/domain provided
    if not company:
        return True  # name alone verified (fallback)

    company_lower = company.lower()
    # Tokenize company, filter common stopwords
    stop = {"inc", "llc", "ltd", "corp", "corporation", "company", "group", "limited", "co", "com", "the", "and", "for"}
    company_tokens = [
        t for t in re.split(r"[^a-z0-9]+", company_lower) if len(t) >= 3 and t not in stop
    ]
    if not company_tokens:
        # company is short like "Vapi" -> use len>=2
        company_tokens = [t for t in re.split(r"[^a-z0-9]+", company_lower) if len(t) >= 2]

    if not company_tokens:
        # company was e.g. "a" – ignore company check, name verified is enough
        return True

    # Check company tokens in context OR full company string
    company_in_context = any(tok in ctx_lower for tok in company_tokens)
    if not company_in_context:
        if company_lower in ctx_lower:
            company_in_context = True

    # Also allow domain-style company: "postman.com" -> token "postman" already covered
    # If still not found, strictly fail (do not guess)
    return company_in_context


def _extract_verified_linkedin(html: str, name: str, company: str) -> Optional[str]:
    """
    Inspect DuckDuckGo HTML for linkedin.com/in/ URLs and verify.

    Returns first verified URL or None.
    """
    if not html or not isinstance(html, str):
        return None
    if not name or not name.strip():
        return None

    # Quick fail if html doesn't contain linkedin at all
    if "linkedin.com" not in html.lower():
        return None

    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    # 1) Prefer DuckDuckGo uddg redirect URLs (URL-decoded) — e.g., //duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fabhinavasthana
    for m in _UDDG_RE.finditer(html):
        encoded = m.group(1)
        try:
            decoded = unquote(encoded)
        except Exception:
            continue
        inner = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s\"'<>\)\]\?&]+", decoded, re.IGNORECASE)
        if not inner:
            continue
        raw = inner.group(0)
        cleaned = _clean_linkedin_url(raw)
        key = cleaned.lower().rstrip("/")
        if "/in/" not in key:
            continue
        if key in seen:
            continue
        seen.add(key)
        candidates.append((cleaned, m.start(), m.end()))

    # 2) Also capture direct/bare linkedin URLs (https://, http://, www.) appearing in HTML text/href
    for m in _LINKEDIN_BARE_RE.finditer(html):
        raw = m.group(0)
        cleaned = _clean_linkedin_url(raw)
        key = cleaned.lower().rstrip("/")
        if "/in/" not in key:
            continue
        if key in seen:
            continue
        seen.add(key)
        candidates.append((cleaned, m.start(), m.end()))

    # Strict: only /in/ profile URLs, never company pages
    if not candidates:
        return None

    for url, start, end in candidates:
        # Build context window around the URL for verification.
        # DuckDuckGo html: each result is about ~2k chars block.
        window_start = max(0, start - 2500)
        window_end = min(len(html), end + 2500)
        context_html = html[window_start:window_end]
        # Strip tags to get text context (title + snippet + URL)
        context_text = re.sub(r"<[^>]+>", " ", context_html)
        context_text = re.sub(r"\s+", " ", context_text).strip()
        # Include URL itself in context already, but ensure url contributes
        # Verify plausibility
        if _verify_candidate(context_text, url, name, company):
            # Final cleaning and ensure https www prefix
            # Return with https
            if url.startswith("http://"):
                url = "https://" + url[len("http://") :]
            return url

    return None


def _build_query(name: str, company: str, domain: str) -> str:
    """
    Build DuckDuckGo query: "<name>" "<company>" LinkedIn
    Fallback to domain if company empty.
    """
    n = (name or "").strip()
    c = (company or domain or "").strip()
    # Quote for exact phrase
    parts: list[str] = []
    if n:
        # Escape quotes inside name
        n_esc = n.replace('"', "")
        parts.append(f'"{n_esc}"')
    if c:
        c_esc = c.replace('"', "")
        parts.append(f'"{c_esc}"')
    parts.append("LinkedIn")
    return " ".join(parts)


def discover_linkedin_profile(name: str, company: str, domain: str) -> Optional[str]:
    """
    Synchronous DuckDuckGo search for LinkedIn profile.

    Searches DuckDuckGo html endpoint for "<name>" "<company|domain>" LinkedIn,
    inspects results for linkedin.com/in/ URLs that appear in the search results,
    verifies plausible association with exact name and company, and returns first
    verified URL or None.

    Resilience: timeout, HTTP error, blocking, malformed -> None, never raises.

    Args:
        name: full name of grounded leadership member (must be exact, grounded by pipeline)
        company: company name or domain fallback (used for query + verification)
        domain: domain of company (e.g., postman.com) – used if company empty

    Returns:
        str | None: verified linkedin.com/in/... URL or None

    Security: Never invents URL; only returns URL that actually appeared in search results.
    Prefer linkedin.com/in/ URLs.
    """
    if not name or not name.strip():
        return None
    # company or domain must be present for meaningful search
    comp = (company or domain or "").strip()
    if not comp:
        return None

    query = _build_query(name, company, domain)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        with httpx.Client(
            timeout=SEARCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            resp = client.get(DUCKDUCKGO_HTML_URL, params={"q": query})
            # HTTP error -> None (resilience)
            if resp.status_code != 200:
                return None
            html = resp.text or ""
            if not html or not isinstance(html, str):
                return None
            # Simple blocking detection: DuckDuckGo sometimes returns captcha/bot page
            lower = html.lower()
            if "anomaly" in lower and "too many requests" in lower:
                return None
            if "captcha" in lower and "duckduckgo" in lower and "linkedin" not in lower:
                # Might be blocking, but if linkedin not present, treat as no result
                # we still try to parse; if no linkedin, returns None anyway
                pass
            return _extract_verified_linkedin(html, name, comp)
    except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout):
        return None
    except httpx.HTTPError:
        return None
    except Exception:
        return None


async def discover_linkedin_profile_async(name: str, company: str, domain: str) -> Optional[str]:
    """
    Async version of discover_linkedin_profile, suitable for orchestrator.

    Same semantics, uses httpx.AsyncClient.
    """
    if not name or not name.strip():
        return None
    comp = (company or domain or "").strip()
    if not comp:
        return None

    query = _build_query(name, company, domain)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        async with httpx.AsyncClient(
            timeout=SEARCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=headers,
        ) as client:
            resp = await client.get(DUCKDUCKGO_HTML_URL, params={"q": query})
            if resp.status_code != 200:
                return None
            html = resp.text or ""
            if not html or not isinstance(html, str):
                return None
            lower = html.lower()
            if "anomaly" in lower and "too many requests" in lower:
                return None
            return _extract_verified_linkedin(html, name, comp)
    except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout):
        return None
    except httpx.HTTPError:
        return None
    except Exception:
        return None


async def enrich_leadership_with_linkedin(
    enrichment,
    domain: str,
    company_name: Optional[str] = None,
    max_searches: int = MAX_LINKEDIN_SEARCHES_PER_DOMAIN,
) -> int:
    """
    Enrich leadership members missing linkedin_url via DuckDuckGo.

    Mutates enrichment in-place. Bounded, fail-soft, never crashes.

    - Only searches grounded leadership records with linkedin is null/missing
    - Deduplicates by normalized name (do not search same person twice)
    - Enforces strict maximum per domain (default 5)
    - Does not replace existing linkedin_url
    - Updates top-level linkedin_urls consistently (deduped)
    - Verification via _verify_candidate before assignment
    - Any search failure returns None and continues (does not fail domain)

    Returns:
        int: number of leadership members enriched

    Args:
        enrichment: CompanyEnrichment (Pydantic model) – must have leadership list
        domain: domain string e.g., "postman.com"
        company_name: optional company name for query; if None, derived from domain
        max_searches: cap for searches per domain
    """
    # Fail-soft: any error returns 0 and leaves enrichment unchanged
    try:
        if enrichment is None:
            return 0
        leadership = getattr(enrichment, "leadership", None)
        if not leadership:
            return 0

        # Derive company name for query if not provided: use domain base as fallback
        if not company_name:
            # Use domain without TLD capitalised; e.g., postman.com -> Postman
            domain_base = (domain or "").strip().split(".")[0]
            if domain_base:
                company_name = domain_base
            else:
                company_name = domain or ""
        company_name = (company_name or "").strip()
        domain = (domain or "").strip()

        # Build list of candidates: linkedin is null/missing, valid grounded name
        candidates: list = []
        seen_names: set[str] = set()
        for member in leadership:
            try:
                name = (getattr(member, "name", "") or "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                # Skip if already has linkedin
                linkedin = getattr(member, "linkedin_url", None)
                if linkedin is not None:
                    # Already has linkedin – do not search/replace (requirement 9)
                    continue
                # Validate grounded name length – must be plausible person name
                if len(name) < 2:
                    continue
                candidates.append(member)
            except Exception:
                continue

        if not candidates:
            return 0

        # Enforce max searches per domain
        candidates = candidates[:max_searches]

        enriched = 0
        # Lazy import to avoid cycle; use TypeAdapter for HttpUrl validation
        from pydantic import TypeAdapter, HttpUrl as _HttpUrl

        _http_adapter = TypeAdapter(_HttpUrl)
        for member in candidates:
            try:
                name = member.name.strip()  # type: ignore
                # Use company_name for query company param, domain for domain param
                result = await discover_linkedin_profile_async(name, company_name, domain)
                if result:
                    try:
                        validated_url = _http_adapter.validate_python(result)
                        member.linkedin_url = validated_url  # type: ignore
                        enriched += 1
                    except Exception:
                        # If validation fails, leave as None
                        try:
                            member.linkedin_url = None  # type: ignore
                        except Exception:
                            pass
                        continue
            except Exception:
                # Search failure does not fail domain – continue
                continue

        # Update top-level linkedin_urls consistently if any enriched
        if enriched > 0:
            try:
                existing = list(getattr(enrichment, "linkedin_urls", []) or [])
                # Collect new member linkedin_urls
                for m in leadership:
                    lu = getattr(m, "linkedin_url", None)
                    if lu is not None:
                        # Ensure validated HttpUrl for top-level list as well
                        try:
                            lu_validated = _http_adapter.validate_python(str(lu))
                        except Exception:
                            lu_validated = lu  # fallback to raw if validation fails
                        lu_str = str(lu_validated)
                        key = lu_str.lower().rstrip("/")
                        if not any(str(u).lower().rstrip("/") == key for u in existing):
                            existing.append(lu_validated)  # type: ignore
                # Dedup preserve order
                deduped: list = []
                seen_urls: set[str] = set()
                for u in existing:
                    # Validate each existing entry as well to ensure correct type
                    try:
                        u_valid = _http_adapter.validate_python(str(u))
                    except Exception:
                        u_valid = u
                    k = str(u_valid).lower().rstrip("/")
                    if k not in seen_urls:
                        seen_urls.add(k)
                        deduped.append(u_valid)
                enrichment.linkedin_urls = deduped  # type: ignore
            except Exception:
                pass

        return enriched
    except Exception:
        return 0
