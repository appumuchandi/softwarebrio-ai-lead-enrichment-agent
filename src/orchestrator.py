"""
Orchestrator - vertical slice.

Fail-soft per domain: one failed domain/URL never terminates the run.
Evidence-aware: source_pages preserved from actual fetched URLs.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import async_playwright

from src.config import get_settings
from src.models import CompanyEnrichment
from src.scraping.discovery import discover_urls
from src.scraping.fetcher import fetch_one, fetch_many
from src.extraction.extractor import extract


async def enrich_one_domain(domain: str) -> CompanyEnrichment:
    """
    Enrich a single domain. Never raises - always returns CompanyEnrichment.
    """
    settings = get_settings()
    raw_domain = domain.strip()
    domain = raw_domain.lower().replace("https://", "").replace("http://", "").split("/")[0].lstrip("www.")
    if not domain:
        return CompanyEnrichment(domain=raw_domain or "unknown", confidence_score=0.0, errors=["empty domain input"])  # type: ignore

    errors: list[str] = []

    # Reuse one Browser for all fetches in this domain (Slice 4)
    playwright = None
    browser = None
    homepage_result = None
    homepage_html: Optional[str] = None
    pages_markdown: dict[str, str] = {}
    successful_urls: list[str] = []
    # We need homepage_url even if browser launch fails
    homepage_url = f"https://{domain}/"

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=settings.browser_headless)

        # Step 1: Fetch homepage first (needed for discovery via actual links) — reuses shared Browser
        homepage_result = await fetch_one(homepage_url, browser)

        homepage_html = homepage_result.html if homepage_result.html else None

        if homepage_result.markdown and not homepage_result.error:
            pages_markdown[str(homepage_result.url)] = homepage_result.markdown
            successful_urls.append(str(homepage_result.url))
        else:
            if homepage_result.error:
                errors.append(f"{homepage_url}: {homepage_result.error}")
            if not homepage_html and homepage_result.html:
                homepage_html = homepage_result.html

        # Step 2: Discover URLs (prioritizes actual homepage links + sitemap, fallback to known paths)
        try:
            discovered = await discover_urls(domain, homepage_html=homepage_html, homepage_url=homepage_url)
        except Exception as e:
            errors.append(f"discovery failed: {e}")
            discovered = [homepage_url, f"https://{domain}/about", f"https://{domain}/contact"]

        # Remove already-fetched homepage from discovered to avoid refetch
        # homepage_result may be None if browser launch failed; handle safely
        homepage_result_url = str(homepage_result.url) if homepage_result and homepage_result.url else homepage_url
        to_fetch = [u for u in discovered if u.rstrip("/") != homepage_url.rstrip("/") and homepage_result_url.rstrip("/") != u.rstrip("/")]
        to_fetch = to_fetch[:4]

        if to_fetch:
            # Reuse same Browser for additional pages (per-URL isolation via new_context/new_page)
            results = await fetch_many(to_fetch, max_concurrency=settings.max_concurrent_pages, browser=browser)
            for r in results:
                url_str = str(r.url)
                if r.markdown and not r.error:
                    pages_markdown[url_str] = r.markdown
                    successful_urls.append(url_str)
                else:
                    if r.error:
                        errors.append(f"{url_str}: {r.error}")
                    if r.is_bot_blocked:
                        errors.append(f"{url_str}: bot challenge detected (graceful fallback, no stealth)")
    except Exception as e:
        # Browser launch failure or unexpected orchestrator error — fail-soft, record and continue to extraction with whatever we have
        errors.append(f"browser setup failed for {domain}: {type(e).__name__}: {e}")
        # If homepage_result was never obtained due to browser failure, ensure we have at least an error
        if not homepage_result:
            errors.append(f"{homepage_url}: failed to fetch homepage (browser failure)")
    finally:
        # Guaranteed cleanup of shared Browser/Playwright for this domain
        try:
            if browser is not None:
                await browser.close()
        except Exception:
            pass
        try:
            if playwright is not None:
                await playwright.stop()
        except Exception:
            pass

    # Step 3: LLM extraction (always called, even if only homepage succeeded; handles empty case)
    # Enforce total markdown budget
    total_chars = sum(len(v) for v in pages_markdown.values())
    if total_chars > settings.max_total_markdown_chars:
        # Truncate longest pages first
        sorted_items = sorted(pages_markdown.items(), key=lambda x: len(x[1]), reverse=True)
        budget = settings.max_total_markdown_chars
        truncated: dict[str, str] = {}
        for url, md in sorted_items:
            if budget <= 0:
                break
            take = min(len(md), budget)
            truncated[url] = md[:take]
            budget -= take
        # Ensure homepage kept if truncated away
        if homepage_url not in truncated and homepage_url in pages_markdown:
            truncated[homepage_url] = pages_markdown[homepage_url][:2000]
        pages_markdown = truncated

    enrichment = await extract(domain=domain, pages_markdown=pages_markdown, errors=errors)

    # Evidence-aware: if extractor didn't set source_pages correctly, ensure it's grounded in what we fetched
    # (extractor already does this, but double-ensure here)
    if not enrichment.source_pages and successful_urls:
        enrichment.source_pages = successful_urls  # type: ignore
    # Ensure domain normalized
    enrichment.domain = domain

    return enrichment


async def enrich_domains(domains: list[str]) -> list[CompanyEnrichment]:
    """
    Enrich multiple domains concurrently with fail-soft per domain.
    """
    settings = get_settings()
    sem = asyncio.Semaphore(settings.max_concurrent_domains)

    async def _guarded(d: str) -> CompanyEnrichment:
        async with sem:
            try:
                return await enrich_one_domain(d)
            except Exception as e:
                # Absolute last resort - should never happen because enrich_one_domain is fail-soft
                norm = d.strip().lower().replace("https://", "").replace("http://", "").split("/")[0].lstrip("www.") or d
                return CompanyEnrichment(domain=norm, confidence_score=0.0, errors=[f"orchestrator crash: {type(e).__name__}: {e}"])  # type: ignore

    # Preserve input order
    results = await asyncio.gather(*[_guarded(d) for d in domains])
    return list(results)
