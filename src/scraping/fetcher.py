"""
Playwright fetcher - reused Browser per fetch_many.

Requirements enforced:
- wait_until="domcontentloaded" as primary (NOT networkidle).
- No playwright-stealth dependency.
- Graceful bot-blocker detection + retry/fallback + error reporting (no crash).
- JS-rendered content handled via Playwright (Chromium).
- Fail-soft: every fetch returns FetchResult with error populated, never raises to caller.

Design (Slice 4 fix): one Browser per fetch_many(), new Context+Page per URL for isolation.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

from playwright.async_api import Browser, async_playwright, TimeoutError as PlaywrightTimeoutError

from src.config import get_settings
from src.models import FetchResult
from src.scraping.cleaner import html_to_markdown

# Signals that likely indicate bot challenge / block
_BOT_BLOCKER_TITLE_RE = re.compile(r"just a moment|checking your browser|cf.?challenge|attention required|access denied|cloudflare", re.I)
_BOT_BLOCKER_BODY_SNIPPETS = [
    "cf-challenge",
    "cf-turnstile",
    "cf-browser-verification",
    "enable javascript and cookies",
    "ddos protection by cloudflare",
]


def _is_bot_blocked(title: str, html: str, status: Optional[int]) -> bool:
    if status in (403, 429, 503):
        if _BOT_BLOCKER_TITLE_RE.search(title or ""):
            return True
        lower = (html or "").lower()
        if any(s in lower for s in _BOT_BLOCKER_BODY_SNIPPETS):
            return True
        return False
    if _BOT_BLOCKER_TITLE_RE.search(title or ""):
        return True
    lower = (html or "").lower()
    if any(s in lower for s in _BOT_BLOCKER_BODY_SNIPPETS):
        return True
    return False


async def fetch_one(url: str, browser: Optional[Browser] = None) -> FetchResult:
    """
    Fetch a single URL with Playwright using the shared Browser.

    Each call creates a new BrowserContext + Page for isolation, with retries.
    Never raises - fail-soft per URL. When browser is provided (preferred, from
    fetch_many/orchestrator), it is reused and NOT closed here. When browser
    is None (direct calls, backward compat), a temporary Browser is created
    for this single fetch and torn down — not optimal but preserves call sites.
    """
    # Backward-compat fallback: direct call without shared Browser
    if browser is None:
        playwright = None
        tmp_browser = None
        try:
            playwright = await async_playwright().start()
            tmp_browser = await playwright.chromium.launch(headless=get_settings().browser_headless)
            return await fetch_one(url, tmp_browser)
        finally:
            try:
                if tmp_browser is not None:
                    await tmp_browser.close()
            except Exception:
                pass
            try:
                if playwright is not None:
                    await playwright.stop()
            except Exception:
                pass
    settings = get_settings()
    if not url.startswith("http"):
        url = "https://" + url

    retries = max(0, settings.page_goto_retries)
    last_error: Optional[str] = None
    last_status: Optional[int] = None

    for attempt in range(retries + 1):
        context = None
        page = None
        try:
            # New context per fetch for isolation + fresh UA (Browser is shared)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                ignore_https_errors=True,
            )
            page = await context.new_page()
            response = await page.goto(
                url,
                wait_until=settings.wait_until,  # "domcontentloaded"
                timeout=settings.navigation_timeout_ms,
            )
            status = response.status if response else None
            last_status = status
            try:
                await page.wait_for_timeout(800)
            except Exception:
                pass

            if status is not None and status >= 400:
                try:
                    title = await page.title()
                except Exception:
                    title = ""
                try:
                    html = await page.content()
                except Exception:
                    html = ""
                is_blocked = _is_bot_blocked(title, html, status)
                err = f"HTTP {status} for {url}"
                if is_blocked:
                    err += " (possible bot challenge detected)"
                elif status == 429:
                    err += " (rate limited)"
                elif status in (500, 502, 503, 504):
                    err += " (server error)"
                transient = status in (429, 500, 502, 503, 504) or (status == 403 and is_blocked)
                if status == 403 and not is_blocked and attempt < retries:
                    transient = True
                if transient and attempt < retries:
                    last_error = err
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue
                return FetchResult(
                    url=url,  # type: ignore
                    status_code=status,
                    html=html[:50000] if html else None,
                    markdown=None,
                    error=err,
                    is_bot_blocked=is_blocked,
                )

            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                html = await page.content()
            except Exception as e:
                last_error = f"failed to get content: {e}"
                if attempt < retries:
                    await asyncio.sleep(0.8 * (attempt + 1))
                    continue
                return FetchResult(url=url, status_code=status, error=last_error)  # type: ignore

            is_blocked = _is_bot_blocked(title, html or "", status)
            if is_blocked:
                err = f"bot challenge detected for {url} (title={title!r})"
                if attempt < retries:
                    last_error = err
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return FetchResult(
                    url=url,  # type: ignore
                    status_code=status,
                    html=(html or "")[:50000],
                    error=err,
                    is_bot_blocked=True,
                )

            md = html_to_markdown(html or "", url=url, max_chars=settings.max_markdown_chars_per_page)
            if len(md.strip()) < 200:
                try:
                    inner = await page.evaluate("() => document.body ? document.body.innerText : ''")
                    if inner and len(inner.strip()) > len(md.strip()):
                        md = inner.strip()[: settings.max_markdown_chars_per_page]
                except Exception:
                    pass

            return FetchResult(
                url=url,  # type: ignore
                status_code=status,
                html=(html or "")[:50000],
                markdown=md if md.strip() else None,
                error=None if md and md.strip() else "empty content after cleaning",
                is_bot_blocked=False,
            )

        except PlaywrightTimeoutError as e:
            last_error = f"timeout ({settings.navigation_timeout_ms}ms, wait_until={settings.wait_until}) for {url}: {e}"
            if attempt < retries:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            return FetchResult(url=url, status_code=last_status, error=last_error, is_bot_blocked=False)  # type: ignore
        except Exception as e:
            last_error = f"fetch exception for {url}: {type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            return FetchResult(url=url, status_code=last_status, error=last_error, is_bot_blocked=False)  # type: ignore
        finally:
            try:
                if page is not None:
                    await page.close()
            except Exception:
                pass
            try:
                if context is not None:
                    await context.close()
            except Exception:
                pass

    return FetchResult(url=url, status_code=last_status, error=last_error or "unknown fetch failure")  # type: ignore


async def fetch_many(urls: list[str], max_concurrency: int = 4, browser: Optional[Browser] = None) -> list[FetchResult]:
    """
    Fetch multiple URLs concurrently with bounded concurrency, reusing one Browser.

    Lifecycle (Slice 4):
    - If browser is None: start Playwright once, launch Chromium once, for each URL new_context/new_page, then close browser/stop Playwright.
    - If browser is provided: reuse it, for each URL new_context/new_page, do NOT close browser (caller owns it).
    Fail-soft: one failure never aborts others. Concurrency via semaphore.
    """
    if not urls:
        return []
    settings = get_settings()
    limit = min(max_concurrency, settings.max_concurrent_pages)
    sem = asyncio.Semaphore(limit)

    # If browser provided, reuse it (orchestrator owns lifecycle)
    if browser is not None:
        async def _guarded(u: str) -> FetchResult:
            async with sem:
                return await fetch_one(u, browser)

        results = await asyncio.gather(*[_guarded(u) for u in urls], return_exceptions=False)
        return results

    # Otherwise, create Playwright + Browser once for this fetch_many call
    playwright = None
    shared_browser = None
    try:
        playwright = await async_playwright().start()
        shared_browser = await playwright.chromium.launch(headless=settings.browser_headless)

        async def _guarded_shared(u: str) -> FetchResult:
            async with sem:
                return await fetch_one(u, shared_browser)

        results = await asyncio.gather(*[_guarded_shared(u) for u in urls], return_exceptions=False)
        return results
    finally:
        # Guaranteed cleanup of shared resources
        try:
            if shared_browser is not None:
                await shared_browser.close()
        except Exception:
            pass
        try:
            if playwright is not None:
                await playwright.stop()
        except Exception:
            pass
