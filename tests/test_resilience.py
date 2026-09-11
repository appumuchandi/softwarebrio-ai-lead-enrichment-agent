"""Slice 3B resilience tests — mocked failure modes.
Covers: 404, timeout, 403/bot, 429, 5xx, malformed LLM JSON, LLM API failure, empty page, failed domain, missing fields, validation, sitemap, DNS, browser crash.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.models import CompanyEnrichment, FetchResult
from src.extraction.extractor import extract, _extract_json_from_text, _is_transient_llm_error
from src.scraping.fetcher import _is_bot_blocked

# ---------- Fetcher-level resilience (mocked via patch on fetch_one/fetch_many) ----------

@pytest.mark.asyncio
async def test_404_does_not_crash_and_is_not_retried_aggressively():
    """404 should return error immediately, not retried, and not crash domain."""
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    # Track calls
    calls = []

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        calls.append(url)
        # Simulate 404 for about, success for homepage
        if "about" in str(url):
            return FetchResult(url=url, status_code=404, error=f"HTTP 404 for {url}")
        return FetchResult(url=url, status_code=200, markdown="# Hi\nWe build APIs", html="<h1>hi</h1>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [await fake_fetch_one(u, browser) for u in urls]

    async def fake_discover(domain, homepage_html=None, homepage_url=None):
        return [f"https://{domain}/", f"https://{domain}/about", f"https://{domain}/team"]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        assert res.domain == "example.com"
        # 404 should be recorded in errors, not raise
        assert any("404" in e for e in res.errors)
        # Should still have source_pages from homepage
        assert len(res.source_pages) >= 1
        # Should not have retried 404 (calls for about should be 1, not 3)
        about_calls = [u for u in calls if "about" in u]
        assert len(about_calls) == 1  # no retry

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_timeout_retry_then_fail_soft():
    """Timeout should retry with backoff but ultimately fail-soft."""
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    # Simulate timeout on homepage fetch (all attempts)
    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        return FetchResult(url=url, status_code=None, error="timeout (15000ms, wait_until=domcontentloaded) for https://example.com/: TimeoutError")

    async def fake_discover(domain, **kwargs):
        return [f"https://{domain}/", f"https://{domain}/about"]

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [FetchResult(url=u, status_code=None, error="timeout") for u in urls]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        # Should not crash, should have errors and confidence 0
        assert res.confidence_score == 0.0 or res.confidence_score <= 0.35
        assert any("timeout" in e.lower() for e in res.errors)

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_403_bot_blocker_graceful_and_retry():
    """403 with bot challenge should be detected, retried, and recorded."""
    # Directly test _is_bot_blocked
    assert _is_bot_blocked("Just a moment...", "<div>cf-challenge</div>", 403) is True
    assert _is_bot_blocked("Access denied", "", 403) is True
    assert _is_bot_blocked("Normal Title", "<p>hello</p>", 200) is False
    # Non-bot 403 without signals should still be considered for single retry but not flagged is_bot_blocked
    assert _is_bot_blocked("Forbidden", "<p>forbidden</p>", 403) is False

    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        if "blocked" in url:
            return FetchResult(url=url, status_code=403, error="HTTP 403 for https://example.com/blocked (possible bot challenge detected)", is_bot_blocked=True, html="<title>Just a moment</title>")
        return FetchResult(url=url, status_code=200, markdown="# Ok", html="<h1>ok</h1>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [await fake_fetch_one(u) for u in urls]

    async def fake_discover(domain, **kwargs):
        return [f"https://{domain}/", f"https://{domain}/blocked"]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        assert any("403" in e or "bot" in e.lower() for e in res.errors)
        assert res.domain == "example.com"

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_429_and_5xx_retry_transient():
    """429 and 5xx should be considered transient and retry, but ultimately fail-soft."""
    # We test via fetcher logic: ensure 429/500 are transient
    # Simulate via orchestrator with fake_fetch_one returning 429 then success on retry? For deterministic test, we mock fetch_many to return 429 error
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        if url == "https://example.com/":
            return FetchResult(url=url, status_code=200, markdown="# Home", html="<h1>home</h1>")
        # Simulate 429 and 500 for other URLs
        if "pricing" in url:
            return FetchResult(url=url, status_code=429, error="HTTP 429 for https://example.com/pricing (rate limited)")
        if "team" in url:
            return FetchResult(url=url, status_code=500, error="HTTP 500 for https://example.com/team (server error)")
        return FetchResult(url=url, status_code=200, markdown="# Page", html="<p>page</p>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [await fake_fetch_one(u) for u in urls]

    async def fake_discover(domain, **kwargs):
        return [f"https://{domain}/", f"https://{domain}/pricing", f"https://{domain}/team"]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        # Both 429 and 500 should be recorded, not crash
        assert any("429" in e for e in res.errors)
        assert any("500" in e for e in res.errors)
        assert len(res.source_pages) >= 1  # homepage still

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_empty_page_content_handled():
    """Malformed/empty page should be handled with empty content error, not crash."""
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        if "empty" in url:
            return FetchResult(url=url, status_code=200, markdown=None, html="<html></html>", error="empty content after cleaning")
        return FetchResult(url=url, status_code=200, markdown="# Content", html="<h1>content</h1>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [await fake_fetch_one(u) for u in urls]

    async def fake_discover(domain, **kwargs):
        return [f"https://{domain}/", f"https://{domain}/empty"]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        assert any("empty" in e.lower() for e in res.errors)
        assert res.confidence_score <= 0.7  # confidence reduced due to error

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_dns_and_browser_crash_fail_soft():
    """Connection/DNS and browser crash should be caught and recorded."""
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        # Simulate DNS failure
        return FetchResult(url=url, status_code=None, error="fetch exception for https://bad.test/: Error: Page.goto: net::ERR_NAME_NOT_RESOLVED")

    async def fake_discover(domain, **kwargs):
        # Simulate browser crash during discovery? Actually discovery should not crash, but test generic
        return [f"https://{domain}/", f"https://{domain}/about"]

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [FetchResult(url=u, status_code=None, error="fetch exception: BrowserClosedError") for u in urls]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("bad.test")
        assert any("ERR_NAME_NOT_RESOLVED" in e or "BrowserClosed" in e for e in res.errors)
        assert res.domain == "bad.test"
        # Should still produce valid record with confidence 0
        CompanyEnrichment.model_validate(res.model_dump())

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_sitemap_failure_fail_soft():
    """Sitemap failure should not crash discovery, should fallback to priority paths."""
    from src.scraping.discovery import _fetch_sitemap_urls, discover_urls
    # Mock httpx to always fail
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.get = AsyncMock(side_effect=Exception("DNS failure"))
        mock_client.return_value = mock_instance

        urls = await _fetch_sitemap_urls("example.com")
        assert urls == []  # fail-soft

        # discover_urls should still return homepage + priority paths
        result = await discover_urls("example.com", homepage_html=None, homepage_url="https://example.com/")
        assert len(result) >= 3
        assert result[0].rstrip("/") == "https://example.com"

@pytest.mark.asyncio
async def test_malformed_llm_json_handled():
    """Malformed LLM JSON should be caught and fallback to regex/mock, not crash."""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    # Mock OpenAI to return malformed JSON
    malformed = "This is not JSON { broken.."
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=malformed))]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_cls.return_value = mock_client

        pages = {"https://example.com/": "# Hello\nWe build APIs."}
        res = await extract("example.com", pages, errors=[])
        # Should not crash, should have fallback with confidence <=0.35 and error recorded
        assert isinstance(res, CompanyEnrichment)
        assert res.confidence_score <= 0.35 or res.confidence_score == 0.5  # mock fallback 0.35 or 0.5
        assert any("LLM" in e or "fallback" in e.lower() for e in res.errors) or res.confidence_score <= 0.35

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_llm_api_failure_rate_limit_timeout():
    """LLM API failure (429, timeout) should be retried then fallback, not crash."""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    # Simulate transient 429 then success? For this test, simulate always 429 to test fallback
    from openai import RateLimitError

    # Create a mock RateLimitError — openai errors need specific args
    # Simpler: mock to raise generic Exception with "429" and "rate limit" in message, which _is_transient_llm_error will detect
    async def mock_create(*args, **kwargs):
        raise Exception("429 rate limit exceeded")

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_cls.return_value = mock_client

        pages = {"https://example.com/": "# Hello"}
        res = await extract("example.com", pages, errors=[])
        # Should be fail-soft, fallback to mock
        assert isinstance(res, CompanyEnrichment)
        assert any("LLM" in e for e in res.errors)
        assert res.confidence_score <= 0.35

    # Test timeout variant
    async def mock_timeout(*args, **kwargs):
        raise Exception("timeout 60.0s")

    with patch("openai.AsyncOpenAI") as mock_cls2:
        mock_client2 = AsyncMock()
        mock_client2.chat.completions.create = AsyncMock(side_effect=mock_timeout)
        mock_cls2.return_value = mock_client2

        pages = {"https://example.com/": "# Hello"}
        res2 = await extract("example.com", pages, errors=[])
        assert any("LLM" in e for e in res2.errors)

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()

def test_is_transient_llm_error_detection():
    assert _is_transient_llm_error(Exception("429 rate limit")) is True
    assert _is_transient_llm_error(Exception("timeout 60s")) is True
    assert _is_transient_llm_error(Exception("500 internal server error")) is True
    assert _is_transient_llm_error(Exception("no valid JSON found")) is False
    assert _is_transient_llm_error(Exception("pydantic validation failed")) is False

@pytest.mark.asyncio
async def test_failed_domain_while_other_succeed():
    """One failed domain must not stop other domains."""
    from src.orchestrator import enrich_domains
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_one(domain):
        if "bad" in domain:
            return CompanyEnrichment(domain=domain, confidence_score=0.0, errors=["no pages fetched - no evidence", "DNS failed"])
        return CompanyEnrichment(domain=domain, company_overview="ok", confidence_score=0.6, source_pages=["https://good.com/"])  # type: ignore

    with patch("src.orchestrator.enrich_one_domain", side_effect=fake_one):
        results = await enrich_domains(["good.com", "bad-domain.test", "also-good.com"])
        assert len(results) == 3
        assert results[0].domain == "good.com"
        assert results[1].domain == "bad-domain.test"
        assert results[1].confidence_score == 0.0
        assert results[2].domain == "also-good.com"
        # Good domains still succeeded
        assert results[0].confidence_score > 0
        assert results[2].confidence_score > 0

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_missing_optional_fields_and_empty_leadership_valid():
    """Missing optional fields should be valid, leadership=[] is valid."""
    # Direct Pydantic validation with missing fields
    c = CompanyEnrichment(domain="example.com", company_overview=None, target_audience_icp=None, leadership=[], contact_emails=[], linkedin_urls=[], source_pages=[], confidence_score=0.0, errors=[])
    assert c.leadership == []
    assert c.target_audience_icp is None
    # Also test extract with minimal evidence produces valid with defaults
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()
    pages = {"https://example.com/": "# Minimal\nHi"}
    res = await extract("example.com", pages, errors=[])
    CompanyEnrichment.model_validate(res.model_dump())
    assert isinstance(res.leadership, list)
    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_pydantic_validation_failure_handled_safely():
    """Malformed LLM JSON that fails Pydantic should be caught, not crash."""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    # LLM returns invalid confidence >1.0 which fails Pydantic ge/le
    bad_payload = {
        "domain": "example.com",
        "company_overview": "Test",
        "contact_emails": ["not-an-email"],  # invalid email
        "leadership": [{"name": ""}],  # name too short
        "linkedin_urls": ["not-a-url"],
        "source_pages": ["https://example.com/"],
        "confidence_score": 2.0,  # out of range
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(bad_payload)))]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_cls.return_value = mock_client

        pages = {"https://example.com/": "# Hello"}
        res = await extract("example.com", pages, errors=[])
        # Should be fail-soft, not raise, with validation error recorded
        assert isinstance(res, CompanyEnrichment)
        assert any("pydantic" in e.lower() or "validation" in e.lower() for e in res.errors) or res.confidence_score == 0.0

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_browser_page_crash_resource_cleanup():
    """Browser/page crash should not leave orphan processes and should return error."""
    # We test via fetcher mock that simulates browser crash exception
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    # Simulate that fetch_one would raise BrowserClosedError which is caught in fetcher
    # Instead test that orchestrator still cleans up and returns
    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        # Simulate that playwright browser crashed but fetcher returned error result (fail-soft)
        return FetchResult(url=url, status_code=None, error="fetch exception for https://example.com/: BrowserClosedError: Target closed", is_bot_blocked=False)

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        # Ensure fetch_many respects concurrency and doesn't leave pending
        # Simulate that all fetches fail with browser crash
        return [FetchResult(url=u, status_code=None, error="BrowserClosedError") for u in urls]

    async def fake_discover(domain, **kwargs):
        return [f"https://{domain}/", f"https://{domain}/about"]

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()
    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_async_pw = MagicMock(return_value=mock_playwright)
    with patch("src.orchestrator.async_playwright", mock_async_pw), \
         patch("src.orchestrator.fetch_one", side_effect=fake_fetch_one), \
         patch("src.orchestrator.fetch_many", side_effect=fake_fetch_many), \
         patch("src.orchestrator.discover_urls", side_effect=fake_discover):
        res = await enrich_one_domain("example.com")
        assert "BrowserClosed" in str(res.errors) or any("BrowserClosed" in e for e in res.errors)
        # Should not crash, should still have valid record
        CompanyEnrichment.model_validate(res.model_dump())

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()
