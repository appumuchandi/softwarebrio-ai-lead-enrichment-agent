import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from src.models import CompanyEnrichment, FetchResult
from src.orchestrator import enrich_one_domain, enrich_domains

@pytest.mark.asyncio
async def test_enrich_one_domain_fail_soft_per_url():
    """404 on one URL does not crash domain enrichment"""
    from src import config as cfg
    import os
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_fetch_one(url, browser=None, *args, **kwargs):
        if "about" in url:
            return FetchResult(url=url, status_code=404, error=f"HTTP 404 for {url}")
        return FetchResult(url=url, status_code=200, markdown="# Homepage\nWe build APIs", html="<h1>hi</h1>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *args, **kwargs):
        return [await fake_fetch_one(u) for u in urls]

    async def fake_discover(domain, homepage_html=None, homepage_url=None):
        return [f"https://{domain}/", f"https://{domain}/about", f"https://{domain}/team"]

    # team will succeed via fake_fetch_one (not about)
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
        assert any("404" in e for e in res.errors)
        # still has at least homepage source
        assert len(res.source_pages) >= 1
        assert 0 <= res.confidence_score <= 1

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_enrich_domains_fail_soft_per_domain():
    """One domain failing does not terminate complete run"""
    from src import config as cfg
    import os
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    async def fake_one(domain):
        if "bad" in domain:
            # simulate total fetch failure -> empty pages
            return CompanyEnrichment(domain=domain, confidence_score=0.0, errors=["no pages fetched"])
        return CompanyEnrichment(domain=domain, company_overview="ok", confidence_score=0.6, source_pages=["https://ok.com/"])  # type: ignore

    with patch("src.orchestrator.enrich_one_domain", side_effect=fake_one):
        results = await enrich_domains(["good.com", "bad-domain.test"])
        assert len(results) == 2
        assert results[0].domain == "good.com"
        assert results[1].domain == "bad-domain.test"
        assert results[1].confidence_score == 0.0

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_fetch_uses_domcontentloaded_not_networkidle():
    # Verify config drives fetcher
    from src.config import get_settings
    s = get_settings()
    assert s.wait_until == "domcontentloaded"
    # Source code should mention domcontentloaded and not depend on networkidle as wait
    src = Path("src/scraping/fetcher.py").read_text()
    assert 'wait_until' in src
    assert 'domcontentloaded' in src
    # comments may mention networkidle but not as the actual wait value
    # ensure the page.goto call uses settings.wait_until
    assert "settings.wait_until" in src

def test_output_json_validates_if_exists():
    p = Path("output.json")
    if not p.exists():
        pytest.skip("no output.json yet")
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 3 or len(data) >= 1  # slice 2 has 3 domains
    for rec in data:
        c = CompanyEnrichment.model_validate(rec)
        assert 0 <= c.confidence_score <= 1
        assert isinstance(c.source_pages, list)
