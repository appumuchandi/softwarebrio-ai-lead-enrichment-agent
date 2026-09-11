import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.scraping.discovery import discover_urls, _score_url, PRIORITY_PATHS

@pytest.mark.asyncio
async def test_discover_prioritizes_same_domain_over_cross_domain():
    html = '''
    <html><body>
    <a href="https://postman.com/about">About</a>
    <a href="https://postman.com/team">Team</a>
    <a href="https://external.com/page">External</a>
    <a href="/pricing">Pricing</a>
    <a href="/careers">Careers</a>
    </body></html>
    '''
    urls = await discover_urls("postman.com", homepage_html=html, homepage_url="https://postman.com/")
    # should include same-domain, not external
    assert any("postman.com/about" in u for u in urls)
    assert any("postman.com/team" in u for u in urls)
    assert not any("external.com" in u for u in urls)
    # homepage first
    assert urls[0].rstrip("/") == "https://postman.com"

@pytest.mark.asyncio
async def test_discover_includes_fallback_priority_paths():
    # no homepage html, no sitemap -> should fallback to priority paths
    urls = await discover_urls("example-xyz-totally-not-real.test", homepage_html=None, homepage_url="https://example-xyz-totally-not-real.test/")
    # should contain homepage + at least 2 fallback
    assert urls[0].rstrip("/") == "https://example-xyz-totally-not-real.test"
    assert len(urls) >= 3
    # contains known paths
    assert any("/about" in u for u in urls)

def test_score_url_prefers_team_about():
    assert _score_url("https://example.com/team") > _score_url("https://example.com/blog/random-post-123")
    assert _score_url("https://example.com/about") > _score_url("https://example.com/legal")

@pytest.mark.asyncio
async def test_discover_dedup_and_limit():
    # many duplicate links
    html = '<a href="/about">About</a>' * 10 + '<a href="/team">Team</a>' * 5
    urls = await discover_urls("example.com", homepage_html=html, homepage_url="https://example.com/")
    assert len(urls) == len(set(u.lower().rstrip("/") for u in urls))
    assert len(urls) <= 6
