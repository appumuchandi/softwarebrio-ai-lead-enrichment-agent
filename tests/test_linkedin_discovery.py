import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from src.models import CompanyEnrichment, LeadershipMember
from src.discovery.linkedin import (
    discover_linkedin_profile,
    discover_linkedin_profile_async,
    enrich_leadership_with_linkedin,
    _extract_verified_linkedin,
    _verify_candidate,
    MAX_LINKEDIN_SEARCHES_PER_DOMAIN,
)


# Helper: build minimal DuckDuckGo HTML containing result
def _ddg_html_with_results(results):
    """
    results: list of dict with keys: href, title, snippet
    Build minimal html that our parser will inspect.
    """
    parts = ['<html><body>']
    for r in results:
        href = r.get("href", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        parts.append(
            f'<div class="result"><a class="result__url" href="{href}">{title}</a>'
            f'<a class="result__snippet">{snippet}</a></div>'
        )
    parts.append('</body></html>')
    return "\n".join(parts)


# 1. valid LinkedIn result
def test_discover_valid_linkedin_result():
    html = _ddg_html_with_results([
        {
            "href": "https://www.linkedin.com/in/abhinavasthana",
            "title": "Abhinav Asthana - Postman - LinkedIn",
            "snippet": "Abhinav Asthana is CEO and Co-Founder at Postman. View profile on LinkedIn."
        }
    ])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Abhinav Asthana", "Postman", "postman.com")
        assert url == "https://www.linkedin.com/in/abhinavasthana"


@pytest.mark.asyncio
async def test_discover_valid_linkedin_result_async():
    html = _ddg_html_with_results([
        {
            "href": "https://www.linkedin.com/in/janedoe",
            "title": "Jane Doe - Example - LinkedIn",
            "snippet": "Jane Doe is CEO at Example. View her LinkedIn profile for Example company."
        }
    ])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        url = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url == "https://www.linkedin.com/in/janedoe"


# 2. no LinkedIn result
def test_discover_no_linkedin_result():
    html = _ddg_html_with_results([
        {"href": "https://example.com/about", "title": "About Example", "snippet": "We build APIs."}
    ])
    # Also add html with no linkedin at all
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url is None


@pytest.mark.asyncio
async def test_discover_no_linkedin_result_async():
    html = "<html><body>No results found for your query.</body></html>"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        url = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url is None


# 3. unrelated LinkedIn result (should not assign)
def test_discover_unrelated_linkedin_result():
    html = _ddg_html_with_results([
        {
            "href": "https://www.linkedin.com/in/johnsmith",
            "title": "John Smith - Example - LinkedIn",
            "snippet": "John Smith is CTO at Example. View profile."
        }
    ])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        # Search for Jane Doe, but result is John Smith -> should be rejected
        url = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url is None

    # Also test company mismatch: same name but wrong company
    html2 = _ddg_html_with_results([
        {
            "href": "https://www.linkedin.com/in/janedoe",
            "title": "Jane Doe - OtherCorp - LinkedIn",
            "snippet": "Jane Doe works at OtherCorp as Senior Engineer."
        }
    ])
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.text = html2
    mock_client2 = MagicMock()
    mock_client2.get.return_value = mock_resp2
    mock_client2.__enter__ = MagicMock(return_value=mock_client2)
    mock_client2.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client2):
        url2 = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        # Our verification requires company token in context; OtherCorp snippet has OtherCorp not Example -> should fail
        assert url2 is None


@pytest.mark.asyncio
async def test_unrelated_linkedin_rejected_async():
    html = _ddg_html_with_results([
        {"href": "https://www.linkedin.com/in/otherperson", "title": "Other Person - Other - LinkedIn", "snippet": "Other Person at Other"}
    ])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        url = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url is None


# 4. search timeout
def test_discover_search_timeout_returns_none():
    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.ReadTimeout("timeout")
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url is None


@pytest.mark.asyncio
async def test_discover_timeout_async():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        url = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url is None

    # Also test ConnectTimeout
    mock_client2 = AsyncMock()
    mock_client2.get = AsyncMock(side_effect=httpx.ConnectTimeout("connect timeout"))
    mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
    mock_client2.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client2):
        url2 = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url2 is None


# 5. HTTP failure
def test_discover_http_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)

    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url is None

    # Also test 403
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 403
    mock_resp2.text = "Forbidden"
    mock_client2 = MagicMock()
    mock_client2.get.return_value = mock_resp2
    mock_client2.__enter__ = MagicMock(return_value=mock_client2)
    mock_client2.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client2):
        url2 = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url2 is None

    # HTTPError exception
    mock_client3 = MagicMock()
    mock_client3.get.side_effect = httpx.HTTPError("network error")
    mock_client3.__enter__ = MagicMock(return_value=mock_client3)
    mock_client3.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client3):
        url3 = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url3 is None


@pytest.mark.asyncio
async def test_http_failure_async():
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        url = await discover_linkedin_profile_async("Jane Doe", "Example", "example.com")
        assert url is None


# 6. malformed search response
def test_malformed_search_response():
    # Empty text
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = ""
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        assert discover_linkedin_profile("Jane Doe", "Example", "example.com") is None

    # None text
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.text = None
    mock_client2 = MagicMock()
    mock_client2.get.return_value = mock_resp2
    mock_client2.__enter__ = MagicMock(return_value=mock_client2)
    mock_client2.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client2):
        assert discover_linkedin_profile("Jane Doe", "Example", "example.com") is None

    # Garbled HTML without linkedin
    mock_resp3 = MagicMock()
    mock_resp3.status_code = 200
    mock_resp3.text = "<html><body>$$$ <<< >>> not valid html <<<</body></html>"
    mock_client3 = MagicMock()
    mock_client3.get.return_value = mock_resp3
    mock_client3.__enter__ = MagicMock(return_value=mock_client3)
    mock_client3.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client3):
        assert discover_linkedin_profile("Jane Doe", "Example", "example.com") is None

    # Direct test of _extract_verified_linkedin with malformed
    assert _extract_verified_linkedin("", "Jane Doe", "Example") is None
    assert _extract_verified_linkedin(None, "Jane Doe", "Example") is None  # type: ignore
    assert _extract_verified_linkedin("<html></html>", "", "Example") is None
    # Malformed linkedin URL (company page not /in/)
    html_no_in = '<html><a href="https://linkedin.com/company/example">company</a></html>'
    assert _extract_verified_linkedin(html_no_in, "Jane Doe", "Example") is None


@pytest.mark.asyncio
async def test_malformed_async():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = ""  # empty
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.AsyncClient", return_value=mock_client):
        assert await discover_linkedin_profile_async("Jane Doe", "Example", "example.com") is None


# 7. duplicate person/search prevention
@pytest.mark.asyncio
async def test_duplicate_person_search_prevention():
    enrichment = CompanyEnrichment(
        domain="example.com",
        leadership=[
            {"name": "Jane Doe", "role": "CEO"},  # type: ignore
            {"name": "Jane Doe", "role": "CEO"},  # duplicate
            {"name": "JANE DOE", "role": "CTO"},  # same normalized
            {"name": "John Smith", "role": "CTO"},  # distinct
        ],
        linkedin_urls=[],  # type: ignore
        source_pages=["https://example.com/"],  # type: ignore
        confidence_score=0.7,
    )
    # Ensure linkedin is None initially
    for m in enrichment.leadership:
        assert m.linkedin_url is None

    calls = []

    async def fake_discover(name, company, domain):
        calls.append(name.lower())
        return f"https://www.linkedin.com/in/{name.lower().replace(' ', '')}"

    # Need to ensure our fake matches verification? But enrich_leadership_with_linkedin calls discover_linkedin_profile_async
    # We patch that to bypass verification, so any returned URL will be assigned
    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com", company_name="Example")

    # Should have searched only 2 unique names (Jane Doe, John Smith), not 3 duplicates
    assert len(calls) == 2
    assert calls.count("jane doe") == 1
    assert calls.count("john smith") == 1
    # Both members with duplicate name should get same URL? Our code deduplicates candidates before search,
    # so only first Jane Doe is in candidates, second duplicate skipped. Thus enriched count is 2 (unique)
    # Check that at least one Jane Doe got filled
    jane_members = [m for m in enrichment.leadership if m.name.lower() == "jane doe"]
    # First Jane should be filled, duplicates should remain None because they were not searched (deduplicated)
    # Implementation deduplicates by name and only searches first occurrence, so second duplicate stays None
    assert jane_members[0].linkedin_url is not None
    # Second duplicate should stay None (since we skipped search for duplicate)
    # This verifies duplicate prevention – not searching same person twice
    assert jane_members[1].linkedin_url is None or str(jane_members[1].linkedin_url) == str(jane_members[0].linkedin_url)


# 8. maximum search limit (strict 5)
@pytest.mark.asyncio
async def test_maximum_search_limit():
    members = [{"name": f"Person {i}", "role": "Engineer"} for i in range(10)]  # 10 members
    enrichment = CompanyEnrichment(
        domain="example.com",
        leadership=members,  # type: ignore
        linkedin_urls=[],  # type: ignore
        source_pages=["https://example.com/"],  # type: ignore
        confidence_score=0.7,
    )
    calls = []

    async def fake_discover(name, company, domain):
        calls.append(name)
        return f"https://www.linkedin.com/in/{name.lower().replace(' ', '-')}"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com", company_name="Example")

    assert len(calls) == MAX_LINKEDIN_SEARCHES_PER_DOMAIN
    assert len(calls) == 5
    assert enriched == 5
    # First 5 should be enriched, rest should remain None
    for i in range(5):
        assert enrichment.leadership[i].linkedin_url is not None
    for i in range(5, 10):
        assert enrichment.leadership[i].linkedin_url is None


# Also test that limit respects config param
@pytest.mark.asyncio
async def test_maximum_search_limit_custom():
    members = [{"name": f"Person {i}", "role": "Engineer"} for i in range(4)]
    enrichment = CompanyEnrichment(domain="example.com", leadership=members, linkedin_urls=[], source_pages=["https://example.com/"], confidence_score=0.7)  # type: ignore
    calls = []

    async def fake_discover(name, company, domain):
        calls.append(name)
        return f"https://www.linkedin.com/in/{name.lower().replace(' ', '-')}"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com", max_searches=2)

    assert len(calls) == 2
    assert enriched == 2


# 9. existing LinkedIn URL is not searched/replaced
@pytest.mark.asyncio
async def test_existing_linkedin_not_searched_or_replaced():
    enrichment = CompanyEnrichment(
        domain="example.com",
        leadership=[
            {"name": "Jane Doe", "role": "CEO", "linkedin_url": "https://www.linkedin.com/in/janedoe"},  # type: ignore
            {"name": "John Smith", "role": "CTO"},  # type: ignore
        ],
        linkedin_urls=["https://www.linkedin.com/in/janedoe"],  # type: ignore
        source_pages=["https://example.com/"],  # type: ignore
        confidence_score=0.7,
    )
    calls = []

    async def fake_discover(name, company, domain):
        calls.append(name)
        # Try to return a different URL for Jane (should not be called)
        return "https://www.linkedin.com/in/should-not-be-used"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com", company_name="Example")

    # Should only search John Smith
    assert calls == ["John Smith"]
    # Existing should be unchanged
    assert str(enrichment.leadership[0].linkedin_url) == "https://www.linkedin.com/in/janedoe"
    # John should now have linkedin
    assert enrichment.leadership[1].linkedin_url is not None
    assert "should-not-be-used" in str(enrichment.leadership[1].linkedin_url)
    # Top-level linkedin_urls should be updated consistently (deduped)
    assert any("janedoe" in str(u).lower() for u in enrichment.linkedin_urls)
    assert any("should-not-be-used" in str(u).lower() for u in enrichment.linkedin_urls)
    # Existing not duplicated
    assert len([u for u in enrichment.linkedin_urls if "janedoe" in str(u).lower()]) == 1


# 10. LinkedIn lookup failure does not fail the company/domain
@pytest.mark.asyncio
async def test_linkedin_lookup_failure_does_not_fail_domain():
    enrichment = CompanyEnrichment(
        domain="example.com",
        leadership=[
            {"name": "Jane Doe", "role": "CEO"},  # type: ignore
            {"name": "John Smith", "role": "CTO"},  # type: ignore
        ],
        linkedin_urls=[],  # type: ignore
        source_pages=["https://example.com/"],  # type: ignore
        confidence_score=0.8,
    )

    async def fake_discover_fail(name, company, domain):
        # Simulate all failures: timeout, HTTP error, etc. -> return None
        return None

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover_fail):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com")

    assert enriched == 0
    # Both remain None, but enrichment still valid, confidence unchanged, no crash
    assert enrichment.leadership[0].linkedin_url is None
    assert enrichment.leadership[1].linkedin_url is None
    assert enrichment.domain == "example.com"
    assert 0 <= enrichment.confidence_score <= 1
    CompanyEnrichment.model_validate(enrichment.model_dump())

    # Also test exception path
    async def fake_discover_exception(name, company, domain):
        raise httpx.TimeoutException("timeout")

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover_exception):
        enriched2 = await enrich_leadership_with_linkedin(enrichment, "example.com")
        assert enriched2 == 0
        # Still valid
        assert enrichment.domain == "example.com"

    # Test partial failure: one succeeds, one fails
    async def fake_partial(name, company, domain):
        if "Jane" in name:
            raise Exception("network failure")
        return "https://www.linkedin.com/in/johnsmith"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_partial):
        enriched3 = await enrich_leadership_with_linkedin(enrichment, "example.com")
        # Should have enriched John despite Jane failure
        assert enriched3 == 1 or enriched3 == 1  # at least John
        # Check that enrichment is still valid
        CompanyEnrichment.model_validate(enrichment.model_dump())


# Additional: test integration with orchestrator – fail-soft
@pytest.mark.asyncio
async def test_orchestrator_integration_fail_soft():
    """Verify that orchestrator enrich_one_domain still succeeds even if LinkedIn discovery fails."""
    from src.orchestrator import enrich_one_domain
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()

    # Mock fetcher to produce a page with leadership name but no linkedin on site
    # Then mock LLM extraction to return grounded leadership with missing linkedin
    # Then force linkedin discovery to raise
    from src.models import FetchResult

    async def fake_fetch_one(url, browser=None, *a, **kw):
        return FetchResult(url=url, status_code=200, markdown="# Hi\nWe build APIs\n## Leadership\nJane Doe - CEO", html="<h1>hi</h1>")

    async def fake_fetch_many(urls, max_concurrency=4, browser=None, *a, **kw):
        return [await fake_fetch_one(u) for u in urls]

    async def fake_discover_urls(domain, homepage_html=None, homepage_url=None):
        return [f"https://{domain}/", f"https://{domain}/about"]

    # Mock extract to return a grounded leadership with missing linkedin
    async def fake_extract(domain, pages_markdown, errors=None):
        return CompanyEnrichment(
            domain=domain,
            company_overview="Test overview",
            leadership=[{"name": "Jane Doe", "role": "CEO"}],  # type: ignore
            linkedin_urls=[],  # type: ignore
            source_pages=list(pages_markdown.keys()),  # type: ignore
            confidence_score=0.7,
            errors=errors or [],
        )

    # Force linkedin discovery to fail
    async def fake_linkedin_fail(*args, **kwargs):
        raise Exception("DuckDuckGo blocking")

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
         patch("src.orchestrator.discover_urls", side_effect=fake_discover_urls), \
         patch("src.extraction.extractor.extract", side_effect=fake_extract) if False else patch("src.orchestrator.extract", side_effect=fake_extract), \
         patch("src.discovery.linkedin.enrich_leadership_with_linkedin", side_effect=fake_linkedin_fail):

        # Also patch where orchestrator imports (it imports inside function, so patch that path)
        # Need to patch both import locations
        with patch("src.discovery.linkedin.enrich_leadership_with_linkedin", side_effect=fake_linkedin_fail):
            res = await enrich_one_domain("example.com")
            # Should still succeed, not crash, even though linkedin discovery threw
            assert res.domain == "example.com"
            assert isinstance(res.leadership, list)
            # Should have at least Jane Doe with null linkedin (since discovery failed, remains null)
            # No exception propagated
            CompanyEnrichment.model_validate(res.model_dump())

    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()


# Test that never invents URL: only returns URLs present in search HTML
def test_never_invents_linkedin_url():
    html = _ddg_html_with_results([
        {"href": "https://example.com/not-linkedin", "title": "Jane Doe Example", "snippet": "Jane Doe at Example"}
    ])
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Jane Doe", "Example", "example.com")
        assert url is None  # Must not invent


def test_clean_linkedin_url_normalizes():
    from src.discovery.linkedin import _clean_linkedin_url
    assert _clean_linkedin_url("https://www.linkedin.com/in/janedoe?trk=profile") == "https://www.linkedin.com/in/janedoe"
    assert _clean_linkedin_url("https://www.linkedin.com/in/janedoe/") == "https://www.linkedin.com/in/janedoe"
    assert _clean_linkedin_url("http://linkedin.com/in/janedoe.,") == "https://www.linkedin.com/in/janedoe"
    assert _clean_linkedin_url("https://www.linkedin.com/in/jane-doe#section") == "https://www.linkedin.com/in/jane-doe"


def test_verify_candidate_helpers():
    # Valid case
    ctx = "Jane Doe is CEO at Example Company. View LinkedIn profile."
    url = "https://www.linkedin.com/in/janedoe"
    assert _verify_candidate(ctx, url, "Jane Doe", "Example") is True
    # Wrong name -> false
    assert _verify_candidate(ctx, url, "John Smith", "Example") is False
    # Wrong company -> false
    assert _verify_candidate("Jane Doe at OtherCorp", url, "Jane Doe", "Example") is False
    # Name in slug passes even if context weak
    ctx2 = " Profile at Example "
    url2 = "https://www.linkedin.com/in/jane-doe"
    # context contains Example and slug contains jane doe -> should pass
    assert _verify_candidate(ctx2, url2, "Jane Doe", "Example") is True


# Test linkedin_urls consistent update
@pytest.mark.asyncio
async def test_linkedin_urls_updated_consistently():
    enrichment = CompanyEnrichment(
        domain="example.com",
        leadership=[
            {"name": "Jane Doe", "role": "CEO"},  # type: ignore
            {"name": "John Smith", "role": "CTO"},  # type: ignore
        ],
        linkedin_urls=["https://linkedin.com/company/example"],  # type: ignore
        source_pages=["https://example.com/"],  # type: ignore
        confidence_score=0.7,
    )

    async def fake_discover(name, company, domain):
        if name == "Jane Doe":
            return "https://www.linkedin.com/in/janedoe"
        return "https://www.linkedin.com/in/johnsmith"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake_discover):
        await enrich_leadership_with_linkedin(enrichment, "example.com", company_name="Example")

    # Should contain company + both persons, deduped
    urls = [str(u).lower() for u in enrichment.linkedin_urls]
    assert any("company/example" in u for u in urls)
    assert any("janedoe" in u for u in urls)
    assert any("johnsmith" in u for u in urls)
    assert len(urls) == len(set(urls))  # deduped


# Test that empty leadership does not trigger search
@pytest.mark.asyncio
async def test_empty_leadership_no_search():
    enrichment = CompanyEnrichment(domain="example.com", leadership=[], linkedin_urls=[], source_pages=["https://example.com/"], confidence_score=0.5)  # type: ignore
    calls = []

    async def fake(name, company, domain):
        calls.append(name)
        return "https://linkedin.com/in/test"

    with patch("src.discovery.linkedin.discover_linkedin_profile_async", side_effect=fake):
        enriched = await enrich_leadership_with_linkedin(enrichment, "example.com")
    assert enriched == 0
    assert calls == []


# Test that function returns None for missing name/company
def test_missing_name_or_company_returns_none():
    assert discover_linkedin_profile("", "Example", "example.com") is None
    assert discover_linkedin_profile("   ", "Example", "example.com") is None
    assert discover_linkedin_profile("Jane Doe", "", "") is None

@pytest.mark.asyncio
async def test_missing_name_or_company_async():
    assert await discover_linkedin_profile_async("", "Example", "example.com") is None
    assert await discover_linkedin_profile_async("Jane Doe", "", "") is None


# --- Regression tests for DuckDuckGo parser fix (encoded uddg, bare www, normal https) ---

def test_parser_encoded_uddg_linkedin_url():
    """Encoded DuckDuckGo uddg redirect must be decoded, normalized, and verified."""
    # Real DDG snippet: href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fabhinavasthana&rut=..."
    html = '''
    <html><body>
    <div class="result">
      <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fabhinavasthana&amp;rut=abc">www.linkedin.com/in/abhinavasthana</a>
      <a class="result__snippet">CEO and Founder at <b>Postman</b> — View <b>Abhinav Asthana</b>'s profile on LinkedIn.</a>
    </div>
    </body></html>
    '''
    result = _extract_verified_linkedin(html, "Abhinav Asthana", "Postman")
    assert result == "https://www.linkedin.com/in/abhinavasthana"

    # Via full discover path (mocked httpx) — sync
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    with patch("src.discovery.linkedin.httpx.Client", return_value=mock_client):
        url = discover_linkedin_profile("Abhinav Asthana", "Postman", "postman.com")
        assert url == "https://www.linkedin.com/in/abhinavasthana"


def test_parser_bare_www_linkedin_url():
    """Bare www.linkedin.com/in/... without scheme must be recognized and normalized to https://www..."""
    html = '''
    <html><body>
    <div class="result">
      <a href="www.linkedin.com/in/janedoe">www.linkedin.com/in/janedoe</a>
      <span>Jane Doe is CEO at Example — View profile.</span>
    </div>
    </body></html>
    '''
    result = _extract_verified_linkedin(html, "Jane Doe", "Example")
    assert result == "https://www.linkedin.com/in/janedoe"

    # Also test bare without www: linkedin.com/in/janedoe
    html2 = '<html><body><a href="linkedin.com/in/janedoe">linkedin.com/in/janedoe</a> Jane Doe at Example</body></html>'
    result2 = _extract_verified_linkedin(html2, "Jane Doe", "Example")
    assert result2 == "https://www.linkedin.com/in/janedoe"


def test_parser_normal_https_linkedin_url():
    """Normal https://www.linkedin.com/in/... must still work and normalize."""
    html = '''
    <html><body>
    <div class="result"><a href="https://www.linkedin.com/in/johndoe">https://www.linkedin.com/in/johndoe</a> John Doe is CTO at Example</div>
    </body></html>
    '''
    result = _extract_verified_linkedin(html, "John Doe", "Example")
    assert result == "https://www.linkedin.com/in/johndoe"

    # http variant should normalize to https
    html2 = '<html><body><a href="http://www.linkedin.com/in/janedoe2">http://www.linkedin.com/in/janedoe2</a> Jane Doe2 at Example</body></html>'
    # Use Jane Doe2 to avoid token mismatch
    result2 = _extract_verified_linkedin(html2, "Jane Doe2", "Example")
    assert result2 == "https://www.linkedin.com/in/janedoe2"

    # https://linkedin.com/in/... without www should normalize to www
    html3 = '<html><body><a href="https://linkedin.com/in/janedoe3">https://linkedin.com/in/janedoe3</a> Jane Doe3 at Example</body></html>'
    result3 = _extract_verified_linkedin(html3, "Jane Doe3", "Example")
    assert result3 == "https://www.linkedin.com/in/janedoe3"


def test_parser_unrelated_linkedin_rejected_even_with_new_parser():
    """Even with new parser, unrelated LinkedIn result must be rejected via verification."""
    # Encoded uddg for John Smith, but searching Jane Doe at Example
    html = '''
    <html><body>
    <div class="result">
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fjohnsmith">www.linkedin.com/in/johnsmith</a>
      <span>John Smith is CTO at Example — View profile.</span>
    </div>
    </body></html>
    '''
    result = _extract_verified_linkedin(html, "Jane Doe", "Example")
    assert result is None

    # Bare www but wrong person
    html2 = '<html><body><a href="www.linkedin.com/in/otherperson">www.linkedin.com/in/otherperson</a> Other Person at Example</body></html>'
    result2 = _extract_verified_linkedin(html2, "Jane Doe", "Example")
    assert result2 is None


def test_parser_verification_still_required():
    """Exact name + company verification must still be required — never accept merely because URL exists."""
    # Valid LinkedIn URL present but company token missing in context
    html = '''
    <html><body>
    <div class="result">
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fjanedoe">www.linkedin.com/in/janedoe</a>
      <span>Jane Doe is CEO at OtherCorp — View profile.</span>
    </div>
    </body></html>
    '''
    result = _extract_verified_linkedin(html, "Jane Doe", "Example")
    assert result is None  # company mismatch -> rejected

    # Name missing in context and also not in slug sufficiently (partial)
    html2 = '''
    <html><body>
    <div class="result">
      <a href="https://www.linkedin.com/in/janedoe">https://www.linkedin.com/in/janedoe</a>
      <span>CEO at Example — View profile.</span>
    </div>
    </body></html>
    '''
    # Context has Example but no Jane Doe tokens; slug does contain janedoe -> our verification allows slug fallback, so this would pass
    # To test strict failure, use slug that doesn't match name: linkedin.com/in/otherperson but context has Example
    html3 = '''
    <html><body>
    <div class="result">
      <a href="https://www.linkedin.com/in/otherperson">https://www.linkedin.com/in/otherperson</a>
      <span>CEO at Example</span>
    </div>
    </body></html>
    '''
    result3 = _extract_verified_linkedin(html3, "Jane Doe", "Example")
    assert result3 is None  # name not in context nor slug -> rejected

    # Positive case must still pass
    html_ok = '''
    <html><body>
    <div class="result">
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fjanedoe">www.linkedin.com/in/janedoe</a>
      <span>Jane Doe is CEO at Example — View profile.</span>
    </div>
    </body></html>
    '''
    result_ok = _extract_verified_linkedin(html_ok, "Jane Doe", "Example")
    assert result_ok == "https://www.linkedin.com/in/janedoe"
