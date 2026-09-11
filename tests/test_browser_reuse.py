import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from src.models import FetchResult

@pytest.mark.asyncio
async def test_fetch_many_uses_one_browser_launch():
    """One browser launch occurs for a fetch_many() call (not per URL)."""
    from src.scraping.fetcher import fetch_many

    # Mock Playwright and Browser
    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_browser.new_context = AsyncMock()

    # Mock context and page
    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock()
    mock_context.close = AsyncMock()
    mock_browser.new_context.return_value = mock_context

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=AsyncMock(status=200))
    mock_page.title = AsyncMock(return_value="Test")
    mock_page.content = AsyncMock(return_value="<html><body><h1>Hi</h1></body></html>")
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.close = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="")
    mock_context.new_page.return_value = mock_page

    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)

    with patch("src.scraping.fetcher.async_playwright", return_value=mock_playwright) as mock_pw_func:
        # Mock get_settings to avoid env issues
        urls = ["https://example.com/page1", "https://example.com/page2", "https://example.com/page3"]
        results = await fetch_many(urls, max_concurrency=2)

        # Should have launched browser exactly once for this fetch_many call
        assert mock_playwright.chromium.launch.call_count == 1, f"Expected 1 launch, got {mock_playwright.chromium.launch.call_count}"
        # async_playwright should have been called once
        assert mock_pw_func.call_count == 1
        # Should have 3 results
        assert len(results) == 3
        # Browser should be closed after fetch_many
        assert mock_browser.close.call_count == 1
        assert mock_playwright.stop.call_count == 1

@pytest.mark.asyncio
async def test_multiple_urls_use_shared_browser_and_separate_contexts():
    """Multiple URLs use the shared browser, each gets a separate BrowserContext."""
    from src.scraping.fetcher import fetch_many

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()

    # Track contexts created
    contexts = []

    async def create_context(*args, **kwargs):
        ctx = AsyncMock()
        ctx.new_page = AsyncMock()
        ctx.close = AsyncMock()
        # Create page mock
        page = AsyncMock()
        page.goto = AsyncMock(return_value=AsyncMock(status=200))
        page.title = AsyncMock(return_value="Test")
        page.content = AsyncMock(return_value="<html><body>content</body></html>")
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()
        page.evaluate = AsyncMock(return_value="")
        ctx.new_page.return_value = page
        contexts.append(ctx)
        return ctx

    mock_browser.new_context.side_effect = create_context

    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)

    with patch("src.scraping.fetcher.async_playwright", return_value=mock_playwright):
        urls = ["https://a.com/1", "https://a.com/2", "https://a.com/3", "https://a.com/4"]
        results = await fetch_many(urls, max_concurrency=2)

        # Each URL should have gotten a separate context (4 contexts)
        assert len(contexts) == 4, f"Expected 4 contexts, got {len(contexts)}"
        # Each context should have been closed
        for ctx in contexts:
            assert ctx.close.call_count == 1
        # All should succeed
        assert len(results) == 4
        assert all(r.error is None or "empty" not in (r.error or "") for r in results)

@pytest.mark.asyncio
async def test_failed_url_does_not_terminate_other_urls_with_shared_browser():
    """A failure in one context/page must not close the shared browser or terminate other fetches."""
    from src.scraping.fetcher import fetch_many

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()

    call_count = 0

    async def create_context(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        ctx = AsyncMock()
        ctx.close = AsyncMock()
        page = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()
        # Make the second URL fail with 404, others succeed
        if call_count == 2:
            # Simulate 404 for second URL
            page.goto = AsyncMock(return_value=AsyncMock(status=404))
            page.title = AsyncMock(return_value="Not Found")
            page.content = AsyncMock(return_value="<html>404</html>")
        else:
            page.goto = AsyncMock(return_value=AsyncMock(status=200))
            page.title = AsyncMock(return_value="OK")
            page.content = AsyncMock(return_value="<html><body><h1>OK</h1><p>content</p></body></html>")
            page.evaluate = AsyncMock(return_value="")
        ctx.new_page = AsyncMock(return_value=page)
        return ctx

    mock_browser.new_context.side_effect = create_context

    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)

    with patch("src.scraping.fetcher.async_playwright", return_value=mock_playwright):
        urls = ["https://example.com/ok1", "https://example.com/bad", "https://example.com/ok2"]
        results = await fetch_many(urls, max_concurrency=3)

        assert len(results) == 3
        # bad should be 404, others 200
        assert results[0].status_code == 200
        assert results[1].status_code == 404
        assert "404" in (results[1].error or "")
        assert results[2].status_code == 200
        # Browser should still be closed only once at the end, not per failure
        assert mock_browser.close.call_count == 1
        # All contexts should be closed even for failed URL
        assert call_count == 3

@pytest.mark.asyncio
async def test_browser_context_page_cleanup_occurs():
    """Resource cleanup must be guaranteed: page closes, context closes, browser closes, playwright stops."""
    from src.scraping.fetcher import fetch_many

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()

    contexts = []
    pages = []

    async def create_context(*args, **kwargs):
        ctx = AsyncMock()
        ctx.close = AsyncMock()
        contexts.append(ctx)
        page = AsyncMock()
        page.goto = AsyncMock(return_value=AsyncMock(status=200))
        page.title = AsyncMock(return_value="OK")
        page.content = AsyncMock(return_value="<html><body>hi</body></html>")
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()
        page.evaluate = AsyncMock(return_value="")
        ctx.new_page = AsyncMock(return_value=page)
        pages.append(page)
        return ctx

    mock_browser.new_context.side_effect = create_context

    mock_playwright = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright.stop = AsyncMock()
    mock_playwright.start = AsyncMock(return_value=mock_playwright)

    with patch("src.scraping.fetcher.async_playwright", return_value=mock_playwright) as mock_pw_func:
        urls = ["https://example.com/a", "https://example.com/b"]
        results = await fetch_many(urls, max_concurrency=2)

        # Each page and context should be closed
        assert len(pages) == 2
        assert len(contexts) == 2
        for p in pages:
            assert p.close.call_count == 1, "page.close not called"
        for c in contexts:
            assert c.close.call_count == 1, "context.close not called"
        # Browser and playwright should be closed/stopped exactly once
        assert mock_browser.close.call_count == 1
        assert mock_playwright.stop.call_count == 1
        assert mock_pw_func.call_count == 1

@pytest.mark.asyncio
async def test_fetch_one_uses_shared_browser_directly():
    """fetch_one with shared Browser should create new context/page but not launch new Browser."""
    from src.scraping.fetcher import fetch_one

    mock_browser = AsyncMock()
    mock_browser.close = AsyncMock()
    mock_context = AsyncMock()
    mock_context.close = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=AsyncMock(status=200))
    mock_page.title = AsyncMock(return_value="Test")
    mock_page.content = AsyncMock(return_value="<html><body><h1>Test</h1></body></html>")
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.close = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="")
    mock_context.new_page = AsyncMock(return_value=mock_page)

    # Should use the shared browser, not launch a new one
    with patch("src.scraping.fetcher.async_playwright") as mock_pw:
        result = await fetch_one("https://example.com/test", browser=mock_browser)
        # Should not have called async_playwright (since browser provided)
        assert mock_pw.call_count == 0
        # Should have created a new context
        assert mock_browser.new_context.call_count == 1
        assert mock_context.new_page.call_count == 1
        assert result.status_code == 200
        # Context and page should be closed, but browser should NOT be closed (caller owns it)
        assert mock_context.close.call_count == 1
        assert mock_page.close.call_count == 1
        assert mock_browser.close.call_count == 0  # not closed by fetch_one
