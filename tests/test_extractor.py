import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.extraction.extractor import extract, _extract_json_from_text, _heuristic_confidence_cap
from src.extraction.prompts import build_user_prompt, SYSTEM_PROMPT
from src.models import CompanyEnrichment

def test_build_user_prompt_contains_source_urls():
    pages = {"https://example.com/": "# Hello", "https://example.com/about": "About us"}
    prompt = build_user_prompt("example.com", pages)
    assert "[SOURCE: https://example.com/]" in prompt
    assert "[SOURCE: https://example.com/about]" in prompt
    assert "example.com" in prompt
    # no raw html
    assert "<html" not in prompt.lower()

def test_extract_json_from_text_handles_wrapped():
    wrapped = '```json\n{"domain":"example.com","confidence_score":0.8}\n```'
    data = _extract_json_from_text(wrapped)
    assert data["domain"] == "example.com"
    # plain
    plain = '{"domain":"example.com","confidence_score":0.5}'
    assert _extract_json_from_text(plain)["confidence_score"] == 0.5
    # extra text surrounding
    extra = 'Here is JSON: {"domain":"x.com","confidence_score":0.7} hope it helps'
    assert _extract_json_from_text(extra)["domain"] == "x.com"

def test_heuristic_confidence_cap():
    base = CompanyEnrichment(domain="example.com", company_overview=None, source_pages=["https://example.com/"])  # type: ignore
    cap = _heuristic_confidence_cap(base, num_sources=1, num_errors=0)
    assert cap <= 0.6  # missing overview caps

    base2 = CompanyEnrichment(domain="example.com", company_overview="Overview", source_pages=["https://example.com/","https://example.com/about"])  # type: ignore
    cap2 = _heuristic_confidence_cap(base2, num_sources=2, num_errors=0)
    # With 2 sources but no leadership/emails/linkedin, heuristic caps to 0.7 (overview-only)
    assert cap2 == 0.7
    # Full case with leadership would allow 1.0
    base2_full = CompanyEnrichment(domain="example.com", company_overview="Overview", leadership=[{"name": "Jane Doe", "role": "CEO"}], source_pages=["https://example.com/","https://example.com/about"])  # type: ignore
    assert _heuristic_confidence_cap(base2_full, 2, 0) == 1.0

    base3 = CompanyEnrichment(domain="example.com", company_overview="Overview", source_pages=[])  # type: ignore
    assert _heuristic_confidence_cap(base3, 0, 0) == 0.0

@pytest.mark.asyncio
async def test_extract_with_mock_provider():
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()
    pages = {"https://example.com/": "# Example Inc\nWe build APIs.\nContact: support@example.com\nhttps://linkedin.com/company/example"}
    res = await extract("example.com", pages, errors=[])
    assert res.domain == "example.com"
    assert res.source_pages is not None
    assert 0 <= res.confidence_score <= 1
    assert res.contact_emails == ["support@example.com"] or "support@example.com" in [str(x) for x in res.contact_emails]
    # cleanup
    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_extract_evidence_aware_source_pages_filtered():
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()
    pages = {"https://example.com/": "content", "https://example.com/about": "about content"}
    res = await extract("example.com", pages, errors=["fetch error"])
    # source_pages should be subset of input keys
    for u in res.source_pages:
        assert str(u) in pages
    assert "fetch error" in res.errors
    os.environ.pop("LLM_PROVIDER", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_extract_with_mocked_openai_success():
    """Verify real OpenAI path via mocked AsyncOpenAI (structured-output)"""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test-mock"
    os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    cfg.reset_settings()

    # Mock response payload that LLM would return
    mock_payload = {
        "domain": "example.com",
        "company_overview": "Example builds APIs for developers.",
        "target_audience_icp": "API developers",
        "contact_emails": ["support@example.com"],
        "leadership": [{"name": "Jane Doe", "role": "CEO", "linkedin_url": "https://linkedin.com/in/janedoe"}],
        "linkedin_urls": ["https://linkedin.com/company/example"],
        "source_pages": ["https://example.com/"],
        "confidence_score": 0.85,
        "errors": []
    }

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(mock_payload)))]

    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_client_cls.return_value = mock_client

        pages = {"https://example.com/": "# Example\nWe build APIs.\nsupport@example.com"}
        res = await extract("example.com", pages, errors=[])

        # Verify OpenAI was called with json_schema
        assert mock_client.chat.completions.create.called
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert "response_format" in call_kwargs
        # Verify Pydantic validation and evidence-aware filtering
        assert res.domain == "example.com"
        assert res.company_overview == "Example builds APIs for developers."
        assert res.confidence_score <= 0.85  # may be capped but should be <= original
        assert str(res.source_pages[0]) == "https://example.com/"

    # cleanup
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_MODEL", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_extract_confidence_capped_after_llm():
    """LLM returns 1.0 but heuristic caps because only overview present"""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    payload = {
        "domain": "example.com",
        "company_overview": "Overview only",
        "target_audience_icp": None,
        "contact_emails": [],
        "leadership": [],
        "linkedin_urls": [],
        "source_pages": ["https://example.com/"],
        "confidence_score": 1.0,
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_client_cls.return_value = mock_client
        pages = {"https://example.com/": "overview only content"}
        res = await extract("example.com", pages, errors=[])
        # Should be capped to 0.7 because only overview, single source
        assert res.confidence_score < 1.0
        assert res.confidence_score <= 0.7

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()
