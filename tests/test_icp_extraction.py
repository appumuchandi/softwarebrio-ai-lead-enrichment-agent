import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from src.extraction.extractor import extract

def test_system_prompt_contains_icp_cues():
    """Verify prompt explicitly guides ICP extraction from 'for developers' etc."""
    assert "for developers" in SYSTEM_PROMPT.lower()
    assert "for teams" in SYSTEM_PROMPT.lower() or "for enterprises" in SYSTEM_PROMPT.lower()
    assert "built for" in SYSTEM_PROMPT.lower() or "designed for" in SYSTEM_PROMPT.lower()
    assert "target_audience_icp" in SYSTEM_PROMPT

def test_build_user_prompt_contains_evidence():
    pages = {
        "https://postman.com/about": "The Postman API Platform delivers high productivity for developers, great quality for APIs, and airtight governance for organizations."
    }
    prompt = build_user_prompt("postman.com", pages)
    # Evidence must be in prompt
    assert "high productivity for developers" in prompt.lower()
    assert "[SOURCE: https://postman.com/about]" in prompt

@pytest.mark.asyncio
async def test_icp_extracted_when_evidence_present():
    """When LLM returns ICP grounded in evidence, it is preserved (not nulled by grounding)."""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    payload = {
        "domain": "postman.com",
        "company_overview": "Postman is an API platform.",
        "target_audience_icp": "Developers and organizations using APIs",
        "contact_emails": [],
        "leadership": [],
        "linkedin_urls": [],
        "source_pages": ["https://postman.com/about"],
        "confidence_score": 0.75,
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_cls.return_value = mock_client

        pages = {
            "https://postman.com/about": "The Postman API Platform delivers high productivity for developers, great quality for APIs, and airtight governance for organizations."
        }
        res = await extract("postman.com", pages, errors=[])
        assert res.target_audience_icp == "Developers and organizations using APIs"
        assert res.domain == "postman.com"

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()

@pytest.mark.asyncio
async def test_icp_null_when_no_evidence():
    """When evidence has no audience cues, LLM should return null and we preserve null."""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    payload = {
        "domain": "example.com",
        "company_overview": "Example builds widgets.",
        "target_audience_icp": None,
        "contact_emails": [],
        "leadership": [],
        "linkedin_urls": [],
        "source_pages": ["https://example.com/"],
        "confidence_score": 0.5,
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_cls.return_value = mock_client

        pages = {"https://example.com/": "Example builds widgets. No audience mentioned."}
        res = await extract("example.com", pages, errors=[])
        assert res.target_audience_icp is None

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()
