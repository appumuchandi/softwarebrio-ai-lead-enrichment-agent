import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.extraction.extractor import _ground_leadership, extract
from src.models import LeadershipMember

# Helper markdown evidence
POSTMAN_MARKDOWN = {
    "https://postman.com/about": """
# About Postman
Postman is the world's leading API platform.
## Leadership
Abhinav Asthana - CEO and Co-Founder
Ankit Sobti - Co-Founder & CTO
Abhijit Kane - Co-Founder
Contact: info@postman.com
[LinkedIn](https://linkedin.com/company/postman)
"""
}

SUPABASE_MARKDOWN = {
    "https://supabase.com/": """
# Supabase
The Postgres Development Platform
Careers page lists open roles but no leadership names.
Pricing and docs only.
""",
    "https://supabase.com/careers": """
# Careers
Join us. Open roles: Software Engineer, Product Manager.
"""
}

VAPI_MARKDOWN = {
    "https://vapi.ai/": """
# Vapi - Build Advanced Voice AI Agents
Team: Jordan Rudess - Head of Design? Actually no real leadership listed.
Pricing and platform details.
""",
    "https://vapi.ai/careers": """
# Careers at Vapi
We are hiring engineers. Jason is VP? Not clearly listed.
"""
}

def test_grounded_leadership_retained():
    leadership = [
        LeadershipMember(name="Abhinav Asthana", role="CEO and Co-Founder", linkedin_url=None),
        LeadershipMember(name="Ankit Sobti", role="Co-Founder", linkedin_url=None),
    ]
    grounded, removed = _ground_leadership(leadership, POSTMAN_MARKDOWN)
    assert len(grounded) == 2
    assert len(removed) == 0
    assert grounded[0].name == "Abhinav Asthana"
    assert grounded[0].role == "CEO and Co-Founder"

def test_hallucinated_leadership_removed():
    hallucinated = [
        LeadershipMember(name="Bryan Byrne", role="Product Manager, Lovable"),
        LeadershipMember(name="Seth Siegler", role="Chief Innovation Officer"),
        LeadershipMember(name="Bryan Byrne", role="CTO"),
    ]
    grounded, removed = _ground_leadership(hallucinated, SUPABASE_MARKDOWN)
    assert len(grounded) == 0
    assert len(removed) == 3
    assert all("not in evidence" in r for r in removed)

def test_partially_grounded_handled_safely():
    # Name grounded but role not in evidence -> keep member but role nulled
    leadership = [
        LeadershipMember(name="Abhinav Asthana", role="Astronaut Commander"),
    ]
    grounded, removed = _ground_leadership(leadership, POSTMAN_MARKDOWN)
    # Name is grounded, so member kept, but role "Astronaut Commander" not in markdown -> role should be None
    assert len(grounded) == 1
    assert grounded[0].name == "Abhinav Asthana"
    assert grounded[0].role is None  # safely nulled, not dropped
    assert len(removed) == 0

    # Name not grounded -> dropped entirely
    leadership2 = [
        LeadershipMember(name="Unknown Person", role="CEO"),
    ]
    grounded2, removed2 = _ground_leadership(leadership2, POSTMAN_MARKDOWN)
    assert len(grounded2) == 0
    assert len(removed2) == 1

def test_empty_leadership_valid():
    grounded, removed = _ground_leadership([], POSTMAN_MARKDOWN)
    assert grounded == []
    assert removed == []
    # Also test that CompanyEnrichment with empty leadership validates
    from src.models import CompanyEnrichment
    c = CompanyEnrichment(domain="example.com", leadership=[])  # type: ignore
    assert c.leadership == []

def test_unknown_name_removed():
    leadership = [
        LeadershipMember(name="Unknown", role="VP of Software"),
        LeadershipMember(name="Unknown Name", role="Engineer"),
        LeadershipMember(name="N/A", role="CEO"),
    ]
    grounded, removed = _ground_leadership(leadership, POSTMAN_MARKDOWN)
    assert len(grounded) == 0
    assert len(removed) == 3

def test_linkedin_url_grounding_for_member():
    # Member with linkedin not in evidence should have linkedin nulled but member kept if name grounded
    leadership = [
        LeadershipMember(name="Abhinav Asthana", role="CEO", linkedin_url="https://linkedin.com/in/abhinavasthana"),
    ]
    # No linkedin in markdown evidence, so should be nulled
    grounded, _ = _ground_leadership(leadership, POSTMAN_MARKDOWN)
    assert len(grounded) == 1
    assert grounded[0].linkedin_url is None

    # With grounded linkedin in markdown and leadership context
    markdown_with_li = {
        "https://example.com/about": "## Leadership\nAbhinav Asthana CEO https://linkedin.com/in/abhinavasthana"
    }
    leadership2 = [
        LeadershipMember(name="Abhinav Asthana", role="CEO", linkedin_url="https://linkedin.com/in/abhinavasthana"),
    ]
    grounded2, _ = _ground_leadership(leadership2, markdown_with_li)
    assert len(grounded2) == 1
    assert str(grounded2[0].linkedin_url) == "https://linkedin.com/in/abhinavasthana"

@pytest.mark.asyncio
async def test_extract_applies_grounding_end_to_end():
    """Verify extract() filters hallucinated leadership via deterministic grounding"""
    import os
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "mock"
    cfg.reset_settings()
    # Mock provider returns empty leadership, so grounding not tested; use openai mocked
    import json
    from unittest.mock import patch, AsyncMock, MagicMock
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    # LLM returns hallucinated supabase leadership
    payload = {
        "domain": "supabase.com",
        "company_overview": "Supabase is Postgres platform",
        "target_audience_icp": None,
        "contact_emails": [],
        "leadership": [
            {"name": "Bryan Byrne", "role": "Product Manager"},
            {"name": "Abhinav Asthana", "role": "CEO"},
        ],
        "linkedin_urls": [],
        "source_pages": ["https://supabase.com/"],
        "confidence_score": 0.85,
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    with patch("openai.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_cls.return_value = mock_client

        # Evidence does NOT contain Bryan Byrne, but does contain Abhinav? For supabase, neither is grounded, so both should be removed
        # For this test, use POSTMAN markdown where Abhinav is grounded
        pages = POSTMAN_MARKDOWN
        # But domain is supabase.com - grounding checks name in pages, so Bryan not found, Abhinav found (in postman markdown, but domain mismatch - for test we use postman markdown for supabase domain to make Abhinav grounded)
        # Actually we want to test that hallucinated is removed even if domain is supabase, but evidence has Abhinav (from postman) -> Abhinav would be kept (grounded), Bryan removed
        res = await extract("supabase.com", pages, errors=[])
        # Bryan Byrne should be removed, only Abhinav kept
        names = [m.name for m in res.leadership]
        assert "Bryan Byrne" not in names
        # Abhinav should remain if we consider postman markdown contains it
        # In this test, pages is POSTMAN, so Abhinav is grounded
        assert "Abhinav Asthana" in names or len(names) == 1

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()

def test_source_pages_not_fabricated_after_grounding():
    """Source pages after grounding must be subset of fetched evidence"""
    import os, json
    from unittest.mock import patch, AsyncMock, MagicMock
    from src import config as cfg
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    cfg.reset_settings()

    payload = {
        "domain": "example.com",
        "company_overview": "Test",
        "target_audience_icp": None,
        "contact_emails": [],
        "leadership": [{"name": "Jane Doe", "role": "CEO"}],
        "linkedin_urls": [],
        "source_pages": ["https://evil.com/fake"],
        "confidence_score": 0.9,
        "errors": []
    }
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]

    async def run():
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_cls.return_value = mock_client
            pages = {"https://example.com/": "Jane Doe CEO"}
            res = await extract("example.com", pages, errors=[])
            # source_pages should be filtered to actual evidence, not evil.com
            assert "https://evil.com/fake" not in [str(u) for u in res.source_pages]
            assert "https://example.com/" in [str(u) for u in res.source_pages]

    import asyncio
    asyncio.run(run())

    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    cfg.reset_settings()


def test_grounded_leadership_source_url_uses_found_page():
    """Regression: grounded member with LLM source_url=None should be attributed to found_page."""
    pages = {
        "https://postman.com/team": "## Leadership\nAbhinav Asthana - CEO and Co-Founder\nOur team includes Abhinav Asthana as CEO.",
        "https://postman.com/": "# Postman\nWe build APIs.",
    }
    leadership = [
        LeadershipMember(name="Abhinav Asthana", role="CEO and Co-Founder", linkedin_url=None, source_url=None),
    ]
    grounded, removed = _ground_leadership(leadership, pages)
    assert len(grounded) == 1
    assert len(removed) == 0
    # Must be attributed to the exact fetched page where name was grounded
    assert str(grounded[0].source_url) == "https://postman.com/team"
    # Also ensure not null and is HttpUrl-grounded, not LLM-provided
    assert grounded[0].source_url is not None
    assert "postman.com/team" in str(grounded[0].source_url)
