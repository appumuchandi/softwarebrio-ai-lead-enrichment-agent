import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError
from src.models import CompanyEnrichment

def test_default_factory_lists_not_shared():
    a = CompanyEnrichment(domain="a.com")
    b = CompanyEnrichment(domain="b.com")
    assert a.contact_emails is not b.contact_emails
    assert a.leadership is not b.leadership
    assert a.linkedin_urls is not b.linkedin_urls
    assert a.source_pages is not b.source_pages
    assert a.errors is not b.errors
    # mutating one doesn't affect other
    a.contact_emails.append("test@test.com")  # type: ignore
    assert len(b.contact_emails) == 0

def test_confidence_range_and_evidence_aware():
    c = CompanyEnrichment(domain="example.com", confidence_score=0.7, source_pages=["https://example.com/"])  # type: ignore
    assert 0 <= c.confidence_score <= 1
    # validation fails out of range
    try:
        CompanyEnrichment(domain="example.com", confidence_score=1.5)
        assert False, "should have raised"
    except ValidationError:
        pass

def test_generic_email_filter():
    # Only generic prefixes pass, personal filtered
    c = CompanyEnrichment(domain="example.com", contact_emails=["support@example.com", "john.doe@example.com"])  # type: ignore
    # john.doe should be filtered out by validator
    assert "support@example.com" in [str(x) for x in c.contact_emails]
    assert "john.doe@example.com" not in [str(x) for x in c.contact_emails]
    # generic list kept
    assert len(c.contact_emails) == 1

def test_domain_normalization():
    c = CompanyEnrichment(domain="https://WWW.Example.COM/path")
    assert c.domain == "example.com"

def test_output_json_schema_validation_sample():
    sample = {
        "domain": "postman.com",
        "company_overview": "API platform",
        "target_audience_icp": None,
        "contact_emails": ["support@postman.com"],
        "leadership": [{"name": "Abhinav Asthana", "role": "CEO"}],
        "linkedin_urls": ["https://linkedin.com/company/postman"],
        "source_pages": ["https://postman.com/"],
        "confidence_score": 0.8,
        "errors": []
    }
    c = CompanyEnrichment.model_validate(sample)
    assert c.domain == "postman.com"
    # round-trip via json mode
    dumped = c.model_dump(mode="json")
    assert dumped["domain"] == "postman.com"
