"""
Pydantic models for lead enrichment.

Design decisions per review:
- Field(default_factory=list) for ALL list fields (no mutable defaults).
- Pydantic = structural validation only (types, formats, constraints).
- Factual confidence is a separate concern (confidence_score + evidence).
- Evidence-aware: every enrichment preserves source_pages and per-field provenance where applicable.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, HttpUrl, EmailStr, field_validator


class LeadershipMember(BaseModel):
    """Single leadership/team member."""

    name: str = Field(..., description="Full name of the person", min_length=1)
    role: Optional[str] = Field(
        default=None, description="Role / title if discoverable, else null"
    )
    linkedin_url: Optional[HttpUrl] = Field(
        default=None, description="LinkedIn URL if discoverable"
    )
    source_url: Optional[HttpUrl] = Field(
        default=None, description="Page URL where this member was found"
    )


class CompanyEnrichment(BaseModel):
    """
    Structured output for one domain.

    Pydantic validates SHAPE (types, URL/email format, score range).
    It does NOT validate factual correctness - that is represented by confidence_score
    and evidence (source_pages / errors).
    """

    domain: str = Field(..., description="Input domain, e.g. postman.com", min_length=3)
    company_overview: Optional[str] = Field(
        default=None,
        description="2-4 sentence overview of what the company does, from homepage/about",
    )
    target_audience_icp: Optional[str] = Field(
        default=None,
        description="Target audience / Ideal Customer Profile if discoverable",
    )
    # Requirement 3: Strict list defaults via default_factory
    contact_emails: Annotated[list[EmailStr], Field(default_factory=list)] = Field(
        default_factory=list,
        description="Generic/public contact emails only (support@, info@, hello@, contact@). Never personal emails.",
    )
    leadership: Annotated[list[LeadershipMember], Field(default_factory=list)] = Field(
        default_factory=list, description="Leadership / team members if discoverable"
    )
    linkedin_urls: Annotated[list[HttpUrl], Field(default_factory=list)] = Field(
        default_factory=list, description="Company and team LinkedIn URLs where discoverable"
    )
    # Evidence-aware: preserve URLs that contributed to extraction
    source_pages: Annotated[list[HttpUrl], Field(default_factory=list)] = Field(
        default_factory=list,
        description="URLs that were successfully fetched and used as evidence for extraction",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data confidence 0.0-1.0. Structural validation passes even at 0.0; factual confidence is separate.",
    )
    errors: Annotated[list[str], Field(default_factory=list)] = Field(
        default_factory=list, description="Non-fatal errors per URL/page (404, timeout, bot blocker, etc.)"
    )

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, v: str) -> str:
        v = v.strip().lower()
        # strip scheme if user passed https:// (loop to handle https://www.)
        for _ in range(2):
            for prefix in ("https://", "http://", "www."):
                if v.startswith(prefix):
                    v = v[len(prefix) :]
        # strip path, query, fragment
        v = v.split("/")[0].split("?")[0].split("#")[0]
        v = v.rstrip("/")
        if not v:
            raise ValueError("domain must be non-empty")
        return v

    @field_validator("contact_emails", mode="after")
    @classmethod
    def filter_generic_emails(cls, v: list[EmailStr]) -> list[EmailStr]:
        """
        Structural guard: keep only generic/public emails.
        This is still structural (pattern-based), not factual verification.
        factual confidence handles whether email is actually correct/verified.
        """
        generic_prefixes = ("info@", "hello@", "contact@", "support@", "sales@", "team@", "press@", "help@", "general@")
        filtered: list[EmailStr] = []
        for email in v:
            lower = str(email).lower()
            # keep if generic prefix OR allow but confidence will be low
            # For MVP: keep only generic to avoid PII leakage; log non-generic as dropped via confidence
            if any(lower.startswith(p) for p in generic_prefixes):
                filtered.append(email)
            # optionally: include other public emails like @domain but not personal
            # we drop non-generic silently; errors/confidence will reflect uncertainty
        # If LLM returned only non-generic, filtered may be empty -> still valid structurally, confidence low
        return filtered


class FetchResult(BaseModel):
    """Result of a single page fetch (internal, not LLM output)."""

    url: HttpUrl
    status_code: Optional[int] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    error: Optional[str] = None
    is_bot_blocked: bool = False


class DomainEnrichmentResult(BaseModel):
    """Wrapper for fail-soft per-domain execution."""

    domain: str
    enrichment: Optional[CompanyEnrichment] = None
    success: bool = False
    error: Optional[str] = None
