"""
LLM extractor - vertical slice.

- Provider configurable via Settings (openai | anthropic | mock).
- Uses Pydantic for structural validation; factual confidence is separate (LLM score + heuristic cap).
- Evidence-aware: source_pages preserved; per-member source_url also.
- Fail-soft: never raises to orchestrator; returns CompanyEnrichment with errors + confidence 0 on LLM failure.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from pydantic import ValidationError

from src.config import get_settings
from src.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from src.models import CompanyEnrichment, LeadershipMember
from src.scraping.cleaner import extract_emails_from_text


def _heuristic_confidence_cap(
    enrichment: CompanyEnrichment,
    num_sources: int,
    num_errors: int,
) -> float:
    """
    Apply heuristic cap to LLM confidence. LLM structural validation passes even at 0.0;
    factual confidence is adjusted here.

    Caps: missing overview -> max 0.6, no sources -> 0.0, sparse sources -> reduction.
    """
    cap = 1.0
    if not enrichment.company_overview:
        cap = min(cap, 0.6)
    if not enrichment.source_pages:
        return 0.0
    if num_sources == 1:
        cap = min(cap, 0.75)
    if num_errors >= 2:
        cap = min(cap, 0.65)
    if not enrichment.leadership and not enrichment.contact_emails and not enrichment.linkedin_urls:
        # Only overview present -> modest confidence
        cap = min(cap, 0.7)
    return cap


def _regex_fallback_emails(pages_markdown: dict[str, str]) -> list[str]:
    """Fallback deterministic email extraction from markdown (evidence-aware)."""
    emails: list[str] = []
    for md in pages_markdown.values():
        for e in extract_emails_from_text(md):
            if e.lower() not in [x.lower() for x in emails]:
                emails.append(e)
    # Filter generic only
    generic_prefixes = ("info@", "hello@", "contact@", "support@", "sales@", "team@", "press@", "help@", "general@")
    return [e for e in emails if any(e.lower().startswith(p) for p in generic_prefixes)]


def _regex_fallback_linkedin(pages_markdown: dict[str, str]) -> list[str]:
    pattern = r"https?://(?:www\.)?linkedin\.com/[^\s\"'<>\)]+"
    found: list[str] = []
    seen = set()
    for md in pages_markdown.values():
        for m in re.findall(pattern, md, flags=re.I):
            # Clean trailing punctuation
            cleaned = m.rstrip(".,;)]")
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                found.append(cleaned)
    return found


LEADERSHIP_CONTEXT_KEYWORDS = [
    "leadership",
    "our team",
    "meet the team",
    "executive team",
    "founders",
    "co-founder",
    "co founder",
    "management team",
    "board of directors",
    "leadership team",
]

TEAM_URL_KEYWORDS = ["team", "about", "leadership", "company", "people", "founders"]

def _is_leadership_context_nearby(page_url: str, page_text_lower: str, name_lower: str) -> bool:
    """Check if name appears in leadership context (near leadership keywords or on team/about page)."""
    # URL hint: team/about pages are more likely to contain true leadership
    url_lower = page_url.lower()
    # Use path-aware check: check if any TEAM_URL_KEYWORDS appears as path segment
    # For homepage "/", not team; for "/about" etc, true
    is_team_url = any(kw in url_lower for kw in TEAM_URL_KEYWORDS)
    # Find name position
    idx = page_text_lower.find(name_lower)
    if idx == -1:
        return False
    # Check window 1500 chars before name for leadership keywords with word boundaries
    window_start = max(0, idx - 1500)
    window = page_text_lower[window_start:idx]
    has_context = any(re.search(r"\b" + re.escape(kw) + r"\b", window) for kw in LEADERSHIP_CONTEXT_KEYWORDS)
    return is_team_url or has_context


def _ground_leadership(
    leadership: list[LeadershipMember], pages_markdown: dict[str, str]
) -> tuple[list[LeadershipMember], list[str]]:
    """
    Deterministic post-extraction grounding for leadership.

    - Name must appear in evidence (case-insensitive exact substring).
    - Name must appear in leadership context (near leadership keywords or on team/about URL) — prevents customer testimonials being misclassified as leadership.
    - If name is 'Unknown' or generic, drop.
    - If role is provided, at least one significant token of role must appear in same page where name was found
      (otherwise role is nulled but member kept if name grounded — safe partial handling).
    - LinkedIn URL for member must be grounded in evidence (same as global linkedin grounding), else nulled.
    Returns (grounded_list, removed_names_for_reporting).
    """
    if not leadership:
        return [], []

    # Pre-lower all markdown for fast lookup, keep per-page
    lowered_pages = {url: md.lower() for url, md in pages_markdown.items()}
    combined_lower = "\n\n".join(lowered_pages.values())

    grounded: list[LeadershipMember] = []
    removed: list[str] = []

    for member in leadership:
        name = (member.name or "").strip()
        if not name or name.lower() in ("unknown", "unknown name", "n/a", "not applicable"):
            removed.append(f"{name} (unknown)")
            continue

        name_lower = name.lower()
        # Find page where name appears (exact substring required for leadership - prevents token-spread false positives)
        found_page: str | None = None
        found_page_text: str | None = None
        for url, text in lowered_pages.items():
            if name_lower in text:
                # Strict leadership context check: name must be near leadership keywords or on team URL
                if _is_leadership_context_nearby(url, text, name_lower):
                    found_page = url
                    found_page_text = text
                    break
                else:
                    # Name found but not in leadership context -> not grounded as leadership
                    # Continue searching other pages for leadership context
                    continue

        if not found_page:
            # Check if name exists at all but without context -> treat as not grounded for leadership
            exists_anywhere = any(name_lower in t for t in lowered_pages.values())
            if exists_anywhere:
                removed.append(f"{name} (name not in leadership context)")
            else:
                removed.append(f"{name} (name not in evidence)")
            continue

        # Role grounding: if role provided, check at least one significant token appears in same page (or combined if strict fails)
        role = member.role
        grounded_role = role
        if role:
            role_lower = role.lower()
            # significant tokens: words >=3 chars, not stopwords
            stop = {"and", "the", "for", "with", "of", "a", "an", "in", "on", "at", "to", "as"}
            role_tokens = [t for t in re.split(r"[^a-z0-9]+", role_lower) if len(t) >= 3 and t not in stop]
            # If role is very generic like "CEO", check "ceo" in same page
            if role_tokens:
                # check same page first
                page_text = found_page_text or ""
                token_found_in_page = any(tok in page_text for tok in role_tokens)
                # fallback: check combined (allows role to be mentioned elsewhere on site)
                token_found_anywhere = any(tok in combined_lower for tok in role_tokens)
                if not token_found_in_page and not token_found_anywhere:
                    # Role not grounded — null it rather than drop member (safe partial handling)
                    grounded_role = None
                # else keep role as is
            else:
                # role too short/generic, keep as is if name grounded
                pass

        # LinkedIn URL grounding for member: must be in evidence linkedin set
        member_linkedin = member.linkedin_url
        grounded_linkedin = member_linkedin
        if member_linkedin:
            grounded_linkedin_set = set(u.lower().rstrip("/") for u in _regex_fallback_linkedin(pages_markdown))
            # Normalize member URL
            m_url = str(member_linkedin).lower().rstrip("/").rstrip(".,)")
            if m_url not in grounded_linkedin_set:
                # Check if any grounded linkedin contains member name? No — just drop linkedin
                grounded_linkedin = None

        # Source attribution: record the exact fetched page where the
        # leadership member was grounded.
        member_source = found_page
        grounded.append(
            LeadershipMember(
                name=name,
                role=grounded_role,
                linkedin_url=grounded_linkedin,  # type: ignore
                source_url=member_source,  # type: ignore
            )
        )

    return grounded, removed


def _extract_json_from_text(text: str) -> dict:
    """Robust JSON extraction: try direct parse, then markdown code block, then outermost {...}."""
    if not text or not text.strip():
        raise ValueError("empty LLM response")
    # 1. Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. Code block ```json {...}```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. Outermost JSON object (first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 4. Give up
    raise ValueError(f"no valid JSON found in LLM response (first 500 chars): {text[:500]!r}")


def _is_transient_llm_error(exc: Exception) -> bool:
    """Deterministic transient check for LLM: timeout, 429, 5xx, connection."""
    msg = str(exc).lower()
    # Timeout
    if "timeout" in msg or "timed out" in msg:
        return True
    # Rate limit
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True
    # 5xx server errors
    if any(code in msg for code in ["500", "502", "503", "504"]):
        return True
    if "internal server error" in msg or "bad gateway" in msg or "service unavailable" in msg or "gateway timeout" in msg:
        return True
    # Connection / network transient
    if "connection" in msg and ("reset" in msg or "closed" in msg or "failed" in msg):
        return True
    # OpenAI specific exception names
    name = type(exc).__name__.lower()
    if "timeout" in name or "ratelimit" in name or "apiconnection" in name or "internalserver" in name:
        return True
    return False


async def _call_openai(domain: str, pages_markdown: dict[str, str]) -> CompanyEnrichment:
    settings = get_settings()
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed. pip install openai") from e

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=60.0,  # explicit LLM timeout (was implicit)
    )
    user_prompt = build_user_prompt(domain, pages_markdown, max_total_chars=settings.max_total_markdown_chars)

    schema = CompanyEnrichment.model_json_schema()
    # Retry transient LLM failures (timeout, 429, 5xx) with backoff, not for malformed JSON/validation
    max_retries = 2
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            try:
                completion = await client.chat.completions.create(
                    model=settings.openai_model,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "CompanyEnrichment",
                            "strict": False,
                            "schema": schema,
                        },
                    },
                )
                content = completion.choices[0].message.content or "{}"
                try:
                    data = json.loads(content)
                except Exception:
                    data = _extract_json_from_text(content)
            except Exception as e:
                # json_schema not supported -> fallback to json_object (deterministic, not transient)
                # Only fallback once per attempt; if fallback also transient, will retry outer
                try:
                    completion = await client.chat.completions.create(
                        model=settings.openai_model,
                        temperature=settings.llm_temperature,
                        max_tokens=settings.llm_max_tokens,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT + "\nReturn valid JSON matching CompanyEnrichment schema."},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    content = completion.choices[0].message.content or "{}"
                    try:
                        data = json.loads(content)
                    except Exception:
                        data = _extract_json_from_text(content)
                except Exception as e2:
                    raise RuntimeError(f"OpenAI call failed: {e} | fallback also failed: {e2}") from e2
            # Success path: validate and return
            data["domain"] = domain
            if not data.get("source_pages"):
                data["source_pages"] = list(pages_markdown.keys())
            enrichment = CompanyEnrichment.model_validate(data)
            return enrichment
        except Exception as e:
            # Distinguish transient vs deterministic (malformed JSON, validation)
            # ValidationError and JSON extraction errors are not transient
            if isinstance(e, ValidationError):
                raise
            if "no valid JSON" in str(e).lower() or "pydantic" in str(e).lower():
                raise
            last_exc = e
            if _is_transient_llm_error(e) and attempt < max_retries:
                # Backoff: 1.5s, 3.0s
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    # Should not reach, but fail-soft
    assert last_exc is not None
    raise last_exc


async def _call_anthropic(domain: str, pages_markdown: dict[str, str]) -> CompanyEnrichment:
    settings = get_settings()
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. pip install anthropic") from e

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    user_prompt = build_user_prompt(domain, pages_markdown, max_total_chars=settings.max_total_markdown_chars)
    schema_str = json.dumps(CompanyEnrichment.model_json_schema(), indent=2)
    resp = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        system=SYSTEM_PROMPT + f"\nYou must return JSON matching this JSON Schema:\n{schema_str}\nReturn JSON only, no markdown.",
        messages=[{"role": "user", "content": user_prompt}],
    )
    content = "".join(block.text for block in resp.content if hasattr(block, "text"))
    # Extract JSON block if wrapped in markdown
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.S | re.I)
    if m:
        content = m.group(1)
    data = json.loads(content)
    data["domain"] = domain
    if not data.get("source_pages"):
        data["source_pages"] = list(pages_markdown.keys())
    return CompanyEnrichment.model_validate(data)


def _mock_extraction(domain: str, pages_markdown: dict[str, str]) -> CompanyEnrichment:
    """
    Deterministic mock for offline dev / CI without API keys.
    Extracts via regex heuristics but still validates through Pydantic.
    """
    all_text = "\n\n".join(pages_markdown.values())
    # Very simple overview: first 2 sentences-ish from first page
    first_md = next(iter(pages_markdown.values()), "")
    overview = None
    if first_md and len(first_md.strip()) > 80:
        # Take first 300 chars as pseudo-overview
        snippet = first_md.strip().replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        overview = snippet[:400].strip()
        if len(overview) == 400:
            overview = overview.rsplit(" ", 1)[0] + "..."
    emails = _regex_fallback_emails(pages_markdown)
    linkedins = _regex_fallback_linkedin(pages_markdown)
    # Leadership mock: try to find lines like "Name - Role" near team keywords
    leadership: list[dict] = []
    # Don't hallucinate - keep empty for mock unless we can parse
    confidence = 0.5 if overview else 0.25
    if not pages_markdown:
        confidence = 0.0

    return CompanyEnrichment(
        domain=domain,
        company_overview=overview,
        target_audience_icp=None,
        contact_emails=emails,  # type: ignore
        leadership=[],  # type: ignore
        linkedin_urls=linkedins,  # type: ignore
        source_pages=list(pages_markdown.keys()),  # type: ignore
        confidence_score=confidence,
        errors=[],
    )


async def extract(domain: str, pages_markdown: dict[str, str], errors: Optional[list[str]] = None) -> CompanyEnrichment:
    """
    Main extraction entry. Provider configurable.

    Returns CompanyEnrichment always (fail-soft). On LLM failure, returns fallback with confidence 0.0-0.3 and errors.
    """
    settings = get_settings()
    errors = errors or []
    domain = domain.strip().lower().replace("https://", "").replace("http://", "").split("/")[0]

    if not pages_markdown:
        # No evidence -> empty enrichment, confidence 0, but structurally valid
        return CompanyEnrichment(
            domain=domain,
            company_overview=None,
            target_audience_icp=None,
            source_pages=[],  # type: ignore
            confidence_score=0.0,
            errors=errors + ["no pages fetched - no evidence"],
        )

    provider = settings.llm_provider.value if hasattr(settings.llm_provider, "value") else str(settings.llm_provider)

    try:
        if provider == "mock":
            enrichment = _mock_extraction(domain, pages_markdown)
        elif provider == "openai":
            enrichment = await _call_openai(domain, pages_markdown)
        elif provider == "anthropic":
            enrichment = await _call_anthropic(domain, pages_markdown)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

        # Post-process: heuristic confidence cap (structural vs factual)
        cap = _heuristic_confidence_cap(enrichment, num_sources=len(pages_markdown), num_errors=len(errors))
        if enrichment.confidence_score > cap:
            enrichment.confidence_score = round(cap, 2)

        # Evidence-aware: ensure source_pages is subset of actually fetched URLs
        # Filter to fetched keys to prevent hallucinated URLs
        fetched_set = set(pages_markdown.keys())
        filtered_sources = [u for u in enrichment.source_pages if str(u) in fetched_set or str(u).rstrip("/") in {k.rstrip("/") for k in fetched_set}]
        if filtered_sources:
            enrichment.source_pages = filtered_sources  # type: ignore
        else:
            # If LLM returned no valid source, default to what we fetched
            enrichment.source_pages = list(pages_markdown.keys())  # type: ignore

        # Evidence-aware grounded filtering: drop hallucinated emails/linkedins not present in markdown
        # This enforces "extract ONLY from evidence" even if LLM hallucinates generic emails.
        all_text = "\n\n".join(pages_markdown.values())
        grounded_email_set = set(e.lower() for e in _regex_fallback_emails(pages_markdown))
        # also consider any email in text that is generic but regex fallback already filtered generic, so this is grounded generic
        if enrichment.contact_emails:
            filtered_emails = [e for e in enrichment.contact_emails if str(e).lower() in grounded_email_set]
            # If LLM hallucinated emails not in evidence, filtered will be smaller; keep filtered (may be empty)
            if len(filtered_emails) != len(enrichment.contact_emails):
                enrichment.contact_emails = filtered_emails  # type: ignore

        grounded_linkedin_set = set(u.lower() for u in _regex_fallback_linkedin(pages_markdown))
        if enrichment.linkedin_urls:
            filtered_linkedin = [u for u in enrichment.linkedin_urls if str(u).lower().rstrip("/") in grounded_linkedin_set or any(str(u).lower() in v for v in grounded_linkedin_set)]
            # More precise: exact lower match
            filtered_linkedin_exact = [u for u in enrichment.linkedin_urls if str(u).lower().rstrip(".,)").lower() in grounded_linkedin_set]
            if len(filtered_linkedin_exact) != len(enrichment.linkedin_urls):
                enrichment.linkedin_urls = filtered_linkedin_exact  # type: ignore

        # Deterministic leadership grounding (Slice 3A): name and role must be in evidence
        if enrichment.leadership:
            grounded, removed = _ground_leadership(enrichment.leadership, pages_markdown)
            # Always replace with grounded (may be empty)
            enrichment.leadership = grounded  # type: ignore
            # Do NOT add removed to errors to keep errors for fetch issues only; grounding is structural

        # Merge errors (fetcher errors) into enrichment.errors
        if errors:
            enrichment.errors = list(dict.fromkeys(list(enrichment.errors) + errors))

        # Fallback deterministic enhancers: if LLM left emails/linkedin empty but regex found them, merge (evidence-grounded)
        # Only merge if regex found evidence in markdown (grounded)
        regex_emails = _regex_fallback_emails(pages_markdown)
        if not enrichment.contact_emails and regex_emails:
            # Validate through Pydantic EmailStr will filter generic only again
            try:
                enrichment.contact_emails = regex_emails  # type: ignore
            except Exception:
                pass
        regex_linkedins = _regex_fallback_linkedin(pages_markdown)
        if not enrichment.linkedin_urls and regex_linkedins:
            try:
                enrichment.linkedin_urls = regex_linkedins  # type: ignore
            except Exception:
                pass

        return enrichment

    except ValidationError as ve:
        # Structural validation failed - return fail-soft with errors
        return CompanyEnrichment(
            domain=domain,
            company_overview=None,
            target_audience_icp=None,
            source_pages=list(pages_markdown.keys()),  # type: ignore
            confidence_score=0.0,
            errors=errors + [f"pydantic validation failed: {ve.errors()[:2]}"],
        )
    except Exception as e:
        # Any LLM/provider failure -> deterministic fallback + low confidence, never crash
        # Try mock as graceful fallback so run still produces useful output
        try:
            fallback = _mock_extraction(domain, pages_markdown)
            fallback.errors = errors + [f"LLM extraction failed ({provider}): {type(e).__name__}: {e} - used regex fallback"]
            fallback.confidence_score = min(fallback.confidence_score, 0.35)
            return fallback
        except Exception as e2:
            return CompanyEnrichment(
                domain=domain,
                company_overview=None,
                target_audience_icp=None,
                source_pages=list(pages_markdown.keys()),  # type: ignore
                confidence_score=0.0,
                errors=errors + [f"LLM failed: {e}", f"fallback failed: {e2}"],
            )
