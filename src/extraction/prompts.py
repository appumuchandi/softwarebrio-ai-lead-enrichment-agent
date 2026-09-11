"""
Versioned prompts for LLM extraction.

Separated from extractor.py for testability and rubric clarity (25% LLM + structured output).
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a precise lead enrichment analyst for B2B sales.

Rules:
- Extract ONLY from the provided markdown evidence. Never hallucinate or invent.
- If information is not present in the evidence, return null for strings and [] for lists.
- For contact_emails: return ONLY generic/public emails like info@, hello@, contact@, support@, sales@, team@, press@, help@. Never return personal emails (e.g., firstname.lastname@). If no generic email is visible, return [].
- For leadership: only include people explicitly listed as leadership/team on the evidence pages. Include name and role if present. For linkedin_url, only include if a linkedin.com URL is visible in the evidence for that person. Never guess.
- For linkedin_urls: collect only linkedin.com URLs that appear in the evidence (company page or team). Deduplicate.
 - For company_overview: 2-4 sentences summarizing what the company does, based on homepage/about evidence.
- For target_audience_icp: describe ideal customer / target audience if explicitly stated or strongly implied in evidence; else null. Look for explicit cues like "for developers", "for teams", "for enterprises", "built for", "designed for", "trusted by", "ideal for", "helps ...", "delivers ... for". If such audience is clearly described (e.g., "high productivity for developers" implies ICP is developers), summarize concisely (e.g., "Developers and engineering teams", "Enterprises"). Do not invent beyond evidence.
- For confidence_score: 0.0-1.0 calibration: 1.0 = all fields well-evidenced from multiple pages, 0.7 = overview + one other field evidenced, 0.4 = only homepage sparse, 0.0 = no evidence. Be conservative. Do NOT inflate.
- Evidence matters: you will be penalized for facts not grounded in source pages.
"""

def build_user_prompt(domain: str, pages_markdown: dict[str, str], max_total_chars: int = 30000) -> str:
    """
    Build user prompt with evidence-aware delimiters.

    Each page is prefixed with SOURCE: <url> so LLM can ground extraction.
    Pages are truncated and concatenated to stay within token budget.
    """
    parts: list[str] = []
    parts.append(f"Domain: {domain}")
    parts.append(f"Evidence pages ({len(pages_markdown)} sources):")
    parts.append("Extract structured fields grounded ONLY in these sources. Preserve source URLs in your reasoning.")
    parts.append("")

    total = 0
    for url, md in pages_markdown.items():
        header = f"\n---\n[SOURCE: {url}]\n"
        # Reserve for header
        remaining = max_total_chars - total - len(header) - 500  # 500 buffer for prompt overhead
        if remaining <= 200:
            parts.append(f"{header}[omitted - token budget exceeded]")
            break
        chunk = md[:remaining] if len(md) > remaining else md
        parts.append(header + chunk)
        total += len(header) + len(chunk)
        if total >= max_total_chars:
            break

    parts.append("\n---\n")
    parts.append(
        "Return JSON matching the CompanyEnrichment schema. "
        "Set source_pages to the subset of SOURCE URLs that actually supported your extraction. "
        "Set confidence_score honestly."
    )
    return "\n".join(parts)
