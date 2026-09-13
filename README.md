# Lead Enrichment Agent — SoftwareBrio AI Intern

Autonomous Python agent that enriches a list of company domains into structured leads via browser automation + LLM structured extraction. Built for the SoftwareBrio AI Engineer Intern take-home (postman.com, supabase.com, vapi.ai).

## Project

Given `domains.txt` (one domain per line), the agent:

1. Discovers relevant same-domain pages (homepage, about, team, pricing, etc.)
2. Renders them with Playwright to handle JavaScript
3. Converts HTML to clean Markdown (no raw HTML sent to LLM)
4. Calls an LLM for **strict Pydantic-structured extraction** (overview, ICP, generic contact emails, leadership/team, LinkedIn URLs, confidence + evidence)
5. Applies **deterministic evidence grounding** to drop hallucinations
6. Writes `output.json` — one `CompanyEnrichment` per domain, never crashing the run.

## Features

- **Domain input** — `domains.txt` or `--domains` CLI
- **Website discovery** — same-domain homepage links (regex) + sitemap.xml filtered to relevant paths + 2–3 known paths as fallback (not blind)
- **Browser automation** — Playwright Chromium, `wait_until="domcontentloaded"` + 800ms hydrate
- **Clean text/Markdown extraction** — `src/scraping/cleaner.py` strips scripts/styles, preserves headings/lists/links, truncates per budget
- **LLM structured extraction** — `openai`/`anthropic`/`mock` via env, `response_format=json_schema` (fallback `json_object`), `temperature=0.0`
- **Pydantic validation** — `CompanyEnrichment` is single source of truth (`Field(default_factory=list)` for all lists, `0.0≤confidence≤1.0`)
- **Evidence grounding** — deterministic post-processing (see below) drops hallucinated emails/LinkedIns/leadership not in fetched Markdown
- **Confidence scoring** — LLM score capped by heuristic (`no overview→0.6`, `≥2 errors→0.65`, `only overview→0.7`)
- **Resilience / fail-soft** — per-URL and per-domain isolation, always writes `output.json`

## Architecture

```
Input Domains (domains.txt)
        │
        ▼
    Discovery (src/scraping/discovery.py)
    ──► homepage <a> links (same-domain, scored) ─┐
    ──► sitemap.xml (best-effort, filtered)        ├─► 3–6 URLs, homepage first
    ──► PRIORITY_PATHS fallback (top 2–3) ─────────┘
        │
        ▼
    Playwright (src/scraping/fetcher.py)
    ──► wait_until="domcontentloaded" (not networkidle) + 800ms hydrate
    ──► bot-blocker detection (title/body), per-URL retries, bounded concurrency (4)
        │
        ▼
Content Cleaning (src/scraping/cleaner.py)
    ──► HTML → Markdown (no <html>/<script>), 12k/page, total budget 30k
        │
        ▼
  LLM Extraction (src/extraction/extractor.py + prompts.py)
    ──► SYSTEM_PROMPT + [SOURCE: <url>] markdown chunks (evidence-aware)
    ──► OpenAI json_schema (strict false → json_object fallback)
        │
        ▼
Pydantic Validation (src/models.py)
    ──► CompanyEnrichment validates shape only (types, EmailStr, HttpUrl)
        │
        ▼
Evidence Grounding (deterministic post-processing)
    ──► emails/LinkedIn: keep only if in markdown via regex
    ──► leadership: name must be exact substring in leadership context
    │    (near “leadership/our team/founders” or on /team/about URL) → else drop
    ──► source_pages filtered to fetched set, not fabricated
        │
        ▼
  Confidence (heuristic cap)
        │
        ▼
  JSON Output (output.json) — list[CompanyEnrichment]
```

Modular layout: `src/models.py` (schema) → `config.py` (env) → `scraping/{fetcher,cleaner,discovery}` → `extraction/{extractor,prompts}` → `orchestrator.py` → `scripts/run.py`. Each stage is testable and explainable in <30s for Loom.

## Why Browser Automation

Marketing sites (Postman, Supabase, Vapi) are React/Next.js — content is rendered client-side. Plain `httpx` misses it. Playwright Chromium renders JavaScript, waits for `domcontentloaded` (not `networkidle`) plus a short hydrate, then extracts `page.content()` or `innerText` fallback. This satisfies the assignment’s “handle JavaScript-rendered content” without `playwright-stealth`.

## Important Design Decision

**Raw HTML is never sent to the LLM.** `html_to_markdown` strips `script/style/noscript`, preserves headings/lists/links, collapses whitespace, and truncates. LLM sees only clean Markdown with `[SOURCE: <url>]` delimiters. This reduces tokens, hallucination, and cost, and makes evidence grounding deterministic.

## Setup

**Windows-friendly (PowerShell) — Python 3.10+ required**

```powershell
# 1) Create venv (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # macOS/Linux

# 2) Install deps
pip install -r requirements.txt
playwright install chromium

# 3) Configure env
Copy-Item .env.example .env
# Edit .env in Notepad:
# LLM_PROVIDER=mock            # offline, no key, deterministic
# LLM_PROVIDER=openai          # real LLM
# OPENAI_API_KEY=sk-...        # required if openai
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1  # optional override for OpenAI-compatible (e.g., NVIDIA)
```

No other tooling needed. Reviewer can run entirely offline with `mock`.

## Environment Variables

All via `.env` or shell (`src/config.py:Settings`, env_file `.env`):

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `LLM_PROVIDER` | `openai` | — | `openai` \| `anthropic` \| `mock` |
| `OPENAI_API_KEY` | — | if `openai` | `sk-...` (or `nvapi-...` with compatible `BASE_URL`) |
| `OPENAI_MODEL` | `gpt-4o-mini` | — | Verified with `gpt-4o-mini` and `meta/llama-3.2-11b-vision-instruct` via NVIDIA |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | — | Override for OpenAI-compatible endpoints |
| `ANTHROPIC_API_KEY` | — | if `anthropic` | `sk-ant-...` |
| `WAIT_UNTIL` | `domcontentloaded` | — | Do not use `networkidle` (per spec) |
| `NAV_TIMEOUT_MS` | `15000` | — | Playwright goto timeout |
| `HEADLESS` | `true` | — | |
| `MAX_CONCURRENT_DOMAINS` | `3` | — | Fail-soft semaphore per domain |
| `MAX_CONCURRENT_PAGES` | `4` | — | Per-domain page concurrency |
| `MAX_MD_CHARS_PER_PAGE` | `12000` | — | Cleaner truncation |
| `MAX_TOTAL_MD_CHARS` | `30000` | — | LLM prompt budget (verified run used 15000/9000 for NVIDIA to avoid hang) |

No secrets are committed; `.env` is gitignored. See `.env.example`.

## Run

```powershell
# Offline (no key) — writes output.json for the 3 required domains
python -m scripts.run --mock --domains postman.com supabase.com vapi.ai
python -m scripts.run --mock --input domains.txt --output output.json

# Real LLM — OpenAI
$env:LLM_PROVIDER="openai"; $env:OPENAI_API_KEY="sk-..."
python -m scripts.run --input domains.txt --output output.json

# Real LLM — NVIDIA OpenAI-compatible (verified: meta/llama-3.2-11b-vision-instruct)
$env:LLM_PROVIDER="openai"; $env:OPENAI_API_KEY="nvapi-..."; $env:OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"; $env:OPENAI_MODEL="meta/llama-3.2-11b-vision-instruct"
python -m scripts.run --input domains.txt --output output.json

# Alternative entry
python scripts/run.py --input domains.txt --output output.json
```

Output is always `output.json` (or custom `--output`), even if all fetches fail.

## Output Schema

`output.json` is `list[CompanyEnrichment]` (`src/models.py:CompanyEnrichment`):

```json
{
  "domain": "postman.com",
  "company_overview": "Postman is an API platform ...",
  "target_audience_icp": "Developers, organizations ... or null",
  "contact_emails": ["info@postman.com"],
  "leadership": [{"name": "Abhinav Asthana", "role": "CEO and Co-Founder", "linkedin_url": null, "source_url": null}],
  "linkedin_urls": ["https://linkedin.com/company/postman"],
  "source_pages": ["https://postman.com/", "https://postman.com/about"],
  "confidence_score": 0.7,
  "errors": []
}
```

- `company_overview` — 2–4 sentence overview from homepage/about (or `null` if not found)
- `target_audience_icp` — ICP description or `null`
- `contact_emails` — **generic/public only** (`info@`, `hello@`, `contact@`, `support@`, `sales@`, `team@`, `press@`, `help@`, `general@`), filtered via `EmailStr` + `filter_generic_emails`; empty if none grounded
- `leadership` — `[{name, role, linkedin_url, source_url}]`; deterministically grounded (see below); `[]` if none grounded
- `linkedin_urls` — deduped `https://linkedin.com/...` from evidence; `[]` if none
- `confidence_score` — `0.0–1.0` (LLM-generated then heuristic-capped); **not** a factual guarantee
- `source_pages` — **only URLs actually fetched** (evidence), never fabricated; filtered to `fetched_set`
- `errors` — concise per-URL messages (`HTTP 404`, `HTTP 403 (possible bot challenge)`, `timeout (...)`, `fetch exception: ...`, `HTTP 429 (rate limited)`, `HTTP 500 (server error)`)

`Pydantic` validates **shape** (types, formats, `ge/le`), **not facts**. Factual trust is represented by `confidence_score` + `errors`, and reduced to `0.0–0.35` on `no evidence` or LLM failure.

## Evidence Grounding (Deterministic Post-Processing)

LLM sometimes hallucinates. After validation, **deterministic** grounding (no LLM) enforces:

- **Emails/LinkedIn**: keep only if substring exists in fetched Markdown (`_regex_fallback_*` sets). If LLM empty but regex found, merge grounded value.
- **Leadership**: name must be **exact substring** case-insensitive **and** in **leadership context**: either URL contains `team/about/leadership/company/people/founders` **or** 1500-char window before name contains `\bleadership\b`/`\bour team\b`/`\bfounders\b`/… via word-boundary regex. Prevents customer testimonials (“Bryan Byrne, Product Manager, Lovable” on Supabase homepage) being misclassified as leadership. If name grounded but role tokens not in same page, `role` is nulled but member kept (safe partial). `linkedin_url` for member nulled if not in evidence. If no member grounded → `leadership=[]`.

This is why `supabase.com` correctly yields `leadership: []` after Slice 3A despite NVIDIA hallucinating 5 customer names, while `postman.com` retains its 3 co-founders (found in `/about` near `leadership`).

## Resilience

| Case | Behavior |
|------|----------|
| **404** (e.g., `supabase.com/about` 404) | No retry (deterministic), `errors: ["https://.../about: HTTP 404 ..."]`, other URLs continue |
| **403 / bot block** (`Just a moment`, `cf-challenge`) | Detect via `_is_bot_blocked`, retry up to 2× with `1.2*(attempt+1)` backoff, `is_bot_blocked=True`, recorded |
| **429 rate-limit** | Transient → retry 2×, `"(rate limited)"`, backoff |
| **5xx (500/502/503/504)** | Transient → retry 2×, `"(server error)"`, backoff |
| **Timeout** (`15000ms`) | `PlaywrightTimeoutError` retry 2× with `1.0*(attempt+1)` |
| **DNS / connection** (`ERR_NAME_NOT_RESOLVED`, `BrowserClosedError`) | Generic `fetch exception` retry 2×, then `FetchResult(error=...)` |
| **Empty/malformed page** | `html_to_markdown` → `innerText` fallback → `error: "empty content after cleaning"` |
| **Sitemap failure** | `httpx` timeout 6s, `try/except` per sitemap candidate → `[]`, discovery falls back to priority paths |
| **LLM timeout / 429 / 5xx** | `_is_transient_llm_error` (timeout/429/5xx/connection) → retry 2× `1.5*(attempt+1)`; `AsyncOpenAI(timeout=60.0)` |
| **Malformed LLM JSON** | `_extract_json_from_text` (direct → ```json block → outermost `{}`) → if still fails, exception is **not** transient → no retry, fallback to mock `confidence ≤0.35` |
| **Pydantic validation** (`confidence 2.0`, bad email) | `except ValidationError` → `confidence 0.0`, `errors: ["pydantic validation failed: ..."]` |
| **Fail-soft** | One failed URL never stops other URLs (`fetch_many` semaphore); one failed domain never stops other domains (`enrich_domains` semaphore + `_guarded`); domain with no evidence still returns `CompanyEnrichment(domain, confidence 0.0, errors[...])`. |

Errors are **never swallowed silently** — concise messages remain in `errors` and `output.json`. Retry only for transient; deterministic 404/malformed JSON not retried; backoff respected; browser/context/page always closed in `finally` (no orphan Playwright).

## Testing

```powershell
python -m pytest tests/ -q
python -m pytest tests/test_resilience.py -v
```

**Currently verified: 81 passed** (no failures)

- `test_cleaner` (5) — no raw HTML, headings/links preserved, truncates
- `test_discovery` (4) — same-domain priority, sitemap fallback, scoring
- `test_extractor` (7) — `SOURCE:` evidence, JSON extraction, mock & mocked OpenAI `json_schema`, confidence cap
- `test_leadership_grounding` (8) — grounded retained, hallucinated removed, partial (role nulled), empty valid, `Unknown` removed, LinkedIn grounded, end-to-end grounding, source_pages not fabricated
- `test_models` (5) — `default_factory`, `EmailStr` generic filter, `confidence 0–1`, domain normalization
- `test_orchestrator` (4) — fail-soft per URL/domain, `domcontentloaded` not `networkidle`, `output.json` validates
- `test_resilience` (14) — 404, timeout, 403/bot, 429, 5xx, empty page, DNS/browser crash, sitemap, malformed JSON, LLM 429/timeout, failed domain while others succeed, missing fields, validation failure
- `test_linkedin_discovery` (25) — valid result, no result, unrelated, timeout, HTTP failure, malformed, duplicate prevention, max 5 limit, existing linkedin skip, fail-soft domain, never-invent, verification, linkedin_urls consistency
- `test_browser_reuse` (5) — one Browser launch, shared contexts, fail-soft with shared browser, cleanup
- `test_icp_extraction` (4) — ICP cues, evidence, null handling

## Sample Results

`output.json` (committed sample, after `python -m scripts.run --mock` or live NVIDIA) contains **exactly** the three required domains and validates via `CompanyEnrichment.model_validate`:

- `postman.com` — `info@postman.com`, leadership 3 (Abhinav Asthana etc, grounded in `/about`), `confidence 0.7`, `errors []`
- `supabase.com` — `leadership []` (hallucinated 5 customer names correctly removed), `confidence 0.65`, `errors [2× 404]`
- `vapi.ai` — `leadership []` (customer testimonial removed), `confidence 0.65`, `errors [2× 404]`

No API keys, debug dumps, or fabricated `source_pages` in output. `source_pages` are only fetched URLs. No hallucinated leadership survives grounding.

> **Factual correctness:** No guarantee. Pydantic verifies **structure** only. Grounding is deterministic post-processing based on fetched Markdown. Always check `confidence_score` and `errors`.

## Reproducibility

Reviewer can reproduce offline:

```powershell
pip install -r requirements.txt
playwright install chromium
python -m scripts.run --mock --input domains.txt --output output.json
python -m pytest tests/ -q   # 81 passed
```

Live LLM (verified): `LLM_PROVIDER=openai` with `OPENAI_API_KEY` (`gpt-4o-mini`) or **NVIDIA OpenAI-compatible** `OPENAI_API_KEY=nvapi-... OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1 OPENAI_MODEL=meta/llama-3.2-11b-vision-instruct` (used for Slice 2–3B verification, requires `MAX_TOTAL_MD_CHARS=15000`/`MAX_MD_CHARS_PER_PAGE=9000` to avoid hang on 30k prompt). Both use same `AsyncOpenAI` path; results vary by model but schema/grounding/resilience remain identical.

## Bonus — DuckDuckGo LinkedIn Discovery (Optional)

External LinkedIn discovery enriches **grounded** leadership members missing `linkedin_url` via DuckDuckGo public web search — no paid API/key.

- **Trigger**: Only after deterministic evidence grounding, for leadership records with valid grounded `name` + `company/domain` where `linkedin_url` is still `null`.
- **Search**: Query `"<person name>" "<company name>" LinkedIn` (company defaults to domain base, e.g., `postman.com` → `Postman`) against `https://html.duckduckgo.com/html/?q=...` via `httpx` (uses existing project HTTP patterns, `httpx` timeout 10s).
- **Inspect**: Parses HTML for `linkedin.com/in/` profile URLs only (prefers `/in/` over `/company/`). Only URLs that **actually appear** in DuckDuckGo results are considered; URLs are never invented.
- **Verify**: Before assignment, the surrounding result title/snippet is checked for plausible association: exact name tokens (and any reordering/hyphenation in slug) plus at least one company token must appear in context. If uncertain, leaves `linkedin` as `null`.
- **Enrichment**: `src/discovery/linkedin.py:discover_linkedin_profile(name: str, company: str, domain: str) -> str | None` (sync + async variant) + `enrich_leadership_with_linkedin(enrichment, domain)` which updates `leadership[].linkedin_url` and keeps top-level `linkedin_urls` deduped and consistent.
- **Limits**: Max **5** DuckDuckGo searches per domain per run, deduplicates same person (case-insensitive), skips members who already have `linkedin_url`, bounded timeout, no extra crawling.
- **Resilience**: Fail-soft — search timeout, HTTP error, DuckDuckGo blocking/captcha, malformed HTML, or no verified LinkedIn result all return `None` and **never crash** the domain pipeline; domain still yields valid `CompanyEnrichment` with `linkedin` as `null`.
- **No replacement**: Existing website-grounded `linkedin_url` values are never overwritten; external search is enrichment only, not a replacement for authoritative website evidence.
- **No key required**: Free public DuckDuckGo HTML search; respects timeout and best-effort semantics.

Output schema unchanged; if verified, `leadership[].linkedin_url = "https://www.linkedin.com/in/..."` and `linkedin_urls` includes it, else remains `null`/`[]`.

## 40% Manual Operations

Confirmed comfortable with ~40% manual operations (website QA, data verification, enrichment review) alongside automation.

## Loom Demo (2–3 min)

Script: `0:00` `domains.txt → output.json` live run; `0:40` architecture pipeline; `1:20` `models.py:Field(default_factory=list)` + `fetcher.py:domcontentloaded` + `extractor.py:json_schema`; `1:50` resilience (kill network → `errors`/`confidence` but run continues, grounding demo); `2:20` output validation.

---

**Stack:** Python 3.10+, Playwright 1.48, Pydantic 2.9, OpenAI 1.55 / Anthropic 0.39, httpx, pytest.
