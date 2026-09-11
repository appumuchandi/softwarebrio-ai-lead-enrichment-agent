"""
CLI entrypoint for vertical slice.

Usage:
  python -m scripts.run --input domains.txt --output output.json
  python -m scripts.run --domains postman.com supabase.com vapi.ai

Env:
  LLM_PROVIDER=mock|openai|anthropic  (default openai)
  OPENAI_API_KEY=...  (required if provider=openai)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure src is importable when run as `python scripts/run.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.models import CompanyEnrichment
from src.orchestrator import enrich_domains


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lead enrichment agent - vertical slice")
    p.add_argument("--input", "-i", type=str, default=None, help="Path to domains.txt (one domain per line)")
    p.add_argument("--output", "-o", type=str, default=None, help="Path to output.json")
    p.add_argument("--domains", nargs="*", default=None, help="Domains as CLI args (overrides --input)")
    p.add_argument("--mock", action="store_true", help="Force LLM_PROVIDER=mock for offline run")
    return p.parse_args()


def load_domains(args: argparse.Namespace) -> list[str]:
    if args.domains:
        return [d.strip() for d in args.domains if d.strip()]
    settings = get_settings()
    input_path = Path(args.input) if args.input else settings.input_path
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}", file=sys.stderr)
        print(f"  Create it or pass --domains postman.com supabase.com vapi.ai", file=sys.stderr)
        sys.exit(1)
    text = input_path.read_text(encoding="utf-8")
    domains = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    return domains


async def main() -> None:
    args = parse_args()
    if args.mock:
        import os
        os.environ["LLM_PROVIDER"] = "mock"
        # Reset settings singleton so it re-reads env
        from src import config as cfg
        cfg.reset_settings()

    settings = get_settings()
    # Validate provider config early, but don't crash - warn and fallback to mock if missing key
    validation_error = settings.validate_for_provider()
    if validation_error and not args.mock:
        print(f"[warn] {validation_error}", file=sys.stderr)
        # For vertical slice: fallback to mock so run still demonstrates pipeline without crashing
        # User can set key to get real LLM. We don't hard-fail per fail-soft principle.
        if settings.llm_provider.value != "mock":
            print(f"[warn] Falling back to LLM_PROVIDER=mock for this run. Set OPENAI_API_KEY to use real LLM.", file=sys.stderr)
            import os
            os.environ["LLM_PROVIDER"] = "mock"
            from src import config as cfg
            cfg.reset_settings()
            settings = get_settings()

    domains = load_domains(args)
    if not domains:
        print("[error] No domains found in input", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else settings.output_path

    print(f"[info] Enriching {len(domains)} domain(s) with provider={settings.llm_provider.value} wait_until={settings.wait_until}")
    print(f"[info] Domains: {', '.join(domains)}")

    results: list[CompanyEnrichment] = await enrich_domains(domains)

    # Serialize: CompanyEnrichment -> json with mode='json' to handle HttpUrl/EmailStr
    output_data = [r.model_dump(mode="json") for r in results]
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] Wrote {len(output_data)} results to {output_path}")

    # Summary for demo
    for r in results:
        status = "ok" if r.confidence_score > 0 else "low-confidence"
        print(f"  - {r.domain}: confidence={r.confidence_score} sources={len(r.source_pages)} errors={len(r.errors)} [{status}]")
        if r.errors:
            for e in r.errors[:2]:
                print(f"      err: {e[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
