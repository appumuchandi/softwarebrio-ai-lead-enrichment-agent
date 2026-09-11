"""
Centralized configuration. LLM provider is fully env-configurable per requirements.

Supported:
- LLM_PROVIDER=openai|anthropic|mock  (mock = no API call, deterministic stub for offline dev)
- OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
- ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    mock = "mock"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM (env-configurable) ---
    llm_provider: LLMProvider = Field(default=LLMProvider.openai, alias="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-haiku-latest", alias="ANTHROPIC_MODEL")

    # temperature 0 for deterministic extraction
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1800, alias="LLM_MAX_TOKENS")

    # --- Scraping ---
    # Primary navigation strategy per architectural requirement
    wait_until: str = Field(default="domcontentloaded", alias="WAIT_UNTIL")
    navigation_timeout_ms: int = Field(default=15000, alias="NAV_TIMEOUT_MS")
    page_goto_retries: int = Field(default=2, alias="PAGE_RETRIES")
    browser_headless: bool = Field(default=True, alias="HEADLESS")
    # concurrency per run (fail-soft: semaphore)
    max_concurrent_domains: int = Field(default=3, alias="MAX_CONCURRENT_DOMAINS")
    max_concurrent_pages: int = Field(default=4, alias="MAX_CONCURRENT_PAGES")

    # --- Paths ---
    output_path: Path = Field(default=Path("output.json"), alias="OUTPUT_PATH")
    input_path: Path = Field(default=Path("domains.txt"), alias="INPUT_PATH")

    # --- Cleaner ---
    max_markdown_chars_per_page: int = Field(default=12000, alias="MAX_MD_CHARS_PER_PAGE")
    max_total_markdown_chars: int = Field(default=30000, alias="MAX_TOTAL_MD_CHARS")

    def validate_for_provider(self) -> Optional[str]:
        """Return error string if config invalid for selected provider, else None."""
        if self.llm_provider == LLMProvider.openai and not self.openai_api_key:
            return "LLM_PROVIDER=openai requires OPENAI_API_KEY"
        if self.llm_provider == LLMProvider.anthropic and not self.anthropic_api_key:
            return "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY"
        return None


# Singleton accessor - lazy to allow .env to be loaded after import
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """For tests."""
    global _settings
    _settings = None
