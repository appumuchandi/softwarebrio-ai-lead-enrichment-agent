"""
HTML -> clean Markdown.

No raw HTML passed to LLM. Uses stdlib html.parser + regex, no heavy deps for vertical slice.
Production upgrade path: trafilatura/readability (noted in docstring) without changing interface.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional


# Tags to drop entirely (including content)
_DROP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
# Tags to unwrap (keep text)
_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "main", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br"}


class _MarkdownConverter(HTMLParser):
    """Minimal HTML -> Markdown converter."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip_depth = 0
        self._in_anchor: Optional[str] = None
        self._anchor_text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href")
            self._in_anchor = href
            self._anchor_text = []
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self._out.append("\n" + "#" * level + " ")
        elif tag == "li":
            self._out.append("\n- ")
        elif tag == "br":
            self._out.append("\n")
        elif tag in ("p", "div", "section", "article", "tr"):
            self._out.append("\n\n")
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if tag == "a" and self._in_anchor is not None:
            text = "".join(self._anchor_text).strip()
            href = self._in_anchor
            self._in_anchor = None
            self._anchor_text = []
            if text and href and href.startswith("http"):
                # Keep markdown link only if meaningful
                if len(text) > 1:
                    self._out.append(f"[{text}]({href})")
                else:
                    self._out.append(text)
            elif text:
                self._out.append(text)
        elif tag == "title":
            self._in_title = False
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_anchor is not None:
            self._anchor_text.append(data)
        else:
            # Collapse whitespace but keep single spaces
            self._out.append(data)

    def get_markdown(self) -> str:
        raw = "".join(self._out)
        # normalize whitespace
        raw = re.sub(r"\r\n", "\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # strip each line trailing
        lines = [ln.rstrip() for ln in raw.split("\n")]
        # drop empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines).strip()


def html_to_markdown(html: str, url: str = "", max_chars: int = 12000) -> str:
    """
    Convert HTML to clean markdown.

    - Strips scripts/styles/nav artifacts via drop tags.
    - Preserves headings, links, lists.
    - Truncates to max_chars (keeps head, which is most relevant for enrichment).
    - Never returns raw HTML.
    """
    if not html or not html.strip():
        return ""
    # Quick path: if html is already plain text (no tags), just normalize. Otherwise parse as HTML.
    # Previously checked only <html>/<body>/<div> which missed <h1>/<p> snippets used in tests and small fetches.
    lower = html.lower()
    has_tag = "<" in html and ">" in html and bool(re.search(r"</?[a-zA-Z][^>]*>", html))
    if not has_tag:
        text = re.sub(r"\n{3,}", "\n\n", html.strip())
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n\n[truncated]"
        return text.strip()

    parser = _MarkdownConverter()
    try:
        parser.feed(html)
    except Exception:
        # Fallback: strip tags brutally
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars].strip()
    md = parser.get_markdown()
    # Add source header for evidence-aware extraction (consumer adds it explicitly, but keep here for trace)
    # Don't add url here; orchestrator adds SOURCE delimiters.
    if len(md) > max_chars:
        md = md[:max_chars].rstrip() + "\n\n[truncated]"
    return md


def extract_emails_from_text(text: str) -> list[str]:
    """Lightweight regex fallback for contact emails visible in markdown."""
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    return list(dict.fromkeys(re.findall(pattern, text)))  # dedup preserve order


def clean_markdown_for_llm(markdown: str, max_chars: int = 12000) -> str:
    """Final normalization before LLM: collapse, truncate, remove excessive blank lines."""
    if not markdown:
        return ""
    md = markdown.strip()
    md = re.sub(r"\n{3,}", "\n\n", md)
    if len(md) > max_chars:
        md = md[:max_chars].rstrip() + "\n\n[truncated to {} chars]".format(max_chars)
    return md
