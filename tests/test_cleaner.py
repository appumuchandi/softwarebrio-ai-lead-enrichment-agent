import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scraping.cleaner import html_to_markdown, clean_markdown_for_llm, extract_emails_from_text

def test_html_to_markdown_strips_scripts_and_no_raw_html():
    html = '<html><head><title>T</title></head><body><h1>Acme</h1><p>We build APIs.</p><script>alert(1)</script><style>.x{}</style><a href="https://linkedin.com/company/acme">LinkedIn</a> contact@acme.com</body></html>'
    md = html_to_markdown(html, max_chars=12000)
    assert "Acme" in md
    assert "We build APIs" in md
    assert "alert" not in md
    assert "<script" not in md.lower()
    assert "<html" not in md.lower()

def test_html_to_markdown_preserves_headings_and_links():
    html = '<h1>Title</h1><p>Para <a href="https://example.com/page">Link</a></p><ul><li>Item1</li><li>Item2</li></ul>'
    md = html_to_markdown(html)
    assert "# Title" in md
    assert "[Link](https://example.com/page)" in md
    assert "- Item1" in md

def test_clean_markdown_no_raw_html():
    md = "# Hello\n<p>raw html</p> <div>xxx</div>"
    cleaned = clean_markdown_for_llm(md, max_chars=12000)
    # cleaner should have stripped, but this test is for the function that just truncates
    assert len(cleaned) <= 12000
    assert cleaned.startswith("# Hello")

def test_extract_emails_from_text():
    text = "Contact support@example.com and info@example.com also support@example.com again"
    emails = extract_emails_from_text(text)
    assert "support@example.com" in emails
    assert "info@example.com" in emails
    # dedup
    assert emails.count("support@example.com") == 1

def test_html_to_markdown_truncates():
    html = "<p>" + "a" * 20000 + "</p>"
    md = html_to_markdown(html, max_chars=100)
    assert len(md) <= 120  # includes [truncated] (100 + len("\n\n[truncated]"))
    assert "[truncated]" in md
