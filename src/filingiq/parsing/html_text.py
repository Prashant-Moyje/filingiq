"""HTML -> analysis-ready text.

Modern SEC filings are inline-XBRL HTML: thousands of nested <div> and <span>
tags, non-breaking spaces, and financial data locked inside <table> elements.

Two requirements drive this module:

1. BLOCK BOUNDARIES MUST SURVIVE. The sectioniser identifies headings by
   requiring "Item 7" to appear at the start of a line. If we flatten the HTML
   into one long string, every cross-reference in body text ("...as discussed
   in Item 7...") becomes indistinguishable from the real heading. Newlines are
   not cosmetic here -- they are the signal.

2. TABLES MUST NOT BE SHREDDED. Naive text extraction turns a financial table
   into a vertical column of orphaned numbers with no row or column context.
   The extraction agent then reads "383,285" with nothing attached to it. We
   render tables as pipe-delimited rows so a row stays a row.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Elements that imply a line break in the rendered document.
_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "table", "thead", "tbody",
    "hr", "blockquote", "dd", "dt", "figcaption",
}

_DROP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}

_WS_CHARS = "\u00a0\u2007\u202f\u2009\u200a\u200b\ufeff"
_WS_TRANS = str.maketrans({c: " " for c in _WS_CHARS})


@dataclass
class ParsedDocument:
    text: str
    n_chars: int
    n_tables: int
    parser_used: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines, without destroying newlines."""
    text = text.translate(_WS_TRANS)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_text(table_el) -> str:
    """Render an lxml <table> as pipe-delimited rows."""
    rows_out: list[str] = []
    for tr in table_el.iter("tr"):
        cells = []
        for cell in tr.iter("td", "th"):
            txt = " ".join(cell.itertext())
            txt = txt.translate(_WS_TRANS)
            txt = re.sub(r"\s+", " ", txt).strip()
            cells.append(txt)
        # SEC tables are heavily padded with empty spacer cells for layout.
        cells = [c for c in cells if c not in ("", "$", ")", "(", "%")]
        if cells:
            rows_out.append(" | ".join(cells))
    if not rows_out:
        return ""
    return "\n[TABLE]\n" + "\n".join(rows_out) + "\n[/TABLE]\n"


def _extract_with_lxml(raw: bytes) -> tuple[str, int]:
    """Parse with lxml.

    IMPORTANT: lxml must receive BYTES, not a decoded str. Many SEC filings
    begin with an XML declaration (<?xml version="1.0" encoding="UTF-8"?>),
    and lxml raises "Unicode strings with encoding declaration are not
    supported" if you hand it an already-decoded string. Decoding first and
    passing the str silently pushed 7 of 16 filings onto the regex fallback,
    destroying every financial table in them. See FAILURE_MODES.md FM-003.
    """
    from lxml import html as lxml_html

    tree = lxml_html.fromstring(raw)

    for el in tree.iter():
        if el.tag in _DROP_TAGS:
            el.drop_tree()

    n_tables = 0
    parts: list[str] = []

    def walk(el) -> None:
        nonlocal n_tables
        tag = el.tag if isinstance(el.tag, str) else ""

        if tag == "table":
            rendered = _table_to_text(el)
            if rendered:
                n_tables += 1
                parts.append(rendered)
            if el.tail:
                parts.append(el.tail)
            return

        if tag in _BLOCK_TAGS:
            parts.append("\n")
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
        if tag in _BLOCK_TAGS:
            parts.append("\n")
        if el.tail:
            parts.append(el.tail)

    walk(tree)
    return "".join(parts), n_tables


def _extract_with_regex(raw: str) -> tuple[str, int]:
    """Dependency-free fallback. Lower quality -- tables are lost -- but it
    keeps the pipeline running if lxml is unavailable."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?i)<(br|/p|/div|/tr|/h[1-6]|/li)[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text), 0


def html_to_text(raw_html: str | bytes) -> ParsedDocument:
    # Keep the bytes intact for lxml; only decode for the regex fallback.
    raw_bytes = raw_html.encode("utf-8") if isinstance(raw_html, str) else raw_html

    try:
        text, n_tables = _extract_with_lxml(raw_bytes)
        parser = "lxml"
    except ImportError:
        log.warning("lxml not installed -- falling back to regex extraction. "
                    "Tables will be degraded. Run: pip install lxml")
        text, n_tables = _extract_with_regex(raw_bytes.decode("utf-8", errors="replace"))
        parser = "regex-fallback"
    except Exception as exc:  # noqa: BLE001
        log.warning("lxml parse failed (%s) -- using regex fallback", exc)
        text, n_tables = _extract_with_regex(raw_bytes.decode("utf-8", errors="replace"))
        parser = "regex-fallback"

    text = html.unescape(text)
    text = normalise_whitespace(text)
    return ParsedDocument(text=text, n_chars=len(text), n_tables=n_tables,
                          parser_used=parser)


def load_filing_text(path) -> ParsedDocument:
    from pathlib import Path

    p = Path(path)
    raw = p.read_bytes()
    if p.suffix.lower() in {".htm", ".html"}:
        return html_to_text(raw)
    text = normalise_whitespace(raw.decode("utf-8", errors="replace"))
    return ParsedDocument(text=text, n_chars=len(text), n_tables=0,
                          parser_used="plaintext")
