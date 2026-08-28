"""Split sections into retrieval units.

Chunking is the single highest-leverage decision in a RAG system and the one
most often made by accident. `text[i:i+1000]` is fast to write and quietly
destroys the thing you are trying to retrieve.

Three rules drive this module:

1. NEVER SPLIT A TABLE. A financial table cut in half yields one chunk of
   column headers with no numbers and another of numbers with no labels.
   Neither is retrievable and neither is answerable. Tables that exceed the
   chunk budget are split BY ROW with the header repeated in each piece, so
   every chunk remains self-describing.

2. SPLIT ON SEMANTIC BOUNDARIES, NOT CHARACTER COUNTS. Risk factors are
   discrete units, each with its own heading. Cutting mid-risk-factor produces
   a chunk that begins mid-argument and ends mid-sentence.

3. CARRY THE CONTEXT DOWN. A chunk reading "increased 12% year over year" is
   useless in isolation -- 12% of what, for whom, when? Every chunk is prefixed
   with its company, fiscal year, and item, so the embedding encodes what the
   passage is ABOUT and not merely what it says.

Rule 3 is what makes cross-company retrieval work at all. Without it, a query
about Apple's FY2024 margins happily retrieves Microsoft's FY2022 chunk,
because the raw sentences are nearly identical.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger(__name__)

TARGET_TOKENS = 450       # sweet spot for most embedding models
MAX_TOKENS = 700          # hard ceiling before forced split
OVERLAP_TOKENS = 60       # continuity across boundaries
MIN_CHUNK_TOKENS = 25     # below this, merge forward rather than emit

_TABLE_RE = re.compile(r"\[TABLE\](.*?)\[/TABLE\]", re.S)

# Orphan markers appear when a section boundary lands inside a table -- an item
# heading can legitimately sit in a table row, since filers wrap page-number
# footers in tables. The section then begins with a dangling [/TABLE] or ends
# with a dangling [TABLE], and that orphan flows into a prose block, producing
# a chunk whose markers do not balance.
#
# A chunk containing "[/TABLE]" with no opening marker tells a retrieval model
# a table is present when none is, and confuses any downstream table-aware
# parsing. Cheaper to strip the orphan than to teach every consumer about it.
_ORPHAN_MARKER_RE = re.compile(r"\[/?TABLE\]")

# Risk-factor style headings: short lines, often bold in the original, that
# introduce a discrete disclosure. Good natural split points.
_HEADING_RE = re.compile(r"^[A-Z][^\n]{10,120}[.:]?$", re.M)


def count_tokens(text: str) -> int:
    """Token count, with a graceful fallback.

    tiktoken is exact for OpenAI-family tokenisers and close enough for others.
    When unavailable, words x 1.33 approximates English prose well; financial
    text with many numbers runs slightly higher, which errs toward smaller
    chunks -- the safe direction.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001
        return int(len(text.split()) * 1.33)


@dataclass
class Block:
    """An atomic unit that must not be split further."""
    text: str
    kind: str              # 'prose' | 'table' | 'heading'
    n_tokens: int
    char_start: int


@dataclass
class Chunk:
    chunk_id: str
    accession: str
    item: str
    chunk_index: int
    text: str              # context-prefixed text, what gets embedded
    raw_text: str          # original passage, what gets shown to a user
    n_tokens: int
    char_start: int
    char_end: int
    has_table: bool
    metadata: dict = field(default_factory=dict)


def _split_into_blocks(text: str) -> list[Block]:
    """Break a section into atomic blocks, keeping tables whole."""
    blocks: list[Block] = []
    cursor = 0

    for m in _TABLE_RE.finditer(text):
        # prose before this table
        before = text[cursor:m.start()]
        if before.strip():
            blocks.extend(_prose_blocks(before, cursor))
        table_text = m.group(0)
        blocks.append(
            Block(text=table_text, kind="table",
                  n_tokens=count_tokens(table_text), char_start=m.start())
        )
        cursor = m.end()

    tail = text[cursor:]
    if tail.strip():
        blocks.extend(_prose_blocks(tail, cursor))
    return blocks


def _prose_blocks(text: str, offset: int) -> Iterator[Block]:
    """Paragraph-level blocks, tagging short title-case lines as headings.

    Any [TABLE] / [/TABLE] marker reaching here is an orphan by definition --
    balanced pairs were consumed by _split_into_blocks before this ran. Strip
    them so no chunk can carry an unbalanced marker.
    """
    pos = 0
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            pos += len(para) + 2
            continue
        start = offset + text.find(para, pos)
        cleaned = _ORPHAN_MARKER_RE.sub("", para).strip()
        pos += len(para) + 2
        if not cleaned:
            continue
        kind = "heading" if _HEADING_RE.fullmatch(cleaned) else "prose"
        yield Block(text=cleaned, kind=kind,
                    n_tokens=count_tokens(cleaned), char_start=start)


def _split_large_table(block: Block) -> list[Block]:
    """Split an oversized table by rows, repeating the header in each piece.

    Without header repetition the second half of a split table is a grid of
    numbers with no column labels -- unretrievable and unanswerable.
    """
    inner = block.text.replace("[TABLE]", "").replace("[/TABLE]", "").strip()
    rows = [r for r in inner.split("\n") if r.strip()]
    if len(rows) <= 2:
        return [block]

    header = rows[0]
    header_tokens = count_tokens(header)
    out: list[Block] = []
    current: list[str] = []
    current_tokens = header_tokens

    for row in rows[1:]:
        rt = count_tokens(row)
        if current and current_tokens + rt > TARGET_TOKENS:
            body = "[TABLE]\n" + header + "\n" + "\n".join(current) + "\n[/TABLE]"
            out.append(Block(text=body, kind="table",
                             n_tokens=count_tokens(body),
                             char_start=block.char_start))
            current, current_tokens = [], header_tokens
        current.append(row)
        current_tokens += rt

    if current:
        body = "[TABLE]\n" + header + "\n" + "\n".join(current) + "\n[/TABLE]"
        out.append(Block(text=body, kind="table", n_tokens=count_tokens(body),
                         char_start=block.char_start))

    log.debug("Split oversized table into %d pieces (header repeated)", len(out))
    return out


def _context_prefix(meta: dict) -> str:
    """The line prepended to every chunk before embedding."""
    bits = [
        meta.get("ticker", ""),
        f"FY{meta.get('fiscal_year')}" if meta.get("fiscal_year") else "",
        f"Item {meta.get('item')}" if meta.get("item") else "",
        meta.get("item_title", ""),
    ]
    return " | ".join(b for b in bits if b)


def chunk_section(
    text: str,
    accession: str,
    item: str,
    metadata: dict | None = None,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Pack a section into overlapping, table-safe, context-prefixed chunks."""
    metadata = dict(metadata or {})
    metadata.setdefault("item", item)
    prefix = _context_prefix(metadata)

    blocks = _split_into_blocks(text)

    # Explode any table too large to fit a chunk.
    expanded: list[Block] = []
    for b in blocks:
        if b.kind == "table" and b.n_tokens > max_tokens:
            expanded.extend(_split_large_table(b))
        else:
            expanded.append(b)

    chunks: list[Chunk] = []
    current: list[Block] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        raw = "\n\n".join(b.text for b in current)
        idx = len(chunks)
        body = f"{prefix}\n\n{raw}" if prefix else raw
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(accession, item, idx),
                accession=accession,
                item=item,
                chunk_index=idx,
                text=body,
                raw_text=raw,
                n_tokens=count_tokens(body),
                char_start=current[0].char_start,
                char_end=current[-1].char_start + len(current[-1].text),
                has_table=any(b.kind == "table" for b in current),
                metadata=metadata,
            )
        )
        # Carry the tail forward as overlap, but never a table (duplicating a
        # table across chunks inflates the index and skews retrieval).
        carry: list[Block] = []
        carried = 0
        for b in reversed(current):
            if b.kind == "table" or carried >= overlap_tokens:
                break
            carry.insert(0, b)
            carried += b.n_tokens
        current = carry
        current_tokens = carried

    for b in expanded:
        # Start a fresh chunk at a heading if the current one is already full-ish.
        if b.kind == "heading" and current_tokens > target_tokens * 0.6:
            flush()
        if current_tokens + b.n_tokens > target_tokens and current:
            flush()
        current.append(b)
        current_tokens += b.n_tokens
        if current_tokens >= max_tokens:
            flush()

    flush()

    # Merge a trailing runt into its predecessor.
    if len(chunks) > 1 and chunks[-1].n_tokens < MIN_CHUNK_TOKENS:
        last = chunks.pop()
        prev = chunks[-1]
        prev.raw_text += "\n\n" + last.raw_text
        prev.text += "\n\n" + last.raw_text
        prev.n_tokens = count_tokens(prev.text)
        prev.char_end = last.char_end

    return chunks


def _chunk_id(accession: str, item: str, idx: int) -> str:
    return hashlib.sha1(f"{accession}:{item}:{idx}".encode()).hexdigest()[:16]
