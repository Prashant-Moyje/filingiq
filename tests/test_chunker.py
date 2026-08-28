"""Tests for the chunker.

The failure these guard against is subtle: a broken chunker still produces
chunks, still embeds them, still returns results. It just returns the WRONG
results, and you discover this in Week 7 when retrieval recall is inexplicably
mediocre and you have no idea which of six components is at fault.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.parsing.chunker import (  # noqa: E402
    MAX_TOKENS,
    chunk_section,
    count_tokens,
    _split_into_blocks,
)


def _para(word: str, n: int) -> str:
    return " ".join([word] * n)


TABLE = """[TABLE]
(in millions) | 2024 | 2023 | 2022
Net sales | 391,035 | 383,285 | 394,328
Cost of sales | 210,352 | 214,137 | 223,546
Gross margin | 180,683 | 169,148 | 170,782
[/TABLE]"""

SECTION = "\n\n".join([
    _para("revenue", 200),
    TABLE,
    _para("margin", 200),
    _para("expenses", 300),
])

META = {"ticker": "AAPL", "fiscal_year": 2024, "item": "7",
        "item_title": "Management's Discussion and Analysis"}


@pytest.fixture
def chunks():
    return chunk_section(SECTION, "0000320193-24-000123", "7", META)


# --- rule 1: never split a table -------------------------------------------

def test_table_is_never_split_across_chunks(chunks):
    """The critical invariant. A half-table is unretrievable AND unanswerable."""
    for c in chunks:
        assert c.raw_text.count("[TABLE]") == c.raw_text.count("[/TABLE]"), (
            f"chunk {c.chunk_index} contains an unbalanced table marker"
        )


def test_table_stays_with_its_numbers(chunks):
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) >= 1
    for c in table_chunks:
        assert "Net sales" in c.raw_text and "391,035" in c.raw_text, (
            "column labels separated from their values"
        )


def test_blocks_identify_tables():
    blocks = _split_into_blocks(SECTION)
    kinds = [b.kind for b in blocks]
    assert "table" in kinds
    table_block = next(b for b in blocks if b.kind == "table")
    assert "Gross margin" in table_block.text


# --- oversized tables split by row, header repeated ------------------------

def test_oversized_table_repeats_header_in_every_piece():
    rows = "\n".join(f"Line item {i} | {i*100} | {i*90} | {i*80}" for i in range(400))
    big = "[TABLE]\n(in millions) | 2024 | 2023 | 2022\n" + rows + "\n[/TABLE]"
    out = chunk_section(big, "acc", "8", META)
    assert len(out) > 1, "oversized table was not split at all"
    for c in out:
        assert "(in millions) | 2024" in c.raw_text, (
            "header not repeated -- this chunk is numbers with no labels"
        )


# --- rule 3: context prefix ------------------------------------------------

def test_every_chunk_carries_company_and_period(chunks):
    """Without this, a query about Apple FY2024 retrieves Microsoft FY2022,
    because the underlying sentences are nearly identical."""
    for c in chunks:
        assert "AAPL" in c.text
        assert "FY2024" in c.text
        assert "Item 7" in c.text


def test_raw_text_excludes_the_prefix(chunks):
    """Embed the prefixed version; show the user the original."""
    for c in chunks:
        assert not c.raw_text.startswith("AAPL |")


# --- sizing ----------------------------------------------------------------

def test_chunks_respect_the_ceiling(chunks):
    for c in chunks:
        assert c.n_tokens <= MAX_TOKENS * 1.6, (
            f"chunk {c.chunk_index} is {c.n_tokens} tokens"
        )


def test_no_empty_or_runt_chunks(chunks):
    assert all(c.raw_text.strip() for c in chunks)
    assert all(c.n_tokens > 20 for c in chunks)


def test_chunk_ids_are_unique_and_stable(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    again = chunk_section(SECTION, "0000320193-24-000123", "7", META)
    assert [c.chunk_id for c in again] == ids, "chunk IDs must be deterministic"


def test_chunk_indices_are_sequential(chunks):
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --- coverage --------------------------------------------------------------

def test_no_content_is_silently_dropped(chunks):
    """Overlap means chunks repeat text, but nothing may VANISH."""
    joined = " ".join(c.raw_text for c in chunks)
    for marker in ["revenue", "margin", "expenses", "391,035"]:
        assert marker in joined, f"{marker!r} lost during chunking"


def test_short_section_yields_one_chunk():
    out = chunk_section("Item 7A. The Company faces interest rate risk.",
                        "acc", "7A", META)
    assert len(out) == 1


def test_token_counter_is_sane():
    assert count_tokens("") == 0
    assert 3 <= count_tokens("the quick brown fox jumps") <= 12


# --- orphan table markers (section boundary landed inside a table) ---------

def test_orphan_closing_marker_is_stripped():
    """A section starting mid-table begins with a dangling [/TABLE]."""
    section = "[/TABLE]\n\n" + _para("revenue", 300)
    out = chunk_section(section, "acc", "5", META)
    for c in out:
        assert "[/TABLE]" not in c.raw_text
        assert c.raw_text.count("[TABLE]") == c.raw_text.count("[/TABLE]")


def test_orphan_opening_marker_is_stripped():
    section = _para("revenue", 300) + "\n\n[TABLE]"
    out = chunk_section(section, "acc", "5", META)
    for c in out:
        assert c.raw_text.count("[TABLE]") == c.raw_text.count("[/TABLE]")


def test_balanced_tables_survive_orphan_stripping():
    """Stripping orphans must not damage real tables."""
    section = _para("intro", 100) + "\n\n" + TABLE + "\n\n" + _para("outro", 100)
    out = chunk_section(section, "acc", "8", META)
    joined = " ".join(c.raw_text for c in out)
    assert "[TABLE]" in joined and "391,035" in joined
    for c in out:
        assert c.raw_text.count("[TABLE]") == c.raw_text.count("[/TABLE]")


def test_mixed_orphans_and_real_tables():
    section = "[/TABLE]\n\n" + _para("a", 150) + "\n\n" + TABLE + "\n\n" + _para("b", 150) + "\n\n[TABLE]"
    out = chunk_section(section, "acc", "8", META)
    for c in out:
        assert c.raw_text.count("[TABLE]") == c.raw_text.count("[/TABLE]"), c.raw_text[:200]
    assert any("Net sales" in c.raw_text for c in out)
