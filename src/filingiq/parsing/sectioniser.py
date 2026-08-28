"""Split a 10-K into its Items (1, 1A, 7, 7A, 8, ...).

THE CENTRAL PROBLEM
-------------------
The string "Item 1A. Risk Factors" appears in a 10-K at least three times:

  1. In the table of contents, near the front.
  2. As the actual section heading, ~30 pages in.
  3. In cross-references scattered through body text
     ("...see the risks described in Item 1A...").

A regex that takes the first match returns the table of contents, giving you a
"Risk Factors section" that is 40 characters long. Every downstream metric --
retrieval recall, extraction accuracy, YoY diffs -- is then computed on garbage,
and the failure is silent because the pipeline still runs.

HOW WE DISAMBIGUATE (three independent filters)
-----------------------------------------------
1. POSITIONAL: a real heading starts a line. Cross-references sit mid-sentence.
   Requires the block-boundary newlines preserved by html_text.py.

2. DENSITY: a table of contents packs many item headings into a small span.
   Body sections are spread across tens of thousands of characters. We find the
   densest cluster and excise it.

3. MONOTONICITY + LENGTH: real sections appear in order and have substance. We
   select the candidate sequence that is strictly increasing in position, then
   reject sections below a minimum length.

Anything the sectioniser is not confident about is reported, not guessed. A
missing section is recoverable; a wrong one poisons everything downstream.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# Ordered canonical item list for a 10-K. Order matters -- it drives the
# monotonicity check and defines where each section ends.
ITEM_SEQUENCE: list[tuple[str, str]] = [
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"),          # required from FY2023 onward
    ("2", "Properties"),
    ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Market for Registrant's Common Equity"),
    ("6", "Selected Financial Data"),
    ("7", "Management's Discussion and Analysis"),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("8", "Financial Statements and Supplementary Data"),
    ("9", "Changes in and Disagreements with Accountants"),
    ("9A", "Controls and Procedures"),
    ("9B", "Other Information"),
    ("10", "Directors and Executive Officers"),
    ("11", "Executive Compensation"),
    ("12", "Security Ownership"),
    ("13", "Certain Relationships and Related Transactions"),
    ("14", "Principal Accountant Fees and Services"),
    ("15", "Exhibits and Financial Statement Schedules"),
]

# The sections we actually care about for FilingIQ.
TARGET_ITEMS = ["1", "1A", "7", "7A", "8"]

# Below this, a "section" is almost certainly a TOC row or a cross-reference.
# Absolute floors, calibrated on standard filers. Scaled by document size at
# runtime: JPM's 10-K is 6.7x larger than Apple's, so a fixed word count is the
# wrong unit. See _scaled_threshold().
MIN_SECTION_CHARS = {
    "1": 3000,
    "1A": 5000,
    "7": 5000,
    "7A": 400,     # legitimately short for many filers
    "8": 3000,
}
DEFAULT_MIN_CHARS = 1000

# A table of contents is a run of ADJACENT lines. Consecutive entries sit a few
# dozen characters apart; the gap between the last TOC row and the first real
# body heading is thousands of characters. Detecting the run by gap rather than
# by a fixed window is what stops the excision from eating the first real
# section. See FAILURE_MODES.md FM-002.
_TOC_MAX_GAP = 600       # max chars between consecutive TOC entries
_TOC_MIN_ITEMS = 6       # distinct items in the run to call it a TOC


@dataclass
class Candidate:
    item: str
    start: int
    matched_text: str


@dataclass
class Section:
    item: str
    title: str
    start: int
    end: int
    text: str
    source: str = "item-heading"   # or "title-fallback"

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class SectionResult:
    sections: dict[str, Section] = field(default_factory=dict)
    toc_span: tuple[int, int] | None = None
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def found_items(self) -> list[str]:
        return sorted(self.sections, key=lambda i: [x[0] for x in ITEM_SEQUENCE].index(i))

    def coverage(self, targets: list[str] | None = None) -> float:
        targets = targets or TARGET_ITEMS
        hit = sum(1 for t in targets if t in self.sections)
        return round(100 * hit / max(len(targets), 1), 1)


def _item_pattern(item: str) -> re.Pattern:
    """Match 'Item 7A' at the start of a line, tolerating the punctuation zoo
    real filings use: 'Item 7A.', 'ITEM 7A -', 'Item 7A:', 'Item7A'.

    The negative lookahead on [A-Z] stops 'Item 1' from matching 'Item 1A'.
    """
    num = re.escape(item)
    return re.compile(
        rf"(?im)^[\s>|]*item\s*{num}(?![0-9A-Za-z])\s*[\.\:\-–—\)]?\s*",
    )


def find_candidates(text: str, items: list[str] | None = None) -> list[Candidate]:
    items = items or [i for i, _ in ITEM_SEQUENCE]
    out: list[Candidate] = []
    for item in items:
        for m in _item_pattern(item).finditer(text):
            out.append(Candidate(item=item, start=m.start(), matched_text=m.group(0).strip()))
    out.sort(key=lambda c: c.start)
    return out


_ITEM_ORDER = {item: i for i, (item, _) in enumerate(ITEM_SEQUENCE)}


# ---------------------------------------------------------------------------
# TITLE FALLBACK
#
# Some filers -- large banks especially -- do not label body sections with Item
# numbers at all. JPMorgan's 10-K is a wrapper: its index cross-references page
# ranges, and the substantive content appears under plain titles like
# "Management's discussion and analysis" with no "Item 7" anywhere near it.
#
# For those filings the item-number sectioniser correctly finds nothing, and
# whichever section precedes the gap absorbs the remainder of the document.
#
# So when an item cannot be located by number, we look for its canonical title
# instead. Sections recovered this way are tagged `title-fallback` so the
# provenance is visible downstream -- they are a weaker signal than a numbered
# heading and should be spot-checked before being trusted.
# ---------------------------------------------------------------------------
# Apostrophes in SEC filings arrive as U+2019, ASCII ', backtick, or mojibake
# from an encoding mismatch ("Management's" rendering as "ManagementÆs" or
# "Managementâ€™s"). Note that Æ, â, €, ™ are LETTERS to Unicode's \w, so a
# plain [^\w\s] class does not catch them -- they must be listed.
_APOS = r"[\W_Æâ€™´`]{0,3}"

TITLE_PATTERNS: dict[str, str] = {
    "1A": rf"risk\s+factors",
    "7": rf"management{_APOS}s\s+discussion\s+and\s+analysis",
    "7A": rf"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk",
    "8": (
        rf"(?:report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm"
        rf"|consolidated\s+financial\s+statements(?:\s+and\s+supplementary\s+data)?"
        rf"|financial\s+statements\s+and\s+supplementary\s+data)"
    ),
}


def _title_pattern(item: str) -> re.Pattern | None:
    body = TITLE_PATTERNS.get(item)
    if not body:
        return None
    # Must occupy its own line: a heading, not a sentence fragment.
    return re.compile(rf"(?im)^[\s>|]*{body}\s*[\.\:\-–—]?\s*$")


def _wrapper_title_pattern(item: str) -> re.Pattern | None:
    """Line-anchored but NOT end-anchored.

    In wrapper filings the content heading is followed on the same line by the
    opening words of the section ("Management's discussion and analysis The
    following is..."), so requiring end-of-line never matches.
    """
    body = TITLE_PATTERNS.get(item)
    if not body:
        return None
    return re.compile(rf"(?im)^[\s>|]*{body}")


# A section whose entire body is a pointer to content elsewhere. JPM's Item 7A
# reads "Refer to the Market Risk Management section on pages 141-149." -- 87
# words, comfortably over any length threshold, and yet not a section at all.
# Length cannot separate these from genuinely short sections (7A is legitimately
# brief for many filers); the referring LANGUAGE can.
_XREF_PATTERNS = re.compile(
    r"(?i)(?:refer\s+to|see\s+(?:the\s+)?|incorporated\s+(?:herein\s+)?by\s+reference"
    r"|appears?\s+on\s+pages?|included\s+(?:herein\s+)?in|set\s+forth\s+(?:in|under))"
)
_PAGE_REF = re.compile(r"(?i)pages?\s+\d+")

# Stubs are SHORT. 150 words is the empirical ceiling: JPM's Item 7A stub is
# 87 words, while their genuinely-brief-but-real Item 2 (351) and Item 9A (328)
# sit well above it. An earlier 400-word ceiling swallowed both, because real
# short sections also say "refer to page N" -- referring language alone cannot
# separate a pointer from a section that merely cites one. See FM-005.
_XREF_MAX_WORDS = 150
_XREF_STRONG_MAX_WORDS = 120


def _is_cross_reference_stub(body: str) -> bool:
    """True if this section is merely a pointer to content located elsewhere.

    Requires BOTH brevity and referring language. Real sections cite other
    sections constantly; what makes a stub a stub is that the citation is
    essentially all there is.
    """
    words = body.split()
    if len(words) > _XREF_MAX_WORDS:
        return False
    # Drop the heading line -- it always restates the item title.
    tail = body.split("\n", 1)[-1] if "\n" in body else body
    if not _XREF_PATTERNS.search(tail):
        return False
    return bool(_PAGE_REF.search(tail)) or len(tail.split()) < _XREF_STRONG_MAX_WORDS


def _looks_like_prose(sample: str) -> bool:
    """Distinguish a real section opening from a contents-index row.

    Wrapper filings contain a SECOND table of contents -- the appended annual
    report's own index -- whose rows match the same title patterns as the real
    headings. The two are easy to tell apart by what follows them:

        index row : "...: | 172 | Consolidated Financial Statements 52 | ..."
        real start: "The following is Management's discussion and analysis of..."

    Pipes and digit density separate them cleanly, with no position tuning.
    """
    sample = sample[:400]
    if not sample.strip():
        return False
    if "[TABLE]" in sample[:120]:
        return False
    if sample.count("|") > 2:
        return False
    n = len(sample)
    if sum(c.isdigit() for c in sample) / n > 0.08:
        return False
    if sum(c.isalpha() for c in sample) / n < 0.60:
        return False
    return True


def _recover_wrapper_sections(
    text: str,
    stub_items: list[str],
    region_start: int,
) -> dict[str, Candidate]:
    """Locate real content for items whose numbered heading was only a stub.

    Searches the trailing region (everything after the last numbered item) for
    line-anchored title matches followed by prose. Enforces ordering among the
    recovered items so Item 8 cannot land before Item 7.
    """
    recovered: dict[str, Candidate] = {}
    cursor = region_start

    for item in sorted(stub_items, key=lambda i: _ITEM_ORDER.get(i, 99)):
        pat = _wrapper_title_pattern(item)
        if pat is None:
            continue
        for m in pat.finditer(text):
            if m.start() <= cursor:
                continue
            if not _looks_like_prose(text[m.end():m.end() + 400]):
                continue
            recovered[item] = Candidate(
                item=item, start=m.start(), matched_text=m.group(0).strip()
            )
            cursor = m.start()
            break

    return recovered


def _build_sections(
    text: str,
    chosen: dict[str, Candidate],
    sources: dict[str, str],
    min_chars: dict[str, int],
) -> tuple[dict[str, Section], list[str], list[str]]:
    """Assign boundaries and apply the minimum-length filter.

    Returns (sections, rejected_messages, stub_items).
    """
    ordered = sorted(chosen.values(), key=lambda c: c.start)
    titles = dict(ITEM_SEQUENCE)
    sections: dict[str, Section] = {}
    rejected: list[str] = []
    stubs: list[str] = []

    for idx, cand in enumerate(ordered):
        start = cand.start
        end = ordered[idx + 1].start if idx + 1 < len(ordered) else len(text)
        body = text[start:end].strip()

        threshold = min_chars.get(cand.item, DEFAULT_MIN_CHARS)
        if len(body) < threshold:
            rejected.append(f"Item {cand.item}: {len(body)} chars < {threshold} minimum")
            stubs.append(cand.item)
            continue
        if _is_cross_reference_stub(body):
            rejected.append(
                f"Item {cand.item}: cross-reference stub "
                f"({len(body.split())} words pointing elsewhere)"
            )
            stubs.append(cand.item)
            continue

        sections[cand.item] = Section(
            item=cand.item, title=titles.get(cand.item, ""), start=start, end=end,
            text=body, source=sources.get(cand.item, "item-heading"),
        )
    return sections, rejected, stubs


def _adjacency_runs(candidates: list[Candidate]) -> list[list[Candidate]]:
    """Group candidates into runs of consecutive, closely-spaced headings."""
    runs: list[list[Candidate]] = []
    current: list[Candidate] = []
    for c in candidates:
        if current and c.start - current[-1].start > _TOC_MAX_GAP:
            runs.append(current)
            current = []
        current.append(c)
    if current:
        runs.append(current)
    return runs


def _truncate_at_order_break(run: list[Candidate]) -> list[Candidate]:
    """Cut a run where item numbering stops ascending.

    A table of contents lists items in order, by definition. So if a run reads
    1, 1A, 1B, 2, 3, 7, 8, 9A, *1*, that final '1' is not a TOC row -- it is the
    first real body heading, which happened to sit close enough to the end of
    the TOC to be swallowed by the adjacency rule.

    Ordering is a structural property of the document, not a tuned threshold,
    which makes it a more reliable signal than any distance heuristic.
    """
    kept: list[Candidate] = []
    last_rank = -1
    for c in run:
        rank = _ITEM_ORDER.get(c.item, -1)
        if rank <= last_rank:
            break
        kept.append(c)
        last_rank = rank
    return kept


def detect_toc_span(candidates: list[Candidate]) -> tuple[int, int] | None:
    """Find the table of contents: a dense RUN of adjacent item headings.

    A TOC lists many items as consecutive lines a few dozen characters apart.
    Body sections are separated by thousands of characters of prose. So we group
    candidates into runs where consecutive entries are within _TOC_MAX_GAP, and
    call the run with the most distinct items the TOC.

    Crucially, the span ends at the LAST ENTRY OF THE RUN -- not at some fixed
    offset from the start. Using a fixed window here silently swallowed the
    first real body heading, which sat just past the end of the TOC.
    """
    if len(candidates) < _TOC_MIN_ITEMS:
        return None

    best: list[Candidate] | None = None
    for raw_run in _adjacency_runs(candidates):
        run = _truncate_at_order_break(raw_run)
        if len({c.item for c in run}) < _TOC_MIN_ITEMS:
            continue
        if best is None or len({c.item for c in run}) > len({c.item for c in best}):
            best = run

    if best is None:
        return None
    last = best[-1]
    return (best[0].start, last.start + len(last.matched_text))


def _select_monotonic(candidates: list[Candidate]) -> dict[str, Candidate]:
    """Keep one candidate per item such that positions increase in item order.

    Walk the canonical item order; for each item take the earliest candidate
    that lies after the previously accepted one. This naturally rejects
    cross-references that appear out of sequence.
    """
    by_item: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_item.setdefault(c.item, []).append(c)

    chosen: dict[str, Candidate] = {}
    cursor = -1
    for item, _title in ITEM_SEQUENCE:
        options = [c for c in by_item.get(item, []) if c.start > cursor]
        if not options:
            continue
        pick = options[0]
        chosen[item] = pick
        cursor = pick.start
    return chosen


def _scaled_thresholds(text_len: int, base: dict[str, int]) -> dict[str, int]:
    """Scale minimum-length thresholds with document size, within bounds.

    A 1.4M-character bank filing and a 220k-character tech filing cannot share
    an absolute floor. Reference size is 250k chars (a typical large-cap 10-K);
    scaling is clamped to [0.5x, 3x] so a very short or very long document
    cannot push the thresholds somewhere absurd.
    """
    factor = max(0.5, min(3.0, text_len / 250_000))
    return {k: int(v * factor) for k, v in base.items()}


def sectionise(
    text: str,
    targets: list[str] | None = None,
    min_chars: dict[str, int] | None = None,
) -> SectionResult:
    targets = targets or TARGET_ITEMS
    min_chars = _scaled_thresholds(len(text), min_chars or MIN_SECTION_CHARS)
    result = SectionResult()

    candidates = find_candidates(text)
    if not candidates:
        result.warnings.append("no item headings found at all -- check the HTML parser")
        return result

    # --- filter 1+2: excise the table of contents ------------------------
    toc = detect_toc_span(candidates)
    result.toc_span = toc
    if toc:
        before = len(candidates)
        candidates = [c for c in candidates if not (toc[0] <= c.start <= toc[1])]
        log.debug("TOC at %s-%s, dropped %d candidates", toc[0], toc[1],
                  before - len(candidates))
    else:
        result.warnings.append(
            "no table of contents detected -- unusual; verify sections manually"
        )

    # --- filter 3: monotonic selection -----------------------------------
    chosen = _select_monotonic(candidates)
    if not chosen:
        result.warnings.append("no monotonic item sequence could be built")
        return result

    sources = {item: "item-heading" for item in chosen}

    # --- pass 2: title fallback for targets we could not find by number ---
    for item in targets:
        if item in chosen:
            continue
        cand = _find_by_title(text, item, chosen, toc)
        if cand is not None:
            chosen[item] = cand
            sources[item] = "title-fallback"
            result.warnings.append(
                f"Item {item} located by title, not by item number -- verify it"
            )

    # --- build sections with boundaries ----------------------------------
    result.sections, result.rejected, stubs = _build_sections(
        text, chosen, sources, min_chars
    )

    # --- pass 3: wrapper recovery ----------------------------------------
    # If a target item was FOUND by number but rejected as too short, it is a
    # cross-reference stub ("...appears on pages 52-167"), not a real section.
    # The content lives in an annual report appended after the numbered items.
    stub_targets = [i for i in stubs if i in targets]
    if stub_targets:
        region_start = max(c.start for c in chosen.values())
        recovered = _recover_wrapper_sections(text, stub_targets, region_start)
        if recovered:
            for item, cand in recovered.items():
                chosen[item] = cand
                sources[item] = "wrapper-recovery"
            result.warnings.append(
                "wrapper filing detected: Items "
                + ", ".join(sorted(recovered))
                + " recovered from the appended report, not the numbered stub"
            )
            result.sections, result.rejected, _ = _build_sections(
                text, chosen, sources, min_chars
            )
        still_stub = [i for i in stub_targets if i not in recovered]
        if still_stub:
            result.warnings.append(
                "Item(s) " + ", ".join(still_stub) + " are cross-reference stubs "
                "with no separately locatable content"
            )

    for t in targets:
        if t not in result.sections:
            result.warnings.append(f"target Item {t} not recovered")

    return result


def _find_by_title(
    text: str,
    item: str,
    chosen: dict[str, Candidate],
    toc: tuple[int, int] | None,
) -> Candidate | None:
    """Locate a section by its canonical title, constrained to the position
    window implied by the items we already found.

    The window matters. Titles like "Risk Factors" recur as running page
    headers and in cross-references dozens of times in a long filing. Bounding
    the search between the preceding and following located items removes almost
    all of those false positives without any scoring heuristic.
    """
    pat = _title_pattern(item)
    if pat is None:
        return None

    rank = _ITEM_ORDER.get(item, -1)
    lower = max(
        (c.start for i, c in chosen.items() if _ITEM_ORDER.get(i, -1) < rank),
        default=0,
    )
    upper = min(
        (c.start for i, c in chosen.items() if _ITEM_ORDER.get(i, -1) > rank),
        default=len(text),
    )

    for m in pat.finditer(text):
        if not (lower < m.start() < upper):
            continue
        if toc and toc[0] <= m.start() <= toc[1]:
            continue
        return Candidate(item=item, start=m.start(), matched_text=m.group(0).strip())
    return None
