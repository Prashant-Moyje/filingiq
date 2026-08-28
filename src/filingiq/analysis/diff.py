"""Year-over-year risk factor diffing.

WHY THIS IS THE POINT OF THE PROJECT
------------------------------------
Any tool can summarise a risk factors section. The signal analysts actually
want is what CHANGED: a risk factor that appeared for the first time this year,
one whose language escalated, one quietly dropped. Nobody tracks this
systematically because it means reading two 200-page documents side by side.

DETECTION IS STATISTICAL, EXPLANATION IS GENERATIVE
---------------------------------------------------
The matching here uses embeddings and cosine similarity, not an LLM. Asking a
model "which of these 70 risk factors are new?" is expensive, non-deterministic,
and unverifiable. Asking it "why does this specific new risk factor matter?"
plays to what it is good at.

That split -- statistics for detection, LLM for interpretation -- is the same
principle applied in the anomaly detection and forecasting projects, and it is
worth being able to defend: use the LLM where fuzziness is the point, not where
a deterministic method already works.

THRESHOLDS
----------
Cosine similarity between BGE embeddings of chunks from consecutive filings:

    >= 0.95  unchanged      boilerplate carried forward verbatim
    0.80-0.95 modified      same risk, rewritten -- often the interesting case
    <  0.80  new            no counterpart in the prior year

These are calibrated on the observation that filings copy risk factors forward
nearly verbatim, so genuine rewrites stand out. They are reported alongside
results rather than buried, because a different embedding model would need
different values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

UNCHANGED_THRESHOLD = 0.95
MODIFIED_THRESHOLD = 0.80

# A section with fewer chunks than this did not parse properly. Real Item 1A
# sections run to dozens of chunks; a handful means the sectioniser captured a
# fragment, and diffing a fragment against a full section measures parsing,
# not disclosure.
MIN_CHUNKS_FOR_DIFF = 5

# If one year has less than this fraction of the other's chunks, the two are
# not comparable. AT&T FY2018 parsed to 5 chunks against FY2019's 33, and the
# resulting drift of 1.0 described the parser, not the company.
MIN_SIZE_RATIO = 0.35


@dataclass
class ChunkRef:
    chunk_id: str
    text: str
    item: str
    chunk_index: int


@dataclass
class DiffEntry:
    status: str                 # 'new' | 'modified' | 'unchanged' | 'removed'
    current: ChunkRef | None
    prior: ChunkRef | None
    similarity: float

    @property
    def text(self) -> str:
        ref = self.current or self.prior
        return ref.text if ref else ""


@dataclass
class DiffResult:
    ticker: str
    current_year: int
    prior_year: int
    entries: list[DiffEntry] = field(default_factory=list)
    # False when the prior year is absent from the corpus. Without this flag
    # the first year of any corpus reports drift=1.0 -- every risk factor
    # looks new because there is nothing to compare against. That is a missing
    # baseline, not a disclosure event.
    has_baseline: bool = True
    # False when the CURRENT year failed to parse. This is the mirror failure
    # and it is more dangerous, because it produces drift=0.0 rather than an
    # obvious anomaly: every prior chunk counts as "removed", total_current is
    # zero, and (new + modified) / max(0, 1) = 0.0 -- which reads as "the
    # company changed nothing this year". BAC FY2023, GS FY2020, MCD FY2019 and
    # NKE FY2024 all produced exactly that. See FM-019.
    has_current: bool = True
    # Set when the two years differ so wildly in size that the comparison is
    # measuring a parsing artifact rather than a disclosure change.
    size_mismatch: bool = False
    # False when the CURRENT year has no Item 1A chunks -- a parser failure,
    # not a disclosure event. Without this the diff reports "0 new, 0 modified,
    # 102 removed, drift 0.0", which reads as a company deleting its entire
    # risk section. That row then becomes the strongest signal in the feature
    # store. Gating on the prior year alone (has_baseline) was not enough.
    has_current: bool = True

    def by_status(self, status: str) -> list[DiffEntry]:
        return [e for e in self.entries if e.status == status]

    def summary(self) -> dict:
        """Counts plus a drift score -- or None when drift is not measurable.

        Three distinct conditions make drift meaningless, and each is reported
        as None rather than a number. The middle one is the dangerous case:
        an unparsed current year yields (new + modified) / max(0, 1) = 0.0,
        which reads as "the company changed nothing" rather than "we could not
        read this filing". See FM-019.
        """
        counts = {s: len(self.by_status(s))
                  for s in ("new", "modified", "unchanged", "removed")}
        total_current = counts["new"] + counts["modified"] + counts["unchanged"]
        base = {**counts, "total_current": total_current,
                "has_baseline": self.has_baseline,
                "has_current": self.has_current,
                "size_mismatch": self.size_mismatch}

        if not self.has_current:
            return {**base, "total_current": 0, "drift_score": None,
                    "note": (f"FY{self.current_year} has no usable Item 1A "
                             "content (parser failure) -- drift is undefined, "
                             "NOT 0.0")}
        if not self.has_baseline:
            return {**base, "drift_score": None,
                    "note": (f"no usable FY{self.prior_year} baseline -- "
                             "drift is undefined, NOT 1.0")}
        if self.size_mismatch:
            return {**base, "drift_score": None,
                    "note": ("section sizes differ too much between years to "
                             "compare -- likely a parsing artifact")}

        return {**base,
                # One comparable number per company-year: the fraction of this
                # year's risk disclosure that is new or rewritten. Feeds the
                # ML layer as `drift_score`.
                "drift_score": round(
                    (counts["new"] + counts["modified"]) / max(total_current, 1),
                    4)}


def diff_sections(current: list[ChunkRef], prior: list[ChunkRef],
                  embedder, unchanged_threshold: float = UNCHANGED_THRESHOLD,
                  modified_threshold: float = MODIFIED_THRESHOLD) -> list[DiffEntry]:
    """Greedy best-match alignment between two years of chunks.

    Greedy rather than optimal (Hungarian) assignment: risk factors are largely
    stable year to year, so the greedy match is near-identical in practice and
    O(n*m) instead of O(n^3). If the corpus grew to where that mattered, the
    swap is local to this function.
    """
    if not current:
        return [DiffEntry("removed", None, p, 0.0) for p in prior]
    if not prior:
        return [DiffEntry("new", c, None, 0.0) for c in current]

    cur_vecs = embedder.embed_passages([c.text for c in current],
                                       show_progress=False)
    pri_vecs = embedder.embed_passages([p.text for p in prior],
                                       show_progress=False)
    # Vectors are L2-normalised by the embedder, so the dot product IS cosine.
    sim = cur_vecs @ pri_vecs.T

    entries: list[DiffEntry] = []
    matched_prior: set[int] = set()

    for i, c in enumerate(current):
        j = int(np.argmax(sim[i]))
        score = float(sim[i, j])
        if score >= unchanged_threshold:
            status = "unchanged"
        elif score >= modified_threshold:
            status = "modified"
        else:
            status = "new"
        if status != "new":
            matched_prior.add(j)
        entries.append(DiffEntry(status, c, prior[j] if status != "new" else None,
                                 score))

    for j, p in enumerate(prior):
        if j not in matched_prior:
            entries.append(DiffEntry("removed", None, p, float(sim[:, j].max())))

    return entries


def load_chunks(con, ticker: str, fiscal_year: int, item: str = "1A") -> list[ChunkRef]:
    rows = con.execute(
        """SELECT c.chunk_id, c.raw_text, c.item, c.chunk_index
           FROM chunks c
           WHERE c.ticker = ? AND c.fiscal_year = ? AND c.item = ?
           ORDER BY c.chunk_index""",
        [ticker.upper(), int(fiscal_year), item],
    ).fetchall()
    return [ChunkRef(r[0], r[1], r[2], r[3]) for r in rows]


def diff_years(con, ticker: str, current_year: int, prior_year: int,
               embedder, item: str = "1A") -> DiffResult:
    current = load_chunks(con, ticker, current_year, item)
    prior = load_chunks(con, ticker, prior_year, item)
    if not current:
        log.warning("%s FY%s has no Item %s chunks", ticker, current_year, item)
    if not prior:
        log.warning("%s: no FY%s baseline -- drift undefined",
                    ticker, prior_year)
    entries = diff_sections(current, prior, embedder)
    return DiffResult(ticker=ticker, current_year=current_year,
                      prior_year=prior_year, entries=entries,
                      has_baseline=bool(prior), has_current=bool(current))
