"""Tests that the diff quality guards fire on the PRODUCTION path.

Why a separate file. tests/test_analysis.py already asserted that a
DiffResult carrying size_mismatch=True yields drift None -- but it built that
DiffResult by hand. Nothing ever checked that diff_years SETS the flag, and
nothing did: MIN_CHUNKS_FOR_DIFF and MIN_SIZE_RATIO were defined and never
read, so the branch was unreachable in production while four tests passed.

AT&T FY2019 -- the case FM-019 names, 33 chunks against a 5-chunk FY2018 --
consequently entered the feature store at drift_score = 1.0, the single
highest drift value in all 155 rows.

The lesson these tests encode: asserting the CONSEQUENCE of a flag is not the
same as asserting that the flag is ever raised. Every test here runs
diff_years against a real (in-memory) database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from filingiq.analysis.diff import (  # noqa: E402
    MIN_CHUNKS_FOR_DIFF, MIN_SIZE_RATIO, diff_years, is_size_mismatch,
)


class StubEmbedder:
    """Deterministic unit vectors keyed on text, so identical text matches
    itself exactly and the test asserts on guard logic, not on a model."""

    def embed_passages(self, texts, show_progress=False):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def con():
    duckdb = pytest.importorskip("duckdb")
    c = duckdb.connect(":memory:")
    c.execute("""CREATE TABLE chunks (
        chunk_id VARCHAR, raw_text VARCHAR, item VARCHAR,
        chunk_index INTEGER, ticker VARCHAR, fiscal_year INTEGER)""")
    yield c
    c.close()


def _load(con, year, n, prefix="risk"):
    for i in range(n):
        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    [f"{year}-{i}", f"{prefix} factor {i} year {year}",
                     "1A", i, "T", year])


def _drift(con):
    return diff_years(con, "T", 2019, 2018, StubEmbedder()).summary()


# --- the regression -------------------------------------------------------

def test_att_shaped_mismatch_yields_undefined_drift(con):
    """FM-019's own example. 33 chunks against 5 is a parser artifact.

    Before the fix this returned drift_score = 1.0 and that row is in
    data/features.csv today as the largest drift in the dataset.
    """
    _load(con, 2019, 33)
    _load(con, 2018, 5)
    s = _drift(con)
    assert s["size_mismatch"] is True, "diff_years must SET the flag, not just honour it"
    assert s["drift_score"] is None, "parser artifact scored as maximum disclosure change"


def test_comparable_sections_still_produce_a_number(con):
    """The guard must not swallow legitimate comparisons."""
    _load(con, 2019, 30)
    _load(con, 2018, 28)
    s = _drift(con)
    assert s["size_mismatch"] is False
    assert s["drift_score"] is not None


def test_ratio_boundary_is_respected(con):
    """Just above the floor compares; just below does not."""
    _load(con, 2019, 100)
    _load(con, 2018, 40)          # ratio 0.40, above 0.35
    assert _drift(con)["drift_score"] is not None


def test_just_below_ratio_floor_is_rejected(con):
    _load(con, 2019, 100)
    _load(con, 2018, 30)          # ratio 0.30, below 0.35
    assert _drift(con)["drift_score"] is None


def test_tiny_sections_are_rejected_even_when_ratios_match(con):
    """4 vs 4 is a perfect ratio and still not a risk section.

    Both sides need MIN_CHUNKS_FOR_DIFF chunks: a ratio test alone passes two
    equally-broken parses.
    """
    _load(con, 2019, 4)
    _load(con, 2018, 4)
    s = _drift(con)
    assert s["size_mismatch"] is True
    assert s["drift_score"] is None


# --- the guard in isolation ----------------------------------------------

class _Ref:
    def __init__(self, i):
        self.chunk_id, self.text, self.item, self.chunk_index = str(i), "x", "1A", i


def _refs(n):
    return [_Ref(i) for i in range(n)]


def test_empty_sides_are_not_this_guards_job():
    """has_current / has_baseline own the empty cases. Claiming them here too
    would report the wrong reason for a missing drift score."""
    assert is_size_mismatch([], _refs(30)) is False
    assert is_size_mismatch(_refs(30), []) is False


def test_guard_is_symmetric():
    """Which year is larger must not change the verdict."""
    assert is_size_mismatch(_refs(33), _refs(5)) == is_size_mismatch(_refs(5), _refs(33))


def test_guard_uses_the_declared_constants():
    """If someone retunes the constants the guard must move with them,
    rather than having the thresholds hardcoded a second time."""
    n = MIN_CHUNKS_FOR_DIFF
    assert is_size_mismatch(_refs(n - 1), _refs(n - 1)) is True
    assert is_size_mismatch(_refs(n), _refs(n)) is False

    big = 100
    just_under = int(big * MIN_SIZE_RATIO) - 1
    just_over = int(big * MIN_SIZE_RATIO) + 2
    assert is_size_mismatch(_refs(big), _refs(just_under)) is True
    assert is_size_mismatch(_refs(big), _refs(just_over)) is False


def test_result_has_exactly_one_has_current_field():
    """The dataclass declared has_current twice. Harmless at runtime -- the
    second wins -- but it is the same duplicated-declaration slip FM-019
    records finding in summary(), and it survived in the field list."""
    import ast
    src = (Path(__file__).resolve().parents[1]
           / "src" / "filingiq" / "analysis" / "diff.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "DiffResult")
    names = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
    assert len(names) == len(set(names)), f"duplicate field(s) in DiffResult: {names}"


# --- documented limitation: chunk-boundary sensitivity ---------------------
#
# Not a bug to fix here, but a property EVALUATION.md section 7 now quantifies.
# These tests exist so that claim cannot quietly become false: if chunking or
# the diff unit changes, the documented numbers must be revisited.

_FACTOR = (
    "Risk Factor {n}: Our business faces material uncertainty.\n"
    "We depend on a limited number of suppliers for critical components, and "
    "disruption in that supply chain could materially and adversely affect our "
    "results of operations, financial condition and cash flows. Competition in "
    "our markets is intense and we may be unable to maintain pricing. Changes "
    "in regulation, including data protection and export controls, may increase "
    "our costs of compliance. Our international operations expose us to "
    "currency fluctuation and geopolitical instability that we cannot control.\n"
)
_NEW = ("Risk Factor NEW: Artificial intelligence regulation may affect us.\n"
        "Emerging rules governing the deployment of artificial intelligence "
        "could impose compliance obligations and restrict product features.\n")


def _chunk_texts(insert_at=None, n=40):
    from filingiq.parsing.chunker import chunk_section
    parts = [_FACTOR.format(n=i) for i in range(n)]
    if insert_at is not None:
        parts.insert(insert_at, _NEW)
    return [c.text for c in chunk_section("\n".join(parts), "acc", "1A",
                                          {"ticker": "T", "fiscal_year": 2019})]


def _displaced(insert_at):
    base = _chunk_texts()
    new = _chunk_texts(insert_at=insert_at)
    return (len(new) - len(set(base) & set(new))) / len(new)


def test_early_insertion_displaces_far_more_than_late_insertion():
    """drift_score diffs chunks, so WHERE text was added changes how much of
    the section stops being byte-identical -- independent of how much actually
    changed. This is the measurement-error argument in section 7."""
    early, late = _displaced(0), _displaced(40)
    assert early > late, "expected front insertion to re-cut more of the section"
    assert early - late > 0.5, (
        f"boundary sensitivity has changed materially (early {early:.0%}, "
        f"late {late:.0%}). EVALUATION.md section 7 quotes these numbers.")


def test_a_single_added_factor_can_displace_most_of_the_section():
    """One new risk factor out of 40 should not make most of the section
    unrecognisable. That it can is the argument for diffing risk factors
    rather than chunks."""
    assert _displaced(0) > 0.5, (
        "front-insertion displacement dropped below 50% -- if the diff unit "
        "changed, EVALUATION.md section 7 needs updating")
