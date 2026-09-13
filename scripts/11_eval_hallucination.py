#!/usr/bin/env python
"""Week 5: measure the hallucination rate. Fills EVALUATION.md section 6.

WHAT THIS MEASURES
------------------
How many numeric claims a generated memo makes that cannot be traced to a
figure this pipeline independently verified -- under three configurations that
add one mitigation at a time.

    1. none    bare prompt, no gate        what a reader would receive
                                           from an unguarded system
    2. xbrl    bare prompt, gate ON        the arithmetic gate alone
    3. cited   full prompt, gate ON        gate plus prompt-level enforcement

THE BASELINE HAS TO BE HONEST
-----------------------------
The shipped MEMO_SYSTEM already ends with "Every number you write will be
checked automatically against SEC XBRL data ... an invented number costs you
the sentence containing it." That sentence is itself a mitigation. A row
labelled "no verification" that keeps it is not measuring an unguarded system;
it is measuring the gate's absence while the prompt quietly does the gate's
job, and it will understate the raw hallucination rate.

So configuration 1 strips that clause. MEMO_SYSTEM_BARE below is the shipped
prompt with every reference to checking and to consequences removed, and
nothing else changed. The user prompt is identical in all three configurations
(nodes.build_memo_prompt), so the only variable is the instruction.

TWO METRICS, NOT ONE
--------------------
Section 6's column is "unsupported numeric claims / memo". Measured on what
reaches the reader, configurations 2 and 3 are 0.0 BY CONSTRUCTION -- the gate
deletes the sentence, so of course nothing unsupported survives. Reporting only
that column produces a table that looks like a triumph and carries no
information.

The informative quantity is how many bad claims were GENERATED before the gate
saw them. That separates two different mechanisms:

    generated   did the instruction stop the model inventing figures?
    shown       did the gate catch what the instruction missed?

Both are reported. The gate's cost -- sentences and words deleted from an
otherwise readable memo -- is reported alongside, because a gate that strips
half the memo is not free even when it is correct.

WHAT THIS CANNOT MEASURE
------------------------
Only numeric claims. "Management's tone has become more cautious" is
unverifiable by arithmetic and is not counted here in either direction. A memo
could be entirely free of unsupported numbers and still be misleading in
prose; this measurement does not speak to that, and section 6 should not be
read as though it does.

Usage:
    python scripts/11_eval_hallucination.py                      # all filings
    python scripts/11_eval_hallucination.py --tickers AAPL MSFT
    python scripts/11_eval_hallucination.py --configs none xbrl
    python scripts/11_eval_hallucination.py --limit 4 --dry-run  # no LLM
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.analysis.claims import (  # noqa: E402
    strip_unsupported_sentences, verify_claims,
)
from filingiq.config import settings  # noqa: E402
from filingiq.graph import nodes  # noqa: E402
from filingiq.llm.router import ModelRouter  # noqa: E402
from filingiq.retrieval.embedder import Embedder  # noqa: E402
from filingiq.storage import db  # noqa: E402

log = logging.getLogger("eval_hallucination")


# ---------------------------------------------------------------------------
# The bare prompt: shipped MEMO_SYSTEM minus every mention of verification.
#
# Written out in full rather than derived by string surgery on MEMO_SYSTEM. A
# .replace() would keep working silently after someone reworded the original,
# and the ablation would then be comparing the shipped prompt against itself
# while still reporting a difference. An explicit constant fails loudly in
# review instead.
# ---------------------------------------------------------------------------
MEMO_SYSTEM_BARE = """You are writing a short analyst note on a company's 10-K.

Rules:
- Use ONLY the figures and risk excerpts provided. Introduce no other numbers.
- If you do not have a figure, do not estimate it -- omit the point entirely.
- State changes plainly. Do not editorialise or give investment advice.
- 150-250 words, plain prose, no headings, no bullet points.
- Write large figures readably ("$391.0 billion"), not as full digit strings.
- Quote per-share amounts to the cent."""

# Configuration 3 adds explicit citation enforcement on top of the shipped
# prompt: qualitative assertions must carry the chunk id they came from, which
# is the only handle a human reviewer has on a claim arithmetic cannot check.
MEMO_SYSTEM_CITED = nodes.MEMO_SYSTEM + """

CITATIONS:
Any statement about risk-factor content must be followed by the chunk id it
came from, in square brackets. A statement you cannot attribute to a supplied
excerpt must be omitted rather than written without one."""

CONFIGS = {
    "none": (MEMO_SYSTEM_BARE, False),
    "xbrl": (MEMO_SYSTEM_BARE, True),
    "cited": (MEMO_SYSTEM_CITED, True),
}

CONFIG_LABELS = {
    "none": "No verification",
    "xbrl": "+ XBRL cross-check",
    "cited": "+ citation enforcement",
}


@dataclass
class MemoMeasurement:
    ticker: str
    fiscal_year: int
    config: str
    numeric_claims: int
    unsupported_generated: int
    unsupported_shown: int
    sentences_removed: int
    words_before: int
    words_after: int
    examples: list[str] = field(default_factory=list)


@dataclass
class ConfigResult:
    config: str
    memos: list[MemoMeasurement] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.memos)
        if not n:
            return {"config": self.config, "n_memos": 0}
        gen = sum(m.unsupported_generated for m in self.memos)
        shown = sum(m.unsupported_shown for m in self.memos)
        claims = sum(m.numeric_claims for m in self.memos)
        wb = sum(m.words_before for m in self.memos)
        wa = sum(m.words_after for m in self.memos)
        return {
            "config": self.config,
            "label": CONFIG_LABELS[self.config],
            "n_memos": n,
            "numeric_claims_per_memo": round(claims / n, 2),
            "unsupported_generated_per_memo": round(gen / n, 3),
            "unsupported_shown_per_memo": round(shown / n, 3),
            "memos_with_any_unsupported": sum(
                1 for m in self.memos if m.unsupported_generated),
            "sentences_removed_total": sum(m.sentences_removed
                                           for m in self.memos),
            # What the gate costs when it fires. A gate that deletes a third of
            # every memo is accurate and unusable.
            "words_retained_pct": round(100 * wa / wb, 1) if wb else None,
        }


def verifiable_values(state: dict) -> dict:
    """Figures a claim may legitimately cite.

    Mirrors verify_node: XBRL ground truth PLUS the counts this pipeline
    computed itself. Diff counts are as verifiable as XBRL figures -- omitting
    them makes the gate strip correct sentences about our own analysis, which
    would inflate the unsupported count with false positives and make the
    mitigations look better than they are.
    """
    out = dict(state.get("verified_values", {}))
    diff = state.get("diff_summary") or {}
    for key in ("new", "modified", "unchanged", "removed", "drift_score"):
        val = diff.get(key)
        if isinstance(val, (int, float)):
            out[f"diff_{key}"] = float(val)
    return out


def measure_memo(memo: str, verified: dict, ticker: str, fiscal_year: int,
                 config: str, gate: bool) -> MemoMeasurement:
    """Count unsupported numeric claims before and after the gate."""
    report = verify_claims(memo, verified)
    summary = report.summary()
    generated = summary["unsupported"]

    if gate:
        cleaned = strip_unsupported_sentences(memo, report)
        shown = verify_claims(cleaned, verified).summary()["unsupported"]
    else:
        cleaned = memo
        shown = generated

    before_sentences = memo.count(".")
    after_sentences = cleaned.count(".")
    return MemoMeasurement(
        ticker=ticker, fiscal_year=fiscal_year, config=config,
        numeric_claims=summary["numeric_claims"],
        unsupported_generated=generated,
        unsupported_shown=shown,
        sentences_removed=max(before_sentences - after_sentences, 0),
        words_before=len(memo.split()),
        words_after=len(cleaned.split()),
        examples=[c.raw for c in report.unsupported[:3]],
    )


def build_state(con, embedder, ticker: str, fiscal_year: int,
                accession: str) -> dict:
    """Run the deterministic nodes only. No LLM, identical across configs."""
    state: dict = {"ticker": ticker, "fiscal_year": fiscal_year,
                   "accession": accession}
    state.update(nodes.load_figures_node(state, con))
    state.update(nodes.diff_node(state, con, embedder))
    state.update(nodes.taxonomy_node(state))
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--limit", type=int, help="max filings per configuration")
    ap.add_argument("--dry-run", action="store_true",
                    help="build states and report scope; no LLM calls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                  "sentence_transformers", "transformers", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not settings.db_path.exists():
        print(f"No database at {settings.db_path}.")
        print("Run scripts/01_ingest.py, 03_parse.py, 06_chunk.py, 07_embed.py "
              "and 10_extract.py first.")
        return 1

    router = ModelRouter()
    if not args.dry_run:
        ok, msg = router.healthcheck()
        if not ok:
            print(f"LLM preflight failed: {msg}")
            print("Set GROQ_API_KEY in .env, or use --dry-run.")
            return 1
        print(f"Preflight OK: {msg}")

    embedder = Embedder()

    with db.connect(read_only=True) as con:
        sql = ("SELECT accession, ticker, fiscal_year FROM filings "
               "WHERE local_path IS NOT NULL")
        params: list = []
        if args.tickers:
            sql += " AND ticker IN ({})".format(
                ",".join("?" for _ in args.tickers))
            params += [t.upper() for t in args.tickers]
        if args.years:
            sql += " AND fiscal_year IN ({})".format(
                ",".join("?" for _ in args.years))
            params += args.years
        sql += " ORDER BY ticker, fiscal_year"
        filings = con.execute(sql, params).fetchall()
        if args.limit:
            filings = filings[:args.limit]
        if not filings:
            print("No filings match.")
            return 1

        print(f"{len(filings)} filings x {len(args.configs)} configurations "
              f"= {len(filings) * len(args.configs)} memos")

        # States are built ONCE and reused across configurations. Rebuilding
        # per configuration would let a non-deterministic diff put different
        # evidence in front of each prompt, and the comparison would no longer
        # isolate the instruction.
        states = {}
        for accession, ticker, fiscal_year in filings:
            try:
                states[accession] = build_state(con, embedder, ticker,
                                                fiscal_year, accession)
            except Exception as exc:  # noqa: BLE001
                log.warning("state build failed for %s %s: %s",
                            ticker, fiscal_year, exc)

        usable = {a: s for a, s in states.items() if s.get("figures")}
        skipped = len(states) - len(usable)
        if skipped:
            print(f"{skipped} filings skipped: no verified figures, so every "
                  f"claim would be unsupported by construction")

        if args.dry_run:
            print(f"\nDry run: {len(usable)} filings ready. "
                  f"Re-run without --dry-run to generate memos.")
            return 0

        results = {c: ConfigResult(config=c) for c in args.configs}
        for config in args.configs:
            system_prompt, gate = CONFIGS[config]
            for accession, state in usable.items():
                out = nodes.memo_node(state, router, system_prompt=system_prompt)
                memo = out.get("memo", "")
                if not memo:
                    log.warning("%s: no memo (%s)", accession,
                                out.get("errors") or out.get("warnings"))
                    continue
                results[config].memos.append(measure_memo(
                    memo, verifiable_values(state), state["ticker"],
                    state["fiscal_year"], config, gate))
            s = results[config].summary()
            print(f"  {config:<8} generated {s['unsupported_generated_per_memo']:>6} "
                  f"/memo   shown {s['unsupported_shown_per_memo']:>6} /memo")

    # ---- the section 6 table ------------------------------------------
    print("\n" + "=" * 78)
    print("SECTION 6: HALLUCINATION RATE")
    print("=" * 78)
    print(f"{'Configuration':<24}{'generated':>11}{'shown':>9}"
          f"{'claims':>9}{'words kept':>12}")
    print("-" * 78)
    for config in args.configs:
        s = results[config].summary()
        if not s.get("n_memos"):
            print(f"{CONFIG_LABELS[config]:<24}{'(no memos)':>11}")
            continue
        kept = s["words_retained_pct"]
        print(f"{s['label']:<24}{s['unsupported_generated_per_memo']:>11.3f}"
              f"{s['unsupported_shown_per_memo']:>9.3f}"
              f"{s['numeric_claims_per_memo']:>9.2f}"
              f"{(f'{kept}%' if kept is not None else '-'):>12}")

    print("""
  generated  = unsupported numeric claims the model WROTE, before the gate.
               This is where the prompt's effect shows.
  shown      = unsupported claims surviving into the memo a reader receives.
               0.000 wherever the gate is on, BY CONSTRUCTION -- the gate
               deletes the sentence. Not evidence of a good model.
  words kept = share of the draft surviving the gate. The gate's cost.""")

    out_path = settings.data_dir / "hallucination_results.json"
    out_path.write_text(json.dumps(
        {"configs": {c: results[c].summary() for c in args.configs},
         "memos": [m.__dict__ for c in args.configs for m in results[c].memos]},
        indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {out_path}")
    print(f"Cost: ${router.usage.summary()['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
