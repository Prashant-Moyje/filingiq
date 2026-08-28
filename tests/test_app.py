"""Smoke test for the demo UI.

A demo that crashes during the three minutes that matter is worse than no
demo. This renders every tab against fixture data with Streamlit stubbed, so
a broken app fails in CI rather than in front of an interviewer.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py"


class StreamlitStub:
    """Minimal stand-in. Returns the first option for every selector so the
    render path is exercised deterministically."""

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def __getattr__(self, name):
        if name == "cache_data":
            def deco(*a, **kw):
                if a and callable(a[0]):
                    return a[0]
                return lambda fn: fn
            return deco

        def call(*a, **kw):
            if name == "columns":
                n = a[0] if a and isinstance(a[0], int) else len(a[0])
                return [StreamlitStub() for _ in range(n)]
            if name == "tabs":
                return [StreamlitStub() for _ in a[0]]
            if name in ("expander", "container", "form", "sidebar", "spinner"):
                return StreamlitStub()
            if name == "selectbox":
                opts = a[1] if len(a) > 1 else kw.get("options", [])
                return opts[0] if len(opts) else None
            if name == "multiselect":
                return kw.get("default", [])
            if name == "text_input":
                return a[1] if len(a) > 1 else ""
            if name == "checkbox":
                return False
            return None
        return call


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Fixture artifacts in a temp dir, so the test never touches real data."""
    data = tmp_path / "data"
    (data / "db").mkdir(parents=True)
    _patch_settings(monkeypatch, data)

    from filingiq.storage import db
    db.init_db(data / "db" / "t.duckdb")
    with db.connect(db_path=data / "db" / "t.duckdb") as con:
        con.execute("""INSERT INTO filings (accession, cik, ticker, form,
            filing_date, period_of_report, fiscal_year, primary_document,
            primary_doc_url, local_path) VALUES ('a1', 1, 'AAPL', '10-K',
            '2024-11-01', '2024-09-28', 2024, 'x.htm', 'u', '/tmp/x')""")
        con.execute("""INSERT INTO ground_truth (cik, ticker, accession, metric,
            xbrl_tag, unit, value, period_start, period_end, fiscal_year,
            fiscal_period, is_instant) VALUES (1,'AAPL','a1','revenue','R','USD',
            391035000000.0,'2023-10-01','2024-09-28',2024,'FY',False)""")
        con.execute("""INSERT INTO chunks (chunk_id, accession, ticker,
            fiscal_year, item, chunk_index, char_start, char_end, n_tokens,
            has_table, text, raw_text) VALUES ('c1','a1','AAPL',2024,'1A',0,0,
            10,5,False,'ctx','supply chain disruption risk')""")

    (data / "extraction_results.json").write_text(json.dumps({
        "summary": {"exact": 0.955, "within_2pct": 0.985, "n_abstained": 18,
                    "n_total": 160, "n_scorable": 134, "abstention_rate": 0.11,
                    "error_types": {}},
        "usage": {"cost_usd": 0.0368, "calls": 48, "tokens_in": 1,
                  "tokens_out": 1, "p50_ms": 1, "p95_ms": 1, "errors": 0},
        "model": "m",
        "rows": [{"ticker": "AAPL", "fiscal_year": 2024, "metric": "revenue",
                  "extracted": 391035000000.0, "truth": 391035000000.0,
                  "band": "exact", "error_type": None, "quote": "Net sales",
                  "accession": "a1"}]}), encoding="utf-8")
    (data / "analysis_results.json").write_text(json.dumps([
        {"ticker": "AAPL", "fiscal_year": 2024, "accession": "a1",
         "diff_summary": {"new": 0, "modified": 14, "unchanged": 49,
                          "removed": 8, "drift_score": 0.22,
                          "has_baseline": True, "has_current": True},
         "risk_themes": {}, "modified_themes": {"supply_chain": 7}}]), encoding="utf-8")
    (data / "retrieval_eval.json").write_text(json.dumps(
        {"metrics": {"hybrid": {"mrr": 0.678, "hit@5": 0.852}}}), encoding="utf-8")
    (data / "model_results.json").write_text(json.dumps(
        {"results": {"combined": {"summary": {"mean_ic": 0.0085,
                                              "mean_auc": 0.45,
                                              "mean_rmse_improvement": -0.11}}},
         "permutation": {"combined": {"p_value": 0.95}}}), encoding="utf-8")
    return data


def _patch_settings(monkeypatch, data_dir):
    """Point every module at a temp data directory.

    Settings is a frozen dataclass, so it cannot be mutated -- a replacement
    object is substituted instead. It must be patched in EVERY module that did
    `from filingiq.config import settings`, because that binds the object at
    import time; patching only filingiq.config would leave storage.db still
    holding the original and writing to the real database.
    """
    import dataclasses
    import filingiq.config as cfg
    import filingiq.storage.db as dbmod

    replacement = dataclasses.replace(
        cfg.settings, data_dir=data_dir, db_path=data_dir / "db" / "t.duckdb")
    monkeypatch.setattr(cfg, "settings", replacement)
    monkeypatch.setattr(dbmod, "settings", replacement)
    return replacement


def _run_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", StreamlitStub())
    src = APP.read_text(encoding="utf-8")
    exec(compile(src, str(APP), "exec"),
         {"__name__": "__main__", "__file__": str(APP)})


def test_app_renders_with_full_artifacts(artifacts, monkeypatch):
    _run_app(monkeypatch)


def test_app_renders_with_no_artifacts_at_all(tmp_path, monkeypatch):
    """A fresh clone has no database and no results. The app must explain what
    to run, not raise."""
    data = tmp_path / "empty"
    data.mkdir()
    _patch_settings(monkeypatch, data)
    _run_app(monkeypatch)


def test_app_renders_with_partial_artifacts(artifacts, monkeypatch):
    """Mid-pipeline state: extraction done, modelling not."""
    (artifacts / "model_results.json").unlink()
    (artifacts / "retrieval_eval.json").unlink()
    _run_app(monkeypatch)
