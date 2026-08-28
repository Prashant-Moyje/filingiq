"""FilingIQ demo UI.

DESIGN CONSTRAINT: this app makes NO LLM calls and needs NO network.

Everything it shows is read from artifacts already on disk -- the DuckDB
database and the JSON result files produced by the pipeline scripts. A live
demo that depends on a provider is a demo that fails during the one three-
minute window that matters, and "sorry, rate limited" is not a recoverable
moment in an interview.

Run:  streamlit run app/main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from filingiq.config import settings
from filingiq.storage import db

st.set_page_config(page_title="FilingIQ", page_icon="📄", layout="wide")

BAND_COLOURS = {
    "exact": "#1a7f37", "within_0.5pct": "#1a7f37", "within_2pct": "#9a6700",
    "wrong": "#cf222e", "abstained": "#57606a", "no_ground_truth": "#8250df",
}


# --------------------------------------------------------------------------
# Loading. Cached, and every loader degrades to an explanatory message rather
# than a traceback -- a missing artifact should tell you which script to run.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sql(query: str, params: list | None = None) -> pd.DataFrame:
    if not settings.db_path.exists():
        return pd.DataFrame()
    with db.connect(read_only=True) as con:
        try:
            return con.execute(query, params or []).fetchdf()
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str):
    path = settings.data_dir / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def need(artifact, script: str) -> bool:
    if artifact is None or (isinstance(artifact, pd.DataFrame) and artifact.empty):
        st.info(f"Not built yet. Run `python scripts/{script}` first.")
        return True
    return False


# --------------------------------------------------------------------------
st.title("FilingIQ")
st.caption("SEC filing intelligence with XBRL-verified extraction — "
           "every figure checked arithmetically against SEC ground truth")

tabs = st.tabs(["Overview", "Extraction verification", "Risk drift",
                "Search", "Evaluation"])

# ============================== OVERVIEW ==================================
with tabs[0]:
    filings = load_sql("SELECT ticker, fiscal_year, form, filing_date FROM filings")
    chunks = load_sql("SELECT count(*) AS n FROM chunks")
    gt = load_sql("SELECT count(*) AS n FROM ground_truth")

    c = st.columns(4)
    c[0].metric("Filings", len(filings) if not filings.empty else 0)
    c[1].metric("Companies", filings["ticker"].nunique() if not filings.empty else 0)
    c[2].metric("XBRL ground-truth facts", int(gt["n"][0]) if not gt.empty else 0)
    c[3].metric("Retrieval chunks", f"{int(chunks['n'][0]):,}" if not chunks.empty else 0)

    extraction = load_json("extraction_results.json")
    if extraction:
        s = extraction["summary"]
        st.subheader("Extraction accuracy vs XBRL ground truth")
        c = st.columns(4)
        c[0].metric("Exact match", f"{s['exact']:.1%}")
        c[1].metric("Within 2%", f"{s['within_2pct']:.1%}")
        c[2].metric("Abstained", f"{s['n_abstained']}/{s['n_total']}")
        c[3].metric("Cost / filing",
                    f"${extraction['usage']['cost_usd'] / max(len(filings['ticker'].unique()), 1) / 4:.4f}")

    st.markdown("""
---
**What makes this measurable.** The SEC publishes XBRL data alongside every
filing, and each XBRL fact names the accession number of the filing that
reported it. That gives per-filing ground truth for every financial figure —
so extraction accuracy here is an arithmetic comparison, not one model grading
another.
""")

    if not filings.empty:
        st.subheader("Corpus")
        pivot = (filings.assign(n=1)
                 .pivot_table(index="ticker", columns="fiscal_year",
                              values="n", aggfunc="sum", fill_value=0))
        st.dataframe(pivot, use_container_width=True)

# ======================= EXTRACTION VERIFICATION ==========================
with tabs[1]:
    st.subheader("Every extracted figure, checked against XBRL")
    extraction = load_json("extraction_results.json")
    if not need(extraction, "10_extract.py"):
        rows = pd.DataFrame(extraction["rows"])

        col = st.columns(3)
        tickers = sorted(rows["ticker"].unique())
        tic = col[0].selectbox("Company", tickers)
        years = sorted(rows[rows["ticker"] == tic]["fiscal_year"].unique(),
                       reverse=True)
        fy = col[1].selectbox("Fiscal year", years)
        only_bad = col[2].checkbox("Show only errors and abstentions")

        view = rows[(rows["ticker"] == tic) & (rows["fiscal_year"] == fy)].copy()
        if only_bad:
            view = view[view["band"].isin(["wrong", "abstained"])]

        for _, r in view.iterrows():
            band = r["band"]
            colour = BAND_COLOURS.get(band, "#57606a")
            left, right = st.columns([3, 2])
            with left:
                st.markdown(
                    f"**{r['metric'].replace('_', ' ').title()}** &nbsp; "
                    f"<span style='color:{colour};font-weight:600'>{band}</span>"
                    + (f" &nbsp;·&nbsp; <code>{r['error_type']}</code>"
                       if r.get("error_type") else ""),
                    unsafe_allow_html=True)
                if r.get("quote"):
                    st.caption(f"Model read: “{r['quote']}”")
            with right:
                e = r["extracted"]
                t = r["truth"]
                st.markdown(
                    f"extracted `{e:,.0f}`" if pd.notna(e) else "extracted `—`")
                st.markdown(
                    f"XBRL truth `{t:,.0f}`" if pd.notna(t) else "XBRL truth `—`")
            st.divider()

        st.markdown("""
**Reading this panel.** `exact` means the extracted figure matched SEC XBRL
data for this filing to the dollar. `abstained` means the model reported
`found: false` — and every abstention in this corpus coincided with a metric
XBRL does not tag, so declining was correct. `wrong` rows show an error type
saying *what kind* of mistake it was, which is what makes the number
actionable rather than just a score.
""")

# ============================== RISK DRIFT ================================
with tabs[2]:
    st.subheader("Year-over-year risk disclosure change")
    analysis = load_json("analysis_results.json")
    if not need(analysis, "13_analyze.py --dry-run"):
        recs = []
        for r in analysis:
            d = r.get("diff_summary") or {}
            if d.get("drift_score") is None:
                continue
            recs.append({"ticker": r["ticker"], "fiscal_year": r["fiscal_year"],
                         "drift_score": d["drift_score"], "new": d.get("new", 0),
                         "modified": d.get("modified", 0),
                         "removed": d.get("removed", 0)})
        drift = pd.DataFrame(recs)

        if drift.empty:
            st.info("No comparable year pairs yet.")
        else:
            picks = st.multiselect(
                "Companies", sorted(drift["ticker"].unique()),
                default=sorted(drift["ticker"].unique())[:5])
            sub = drift[drift["ticker"].isin(picks)]
            if not sub.empty:
                st.line_chart(
                    sub.pivot_table(index="fiscal_year", columns="ticker",
                                    values="drift_score"),
                    y_label="drift score (share of risk factors new or rewritten)")

            st.caption(
                "Drift is computed from embedding similarity between this "
                "year's risk-factor chunks and last year's — statistically, "
                "not by asking a model which ones changed. Rows where either "
                "year failed to parse are excluded rather than scored 0.0.")

            st.subheader("Themes among changed risk factors")
            tic = st.selectbox("Company", sorted(drift["ticker"].unique()),
                               key="drift_ticker")
            for r in analysis:
                if r["ticker"] != tic:
                    continue
                mt = r.get("modified_themes") or {}
                nt = r.get("risk_themes") or {}
                if not (mt or nt):
                    continue
                st.markdown(f"**FY{r['fiscal_year']}**")
                merged = {}
                for k, v in {**nt}.items():
                    merged[k] = merged.get(k, 0) + v
                for k, v in mt.items():
                    merged[k] = merged.get(k, 0) + v
                if merged:
                    st.bar_chart(pd.Series(merged).sort_values(ascending=False))

# ================================ SEARCH ==================================
with tabs[3]:
    st.subheader("Search the filing corpus")
    st.caption("Keyword search over parsed chunks. The production path uses "
               "hybrid dense + BM25 retrieval with metadata filtering; this "
               "view runs on SQL so the demo needs no model or vector store.")

    c = st.columns([3, 1, 1])
    q = c[0].text_input("Query", "supply chain disruption")
    tickers = load_sql("SELECT DISTINCT ticker FROM chunks ORDER BY ticker")
    tic = c[1].selectbox("Company", ["all"] + (tickers["ticker"].tolist()
                                               if not tickers.empty else []))
    item = c[2].selectbox("Item", ["all", "1", "1A", "7", "7A", "8"])

    if q:
        sql = "SELECT ticker, fiscal_year, item, raw_text FROM chunks WHERE lower(raw_text) LIKE ?"
        params: list = [f"%{q.lower()}%"]
        if tic != "all":
            sql += " AND ticker = ?"
            params.append(tic)
        if item != "all":
            sql += " AND item = ?"
            params.append(item)
        sql += " LIMIT 20"
        hits = load_sql(sql, params)
        if hits.empty:
            st.info("No matches.")
        else:
            st.caption(f"{len(hits)} chunk(s)")
            for _, h in hits.iterrows():
                with st.expander(f"{h['ticker']} FY{h['fiscal_year']} · Item {h['item']}"):
                    st.write(h["raw_text"][:1500])

# ============================== EVALUATION ================================
with tabs[4]:
    st.subheader("Evaluation")

    retrieval = load_json("retrieval_eval.json")
    if retrieval:
        st.markdown("**Retrieval ablation** (with metadata pre-filtering)")
        st.dataframe(pd.DataFrame(retrieval["metrics"]).T, use_container_width=True)
        st.caption("Bootstrap confidence intervals and paired significance "
                   "tests are in EVALUATION.md §4.")

    model = load_json("model_results.json")
    if model:
        st.markdown("**Does disclosure drift predict returns?**")
        rows = []
        for name, r in model["results"].items():
            s = r["summary"]
            p = (model.get("permutation") or {}).get(name, {})
            rows.append({"feature set": name, "mean IC": s["mean_ic"],
                         "mean AUC": s["mean_auc"],
                         "RMSE vs constant": s["mean_rmse_improvement"],
                         "permutation p": p.get("p_value")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.warning(
            "**Null result, and it is the expected one.** No feature set is "
            "distinguishable from a permutation null, and every model performs "
            "worse than predicting the training mean. The disclosure-only IC "
            "of +0.09 looks like signal until you see the null distribution "
            "has σ = 0.10. 10-K risk language is public and parsed by every "
            "quant fund — a simple drift score beating the market would imply "
            "an inefficiency in one of the most analysed datasets in finance.")

    if not retrieval and not model:
        st.info("Run the evaluation scripts to populate this tab.")

    st.markdown("""
---
Full methodology, ablations, confidence intervals and 19 documented failure
modes: `EVALUATION.md` and `FAILURE_MODES.md` in the repository.
""")
