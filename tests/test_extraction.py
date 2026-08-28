"""Extraction tests. No API key or network needed -- the LLM is stubbed.

The point of stubbing is that these test OUR logic (scale normalisation,
verification, schema repair, citation mapping), not the model's behaviour.
Tests that call a real LLM are slow, non-deterministic, and test the wrong
thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.extraction.schema import ExtractedFigure  # noqa: E402
from filingiq.extraction.verify import summarise, verify_figure  # noqa: E402


# --- scale normalisation ---------------------------------------------------

def test_millions_scale_is_applied():
    """The single most common extraction failure: reading 391,035 from a table
    headed '(in millions)' and reporting 391035."""
    f = ExtractedFigure(metric="revenue", found=True, value_as_stated=391035,
                        scale="millions", chunk_id="c1", confidence=0.9)
    assert f.normalized_value == 391_035_000_000


def test_thousands_and_billions():
    a = ExtractedFigure(metric="revenue", found=True, value_as_stated=391035000,
                        scale="thousands", chunk_id="c", confidence=0.9)
    b = ExtractedFigure(metric="revenue", found=True, value_as_stated=391.035,
                        scale="billions", chunk_id="c", confidence=0.9)
    assert a.normalized_value == 391_035_000_000
    assert b.normalized_value == pytest.approx(391_035_000_000)


def test_eps_uses_units():
    f = ExtractedFigure(metric="eps_diluted", found=True, value_as_stated=6.13,
                        scale="units", chunk_id="c", confidence=0.9)
    assert f.normalized_value == pytest.approx(6.13)


def test_abstention_yields_no_value():
    f = ExtractedFigure(metric="revenue", found=False)
    assert f.normalized_value is None
    assert not f.is_usable()


def test_citation_is_required_for_usability():
    """A figure without provenance cannot be verified by a human, so it is not
    usable regardless of the model's confidence."""
    f = ExtractedFigure(metric="revenue", found=True, value_as_stated=1,
                        scale="units", chunk_id=None, confidence=0.99)
    assert not f.is_usable()


def test_low_confidence_is_not_usable():
    f = ExtractedFigure(metric="revenue", found=True, value_as_stated=1,
                        scale="units", chunk_id="c", confidence=0.2)
    assert not f.is_usable()


def test_long_quotes_are_truncated():
    f = ExtractedFigure(metric="revenue", found=True, quote=" ".join(["w"] * 60))
    assert len(f.quote.split()) <= 26


# --- verification ----------------------------------------------------------

def test_exact_match():
    assert verify_figure("revenue", 391035000000.0, 391035000000.0).band == "exact"


def test_scale_error_is_named_not_just_wrong():
    v = verify_figure("revenue", 391035.0, 391035000000.0)
    assert v.band == "wrong"
    assert v.error_type == "scale_error_millions"


def test_prior_year_is_wrong_even_inside_tolerance():
    """AAPL FY2023 revenue is 2% from FY2024, so a wrong-column read lands
    inside the 2% band. Reading the wrong column is wrong regardless."""
    v = verify_figure("revenue", 383285000000.0, 391035000000.0,
                      prior_year_truth=383285000000.0)
    assert v.band == "wrong"
    assert v.error_type == "wrong_period_prior_year"
    assert not v.correct


def test_close_but_not_prior_year_still_passes_tolerance():
    """Guard against the prior-year check condemning genuinely-close values."""
    v = verify_figure("revenue", 391100000000.0, 391035000000.0,
                      prior_year_truth=383285000000.0)
    assert v.band == "within_0.5pct"


def test_wrong_metric_is_identified():
    v = verify_figure("net_income", 391035000000.0, 93736000000.0,
                      all_truth={"revenue": 391035000000.0})
    assert v.error_type == "wrong_metric_matched_revenue"


def test_sign_error():
    v = verify_figure("net_income", -93736000000.0, 93736000000.0)
    assert v.error_type == "sign_error"


def test_abstention_is_not_counted_as_wrong():
    v = verify_figure("revenue", None, 391035000000.0)
    assert v.abstained and not v.correct
    s = summarise([v])
    assert s["n_scorable"] == 0
    assert s["abstention_rate"] == 1.0


def test_missing_ground_truth_is_not_scored():
    v = verify_figure("revenue", 1.0, None)
    assert summarise([v])["n_scorable"] == 0


def test_bands_are_cumulative():
    T = 100_000_000_000.0
    vs = [verify_figure("m", T, T),                # exact
          verify_figure("m", T * 1.003, T),        # 0.3%
          verify_figure("m", T * 1.015, T)]        # 1.5%
    s = summarise(vs)
    # summarise() rounds to 3dp, so compare with an absolute tolerance.
    assert s["exact"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["within_0.5pct"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["within_2pct"] == pytest.approx(1.0, abs=1e-3)


# --- agent plumbing (LLM stubbed) -----------------------------------------

class FakeHit:
    def __init__(self, cid, text, item="8"):
        self.chunk_id = cid
        self.score = 1.0
        self.payload = {"chunk_id": cid, "text": text, "item": item}


class FakeRetriever:
    def __init__(self, hits): self.hits = hits
    def search(self, *a, **k): return self.hits


class FakeRouter:
    """Returns queued responses, so schema-repair paths can be exercised."""
    def __init__(self, responses):
        self.responses, self.i, self.model, self.provider = responses, 0, "fake", "fake"
        from filingiq.llm.router import UsageTracker
        self.usage = UsageTracker()

    def complete(self, system, user, **kw):
        from filingiq.llm.router import LLMResponse
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        r = LLMResponse(text=text, model="fake", tokens_in=10, tokens_out=5,
                        latency_ms=1, cost_usd=0.0)
        self.usage.record(r)
        return r


def _agent(responses, hits=None):
    from filingiq.extraction.agent import ExtractionAgent
    hits = hits or [FakeHit("real-chunk-1", "Net sales | 391,035"),
                    FakeHit("real-chunk-2", "other")]
    return ExtractionAgent(FakeRetriever(hits), FakeRouter(responses))


GOOD = ('{"found": true, "value_as_stated": 391035, "scale": "millions", '
        '"currency": "USD", "chunk_id": "1", "quote": "Net sales 391,035", '
        '"confidence": 0.95, "reasoning": "first column"}')


def test_agent_maps_excerpt_number_to_real_chunk_id():
    """The model cites '1'; provenance must resolve to the actual chunk."""
    fig = _agent([GOOD]).extract_metric("revenue", "AAPL", 2024)
    assert fig.chunk_id == "real-chunk-1"
    assert fig.normalized_value == 391_035_000_000


def test_agent_repairs_malformed_json():
    a = _agent(["not json at all", GOOD])
    fig = a.extract_metric("revenue", "AAPL", 2024)
    assert fig.found and a.repairs == 1


def test_agent_gives_up_after_one_repair():
    a = _agent(["garbage", "still garbage"])
    fig = a.extract_metric("revenue", "AAPL", 2024)
    assert not fig.found and a.schema_failures == 1


def test_agent_handles_empty_retrieval():
    from filingiq.extraction.agent import ExtractionAgent
    a = ExtractionAgent(FakeRetriever([]), FakeRouter([GOOD]))
    fig = a.extract_metric("revenue", "AAPL", 2024)
    assert not fig.found
    assert "no chunks" in (fig.reasoning or "")


def test_agent_respects_model_abstention():
    a = _agent(['{"found": false, "confidence": 0.0, "reasoning": "not present"}'])
    fig = a.extract_metric("revenue", "AAPL", 2024)
    assert not fig.found and fig.normalized_value is None


def test_hallucinated_citation_is_downgraded():
    """If the model cites an excerpt that does not exist, confidence is capped
    rather than the citation being accepted at face value."""
    bad = GOOD.replace('"chunk_id": "1"', '"chunk_id": "99"')
    fig = _agent([bad]).extract_metric("revenue", "AAPL", 2024)
    assert fig.confidence <= 0.5


def test_build_context_numbers_excerpts():
    from filingiq.extraction.agent import build_context
    ctx, id_map = build_context([FakeHit("abc", "hello"), FakeHit("def", "world")])
    assert "Excerpt 1" in ctx and "Excerpt 2" in ctx
    assert id_map == {"1": "abc", "2": "def"}


def test_eps_near_miss_is_not_exact():
    """Regression: the absolute-epsilon shortcut used for large balance-sheet
    figures must not apply to per-share amounts, where a difference of 1.0 is
    a 16% error, not floating-point noise."""
    v = verify_figure("eps_diluted", 5.13, 6.13)
    assert v.band == "wrong", "EPS off by a full dollar scored as exact"


def test_large_figures_still_tolerate_sub_dollar_noise():
    v = verify_figure("revenue", 391035000000.4, 391035000000.0)
    assert v.band == "exact"


# --- preflight and fatal-error handling -----------------------------------

def test_config_errors_are_not_retried():
    """A 404 or bad key does not become correct on the third attempt. Burning
    retries on config errors turns a 2-second failure into a 6-minute one."""
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="ollama", model="nope", requests_per_second=100)

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("404 Client Error: Not Found for url: /api/chat")

    r._complete_ollama = boom
    with pytest.raises(RuntimeError):
        r.complete("s", "u", retries=3)
    assert calls["n"] == 1, "retried a fatal configuration error"


def test_transient_errors_are_retried():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="ollama", model="m", requests_per_second=100)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        raise RuntimeError("503 Service Unavailable")

    r._complete_ollama = flaky
    resp = r.complete("s", "u", retries=2)
    assert calls["n"] == 2 and resp.error is not None


def test_healthcheck_reports_failure():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="ollama", model="m", requests_per_second=100)
    r._complete_ollama = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("503 unavailable"))
    ok, msg = r.healthcheck()
    assert not ok and msg


def test_autoselect_prefers_the_first_available():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="groq", model="dead-model", requests_per_second=100)
    r.list_models = lambda: ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "whisper-large-v3"]
    assert r.autoselect_model() == "openai/gpt-oss-120b"


def test_autoselect_falls_back_to_any_chat_model():
    """Preferred names may all be gone. Anything chat-capable beats failing."""
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="groq", model="x", requests_per_second=100)
    r.list_models = lambda: ["some-new-model", "whisper-large-v3", "llama-guard-4"]
    assert r.autoselect_model() == "some-new-model"


def test_autoselect_excludes_non_chat_models():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="groq", model="x", requests_per_second=100)
    r.list_models = lambda: ["whisper-large-v3", "canopylabs/orpheus-v1-english"]
    assert r.autoselect_model() is None


def test_autoselect_returns_none_when_catalogue_unreachable():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="groq", model="x", requests_per_second=100)
    r.list_models = lambda: []
    assert r.autoselect_model() is None


# --- reasoning models ------------------------------------------------------

def test_reasoning_models_are_detected():
    from filingiq.llm.router import is_reasoning_model
    assert is_reasoning_model("openai/gpt-oss-120b")
    assert is_reasoning_model("qwen/qwen3.6-27b")
    assert not is_reasoning_model("llama-3.1-8b-instant")


def test_reasoning_models_get_a_token_floor():
    """A 20-token budget makes a reasoning model spend everything on internal
    thinking and return an empty string, which the provider then rejects as
    invalid JSON. The error names the prompt; the cause is the budget."""
    from filingiq.llm.router import ModelRouter, REASONING_MIN_TOKENS
    captured = {}

    r = ModelRouter(provider="groq", model="openai/gpt-oss-120b",
                    requests_per_second=100)

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            class M: content = '{"ok": true}'
            class C: message = M()
            class U: prompt_tokens, completion_tokens = 5, 5
            class R: choices, usage = [C()], U()
            return R()

    class FakeClient:
        chat = type("X", (), {"completions": FakeCompletions()})()

    r._client = FakeClient()
    r.complete("s", "u", max_tokens=20)
    assert captured["max_tokens"] >= REASONING_MIN_TOKENS
    assert captured.get("reasoning_effort") == "low"


def test_non_reasoning_models_keep_their_budget():
    from filingiq.llm.router import ModelRouter
    captured = {}
    r = ModelRouter(provider="groq", model="llama-3.1-8b-instant",
                    requests_per_second=100)

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            class M: content = "{}"
            class C: message = M()
            class U: prompt_tokens, completion_tokens = 1, 1
            class R: choices, usage = [C()], U()
            return R()

    r._client = type("X", (), {"chat": type("Y", (), {"completions": FakeCompletions()})()})()
    r.complete("s", "u", max_tokens=20)
    assert captured["max_tokens"] == 20
    assert "reasoning_effort" not in captured


def test_json_validate_failure_is_retried_not_fatal():
    from filingiq.llm.router import ModelRouter
    r = ModelRouter(provider="groq", model="openai/gpt-oss-120b",
                    requests_per_second=100)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("Error code: 400 - json_validate_failed")

    r._complete_groq = boom
    resp = r.complete("s", "u", retries=2)
    assert calls["n"] == 2, "treated a recoverable JSON failure as fatal"
    assert resp.error


# --- grouped extraction ----------------------------------------------------

BATCH_GOOD = ('{"figures": ['
  '{"metric": "revenue", "found": true, "value_as_stated": 391035, '
  '"scale": "millions", "chunk_id": "1", "confidence": 0.95},'
  '{"metric": "net_income", "found": true, "value_as_stated": 93736, '
  '"scale": "millions", "chunk_id": "1", "confidence": 0.95}]}')


def test_batch_extraction_returns_all_requested_metrics():
    a = _agent([BATCH_GOOD])
    figs = a.extract_group("income_statement", ["revenue", "net_income"],
                           "AAPL", 2024)
    assert {f.metric for f in figs} == {"revenue", "net_income"}
    assert figs[0].normalized_value == 391_035_000_000


def test_omitted_metric_becomes_an_abstention_not_a_missing_row():
    """If the model silently drops a metric from its array, the count must
    still be right -- a vanished metric would quietly inflate accuracy."""
    a = _agent([BATCH_GOOD])
    figs = a.extract_group("income_statement",
                           ["revenue", "net_income", "rnd_expense"], "AAPL", 2024)
    assert len(figs) == 3
    rnd = next(f for f in figs if f.metric == "rnd_expense")
    assert not rnd.found and "omitted" in (rnd.reasoning or "")


def test_batch_maps_citations_to_real_chunk_ids():
    a = _agent([BATCH_GOOD])
    figs = a.extract_group("income_statement", ["revenue"], "AAPL", 2024)
    assert figs[0].chunk_id == "real-chunk-1"


def test_grouped_filing_uses_fewer_calls_than_per_metric():
    metrics = ["revenue", "net_income", "total_assets", "operating_cash_flow"]
    a = _agent([BATCH_GOOD] * 10)
    a.extract_filing("AAPL", 2024, "acc", metrics, grouped=True)
    grouped_calls = a.router.usage.calls

    b = _agent([GOOD] * 10)
    b.extract_filing("AAPL", 2024, "acc", metrics, grouped=False)
    assert grouped_calls < b.router.usage.calls


def test_batch_parse_failure_falls_back_to_abstention():
    a = _agent(["not json", "still not json"])
    figs = a.extract_group("income_statement", ["revenue", "net_income"],
                           "AAPL", 2024)
    assert len(figs) == 2 and not any(f.found for f in figs)


def test_generation_bottleneck_threshold_is_strict():
    """A 99.3% ceiling with 81.9% accuracy is a generation problem, not a
    balanced system. An 80% threshold called that 'balanced' and pointed the
    next day's work at the wrong layer."""
    ceiling, acc = 0.993, 0.819
    assert acc < ceiling * 0.95, "threshold too lenient to flag the real gap"
    assert not (acc < ceiling * 0.80), "the old threshold missed it"


# --- statement scope -------------------------------------------------------

def test_accounting_identity_does_not_catch_parent_company_error():
    """Documents a real limitation rather than asserting a capability.

    JPM FY2021: the model read Parent-Company-Only figures. Those statements
    balance, and parent-only equity EQUALS consolidated equity, so
    assets = liabilities + equity holds exactly. A coherent wrong answer
    passes every arithmetic check -- which is why the scope guard is lexical."""
    from filingiq.extraction.verify import check_accounting_identities
    parent_only = {"total_assets": 568_481e6, "total_liabilities": 274_354e6,
                   "stockholders_equity": 294_127e6}
    assert abs(sum([parent_only["total_liabilities"],
                    parent_only["stockholders_equity"]])
               - parent_only["total_assets"]) < 1
    assert check_accounting_identities(parent_only) == []


def test_identity_catches_genuinely_inconsistent_figures():
    from filingiq.extraction.verify import check_accounting_identities
    v = check_accounting_identities({"total_assets": 100e9,
                                     "total_liabilities": 20e9,
                                     "stockholders_equity": 10e9})
    assert any(x.rule == "balance_sheet_identity" for x in v)


def test_identity_catches_equity_exceeding_assets():
    from filingiq.extraction.verify import check_accounting_identities
    v = check_accounting_identities({"total_assets": 10e9,
                                     "stockholders_equity": 50e9})
    assert any(x.rule == "equity_exceeds_assets" for x in v)


def test_identity_ignores_missing_figures():
    from filingiq.extraction.verify import check_accounting_identities
    assert check_accounting_identities({"total_assets": 100e9}) == []


def test_parent_company_chunks_are_demoted_not_deleted():
    from filingiq.extraction.agent import deprioritise_non_consolidated
    hits = [FakeHit("seg", "Parent Company Only condensed balance sheet Total assets 568,481"),
            FakeHit("cons", "Consolidated balance sheets Total assets 3,743,567")]
    out = deprioritise_non_consolidated(hits)
    assert out[0].chunk_id == "cons", "consolidated table must come first"
    assert len(out) == 2, "demote, never delete -- it may be the only table"


def test_segment_and_vie_tables_are_also_demoted():
    from filingiq.extraction.agent import deprioritise_non_consolidated
    for phrase in ["reportable segment results", "variable interest entities"]:
        out = deprioritise_non_consolidated(
            [FakeHit("bad", phrase + " Total assets 1"),
             FakeHit("good", "Consolidated balance sheets Total assets 2")])
        assert out[0].chunk_id == "good"


# --- errored vs abstained --------------------------------------------------

def test_failed_call_is_not_an_abstention():
    """A rate-limited call and a cautious model both produce no value. Treating
    them the same let an outage masquerade as caution and corrupted a results
    file with fake 100%-abstention filings."""
    v = verify_figure("revenue", None, 391035000000.0, errored=True)
    assert v.band == "errored"
    assert not v.abstained
    assert v.error_type == "call_failed"


def test_summary_separates_errors_from_abstentions():
    vs = [verify_figure("a", None, 1e9, errored=True),
          verify_figure("b", None, 1e9),
          verify_figure("c", 1e9, 1e9)]
    s = summarise(vs)
    assert s["n_errored"] == 1
    assert s["n_abstained"] == 1
    assert s["n_scorable"] == 1
    assert s["exact"] == 1.0, "a failed call must not dilute accuracy"


def test_errored_figures_are_excluded_from_scoring():
    """Errors must not count as wrong answers either -- that would understate
    accuracy just as counting them as abstentions overstates caution."""
    vs = [verify_figure(f"m{i}", None, 1e9, errored=True) for i in range(9)]
    vs.append(verify_figure("m9", 1e9, 1e9))
    s = summarise(vs)
    assert s["n_scorable"] == 1 and s["exact"] == 1.0
    assert s["error_rate"] == 0.9


def test_no_ground_truth_must_not_count_against_accuracy():
    """JPM has 10 metrics XBRL never tags. Counting those as failures showed
    50% for a company scoring near 100% on every checkable figure, and would
    have penalised precisely the filings that differ most from the norm."""
    vs = [verify_figure("m1", 1e9, 1e9),
          verify_figure("m2", 1e9, 1e9),
          verify_figure("m3", 5e8, None)]     # extracted, but no ground truth
    s = summarise(vs)
    assert s["n_scorable"] == 2
    assert s["exact"] == 1.0, "an unscorable figure diluted the accuracy"
