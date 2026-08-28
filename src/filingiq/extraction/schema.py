"""Extraction schema and scale normalisation.

THE SCALE PROBLEM
-----------------
This is the hard part of financial extraction, and it is invisible until you
measure against ground truth.

A 10-K income statement is headed "(in millions)" and the revenue cell reads
391,035. XBRL records 391035000000. The model that reads "391,035" and reports
391035 is not hallucinating -- it read the page correctly. It just did not
apply the scale, and a naive comparison scores it as catastrophically wrong.

So the schema separates two things the model must report independently:

    value_as_stated : the number exactly as printed on the page
    scale           : the multiplier declared in the table header

Normalisation happens in code, deterministically, not in the model's head.
Asking an LLM to multiply 391,035 by 1,000,000 and return 391035000000
introduces arithmetic errors for no benefit -- the multiplication is trivial in
Python and impossible to get wrong there.

ABSTENTION IS A FIRST-CLASS OUTCOME
-----------------------------------
`found=False` is a valid, desirable answer. A model that invents a plausible
figure when the retrieved context does not contain one is far more dangerous
than one that declines. Refusal rate is therefore reported alongside accuracy:
a system with 99% accuracy on 30% of questions may be preferable to one with
80% accuracy on all of them, depending on what the answer is used for.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCALE_FACTORS = {
    "units": 1.0,
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
}

Scale = Literal["units", "thousands", "millions", "billions"]


class ExtractedFigure(BaseModel):
    """One financial figure, with provenance and an explicit scale."""

    metric: str
    found: bool = Field(description="False if the context does not contain it")
    value_as_stated: float | None = Field(
        default=None, description="The number exactly as printed, unscaled")
    scale: Scale | None = Field(
        default=None, description="Multiplier from the table header")
    currency: str | None = "USD"
    period_end: str | None = Field(
        default=None, description="Fiscal period end, YYYY-MM-DD")
    chunk_id: str | None = Field(
        default=None, description="Chunk the figure was read from")
    quote: str | None = Field(
        default=None, description="Short supporting text, under 15 words")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str | None = None
    # What retrieval actually supplied. Without this you cannot tell a
    # retrieval failure from a generation failure after the fact, and that is
    # the single most important distinction when debugging a RAG pipeline.
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("quote")
    @classmethod
    def _cap_quote(cls, v: str | None) -> str | None:
        # Keeps the citation a citation rather than a reproduction, and stops
        # the model padding output with long verbatim passages.
        if v and len(v.split()) > 25:
            return " ".join(v.split()[:25]) + "..."
        return v

    @property
    def normalized_value(self) -> float | None:
        """value_as_stated x scale factor. Computed in code, never by the model."""
        if not self.found or self.value_as_stated is None:
            return None
        factor = SCALE_FACTORS.get(self.scale or "units", 1.0)
        return self.value_as_stated * factor

    def is_usable(self, min_confidence: float = 0.5) -> bool:
        return (self.found
                and self.normalized_value is not None
                and self.chunk_id is not None
                and self.confidence >= min_confidence)


class ExtractionResult(BaseModel):
    """All figures extracted from one filing."""
    accession: str
    ticker: str
    fiscal_year: int
    figures: list[ExtractedFigure] = Field(default_factory=list)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    def by_metric(self) -> dict[str, ExtractedFigure]:
        return {f.metric: f for f in self.figures}


# Metric definitions given to the model. Precise wording matters: "total
# revenue" is ambiguous for a company reporting products and services
# separately, so each definition states which line to take.
METRIC_DEFINITIONS = {
    "revenue": "Total net sales or total revenues for the full fiscal year "
               "(the consolidated total, not a segment or product line).",
    "net_income": "Net income attributable to the company for the full fiscal year.",
    "operating_income": "Operating income or income from operations for the year.",
    "eps_diluted": "Diluted earnings per share for the year, in dollars per share.",
    "total_assets": "Total assets at the fiscal year end (balance sheet total).",
    "total_liabilities": "Total liabilities at the fiscal year end.",
    "stockholders_equity": "Total shareholders' equity at the fiscal year end.",
    "cash_and_equivalents": "Cash and cash equivalents at the fiscal year end "
                            "(exclude marketable securities).",
    "operating_cash_flow": "Net cash generated by operating activities for the year.",
    "rnd_expense": "Research and development expense for the year.",
}
