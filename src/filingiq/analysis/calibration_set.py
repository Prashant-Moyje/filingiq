"""A labelled risk-factor set for calibrating the diff thresholds.

WHY THIS EXISTS
---------------
`UNCHANGED_THRESHOLD = 0.95` and `MODIFIED_THRESHOLD = 0.80` in diff.py were
asserted, never fitted. Without labels there is no precision or recall for the
diff, so "the drift score is 0.31" has never had an error bar of any kind.

WHAT THE LABELS ARE, AND ARE NOT
--------------------------------
These are ground truth BY CONSTRUCTION, not human judgments on real filings.
Each current-year factor is derived from a prior-year one by a known
transformation, so its correct status is known before any model sees it:

  unchanged  carried forward with no substantive change -- verbatim, or with
             only dates and figures refreshed, which is what the great
             majority of real 10-K risk factors do year to year
  modified   same underlying risk, substantively rewritten -- clauses added,
             language escalated or softened, structure reorganised
  new        a risk with no counterpart in the prior year

The honest limitation: a transformation I write is not drawn from the same
distribution as a transformation a securities lawyer writes. Real edits may be
subtler (a single hedging word) or blunter (wholesale section reorganisation)
than these. Thresholds fitted here should be treated as a calibrated starting
point with a measured basis, not as a substitute for labelled real filings.

The text is written in the register of actual Item 1A prose -- long sentences,
"could materially and adversely affect", explicit forward-looking hedges --
because embedding similarity is sensitive to register and a calibration on
crisp modern prose would not transfer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelledPair:
    topic: str
    prior: str
    current: str
    label: str          # 'unchanged' | 'modified' | 'new'
    transformation: str


# --- prior-year factors ----------------------------------------------------

_SUPPLY = (
    "We depend on a limited number of suppliers, and in some cases a single "
    "supplier, for components that are critical to the manufacture of our "
    "products. We do not have long-term supply agreements with all of these "
    "vendors, and an interruption in supply, whether caused by capacity "
    "constraints, financial difficulty at a supplier, natural disaster or "
    "labor dispute, could delay shipments and increase our costs. Any such "
    "interruption could materially and adversely affect our business, results "
    "of operations and financial condition."
)

_CYBER = (
    "We face ongoing attempts by third parties to gain unauthorized access to "
    "our information technology systems and to the data of our customers. "
    "Techniques used to obtain unauthorized access change frequently and may "
    "not be recognized until launched against a target. A significant breach "
    "could interrupt our operations, damage our reputation, expose us to "
    "notification obligations and regulatory investigation, and result in "
    "litigation and material remediation expense."
)

_REGULATORY = (
    "Our business is subject to extensive regulation in the jurisdictions in "
    "which we operate. Changes in law or in the interpretation or enforcement "
    "of existing law, including rules governing competition, consumer "
    "protection and the collection and use of personal information, may "
    "require changes to our products and business practices and may increase "
    "our cost of compliance. We cannot predict the timing or scope of such "
    "changes."
)

_TALENT = (
    "Our future success depends in substantial part on our ability to attract "
    "and retain highly skilled engineering, product and executive personnel. "
    "Competition for such personnel is intense, particularly in the "
    "geographies in which our principal research facilities are located. The "
    "loss of key employees, or an inability to hire on acceptable terms, could "
    "delay our product development efforts and harm our competitive position."
)

_LIQUIDITY = (
    "We have substantial indebtedness, and our credit facilities contain "
    "covenants that restrict our ability to incur additional debt, make "
    "certain investments and pay dividends. Deterioration in our operating "
    "results, or adverse conditions in the credit markets, could impair our "
    "ability to refinance maturing obligations on acceptable terms or at all, "
    "which would adversely affect our liquidity and financial condition."
)

_LITIGATION = (
    "We are party to various legal proceedings arising in the ordinary course "
    "of business, including claims relating to product liability, "
    "intellectual property and employment matters. Litigation is inherently "
    "uncertain, and an adverse determination in one or more of these matters "
    "could result in damages or injunctive relief that exceeds our insurance "
    "coverage and materially affects our results of operations."
)

_CONCENTRATION = (
    "A limited number of customers account for a substantial portion of our "
    "net revenue. These customers are not obligated to purchase any minimum "
    "volume and may reduce or discontinue orders with limited notice. The "
    "loss of one or more significant customers, or a material reduction in "
    "their purchases, would reduce our revenue and could materially and "
    "adversely affect our operating results."
)

_FX = (
    "A significant portion of our revenue is denominated in currencies other "
    "than the U.S. dollar. Strengthening of the U.S. dollar reduces the "
    "reported value of that revenue upon translation. Our hedging activities, "
    "where undertaken, do not eliminate this exposure and may themselves give "
    "rise to losses. Currency volatility could therefore materially affect our "
    "reported financial results."
)

PRIOR_FACTORS: dict[str, str] = {
    "supply_chain": _SUPPLY,
    "cybersecurity": _CYBER,
    "regulatory": _REGULATORY,
    "talent": _TALENT,
    "liquidity": _LIQUIDITY,
    "litigation": _LITIGATION,
    "concentration": _CONCENTRATION,
    "foreign_exchange": _FX,
}


# --- transformations -------------------------------------------------------
#
# unchanged: verbatim, or only dates/figures refreshed. This is what most real
# risk factors do -- filings copy them forward almost word for word.

_UNCHANGED: dict[str, str] = {
    # Verbatim carry-forward.
    "supply_chain": _SUPPLY,
    "cybersecurity": _CYBER,
    # Figures and dates refreshed, wording otherwise identical.
    "regulatory": _REGULATORY.replace(
        "We cannot predict the timing or scope of such changes.",
        "We cannot predict the timing or scope of such changes. During fiscal "
        "2024 we incurred $47 million of incremental compliance cost."),
    "talent": _TALENT.replace(
        "Competition for such personnel is intense",
        "Competition for such personnel remains intense"),
}

# modified: same risk, substantively rewritten.

_MODIFIED: dict[str, str] = {
    "liquidity": (
        "Our leverage has increased following the acquisitions completed during "
        "the year, and a greater share of our debt now bears interest at "
        "floating rates. Covenants in our amended credit agreement impose "
        "tighter limits on additional borrowing and on distributions to "
        "shareholders than those previously in effect. If earnings decline or "
        "credit spreads widen, we may be unable to refinance maturing "
        "obligations on acceptable terms, and a downgrade of our credit "
        "ratings would further increase our cost of capital."
    ),
    "litigation": (
        "We are defending a putative class action alleging defects in a "
        "discontinued product line, in addition to the ordinary-course "
        "intellectual property and employment matters previously disclosed. "
        "The class action seeks damages substantially in excess of our "
        "available product liability coverage. An adverse determination, or a "
        "settlement on unfavourable terms, could require charges that "
        "materially exceed the amounts we have reserved."
    ),
    "concentration": (
        "Customer concentration increased during the year as our two largest "
        "accounts grew faster than the remainder of our business. Neither "
        "customer is contractually obligated to purchase minimum volumes, and "
        "one has publicly disclosed a strategy of qualifying second sources "
        "for components of the type we supply. A decision by either to shift "
        "volume away from us would reduce revenue materially and could require "
        "us to write down dedicated manufacturing assets."
    ),
}

# new: risks with no counterpart in the prior year. These are the hard cases --
# they are still SEC risk-factor prose, so they share register and hedging
# language with every prior factor, and the diff scores them against their
# BEST prior match, not against a random one.

_NEW: dict[str, str] = {
    "ai_regulation": (
        "Emerging legal frameworks governing artificial intelligence, "
        "including the EU AI Act and comparable proposals in other "
        "jurisdictions, may impose transparency, testing and record-keeping "
        "obligations on systems we develop or deploy. Compliance may require "
        "us to modify product features, restrict availability in certain "
        "markets, or incur costs we cannot presently quantify. Failure to "
        "comply could expose us to substantial administrative penalties."
    ),
    "climate_transition": (
        "Our operations are exposed to the physical effects of climate change, "
        "including flooding and extreme heat at facilities in coastal and arid "
        "regions. Transition risk may also raise our costs, as carbon pricing "
        "mechanisms and disclosure mandates are extended to sectors in which "
        "we operate. Capital expenditure required to reduce the emissions "
        "intensity of our operations may exceed current estimates."
    ),
    "pension_obligation": (
        "Our defined benefit pension plans are sensitive to discount rates and "
        "to the investment performance of plan assets. A decline in long-term "
        "rates increases the measured obligation and may require accelerated "
        "cash contributions. Actuarial assumptions concerning mortality and "
        "salary growth are revised periodically, and unfavourable revisions "
        "would increase our recognised liability."
    ),
}


def build_pairs() -> list[LabelledPair]:
    """Every labelled pair, with its transformation named."""
    out: list[LabelledPair] = []
    for topic, text in _UNCHANGED.items():
        how = "verbatim" if text == PRIOR_FACTORS[topic] else "figures/dates refreshed"
        out.append(LabelledPair(topic, PRIOR_FACTORS[topic], text, "unchanged", how))
    for topic, text in _MODIFIED.items():
        out.append(LabelledPair(topic, PRIOR_FACTORS[topic], text, "modified",
                                "substantive rewrite"))
    for topic, text in _NEW.items():
        out.append(LabelledPair(topic, "", text, "new", "no prior counterpart"))
    return out


def current_year_section() -> list[tuple[str, str, str]]:
    """(topic, text, true_label) for a synthetic current-year Item 1A."""
    return [(p.topic, p.current, p.label) for p in build_pairs()]


def prior_year_section() -> list[tuple[str, str]]:
    """(topic, text) for the prior year. Every prior factor is present, so a
    'new' current factor is scored against the full prior set exactly as
    diff_sections does -- against its BEST match, not a random one."""
    return list(PRIOR_FACTORS.items())
