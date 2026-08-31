"""Dual-track intent routing.

The brief's first pillar asks for an instant split between a high-precision
"Buying" track that locks hard constraints and a diverse "Browsing" track that
opens up cross-category scenario matching. The split is not cosmetic: the two
tracks want genuinely different rankers. A buyer who has named a constraint
wants that constraint enforced and near-duplicates are fine; a browser who has
named nothing wants spread, because ten minor variants of one product waste the
whole turn.

Routing is evidence-driven rather than keyword-driven. The score is a
specificity measure -- how much the customer has actually pinned down -- so it
keeps working when the opening line is phrased differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .state import SessionState

BUY = "buy"
BROWSE = "browse"
BLEND = "blend"

# Between these two thresholds the evidence genuinely does not separate the
# tracks. Committing early is how a session either over-filters a browser down
# to nothing or hands a decided buyer a diverse spread it has to wade through.
BROWSE_MAX = 0.30
BUY_MIN = 0.55


def specificity(state: "SessionState") -> float:
    """How pinned-down the request is, in [0, 1]."""
    score = 0.0
    if state.category:
        score += 0.25
    # Distinct disclosed constraints, saturating: the third and fourth add less
    # than the first, because by then the pool is already small.
    n = len(state.disclosed_keys)
    score += min(n, 4) * 0.145
    if state.scenario == "buying":
        score += 0.14
    elif state.scenario == "browsing" and n == 0:
        score -= 0.12
    if state.override_seen:
        score += 0.08
    return max(0.0, min(1.0, score))


def route(state: "SessionState") -> tuple:
    """Return ``(track, specificity)`` for this turn."""
    value = specificity(state)
    if value >= BUY_MIN:
        return BUY, value
    if value <= BROWSE_MAX:
        return BROWSE, value
    return BLEND, value


def weights(track: str) -> dict:
    """Per-track fusion weights, consumed by :mod:`src.rank`.

    ``phrase`` dominates once anything concrete has been disclosed, because an
    exact rare-constraint match is near-conclusive. ``vector`` carries the
    browsing track, where cosine's length normalisation surfaces short precise
    titles that BM25 buries. ``quality`` is a mild prior that only decides
    otherwise-tied candidates.
    """
    if track == BUY:
        return {"phrase": 1.00, "bm25": 0.30, "vector": 0.22, "quality": 0.030, "mmr": 0.00}
    if track == BROWSE:
        return {"phrase": 1.00, "bm25": 0.36, "vector": 0.46, "quality": 0.115, "mmr": 0.30}
    return {"phrase": 1.00, "bm25": 0.33, "vector": 0.34, "quality": 0.070, "mmr": 0.15}
