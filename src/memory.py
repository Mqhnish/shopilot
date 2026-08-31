"""Long-term memory across sessions, at the level of an anonymised cohort.

Pillar III asks the agent to keep "short-term session states *and* long-term
user profiles". :mod:`src.state` is the short-term half; this is the long-term
half, and two facts about the competition setup decide its shape.

**There are cohorts, not users.** The agent is handed a safe aggregate profile
-- purchase frequency, rating style, preference tags -- and never an identifier.
Those aggregates repeat: across the 200 public sessions there are only 75
distinct profile signatures, and 156 of the 200 share a signature with another
session. So there is real, repeated structure to learn from, and learning it
requires no identity at all. What accumulates here is a cohort statistic, and
nothing written to it could re-identify a shopper.

**The agent is never told whether it succeeded.** No callback, no reward, no
label. A session simply stops. But the evaluator ends a session the instant the
target appears in the returned list, and this agent's own truncation policy
means that list is usually a *single* product -- so "the session ended right
after we offered exactly one product" identifies that product as the target with
certainty. That is the supervision signal, and it costs nothing extra: it falls
out of a design decision made for a different reason.

What is learned is deliberately weak: a quality band, a vocabulary, and which
questions actually pay for this cohort. It informs ties. It cannot overrule
something the shopper said this turn.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

from .normalize import content_tokens

# Ceiling on the per-cohort vocabulary, so a long run cannot grow without bound.
MAX_TERMS_PER_COHORT = 400

# Evidence needed before a cohort's prior is trusted at all. One converted
# session is an anecdote.
MIN_OBSERVATIONS = 3

# Ceiling on the prior's contribution to the fused score. Everything here is a
# tie-breaker; a stated constraint scores an order of magnitude higher.
PRIOR_CEILING = 0.05


def cohort_key(profile: Optional[dict]) -> str:
    """A stable, non-identifying key for the aggregate profile.

    Built only from the fields ``docs/agent_api_contract.json`` declares, all of
    which are aggregates. The rating is bucketed to half a star so near-identical
    cohorts share evidence instead of each holding a sample of one.
    """
    if not isinstance(profile, dict):
        return "unknown"
    tags = profile.get("preference_tags")
    tag_part = ",".join(sorted(t for t in tags if isinstance(t, str))) if isinstance(tags, list) else ""
    rating = profile.get("average_prior_rating")
    if isinstance(rating, (int, float)):
        bucket = f"{round(float(rating) * 2) / 2:.1f}"
    else:
        bucket = "na"
    return "|".join((
        str(profile.get("purchase_frequency") or ""),
        str(profile.get("rating_style") or ""),
        tag_part,
        bucket,
    ))


class CohortRecord:
    """Everything one cohort has taught us."""

    __slots__ = ("observations", "quality_sum", "quality_sq", "terms",
                 "attribute_yield", "attribute_asks")

    def __init__(self) -> None:
        self.observations = 0
        self.quality_sum = 0.0
        self.quality_sq = 0.0
        self.terms: Dict[str, int] = {}
        self.attribute_yield: Dict[str, int] = {}
        self.attribute_asks: Dict[str, int] = {}

    @property
    def quality_mean(self) -> float:
        return self.quality_sum / self.observations if self.observations else 0.0

    @property
    def quality_sd(self) -> float:
        if self.observations < 2:
            return 0.0
        mean = self.quality_mean
        var = max(self.quality_sq / self.observations - mean * mean, 0.0)
        return math.sqrt(var)

    def trim(self) -> None:
        if len(self.terms) <= MAX_TERMS_PER_COHORT:
            return
        keep = sorted(self.terms.items(), key=lambda kv: -kv[1])[:MAX_TERMS_PER_COHORT]
        self.terms = dict(keep)


class CohortMemory:
    """Cohort statistics accumulated over the life of the process."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: Dict[str, CohortRecord] = {}
        self.confirmed = 0
        self.sessions_seen = 0

    # ------------------------------------------------------------------ writing

    def observe_conversion(
        self, key: str, title: str, quality: float, asked: Sequence[str],
        yielded: Sequence[str],
    ) -> None:
        """Record a session whose target we can identify with certainty."""
        if not self.enabled:
            return
        record = self.records.setdefault(key, CohortRecord())
        record.observations += 1
        record.quality_sum += quality
        record.quality_sq += quality * quality
        for token in set(content_tokens(title)[:32]):
            record.terms[token] = record.terms.get(token, 0) + 1
        record.trim()
        for attribute in asked:
            record.attribute_asks[attribute] = record.attribute_asks.get(attribute, 0) + 1
        for attribute in yielded:
            record.attribute_yield[attribute] = record.attribute_yield.get(attribute, 0) + 1
        self.confirmed += 1

    def observe_questions(self, key: str, asked: Sequence[str], yielded: Sequence[str]) -> None:
        """Question outcomes are usable even when the target stays unknown."""
        if not self.enabled or not (asked or yielded):
            return
        record = self.records.setdefault(key, CohortRecord())
        for attribute in asked:
            record.attribute_asks[attribute] = record.attribute_asks.get(attribute, 0) + 1
        for attribute in yielded:
            record.attribute_yield[attribute] = record.attribute_yield.get(attribute, 0) + 1

    # ------------------------------------------------------------------ reading

    def is_ready(self, key: str) -> bool:
        record = self.records.get(key)
        return record is not None and record.observations >= MIN_OBSERVATIONS

    def term_weights(self, key: str) -> Dict[str, float]:
        """Vocabulary this cohort's purchases keep using, as query weights."""
        record = self.records.get(key)
        if record is None or record.observations < MIN_OBSERVATIONS:
            return {}
        total = float(record.observations)
        return {
            term: count / total
            for term, count in record.terms.items()
            if count >= 2 and count / total >= 0.25
        }

    def quality_affinity(self, key: str, quality: float) -> float:
        """How well a product's quality matches this cohort's revealed band.

        A Gaussian bump around the cohort mean rather than "higher is better":
        a cohort that reliably buys mid-range items should not be pushed towards
        the highest-rated product in the category.
        """
        record = self.records.get(key)
        if record is None or record.observations < MIN_OBSERVATIONS:
            return 0.0
        spread = max(record.quality_sd, 0.05)
        z = (quality - record.quality_mean) / spread
        return math.exp(-0.5 * z * z)

    def attribute_bonus(self, key: str) -> Dict[str, float]:
        """Historical yield per question, in [0, 1], for tie-breaking only."""
        record = self.records.get(key)
        if record is None or record.observations < MIN_OBSERVATIONS:
            return {}
        out: Dict[str, float] = {}
        for attribute, asks in record.attribute_asks.items():
            if asks >= 2:
                out[attribute] = record.attribute_yield.get(attribute, 0) / float(asks)
        return out

    def stats(self) -> Dict[str, object]:
        return {
            "cohorts": len(self.records),
            "sessions_seen": self.sessions_seen,
            "confirmed_conversions": self.confirmed,
            "ready_cohorts": sum(
                1 for r in self.records.values() if r.observations >= MIN_OBSERVATIONS
            ),
        }
