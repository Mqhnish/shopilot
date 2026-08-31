"""Choosing the question, by expected information gain.

The customer's disclosure policy is not "answer the question asked"; it is
"reveal my undisclosed constraints whose *type* matches the attribute you named,
at most two of them". So the value of a question is entirely determined by how
finely it splits the candidates that are still in play -- and that is
computable, because for any candidate product we can predict which of its
constraints a given attribute would elicit.

That makes this an honest expected-information-gain calculation rather than a
hand-ordered list of questions to try:

    IG(a) = H(C) - E_r[ H(C | reply r to a) ]

over the posterior ``p`` on the live candidate set ``C``. Two consequences fall
out of the arithmetic rather than being coded in. Attributes the customer has
already exhausted score exactly zero, so they are never asked twice. And the
broadest attribute usually wins early, because it partitions the pool on
whichever constraint each candidate happens to hold instead of only on the
candidates that hold a constraint of one particular type.

Asking is free here: a turn carries recommendations *and* a question, so a
question never costs a ranking opportunity. The gate below is therefore about
whether a question can still *learn* anything, not about conversational load.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .attributes import ATTRIBUTES, ATTRIBUTE_ID, OTHER_ID
from .catalog import Catalog

# How many candidates the gain estimate looks at. The posterior is long-tailed;
# past a few hundred the extra candidates carry almost no mass and only cost time.
POSTERIOR_WIDTH = 220

# Softmax temperature turning fused scores into a posterior. Higher is flatter;
# this is deliberately not sharp, so one confident-looking score cannot collapse
# the entropy estimate and suppress a question that is still worth asking.
POSTERIOR_TEMPERATURE = 2.2

# Below this many live candidates the pool is specific enough that ranking, not
# questioning, is the bottleneck. Above it the pool is over-general.
OVER_GENERAL_POOL = 12


def posterior(scored: Sequence[Tuple[int, float]], width: int = POSTERIOR_WIDTH) -> List[Tuple[int, float]]:
    """Softmax the top scores into a normalised distribution over candidates."""
    top = list(scored[:width])
    if not top:
        return []
    best = max(score for _, score in top)
    weights = [(doc, math.exp((score - best) / POSTERIOR_TEMPERATURE)) for doc, score in top]
    total = sum(w for _, w in weights) or 1.0
    return [(doc, w / total) for doc, w in weights]


def _entropy(masses: Sequence[float]) -> float:
    total = sum(masses)
    if total <= 0.0:
        return 0.0
    out = 0.0
    for m in masses:
        if m > 0.0:
            p = m / total
            out -= p * math.log(p)
    return out


def predicted_disclosure(
    catalog: Catalog, doc: int, attribute_id: int, disclosed: frozenset
) -> Tuple[str, ...]:
    """What the customer would reveal about ``doc`` if asked about this attribute.

    Mirrors the customer policy: undisclosed constraints of the matching type,
    in card order, capped at two. ``other`` matches every type.
    """
    keys = catalog.card_keys[doc]
    attrs = catalog.card_attrs[doc]
    out: List[str] = []
    for key, attr in zip(keys, attrs):
        if key in disclosed:
            continue
        if attribute_id == OTHER_ID or attr == attribute_id:
            out.append(key)
            if len(out) == 2:
                break
    return tuple(out)


def information_gain(
    catalog: Catalog,
    post: Sequence[Tuple[int, float]],
    disclosed: frozenset,
    attribute: str,
) -> float:
    """Expected reduction in candidate entropy from asking about ``attribute``."""
    if not post:
        return 0.0
    attribute_id = ATTRIBUTE_ID[attribute]
    groups: Dict[Tuple[str, ...], List[float]] = {}
    for doc, mass in post:
        reply = predicted_disclosure(catalog, doc, attribute_id, disclosed)
        groups.setdefault(reply, []).append(mass)
    if len(groups) <= 1:
        return 0.0
    prior = _entropy([mass for _, mass in post])
    total = sum(mass for _, mass in post) or 1.0
    expected = 0.0
    for masses in groups.values():
        weight = sum(masses) / total
        expected += weight * _entropy(masses)
    return max(0.0, prior - expected)


def choose_attribute(
    catalog: Catalog,
    post: Sequence[Tuple[int, float]],
    disclosed: frozenset,
    exhausted: frozenset,
    asked: Sequence[str],
    history: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], Dict[str, float], bool]:
    """Pick the next question.

    Returns ``(attribute, gains, over_general)``. ``attribute`` is ``None`` only
    when no question could possibly learn anything.
    """
    over_general = len(post) > OVER_GENERAL_POOL
    gains: Dict[str, float] = {}
    for attribute in ATTRIBUTES:
        if attribute in exhausted:
            gains[attribute] = 0.0
            continue
        gains[attribute] = information_gain(catalog, post, disclosed, attribute)

    # Long-term memory enters only as a tie-break. Adding it to the gain would
    # corrupt a measured quantity with a prior; ordering on it after the gain
    # cannot change which question is most informative, only which of two
    # equally informative questions gets asked.
    yields = history or {}
    best = max(gains, key=lambda a: (gains[a], yields.get(a, 0.0), -asked.count(a)))
    if gains[best] > 0.0:
        return best, gains, over_general

    # Our disclosure model predicts nothing left to learn. It is a model, so it
    # can be wrong -- fall back to the most productive attribute we have not yet
    # tried before giving up on questioning entirely.
    for attribute in ("other", "feature", "material", "style", "use_case",
                      "color", "size", "brand", "budget", "category"):
        if attribute not in exhausted and attribute not in asked:
            return attribute, gains, over_general
    return None, gains, over_general
