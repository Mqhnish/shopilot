"""The attribute vocabulary and the customer's own constraint classifier.

The simulator answers a question about attribute ``a`` by disclosing its
undisclosed constraints ``v`` for which ``classify_constraint(v) == a``. To
choose questions by expected information gain we therefore have to be able to
predict that classification for a *candidate* product, which means reproducing
the classifier. It is replicated here rather than imported so the submission
stays self-contained and never depends on organizer files at run time; a
differential test in ``tests/test_contract.py`` keeps the two in step.
"""

from __future__ import annotations

import re
from typing import Tuple

from .normalize import MATERIALS

# Index-order matters: these ids are stored per product in the catalog.
ATTRIBUTES: Tuple[str, ...] = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)
ATTRIBUTE_ID = {name: i for i, name in enumerate(ATTRIBUTES)}
OTHER_ID = ATTRIBUTE_ID["other"]

_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")
_COLOR_WORDS = ("color", "black", "white", "blue", "red", "pink", "green")
_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
_STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
_USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")


def classify_constraint(value: str) -> str:
    """Reproduce ``evaluator.local_evaluator.classify_constraint``."""
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(w in lowered for w in _COLOR_WORDS):
        return "color"
    if any(w in lowered for w in _SIZE_WORDS):
        return "size"
    if any(w in lowered for w in _STYLE_WORDS):
        return "style"
    if any(w in lowered for w in _USE_CASE_WORDS):
        return "use_case"
    return "feature"


def classify_id(value: str) -> int:
    return ATTRIBUTE_ID[classify_constraint(value)]
