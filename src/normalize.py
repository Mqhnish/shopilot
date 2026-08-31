"""Text canonicalisation shared by every index and every parser.

The single most important property here is that :func:`phrase_key` reproduces
the *exact* normalisation the organizer's evaluator applies when it builds a
customer's intent card. The evaluator lifts constraint strings verbatim out of
a product's ``features`` and ``details`` via ``_flatten_values`` and
``_clean_constraint``; if we canonicalise a disclosed phrase the same way we
canonicalised the catalog, a disclosed phrase becomes a literal key lookup
rather than a fuzzy similarity problem.

We deliberately do not depend on that being true -- :mod:`src.lexical` and
:mod:`src.vector` recover the target from paraphrased text too -- but when the
text *is* verbatim, this is what turns a 50,000-way problem into a 1-way one.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

# Mirrors evaluator.local_evaluator._clean_constraint / intent_card(limit=180).
CONSTRAINT_LIMIT = 180
_WHITESPACE_RE = re.compile(r"\s+")
_STRIP_CHARS = " -;,.\t\n"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?", re.IGNORECASE)

# Kept small on purpose. Aggressive stopword lists delete exactly the words that
# make a constraint phrase discriminative ("for", "with" in "Pull On closure").
STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have i im in is it its me my
of on or please so some that the their then there these this to was were will
with would you your looking want need am
""".split())

# Attribute vocabulary fixed by docs/agent_api_contract.json.
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

# Same vocabularies the evaluator's regexes use, so our attribute reasoning
# lines up with the simulator's ``classify_constraint``.
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool",
             "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green",
          "brown", "gray", "grey", "purple", "yellow", "orange")


def clean_constraint(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    """Reproduce ``evaluator.local_evaluator._clean_constraint`` exactly."""
    return _WHITESPACE_RE.sub(" ", value).strip(_STRIP_CHARS)[:limit].rstrip()


def phrase_key(value: str) -> str:
    """Canonical key for exact-phrase matching.

    Case- and accent-insensitive on top of the evaluator's own cleaning, so a
    disclosed phrase still matches when only capitalisation differs.
    """
    cleaned = clean_constraint(value)
    folded = unicodedata.normalize("NFKD", cleaned)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return folded.casefold()


def soft_key(value: str) -> str:
    """Punctuation-insensitive key, for near-exact matching.

    ``"100% Polyester"`` and ``"100 polyester"`` collapse together here but not
    under :func:`phrase_key`. Used as the second of the three phrase tiers.
    """
    return " ".join(tokenize(value))


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, accents folded, stopwords kept."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [m.group(0).casefold() for m in TOKEN_RE.finditer(folded)]


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords and bare single characters removed."""
    return [t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS]


def flatten_values(value: object) -> List[str]:
    """Reproduce ``evaluator.local_evaluator._flatten_values`` exactly."""
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items() if v not in (None, "", [])]
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def dedupe(items: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication (``dict.fromkeys`` semantics)."""
    return list(dict.fromkeys(items))
