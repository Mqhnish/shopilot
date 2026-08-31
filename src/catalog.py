"""Parse the frozen catalog once into flat, integer-indexed arrays.

Everything downstream addresses products by a dense integer ``doc`` id rather
than by ``parent_asin``; that keeps the hot loops on list indexing instead of
dict hashing, which matters because the whole system is pure-stdlib Python.

The catalog is read-only. Nothing in this module writes to it.
"""

from __future__ import annotations

import json
import re
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .attributes import classify_id
from .normalize import (
    COLORS,
    MATERIALS,
    clean_constraint,
    content_tokens,
    dedupe,
    flatten_values,
    phrase_key,
    soft_key,
)

# Category segments that carry no discriminative signal: every product in this
# catalog is a Clothing/Shoes/Jewelry item, so these never separate anything.
_EXCLUDED_CATEGORY_PARTS = frozenset({
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
})

# Fields the evaluator searches, with the weight we give each for BM25. Title
# and features carry the buying signal; description is long and noisy.
_MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
_COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

_FIELD_WEIGHTS = (
    ("title", 3.0),
    ("categories", 2.0),
    ("features", 2.0),
    ("details", 1.5),
    ("store", 1.5),
    ("description", 1.0),
)


def coarse_category(values: Sequence[str]) -> str:
    """Reproduce ``evaluator.local_evaluator.coarse_category``.

    The evaluator builds the customer's opening line from this string, so it is
    the one piece of the target's own metadata that is disclosed on turn 1 of
    *every* scenario -- browsing sessions included.
    """
    cleaned: List[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED_CATEGORY_PARTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


# Amazon's category tree carries campaign and housekeeping nodes beside real
# product types: "Shoes & Jewelry Westlake" (1,136 products, an unclassified
# bucket), "Men's Watches Under $50", "Swimwear TEST Women's Swimwear",
# "Girls Sneakers (fs no puma)". Two markers select them -- a coarse label whose
# path never reached a product-type node, and a price, percentage, parenthetical
# or campaign word in the name. Measured on the frozen catalog: 189 of 1,115
# coarse categories holding 2,570 of 50,000 products, and not one garment type.
# (173 are root-prefixed and 108 carry a marker; the two sets overlap, and the
# union is what counts -- summing them is how an earlier draft of this comment
# reported 279. tools/check_readme.py now re-counts both figures and fails on
# drift, because prose that quotes a measurement is code that can be wrong.)
#
# This lives here, next to coarse_category, because it is a fact about the
# catalog rather than about any one consumer of it. The ranker uses it to keep
# cross-category discovery out of campaign buckets; the demo server uses the
# same call so the two can never disagree about what is shoppable.
ROOT_LABEL = coarse_category(["Clothing, Shoes & Jewelry"])

_MERCHANDISING_RE = re.compile(
    r"""\d | [$%] | \( |
        \b(?: under | up\ to | off | deal | deals | clearance | sale |
              test | cohort | mfn | pasin | edit | guide | picks? | finds? |
              most-loved | bestsellers? | best\ sellers? | new\ arrivals? |
              shop | discounts? | pricing | essentials )\b""",
    re.IGNORECASE | re.VERBOSE,
)


def is_merchandising(name: str) -> bool:
    """Whether a coarse category label is a campaign slice, not a product type."""
    return bool(
        (name != ROOT_LABEL and name.startswith(ROOT_LABEL + " "))
        or _MERCHANDISING_RE.search(name)
    )


def _searchable_text(product: dict) -> str:
    """Reproduce ``evaluator.local_evaluator.searchable_text`` (field order matters
    only for the material/colour regex, which scans left to right)."""
    parts: List[str] = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def constraint_candidates(product: dict) -> List[str]:
    """The exact pool of strings the simulator can disclose about ``product``.

    Mirrors ``evaluator.local_evaluator.intent_card``: features and details are
    flattened, a material and a colour found anywhere in the searchable text are
    pushed to the front, and a price line is appended. The customer's hard
    constraints are the first two survivors and its soft preferences the next
    two, so this list *is* the disclosure surface.
    """
    candidates = [*flatten_values(product.get("features")),
                  *flatten_values(product.get("details"))]
    corpus = _searchable_text(product)
    # The evaluator takes the *earliest* match in the corpus, not the first
    # entry of the vocabulary, so we mirror its regexes rather than scanning
    # the word lists in order.
    material = _MATERIAL_RE.search(corpus)
    color = _COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    price = product.get("price")
    if price not in (None, ""):
        candidates.append(f"budget around ${price}")
    cleaned = dedupe(c for c in (clean_constraint(v) for v in candidates) if c)
    return cleaned or [clean_constraint(str(product.get("title") or "product"))]


class Catalog:
    """The frozen catalog, parsed once and held in memory."""

    __slots__ = (
        "asins", "index_of", "titles", "coarse", "ratings", "rating_counts",
        "prices", "phrase_keys", "soft_keys", "bucket", "size",
        "vocab", "doc_terms", "doc_tfs", "doc_start", "doc_len", "df",
        "avg_doc_len", "card_keys", "card_attrs",
    )

    def __init__(self, path: str, limit: Optional[int] = None) -> None:
        source = Path(path)
        if not source.exists():
            # The catalog is a 60 MB frozen artifact that is deliberately not
            # committed. A raw FileNotFoundError here is the first thing a new
            # reader would hit, so say what to do about it instead.
            raise FileNotFoundError(
                f"catalog not found: {source}\n\n"
                "The frozen 50,000-product catalog is not committed to this "
                "repository. Fetch and verify it with:\n\n"
                "    make setup\n"
                "    (or: python3 tools/setup_data.py)\n\n"
                "See README.md > Setup."
            )
        self.asins: List[str] = []
        self.titles: List[str] = []
        self.coarse: List[str] = []
        self.ratings: List[float] = []
        self.rating_counts: List[int] = []
        self.prices: List[Optional[float]] = []
        self.phrase_keys: List[Set[str]] = []
        self.soft_keys: List[Set[str]] = []
        # The ordered constraint pool the simulator would draw on for this
        # product, and each entry's attribute class. Needed to predict what a
        # question would elicit -- see src.clarify.
        self.card_keys: List[Tuple[str, ...]] = []
        self.card_attrs: List[Tuple[int, ...]] = []

        # Term postings are kept as three flat arrays rather than one dict per
        # product: 50k Python dicts of ~150 entries cost hundreds of megabytes,
        # while the flat form costs ~50MB and scans faster.
        self.vocab: Dict[str, int] = {}
        self.doc_terms = array("i")
        self.doc_tfs = array("H")
        self.doc_start = array("i", [0])
        self.doc_len: List[int] = []

        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if limit is not None and len(self.asins) >= limit:
                    break
                self._add(json.loads(line))

        self.size = len(self.asins)
        self.index_of: Dict[str, int] = {a: i for i, a in enumerate(self.asins)}
        self.bucket: Dict[str, List[int]] = {}
        for doc, key in enumerate(self.coarse):
            self.bucket.setdefault(key.casefold(), []).append(doc)

        self.df = array("i", bytes(4 * len(self.vocab)))
        for term_id in self.doc_terms:
            self.df[term_id] += 1
        self.avg_doc_len = (sum(self.doc_len) / self.size) if self.size else 0.0

    def _add(self, product: dict) -> None:
        self.asins.append(str(product["parent_asin"]))
        title = str(product.get("title") or "")
        self.titles.append(title)
        self.coarse.append(coarse_category(product.get("categories") or []))

        # Weighted term frequencies: a token occurring in a field of weight w
        # counts w times, which folds field importance into plain BM25 without
        # needing a separate index per field.
        freqs: Dict[int, float] = {}
        vocab = self.vocab
        for field, weight in _FIELD_WEIGHTS:
            text = " ".join(flatten_values(product.get(field)))
            if not text:
                continue
            for token in content_tokens(text):
                term_id = vocab.get(token)
                if term_id is None:
                    term_id = vocab[token] = len(vocab)
                freqs[term_id] = freqs.get(term_id, 0.0) + weight
        self.doc_terms.extend(freqs)
        self.doc_tfs.extend(min(int(round(v)), 65535) or 1 for v in freqs.values())
        self.doc_start.append(len(self.doc_terms))
        self.doc_len.append(int(sum(freqs.values())))

        candidates = constraint_candidates(product)
        keys = [phrase_key(c) for c in candidates]
        self.phrase_keys.append(set(keys))
        self.soft_keys.append({k for k in (soft_key(c) for c in candidates) if k})
        # Mirrors intent_card(): the first two entries are the hard constraints
        # and the next two the soft preferences, with a short-pool fallback.
        card = tuple(keys[:2]) + (tuple(keys[2:4]) or tuple(keys[:1]))
        self.card_keys.append(card)
        self.card_attrs.append(tuple(classify_id(k) for k in card))

        rating = product.get("average_rating")
        self.ratings.append(float(rating) if isinstance(rating, (int, float)) else 0.0)
        count = product.get("rating_number")
        self.rating_counts.append(int(count) if isinstance(count, (int, float)) else 0)
        price = product.get("price")
        self.prices.append(float(price) if isinstance(price, (int, float)) else None)

    def postings(self, doc: int) -> Tuple[memoryview, memoryview]:
        """Term ids and weighted term frequencies for one product."""
        lo, hi = self.doc_start[doc], self.doc_start[doc + 1]
        return memoryview(self.doc_terms)[lo:hi], memoryview(self.doc_tfs)[lo:hi]

    def bucket_for(self, category_text: str) -> Tuple[str, List[int]]:
        """Exact coarse-category bucket for a disclosed category string."""
        key = " ".join(category_text.split()).casefold()
        return key, self.bucket.get(key, [])
