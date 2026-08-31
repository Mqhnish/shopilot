"""Fuse the retrieval routes into one ranked list.

Each route answers a different question -- "does this product literally carry
the constraint the customer named?", "does its text overlap the conversation?",
"is it close in TF-IDF space?" -- and each is normalised to its own maximum
before fusion so that no route's raw scale decides the outcome. The weights come
from :func:`src.route.weights` and therefore differ between the buying and
browsing tracks.

The diversification step only runs on the browsing track. A browser who has
named nothing gains from spread; showing ten colourways of one shirt spends the
turn to learn almost nothing. A buyer who has named three hard constraints wants
those constraints enforced, and near-duplicates among the top ten are fine.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .catalog import Catalog, is_merchandising
from .lexical import Retriever, query_vector
from .memory import CohortMemory
from .normalize import content_tokens
from .state import SessionState

# Bayesian shrinkage for the quality prior: a 5.0 from three raters should not
# outrank a 4.6 from nine thousand.
PRIOR_MEAN = 4.0
PRIOR_COUNT = 30.0

# Bonus for sitting in the exact category the customer opened with. Large, but
# deliberately a bonus and not a filter -- see `candidate_pool`.
CATEGORY_BONUS = 0.55

# MMR trade-off on the browsing track: 1.0 is pure relevance, 0.0 pure novelty.
MMR_LAMBDA = 0.72

# A disclosed phrase may nominate candidates only if it is carried by at most
# this many products. Above it the phrase is a property, not an identifier: it
# still scores candidates, but it no longer drags thousands of them into the
# pool. The largest category in the catalog holds 1,354 products, so a phrase
# less selective than that cannot narrow anything.
PHRASE_RECALL_MAX_DF = 500

# Query weights for the different kinds of text we have heard.
W_CATEGORY = 1.6
W_PHRASE = 1.0
W_FREE_TEXT = 0.35
W_PROFILE_TAG = 0.18
# Vocabulary learned from a cohort's own past purchases. Lower than a stated
# preference tag, because it is inferred rather than declared.
W_MEMORY_TERM = 0.12

# Ceiling on the long-term prior's contribution to a fused score. A stated
# constraint scores around 1.0, so this can only ever decide a near-tie.
W_MEMORY_PRIOR = 0.05


_QUALITY_CACHE: Dict[int, List[float]] = {}


def quality_table(catalog: Catalog) -> List[float]:
    """Precomputed quality prior per document.

    Called once per candidate per turn otherwise, which added up to a third of a
    million calls on a 200-session run for a value that never changes.
    """
    cached = _QUALITY_CACHE.get(id(catalog))
    if cached is None:
        cached = [quality_prior(catalog, doc) for doc in range(catalog.size)]
        _QUALITY_CACHE[id(catalog)] = cached
    return cached


def quality_prior(catalog: Catalog, doc: int) -> float:
    """Shrunk rating in roughly [0, 1]."""
    count = catalog.rating_counts[doc]
    rating = catalog.ratings[doc]
    shrunk = (rating * count + PRIOR_MEAN * PRIOR_COUNT) / (count + PRIOR_COUNT)
    # Popularity matters a little on its own: it is weak evidence that a product
    # is the kind of thing people in this category actually buy.
    popularity = math.log1p(count) / math.log1p(20000.0)
    return 0.78 * (shrunk / 5.0) + 0.22 * min(popularity, 1.0)


def candidate_pool(
    catalog: Catalog,
    retriever: Retriever,
    state: SessionState,
    use_profile: bool = True,
    full_budget: int = 600,
    dense_recall: bool = False,
    dense_budget: int = 120,
) -> Tuple[List[int], Set[int]]:
    """Assemble the live candidate set.

    The category the customer opens with is derived from the target's own
    metadata, so its bucket contains the target by construction and is the
    natural pool. It is still a *bonus* rather than a hard filter downstream,
    for two reasons: the bucket can be large enough that ranking inside it is
    the real work, and if the organizer ever paraphrases the opening line the
    exact bucket may be missed entirely. Products found by phrase or by
    full-catalog search are therefore always allowed in alongside it.
    """
    in_bucket: Set[int] = set()
    pool: Set[int] = set()
    if state.category_key:
        bucket = catalog.bucket.get(state.category_key)
        if bucket:
            in_bucket = set(bucket)
            pool |= in_bucket
    pool |= retriever.phrase_recall_docs(state.weighted_phrases(), PHRASE_RECALL_MAX_DF)

    if len(pool) < 40 or dense_recall:
        # Thin pool: fall back to full-catalog lexical recall so a missed or
        # unusual category cannot strand the session with nothing to rank.
        #
        # ``dense_recall`` runs the same arm unconditionally on the browsing
        # track. Without it the pool on an open-ended session is *only* the
        # category bucket, so "cross-category scenario matching" is impossible
        # however the ranker is tuned -- there is nothing outside the category
        # in the pool to rank. This is the retrieval half of the brief's
        # "diverse dense retrieval track"; the ranking half is _cross_category.
        query = _build_query(catalog, state, use_profile)
        # The two arms want different budgets. The thin-pool fallback is a
        # rescue -- it may be the only source of candidates, so it takes the
        # full budget. The browsing arm only has to fill the two tail slots
        # _cross_category reserves, so a wide sweep is wasted work on the
        # hottest path in the system: full-catalog BM25 is 56% of total runtime.
        budget = full_budget if len(pool) < 40 else dense_budget
        pool |= set(retriever.bm25(query, None, limit=budget))
    return sorted(pool), in_bucket


def _build_query(
    catalog: Catalog,
    state: SessionState,
    use_profile: bool = True,
    memory: Optional[CohortMemory] = None,
) -> Dict[str, float]:
    """Weighted bag of words over everything the session has heard."""
    texts: List[str] = []
    weights: List[float] = []
    if state.category:
        texts.append(state.category)
        weights.append(W_CATEGORY)
    for text, weight in state.weighted_phrases():
        texts.append(text)
        weights.append(W_PHRASE * weight)
    for text in state.free_text:
        texts.append(text)
        weights.append(W_FREE_TEXT)
    # Safe personalisation: the aggregate profile carries preference tags such
    # as "fit" or "comfort". They are weak signals and are weighted as such --
    # enough to break ties, never enough to override a stated constraint.
    tags = state.profile.get("preference_tags") if use_profile else None
    if isinstance(tags, list):
        for tag in tags[:6]:
            if isinstance(tag, str) and tag.strip():
                texts.append(tag)
                weights.append(W_PROFILE_TAG)
    # Long-term half: vocabulary this cohort's own past purchases kept using.
    if memory is not None:
        for term, strength in memory.term_weights(state.cohort).items():
            texts.append(term)
            weights.append(W_MEMORY_TERM * strength)
    return query_vector(texts, weights)


def _normalise(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0.0:
        return {}
    return {doc: value / top for doc, value in scores.items()}


def rank(
    catalog: Catalog,
    retriever: Retriever,
    state: SessionState,
    track_weights: Dict[str, float],
    top_k: int = 10,
    use_exclusions: bool = True,
    use_diversity: bool = True,
    use_profile: bool = True,
    memory: Optional[CohortMemory] = None,
    cross_category: int = 0,
) -> Tuple[List[int], Dict[str, object]]:
    """Produce the ranked candidate list for this turn, plus a trace."""
    phrase_scores = retriever.phrase_hits(state.weighted_phrases())
    pool, in_bucket = candidate_pool(
        catalog, retriever, state, use_profile, dense_recall=bool(cross_category))
    memory_ready = memory is not None and memory.is_ready(state.cohort)

    # Negative evidence: anything already shown without ending the session is
    # provably not the target. Dropped rather than demoted -- there is no
    # uncertainty to hedge against.
    excluded = state.shown if use_exclusions else set()
    if excluded:
        asins = catalog.asins
        filtered = [doc for doc in pool if asins[doc] not in excluded]
        # Never strand the session; if exclusions empty the pool, keep it.
        if len(filtered) >= top_k:
            pool = filtered

    if not pool:
        return [], {"pool": 0, "reason": "empty-pool"}

    query = _build_query(catalog, state, use_profile, memory)
    bm25_scores = retriever.bm25(query, pool, limit=0)
    vector_scores = retriever.cosine(query, pool)

    # A generic phrase such as "Imported" can hit fourteen thousand products,
    # so restrict to the live pool once rather than per candidate.
    pool_set = set(pool)
    norm_phrase = _normalise({d: v for d, v in phrase_scores.items() if d in pool_set})
    norm_bm25 = _normalise(bm25_scores)
    norm_vector = _normalise(vector_scores)

    quality = quality_table(catalog)
    w_phrase = track_weights["phrase"]
    w_bm25 = track_weights["bm25"]
    w_vector = track_weights["vector"]
    w_quality = track_weights["quality"]
    fused: List[Tuple[int, float]] = []
    for doc in pool:
        score = (
            w_phrase * norm_phrase.get(doc, 0.0)
            + w_bm25 * norm_bm25.get(doc, 0.0)
            + w_vector * norm_vector.get(doc, 0.0)
            + w_quality * quality[doc]
        )
        if memory_ready:
            score += W_MEMORY_PRIOR * memory.quality_affinity(state.cohort, quality[doc])
        if doc in in_bucket:
            score += CATEGORY_BONUS
        fused.append((doc, score))
    fused.sort(key=lambda item: (-item[1], catalog.asins[item[0]]))

    mmr_weight = track_weights.get("mmr", 0.0)
    if use_diversity and mmr_weight > 0.0:
        ordered = _diversify(catalog, fused, top_k, mmr_weight)
    else:
        ordered = [doc for doc, _ in fused[:top_k]]

    if cross_category and in_bucket:
        ordered = _cross_category(
            catalog, fused, ordered, in_bucket, top_k, cross_category)

    trace = {
        "memory": bool(memory_ready),
        "pool": len(pool),
        "in_bucket": len(in_bucket),
        "excluded": len(excluded),
        "phrase_docs": len(phrase_scores),
        "top_score": round(fused[0][1], 4) if fused else 0.0,
        "margin": round(fused[0][1] - fused[1][1], 4) if len(fused) > 1 else 0.0,
    }
    return ordered, trace


_TITLE_TOKEN_CACHE: Dict[Tuple[int, int], Set[str]] = {}


def _title_tokens(catalog: Catalog, doc: int) -> Set[str]:
    key = (id(catalog), doc)
    tokens = _TITLE_TOKEN_CACHE.get(key)
    if tokens is None:
        tokens = set(content_tokens(catalog.titles[doc])[:24])
        _TITLE_TOKEN_CACHE[key] = tokens
    return tokens


def _cross_category(
    catalog: Catalog,
    fused: Sequence[Tuple[int, float]],
    ordered: List[int],
    in_bucket: Set[int],
    top_k: int,
    slots: int,
) -> List[int]:
    """Reserve the tail of a browsing list for the best out-of-category matches.

    The brief asks the browsing track to "unlock cross-category scenario
    matching", and measured against the shipped ranker it did nothing of the
    kind: CATEGORY_BONUS is large enough that on 40 browsing sessions the top
    ten were 100% in-category, every time. A shopper exploring "something for
    winter" wants the boots *and* the gloves; a category bonus that can never be
    outvoted cannot give them that.

    So the tail of the list -- never the head -- is opened up. The head is what
    the metric is won on and it stays exactly as the fusion ranked it; the last
    ``slots`` positions go to the highest-scoring candidates the category bonus
    was suppressing. Nothing is re-scored and nothing is filtered: this only
    changes which already-ranked candidates occupy the final slots.
    """
    if slots <= 0 or len(ordered) < top_k:
        return ordered
    chosen = set(ordered)
    # A campaign slice is not a scenario. "Shoes & Jewelry Westlake" is the
    # catalog's unclassified bucket and surfacing it as a discovery would make
    # the spread look broken rather than useful.
    outside = [doc for doc, _score in fused
               if doc not in in_bucket and doc not in chosen
               and not is_merchandising(catalog.coarse[doc])]
    if not outside:
        return ordered
    keep = ordered[:max(0, top_k - slots)]
    return keep + outside[:top_k - len(keep)]


def _diversify(
    catalog: Catalog,
    fused: Sequence[Tuple[int, float]],
    top_k: int,
    mmr_weight: float,
) -> List[int]:
    """Maximal marginal relevance over the head of the ranking."""
    pool = list(fused[: max(top_k * 6, 40)])
    if not pool:
        return []
    best = pool[0][1] or 1.0
    selected: List[int] = []
    selected_tokens: List[Set[str]] = []
    remaining = list(pool)
    while remaining and len(selected) < top_k:
        best_doc = None
        best_value = -1e9
        best_index = 0
        for index, (doc, score) in enumerate(remaining):
            relevance = score / best if best else 0.0
            tokens = _title_tokens(catalog, doc)
            penalty = 0.0
            for chosen in selected_tokens:
                union = len(tokens | chosen)
                if union:
                    penalty = max(penalty, len(tokens & chosen) / union)
            value = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * mmr_weight * penalty
            if value > best_value:
                best_value, best_doc, best_index = value, doc, index
        remaining.pop(best_index)
        selected.append(best_doc)
        selected_tokens.append(_title_tokens(catalog, best_doc))
    return selected
