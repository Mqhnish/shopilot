"""Multi-route retrieval over the frozen catalog.

Three independent routes, fused later in :mod:`src.rank`:

``phrase``
    Exact and punctuation-insensitive lookup of a disclosed constraint string
    against the pool of strings the simulator could have drawn it from. When the
    customer's wording is verbatim -- which it is whenever the organizer has not
    paraphrased -- a single rare phrase identifies the product outright.

``bm25``
    Okapi BM25 over a weighted bag of words covering title, categories,
    features, details, store and description. This is the route that survives
    paraphrase, because it needs word overlap rather than string identity.

``vector``
    Cosine similarity in TF-IDF space. A sparse vector-space model rather than
    neural embeddings: the organizer may run the final scoring with no network,
    and shipping model weights is both heavy and disallowed-adjacent. Cosine
    normalises away document length, so it ranks short, precise titles very
    differently from BM25 and genuinely adds to the fusion.
"""

from __future__ import annotations

import math
from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .catalog import Catalog
from .normalize import content_tokens, phrase_key, soft_key

BM25_K1 = 1.4
BM25_B = 0.72


class Retriever:
    """Owns the catalog and every index built over it."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._idf = self._build_idf()
        self._doc_norm = self._build_doc_norms()
        self._phrase_exact = self._build_phrase_index(catalog.phrase_keys)
        self._phrase_soft = self._build_phrase_index(catalog.soft_keys)
        self._inv_docs, self._inv_tfs, self._inv_start = self._build_inverted()

    # ---------------------------------------------------------------- indexes

    def _build_idf(self) -> List[float]:
        n = float(self.catalog.size)
        return [
            math.log(1.0 + (n - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
            for df in self.catalog.df
        ]

    def _build_doc_norms(self) -> array:
        """L2 norm of every document's TF-IDF vector, for cosine similarity."""
        cat, idf = self.catalog, self._idf
        norms = array("d", bytes(8 * cat.size))
        terms, tfs, start = cat.doc_terms, cat.doc_tfs, cat.doc_start
        for doc in range(cat.size):
            total = 0.0
            for i in range(start[doc], start[doc + 1]):
                w = (1.0 + math.log(tfs[i])) * idf[terms[i]]
                total += w * w
            norms[doc] = math.sqrt(total) or 1.0
        return norms

    @staticmethod
    def _build_phrase_index(per_doc: Sequence[Iterable[str]]) -> Dict[str, List[int]]:
        index: Dict[str, List[int]] = {}
        for doc, keys in enumerate(per_doc):
            for key in keys:
                index.setdefault(key, []).append(doc)
        return index

    def _build_inverted(self) -> Tuple[array, array, array]:
        """Transpose the flat per-document postings into per-term postings.

        A counting sort over term ids; no comparison sort and no intermediate
        Python objects, which keeps this a couple of seconds rather than a
        couple of minutes.
        """
        cat = self.catalog
        n_terms = len(cat.vocab)
        start = array("i", bytes(4 * (n_terms + 1)))
        for term_id in cat.doc_terms:
            start[term_id + 1] += 1
        for i in range(1, n_terms + 1):
            start[i] += start[i - 1]

        cursor = array("i", start[:n_terms])
        total = len(cat.doc_terms)
        inv_docs = array("i", bytes(4 * total))
        inv_tfs = array("H", bytes(2 * total))
        doc_start, terms, tfs = cat.doc_start, cat.doc_terms, cat.doc_tfs
        for doc in range(cat.size):
            for i in range(doc_start[doc], doc_start[doc + 1]):
                term_id = terms[i]
                pos = cursor[term_id]
                inv_docs[pos] = doc
                inv_tfs[pos] = tfs[i]
                cursor[term_id] = pos + 1
        return inv_docs, inv_tfs, start

    # ----------------------------------------------------------------- routes

    def phrase_hits(self, phrases: Iterable[Tuple[str, float]]) -> Dict[int, float]:
        """Score documents by the rarity of the disclosed phrases they carry.

        A phrase held by one product in fifty thousand is overwhelming evidence;
        one held by fourteen thousand ("Imported") is nearly none. Weighting by
        inverse document frequency separates the two automatically, with no
        hand-maintained list of generic phrases.

        Matching runs in two tiers. The exact tier keys on the evaluator's own
        cleaning, so verbatim wording is a dictionary lookup. The soft tier
        strips punctuation, which recovers "100% Polyester" from "100
        polyester". The exact tier wins outright when it fires -- counting both
        would double-weight the same piece of evidence.
        """
        scores: Dict[int, float] = {}
        n = float(self.catalog.size)
        seen_keys: set = set()
        for phrase, weight in phrases:
            if weight <= 0.0:
                continue
            exact_key = phrase_key(phrase)
            docs = self._phrase_exact.get(exact_key)
            tier = 1.0
            key = exact_key
            if not docs:
                key = soft_key(phrase)
                docs = self._phrase_soft.get(key)
                tier = 0.65
            if not docs or key in seen_keys:
                continue
            seen_keys.add(key)
            gain = weight * tier * math.log(1.0 + n / len(docs))
            for doc in docs:
                scores[doc] = scores.get(doc, 0.0) + gain
        return scores

    def match_spans(
        self, text: str, max_words: int = 32, max_spans: int = 4000
    ) -> List[str]:
        """Find every constraint string the catalog knows that occurs in ``text``.

        Frame-independent by construction. Rather than recognising *how* the
        customer phrased a disclosure, this enumerates the contiguous word spans
        of the message and keeps the ones that are literally constraint strings
        in the catalog. A shopper can say "must-have: Buckle closure", "what
        matters there is Buckle closure", or "it needs a Buckle closure" and all
        three yield the same span.

        Over-generation is free: a span is kept only if it is a real constraint
        string carried by some product, so a wrong split matches nothing. This is
        what makes the agent survive rewording of the customer's frames, which
        measured as the single largest robustness risk (see tools/paraphrase.py).
        """
        words = text.split()
        if not words:
            return []
        found: List[str] = []
        seen: set = set()
        budget = max_spans
        exact = self._phrase_exact
        for start in range(len(words)):
            for end in range(start + 1, min(start + max_words, len(words)) + 1):
                budget -= 1
                if budget < 0:
                    return found
                span = " ".join(words[start:end])
                key = phrase_key(span)
                if key and key not in seen and key in exact:
                    seen.add(key)
                    found.append(span)
        return found

    def phrase_df(self, phrase: str) -> int:
        """How many products carry this constraint string, 0 if none do.

        The rarity that :meth:`phrase_hits` already weights by, exposed so a
        caller can report *why* a product ranked where it did without
        re-implementing the two-tier lookup and drifting from it. Read-only and
        off the hot path.
        """
        docs = self._phrase_exact.get(phrase_key(phrase))
        if docs is None:
            docs = self._phrase_soft.get(soft_key(phrase))
        return len(docs) if docs else 0

    def phrase_recall_docs(self, phrases: Iterable[Tuple[str, float]], max_df: int) -> set:
        """Documents worth *retrieving* on the strength of a disclosed phrase.

        Rarity decides. "Buckle closure" is carried by thousands of products, so
        it localises nothing -- pulling all of them into the candidate pool
        replaces a 258-product category with an 18,000-product haystack and
        makes the ranking worse, not better. Such a phrase still contributes to
        the *score* of candidates found by other routes; it just no longer gets
        to nominate them.
        """
        docs: set = set()
        for phrase, weight in phrases:
            if weight <= 0.0:
                continue
            hits = self._phrase_exact.get(phrase_key(phrase))
            if hits is None:
                hits = self._phrase_soft.get(soft_key(phrase))
            if hits is not None and len(hits) <= max_df:
                docs.update(hits)
        return docs

    def bm25(
        self,
        query: Dict[str, float],
        candidates: Optional[Sequence[int]] = None,
        limit: int = 400,
    ) -> Dict[int, float]:
        """Okapi BM25. Restricted to ``candidates`` when a category is known."""
        cat = self.catalog
        avg = cat.avg_doc_len or 1.0
        allowed = set(candidates) if candidates is not None else None
        scores: Dict[int, float] = {}
        for token, qw in query.items():
            term_id = cat.vocab.get(token)
            if term_id is None:
                continue
            idf = self._idf[term_id]
            if idf <= 0.0:
                continue
            lo, hi = self._inv_start[term_id], self._inv_start[term_id + 1]
            if allowed is not None and (hi - lo) > 8 * (len(allowed) + 1):
                # Rare-candidate case: probing each candidate beats scanning a
                # long postings list.
                continue
            for i in range(lo, hi):
                doc = self._inv_docs[i]
                if allowed is not None and doc not in allowed:
                    continue
                tf = self._inv_tfs[i]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * cat.doc_len[doc] / avg)
                scores[doc] = scores.get(doc, 0.0) + qw * idf * tf * (BM25_K1 + 1.0) / denom
        if allowed is not None:
            scores.update(self._bm25_probe(query, allowed, scores))
        if limit and len(scores) > limit:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
            return dict(top)
        return scores

    def _bm25_probe(
        self, query: Dict[str, float], allowed: set, already: Dict[int, float]
    ) -> Dict[int, float]:
        """Score candidates directly for query terms whose postings list is long.

        Walking 14,000 postings to find which of 180 candidates contain
        "polyester" is wasteful; walking the 180 candidates' own postings is not.
        """
        cat = self.catalog
        avg = cat.avg_doc_len or 1.0
        wanted: Dict[int, float] = {}
        for token, qw in query.items():
            term_id = cat.vocab.get(token)
            if term_id is None:
                continue
            lo, hi = self._inv_start[term_id], self._inv_start[term_id + 1]
            if (hi - lo) > 8 * (len(allowed) + 1) and self._idf[term_id] > 0.0:
                wanted[term_id] = qw
        if not wanted:
            return {}
        out: Dict[int, float] = {}
        terms, tfs, start = cat.doc_terms, cat.doc_tfs, cat.doc_start
        for doc in allowed:
            total = already.get(doc, 0.0)
            length = cat.doc_len[doc]
            for i in range(start[doc], start[doc + 1]):
                qw = wanted.get(terms[i])
                if qw is None:
                    continue
                tf = tfs[i]
                denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg)
                total += qw * self._idf[terms[i]] * tf * (BM25_K1 + 1.0) / denom
            if total:
                out[doc] = total
        return out

    def cosine(self, query: Dict[str, float], candidates: Sequence[int]) -> Dict[int, float]:
        """TF-IDF cosine similarity against an explicit candidate set."""
        cat, idf = self.catalog, self._idf
        qvec: Dict[int, float] = {}
        for token, qw in query.items():
            term_id = cat.vocab.get(token)
            if term_id is not None and idf[term_id] > 0.0:
                qvec[term_id] = qvec.get(term_id, 0.0) + qw * idf[term_id]
        qnorm = math.sqrt(sum(v * v for v in qvec.values()))
        if not qvec or qnorm == 0.0:
            return {}
        terms, tfs, start = cat.doc_terms, cat.doc_tfs, cat.doc_start
        out: Dict[int, float] = {}
        for doc in candidates:
            dot = 0.0
            for i in range(start[doc], start[doc + 1]):
                qw = qvec.get(terms[i])
                if qw is not None:
                    dot += qw * (1.0 + math.log(tfs[i])) * idf[terms[i]]
            if dot:
                out[doc] = dot / (qnorm * self._doc_norm[doc])
        return out


def query_vector(texts: Iterable[str], weights: Optional[Iterable[float]] = None) -> Dict[str, float]:
    """Turn weighted pieces of dialog into a bag-of-words query vector."""
    weights = list(weights) if weights is not None else None
    query: Dict[str, float] = {}
    for i, text in enumerate(texts):
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        if w <= 0.0:
            continue
        for token in content_tokens(text):
            query[token] = query.get(token, 0.0) + w
    return query
