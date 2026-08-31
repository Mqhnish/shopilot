"""The three retrieval routes and the fusion on top of them."""

from __future__ import annotations

import unittest

from tests.fixtures import catalog_rows, subset_catalog

from src.catalog import Catalog, coarse_category
from src.lexical import Retriever, query_vector
from src.rank import quality_prior, rank
from src.route import BROWSE, BUY, route, weights
from src.state import SessionState


class TestIndexes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.samples = subset_catalog()
        cls.catalog = Catalog(cls.path)
        cls.retriever = Retriever(cls.catalog)

    def test_every_product_is_reachable_by_its_own_constraints(self) -> None:
        """The phrase index must contain what the customer could quote."""
        for doc in range(0, self.catalog.size, 97):
            for key in self.catalog.card_keys[doc]:
                self.assertIn(key, self.retriever._phrase_exact)
                self.assertIn(doc, self.retriever._phrase_exact[key])

    def test_rare_phrase_outranks_a_generic_one(self) -> None:
        """IDF weighting, not a hand-written stoplist, is what suppresses 'Imported'."""
        common = max(self.retriever._phrase_exact.items(), key=lambda kv: len(kv[1]))
        rare = next(k for k, v in self.retriever._phrase_exact.items() if len(v) == 1)
        rare_score = max(self.retriever.phrase_hits([(rare, 1.0)]).values())
        common_score = max(self.retriever.phrase_hits([(common[0], 1.0)]).values())
        self.assertGreater(rare_score, common_score * 3)

    def test_zero_weight_phrases_are_ignored(self) -> None:
        rare = next(k for k, v in self.retriever._phrase_exact.items() if len(v) == 1)
        self.assertEqual(self.retriever.phrase_hits([(rare, 0.0)]), {})

    def test_bucket_lookup_is_case_insensitive(self) -> None:
        name = self.catalog.coarse[0]
        _key, docs = self.catalog.bucket_for(name.upper())
        self.assertIn(0, docs)

    def test_bucket_membership_matches_coarse_category(self) -> None:
        rows = {str(r["parent_asin"]): r for r in catalog_rows()}
        for doc in range(0, self.catalog.size, 211):
            product = rows[self.catalog.asins[doc]]
            key, docs = self.catalog.bucket_for(coarse_category(product.get("categories") or []))
            self.assertIn(doc, docs)
            self.assertEqual(key, self.catalog.coarse[doc].casefold())

    def test_bm25_restricted_to_candidates_stays_inside_them(self) -> None:
        _key, docs = self.catalog.bucket_for(self.catalog.coarse[0])
        scores = self.retriever.bm25(query_vector(["cotton shirt"]), docs)
        self.assertTrue(set(scores).issubset(set(docs)))

    def test_cosine_is_bounded(self) -> None:
        docs = list(range(min(300, self.catalog.size)))
        for value in self.retriever.cosine(query_vector(["black leather belt"]), docs).values():
            self.assertGreaterEqual(value, -1e-9)
            self.assertLessEqual(value, 1.0 + 1e-9)

    def test_unknown_query_terms_are_harmless(self) -> None:
        self.assertEqual(self.retriever.bm25(query_vector(["zzzqqqxyzzy"]), None), {})
        self.assertEqual(self.retriever.cosine(query_vector(["zzzqqqxyzzy"]), [0, 1, 2]), {})

    def test_quality_prior_prefers_well_reviewed_products(self) -> None:
        best = max(range(self.catalog.size), key=lambda d: quality_prior(self.catalog, d))
        self.assertGreaterEqual(self.catalog.ratings[best], 4.0)
        for doc in range(self.catalog.size):
            self.assertGreaterEqual(quality_prior(self.catalog, doc), 0.0)
            self.assertLessEqual(quality_prior(self.catalog, doc), 1.05)


class TestRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.samples = subset_catalog()
        cls.catalog = Catalog(cls.path)
        cls.retriever = Retriever(cls.catalog)

    def _state(self, doc: int) -> SessionState:
        state = SessionState("s", {})
        state.category = self.catalog.coarse[doc]
        state.category_key, _docs = self.catalog.bucket_for(state.category)
        return state

    def test_disclosed_constraints_pull_the_right_product_to_the_top(self) -> None:
        found = 0
        for doc in range(0, self.catalog.size, 401):
            state = self._state(doc)
            for key in self.catalog.card_keys[doc]:
                state.add_phrase(key)
            docs, _trace = rank(self.catalog, self.retriever, state, dict(weights(BUY)), top_k=10)
            if docs and docs[0] == doc:
                found += 1
        self.assertGreater(found, 0)

    def test_excluded_products_are_dropped(self) -> None:
        doc = 0
        state = self._state(doc)
        docs, _trace = rank(self.catalog, self.retriever, state, dict(weights(BROWSE)), top_k=10)
        state.shown.update(self.catalog.asins[d] for d in docs)
        again, _trace = rank(self.catalog, self.retriever, state, dict(weights(BROWSE)), top_k=10)
        self.assertEqual(set(docs) & set(again), set())

    def test_exclusions_never_empty_the_list(self) -> None:
        state = self._state(0)
        state.shown.update(self.catalog.asins)
        docs, _trace = rank(self.catalog, self.retriever, state, dict(weights(BROWSE)), top_k=10)
        self.assertTrue(docs, "excluding everything must not strand the session")

    def test_routing_moves_from_browse_to_buy_as_evidence_arrives(self) -> None:
        state = SessionState("s", {})
        state.scenario = "browsing"
        self.assertEqual(route(state)[0], BROWSE)
        state.scenario = "buying"
        state.category = "Accessories Belts"
        for key in ("leather", "100% Leather", "Buckle closure"):
            state.add_phrase(key)
        self.assertEqual(route(state)[0], BUY)

    def test_browse_track_diversifies_more_than_buy(self) -> None:
        self.assertGreater(weights(BROWSE)["mmr"], weights(BUY)["mmr"])
        self.assertGreater(weights(BROWSE)["vector"], weights(BUY)["vector"])


if __name__ == "__main__":
    unittest.main()
