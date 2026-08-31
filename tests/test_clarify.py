"""Question selection by expected information gain.

The properties asserted here are the ones the score depends on: a question the
customer has already refused must be worth exactly nothing, a question that
cannot separate the candidates must be worth exactly nothing, and the ranking of
questions must follow how finely each one splits the pool.
"""

from __future__ import annotations

import unittest

from tests.fixtures import subset_catalog

from src.attributes import ATTRIBUTE_ID
from src.catalog import Catalog
from src.clarify import (choose_attribute, information_gain, posterior,
                         predicted_disclosure)


class TestClarify(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _ = subset_catalog()
        cls.catalog = Catalog(path)
        biggest = max(cls.catalog.bucket.values(), key=len)
        cls.post = posterior([(doc, 0.0) for doc in biggest])

    def test_posterior_is_normalised(self) -> None:
        self.assertAlmostEqual(sum(mass for _, mass in self.post), 1.0, places=9)

    def test_other_matches_any_constraint_type(self) -> None:
        doc = self.post[0][0]
        other = predicted_disclosure(self.catalog, doc, ATTRIBUTE_ID["other"], frozenset())
        self.assertTrue(other)
        self.assertLessEqual(len(other), 2, "the customer discloses at most two at a time")

    def test_disclosed_constraints_are_not_predicted_again(self) -> None:
        doc = self.post[0][0]
        first = predicted_disclosure(self.catalog, doc, ATTRIBUTE_ID["other"], frozenset())
        second = predicted_disclosure(self.catalog, doc, ATTRIBUTE_ID["other"], frozenset(first))
        self.assertEqual(set(first) & set(second), set())

    def test_gain_is_zero_when_nothing_can_be_learned(self) -> None:
        everything = frozenset(
            key for doc, _ in self.post for key in self.catalog.card_keys[doc]
        )
        for attribute in ("other", "feature", "material", "color"):
            self.assertEqual(
                information_gain(self.catalog, self.post, everything, attribute), 0.0
            )

    def test_exhausted_attributes_score_zero(self) -> None:
        _attr, gains, _og = choose_attribute(
            self.catalog, self.post, frozenset(), frozenset({"material", "feature"}), []
        )
        self.assertEqual(gains["material"], 0.0)
        self.assertEqual(gains["feature"], 0.0)

    def test_high_yield_attributes_beat_ones_the_catalog_rarely_carries(self) -> None:
        """The gain ordering has to follow the catalog, not a hand-written list.

        ``other`` and ``feature`` are the two that pay: ``other`` because it
        matches any constraint type, ``feature`` because the descriptive strings
        it elicits are the discriminative ones. Which of the two wins depends on
        the bucket -- when a material fills both leading card slots, ``other``
        only elicits the material and ``feature`` cuts the pool far more finely.
        What must always hold is that both beat the attributes almost nothing in
        this catalog classifies as.
        """
        attribute, gains, over_general = choose_attribute(
            self.catalog, self.post, frozenset(), frozenset(), []
        )
        self.assertTrue(over_general)
        self.assertIn(attribute, {"other", "feature"})
        for barren in ("brand", "category", "budget"):
            self.assertGreater(gains["other"], gains[barren])
            self.assertGreater(gains["feature"], gains[barren])
        self.assertEqual(gains[attribute], max(gains.values()))

    def test_falls_back_to_an_untried_attribute_when_the_model_says_nothing_is_left(self) -> None:
        everything = frozenset(
            key for doc, _ in self.post for key in self.catalog.card_keys[doc]
        )
        attribute, gains, _og = choose_attribute(
            self.catalog, self.post, everything, frozenset(), []
        )
        self.assertEqual(max(gains.values()), 0.0)
        self.assertIsNotNone(attribute, "our disclosure model can be wrong; still ask")

    def test_gives_up_only_when_every_attribute_is_exhausted(self) -> None:
        every = frozenset(ATTRIBUTE_ID)
        attribute, _gains, _og = choose_attribute(
            self.catalog, self.post, frozenset(), every, list(every)
        )
        self.assertIsNone(attribute)

    def test_empty_posterior_is_safe(self) -> None:
        attribute, gains, over_general = choose_attribute(
            self.catalog, [], frozenset(), frozenset(), []
        )
        self.assertFalse(over_general)
        self.assertEqual(max(gains.values()), 0.0)
        self.assertIsNotNone(attribute)


if __name__ == "__main__":
    unittest.main()
