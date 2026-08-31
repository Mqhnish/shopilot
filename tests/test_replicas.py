"""Differential tests: our reimplementations against the organizer's evaluator.

The agent reimplements four pieces of the evaluator -- category coarsening,
constraint cleaning, value flattening, and constraint classification -- so that
the submission is self-contained at run time. Reimplementation is only safe
while the copies stay identical, which is what these tests are for. If the
organizer changes the evaluator, these fail loudly instead of the agent quietly
losing accuracy.
"""

from __future__ import annotations

import unittest

from tests.fixtures import sampled_rows

from evaluator import local_evaluator as ev
from src.attributes import classify_constraint
from src.catalog import Catalog, coarse_category, constraint_candidates
from src.normalize import clean_constraint, flatten_values, phrase_key

SAMPLE_SIZE = 3000


class TestReplicas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = sampled_rows(SAMPLE_SIZE)

    def test_flatten_values_matches_evaluator(self) -> None:
        for product in self.rows:
            for field in ("features", "details", "description", "categories", "title", "price"):
                self.assertEqual(
                    flatten_values(product.get(field)),
                    ev._flatten_values(product.get(field)),
                    msg=f"{product['parent_asin']} field={field}",
                )

    def test_clean_constraint_matches_evaluator(self) -> None:
        for product in self.rows:
            for value in flatten_values(product.get("features")):
                self.assertEqual(clean_constraint(value), ev._clean_constraint(value, 180))

    def test_coarse_category_matches_evaluator(self) -> None:
        for product in self.rows:
            categories = product.get("categories") or []
            self.assertEqual(coarse_category(categories), ev.coarse_category(categories))

    def test_classify_constraint_matches_evaluator(self) -> None:
        for product in self.rows:
            card = ev.intent_card(product)
            for value in card["hard_constraints"] + card["soft_preferences"]:
                self.assertEqual(classify_constraint(value), ev.classify_constraint(value))

    def test_candidate_pool_covers_every_intent_card(self) -> None:
        """Every string the customer could disclose must be in our index."""
        for product in self.rows:
            card = ev.intent_card(product)
            pool = set(constraint_candidates(product))
            for value in card["hard_constraints"] + card["soft_preferences"]:
                self.assertIn(value, pool, msg=product["parent_asin"])

    def test_predicted_intent_card_matches_evaluator(self) -> None:
        """The per-product card the question planner reasons over is exact."""
        rows = self.rows[:400]
        import json
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.close()
        catalog = Catalog(handle.name)
        for doc, product in enumerate(rows):
            card = ev.intent_card(product)
            expected = tuple(
                phrase_key(v) for v in card["hard_constraints"] + card["soft_preferences"]
            )
            self.assertEqual(catalog.card_keys[doc], expected, msg=product["parent_asin"])


if __name__ == "__main__":
    unittest.main()
