"""Parsing every frame the simulated customer can produce.

These are the exact strings ``evaluator.local_evaluator`` emits. The awkward one
is the disclosure frame: constraints are joined with ``"; "`` but may themselves
contain ``"; "``, so the split is genuinely ambiguous and the parser has to
offer every contiguous span rather than guess.
"""

from __future__ import annotations

import unittest

from tests.fixtures import ROOT  # noqa: F401  (path setup)

from src.parse import parse_turn

KNOWN = frozenset({"jewelry necklaces", "accessories belts", "tops & tees tanks & camis",
                   "watches wrist watches", "outdoor & work snow & cold weather"})


class TestParse(unittest.TestCase):
    def test_buying_opener(self) -> None:
        obs = parse_turn(
            "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.",
            KNOWN, first_turn=True,
        )
        self.assertEqual(obs.category, "Jewelry Necklaces")
        self.assertEqual(obs.scenario_hint, "buying")
        self.assertIn("Material:alloy", obs.phrases)

    def test_browsing_opener(self) -> None:
        obs = parse_turn(
            "I'm looking for Accessories Belts, but I'm still exploring.", KNOWN, first_turn=True
        )
        self.assertEqual(obs.category, "Accessories Belts")
        self.assertEqual(obs.scenario_hint, "browsing")
        self.assertEqual(obs.phrases, [])

    def test_intent_override_opener_keeps_category_and_constraint(self) -> None:
        obs = parse_turn(
            "I'm looking for Accessories Belts. Buckle closure", KNOWN, first_turn=True
        )
        self.assertEqual(obs.category, "Accessories Belts")
        self.assertEqual(obs.scenario_hint, "intent_override")
        self.assertIn("Buckle closure", obs.phrases)

    def test_category_containing_a_period_is_not_split(self) -> None:
        """A naive split on the first '. ' would truncate the category."""
        known = frozenset({"u.s. polo assn. belts"})
        obs = parse_turn("I'm looking for U.S. Polo Assn. Belts. Buckle closure", known, first_turn=True)
        self.assertEqual(obs.category, "U.S. Polo Assn. Belts")
        self.assertIn("Buckle closure", obs.phrases)

    def test_disclosure_frame(self) -> None:
        obs = parse_turn("For that, what matters is: leather; 100% Leather.", KNOWN)
        self.assertIn("leather", obs.phrases)
        self.assertIn("100% Leather", obs.phrases)

    def test_disclosure_with_internal_semicolons_offers_every_span(self) -> None:
        text = ("For that, what matters is: cotton; "
                "Solids: 100% Cotton; Heathers: 60% Cotton, 40% Polyester.")
        obs = parse_turn(text, KNOWN)
        self.assertIn("cotton", obs.phrases)
        self.assertIn("Solids: 100% Cotton; Heathers: 60% Cotton, 40% Polyester", obs.phrases)

    def test_override_frame(self) -> None:
        obs = parse_turn(
            "Actually, ignore my earlier preference. What I need is: Water Resistant.", KNOWN
        )
        self.assertEqual(obs.override_value, "Water Resistant")
        self.assertIn("Water Resistant", obs.phrases)
        self.assertEqual(obs.scenario_hint, "intent_override")

    def test_exhausted_attribute_frame(self) -> None:
        obs = parse_turn("I don't have an additional preference for color.", KNOWN)
        self.assertEqual(obs.exhausted, "color")
        self.assertIsNone(obs.no_preference)

    def test_boundary_no_preference_frame(self) -> None:
        obs = parse_turn("I don't have a preference for size; please use your judgment.", KNOWN)
        self.assertEqual(obs.no_preference, "size")

    def test_nudge_frame(self) -> None:
        obs = parse_turn(
            "Those options are not quite right yet. Ask me about one specific attribute.", KNOWN
        )
        self.assertTrue(obs.nudge)
        self.assertEqual(obs.phrases, [])

    def test_unknown_phrasing_degrades_to_free_text(self) -> None:
        """Paraphrase must not break the turn; it should still carry signal."""
        obs = parse_turn("hey, got any warm waterproof winter boots for hiking?", KNOWN)
        self.assertTrue(obs.free_text)
        self.assertIsNone(obs.exhausted)

    def test_empty_and_none_are_safe(self) -> None:
        for value in ("", "   ", None):
            obs = parse_turn(value, KNOWN, first_turn=True)
            self.assertEqual(obs.phrases, [])


if __name__ == "__main__":
    unittest.main()
