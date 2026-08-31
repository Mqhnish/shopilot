"""Session state: negative evidence, intent override, attribute bookkeeping."""

from __future__ import annotations

import unittest

from tests.fixtures import ROOT  # noqa: F401  (path setup)

from src.state import SessionState


class TestNegativeEvidence(unittest.TestCase):
    def test_shown_items_are_recorded(self) -> None:
        state = SessionState("s", {})
        state.record_shown(["A", "B"])
        self.assertEqual(state.shown, {"A", "B"})

    def test_override_sessions_withhold_exclusions_until_override_lands(self) -> None:
        """Those sessions cannot convert early, so 'shown and missed' proves nothing."""
        state = SessionState("s", {})
        state.scenario = "intent_override"
        self.assertFalse(state.exclusions_active())
        state.record_shown(["A"])
        self.assertEqual(state.shown, set())
        state.override_seen = True
        self.assertTrue(state.exclusions_active())
        state.record_shown(["B"])
        self.assertEqual(state.shown, {"B"})

    def test_other_scenarios_exclude_immediately(self) -> None:
        for scenario in ("buying", "browsing", "boundary"):
            state = SessionState("s", {})
            state.scenario = scenario
            state.record_shown(["A"])
            self.assertEqual(state.shown, {"A"}, msg=scenario)


class TestSlots(unittest.TestCase):
    def test_phrases_accumulate_and_keep_strongest_weight(self) -> None:
        state = SessionState("s", {})
        state.add_phrase("leather", 1.0)
        state.add_phrase("leather", 1.25)
        state.add_phrase("Buckle closure", 1.0)
        self.assertEqual(dict(state.weighted_phrases())["leather"], 1.25)
        self.assertEqual(len(state.phrase_order), 2)

    def test_learned_flag_only_fires_on_genuinely_new_evidence(self) -> None:
        state = SessionState("s", {})
        state.add_phrase("leather")
        self.assertTrue(state.learned_this_turn)
        state.learned_this_turn = False
        state.add_phrase("LEATHER")  # same constraint, different case
        self.assertFalse(state.learned_this_turn)

    def test_demotion_scales_without_erasing(self) -> None:
        state = SessionState("s", {})
        state.add_phrase("Buckle closure", 1.0)
        state.demote_existing(0.45)
        self.assertAlmostEqual(dict(state.weighted_phrases())["Buckle closure"], 0.45)

    def test_erasure_clears_everything(self) -> None:
        state = SessionState("s", {})
        state.add_phrase("Buckle closure")
        state.clear_slots()
        self.assertEqual(state.weighted_phrases(), [])
        self.assertEqual(state.disclosed_keys, set())

    def test_exhausted_attributes_are_tracked(self) -> None:
        state = SessionState("s", {})
        state.mark_exhausted("color")
        state.mark_no_preference("size")
        self.assertIn("color", state.exhausted)
        self.assertIn("size", state.exhausted)
        self.assertIn("size", state.no_preference)

    def test_empty_phrase_is_ignored(self) -> None:
        state = SessionState("s", {})
        state.add_phrase("   ")
        self.assertEqual(state.phrase_order, [])


if __name__ == "__main__":
    unittest.main()
