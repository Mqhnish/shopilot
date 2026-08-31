"""Whole sessions, driven by the organizer's own evaluator.

Unit tests can all pass while the agent still loses sessions -- the interesting
failures live in the interaction between routing, disclosure and the turn
budget. These run the real thing: real catalog rows, real public sessions, the
unmodified evaluator.
"""

from __future__ import annotations

import unittest

from tests.fixtures import subset_catalog

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate
from src.agent import Options


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.samples = subset_catalog()
        cls.catalog_ids, cls.categories, cls.products = catalog_index(cls.path)

    def _run(self, **overrides) -> dict:
        agent = Agent(self.path, options=Options(**overrides))
        return evaluate(agent, self.samples, self.catalog_ids, self.categories, self.products)

    def test_finds_every_target_within_the_turn_budget(self) -> None:
        result = self._run()
        self.assertEqual(result["hit_rate_at_10"], 1.0)

    def test_beats_the_published_weak_baseline_by_a_wide_margin(self) -> None:
        """docs/baseline_results.json reports 0.10671 for the starter agent."""
        result = self._run()
        self.assertGreater(result["recommended_technical_score"], 0.5)

    def test_respects_the_ten_turn_limit(self) -> None:
        result = self._run()
        for session in result["sessions"]:
            if session["first_hit_turn"] is not None:
                self.assertLessEqual(session["first_hit_turn"], 10)
                self.assertGreaterEqual(session["first_hit_turn"], 1)

    def test_intent_override_sessions_never_convert_before_the_override(self) -> None:
        """The evaluator bars it, so a hit before turn 3 would mean we misread it."""
        result = self._run()
        for session in result["sessions"]:
            if session["scenario_type"] == "intent_override" and session["first_hit_turn"]:
                self.assertGreaterEqual(session["first_hit_turn"], 3)

    def test_reports_zero_tokens_because_no_model_is_called(self) -> None:
        result = self._run()
        self.assertEqual(result["reported_token_usage"]["total_tokens"], 0)

    def test_still_works_with_every_optional_component_disabled(self) -> None:
        """Graceful degradation: the fallbacks must be able to carry a session."""
        result = self._run(
            use_phrase=False, use_exclusions=False, use_routing=False,
            use_clarify=False, use_diversity=False, use_truncation=False,
        )
        self.assertGreater(result["hit_rate_at_10"], 0.0)

    def test_withholding_costs_turns_when_questions_teach_nothing(self) -> None:
        """Without a question policy there is nothing to wait for, so the agent
        must stop holding candidates back rather than trickle one per turn.

        Asserted on turns-to-conversion, which is the mechanism and is true on
        any catalog. Whether it *pays* is not: on this deliberately easy subset
        the target is almost always rank 1, so trickling one candidate per turn
        costs turns but still converts, and withholding actually scores higher.
        On the full 50,000-product public set the same behaviour drops hit rate
        from 0.735 to 0.640 and costs 0.08 of composite score, because there the
        agent runs out of turns before it walks far enough down the ranking.
        The guard exists for that regime, and hit rate carries half the score.
        """
        without = self._run(use_clarify=False)
        starved = self._run(use_clarify=False, low_gain=-1.0)
        self.assertLess(without["mttc"], starved["mttc"])

    def test_full_width_mode_still_finds_everything(self) -> None:
        """The 'always return ten' configuration trades MRR, not hit rate."""
        result = self._run(use_truncation=False)
        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertLess(result["mrr"], self._run()["mrr"])


if __name__ == "__main__":
    unittest.main()
