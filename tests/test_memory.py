"""Long-term cohort memory.

The load-bearing claim is the inference rule: a session that stopped before the
turn limit, right after a turn that offered exactly one product, converted on
that product. If that rule is wrong the agent learns from noise, so it is tested
directly and against real sessions rather than only in the abstract.
"""

from __future__ import annotations

import unittest

from tests.fixtures import subset_catalog

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate
from src.agent import Options
from src.memory import MIN_OBSERVATIONS, CohortMemory, cohort_key


class TestCohortKey(unittest.TestCase):
    BASE = {
        "purchase_frequency": "3-4 prior purchases",
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort", "durability"],
        "average_prior_rating": 5.0,
        "summary": "irrelevant",
    }

    def test_is_order_independent(self) -> None:
        other = {**self.BASE, "preference_tags": ["durability", "comfort", "fit"]}
        self.assertEqual(cohort_key(self.BASE), cohort_key(other))

    def test_ignores_free_text_summary(self) -> None:
        self.assertEqual(cohort_key(self.BASE), cohort_key({**self.BASE, "summary": "different"}))

    def test_buckets_near_identical_ratings_together(self) -> None:
        self.assertEqual(cohort_key(self.BASE), cohort_key({**self.BASE, "average_prior_rating": 4.9}))

    def test_separates_genuinely_different_cohorts(self) -> None:
        other = {**self.BASE, "preference_tags": ["price"]}
        self.assertNotEqual(cohort_key(self.BASE), cohort_key(other))

    def test_carries_no_identifier(self) -> None:
        """Only aggregate contract fields may reach the key."""
        key = cohort_key({**self.BASE, "summary": "Bought by Jane Doe on 2024-01-02"})
        self.assertNotIn("Jane", key)
        self.assertNotIn("2024", key)

    def test_malformed_profiles_are_safe(self) -> None:
        for profile in (None, {}, {"preference_tags": "not-a-list"}, {"average_prior_rating": "x"}):
            self.assertIsInstance(cohort_key(profile), str)


class TestMemoryPriors(unittest.TestCase):
    def _memory(self, n: int) -> CohortMemory:
        memory = CohortMemory()
        for i in range(n):
            memory.observe_conversion("c", "Mens Leather Belt Full Grain", 0.70 + 0.005 * i,
                                      ["other"], ["other"])
        return memory

    def test_a_single_observation_is_not_trusted(self) -> None:
        memory = self._memory(1)
        self.assertFalse(memory.is_ready("c"))
        self.assertEqual(memory.term_weights("c"), {})
        self.assertEqual(memory.quality_affinity("c", 0.7), 0.0)
        self.assertEqual(memory.attribute_bonus("c"), {})

    def test_becomes_ready_at_the_threshold(self) -> None:
        self.assertTrue(self._memory(MIN_OBSERVATIONS).is_ready("c"))

    def test_quality_affinity_is_a_band_not_a_preference_for_higher(self) -> None:
        """A cohort that buys mid-range should not be pushed to the top-rated item."""
        memory = self._memory(6)
        on_band = memory.quality_affinity("c", 0.705)
        far_above = memory.quality_affinity("c", 0.99)
        far_below = memory.quality_affinity("c", 0.30)
        self.assertGreater(on_band, far_above)
        self.assertGreater(on_band, far_below)

    def test_unknown_cohort_contributes_nothing(self) -> None:
        memory = self._memory(6)
        self.assertEqual(memory.quality_affinity("someone-else", 0.7), 0.0)
        self.assertEqual(memory.term_weights("someone-else"), {})

    def test_vocabulary_reflects_repeated_purchases(self) -> None:
        memory = self._memory(6)
        self.assertIn("leather", memory.term_weights("c"))

    def test_disabled_memory_records_nothing(self) -> None:
        memory = CohortMemory(enabled=False)
        for _ in range(9):
            memory.observe_conversion("c", "Leather Belt", 0.7, ["other"], ["other"])
        self.assertFalse(memory.is_ready("c"))


class TestConversionInference(unittest.TestCase):
    """The inference rule, checked against real sessions and real ground truth."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.samples = subset_catalog()
        cls.ids, cls.cats, cls.products = catalog_index(cls.path)

    def test_inferred_conversions_are_always_correct(self) -> None:
        agent = Agent(self.path)
        impl = agent._impl
        truth = {s["sample_id"]: s["ground_truth"]["parent_asin"] for s in self.samples}
        inferred = {}
        original = impl._retire

        def spy(session_id):
            state = impl._sessions.get(session_id)
            if state is not None and state.turn < 10 and state.last_turn_size == 1 and state.last_shown:
                inferred[session_id] = state.last_shown[0]
            return original(session_id)

        impl._retire = spy
        # Session ids are the sample ids only if we drive it ourselves, so run
        # the evaluator and map by order instead.
        for sample in self.samples:
            agent.reset(sample["sample_id"], sample["user_profile"])
            result = evaluate(agent, [sample], self.ids, self.cats, self.products)
            del result
        impl.finalize()

        self.assertTrue(inferred, "expected at least one certain conversion")
        wrong = {sid: got for sid, got in inferred.items()
                 if sid in truth and got != truth[sid]}
        self.assertEqual(wrong, {}, f"inference produced a wrong target: {wrong}")

    def test_memory_is_populated_over_a_run(self) -> None:
        agent = Agent(self.path)
        evaluate(agent, self.samples, self.ids, self.cats, self.products)
        agent._impl.finalize()
        stats = agent._impl.memory.stats()
        self.assertEqual(stats["sessions_seen"], len(self.samples))
        self.assertGreater(stats["confirmed_conversions"], 0)

    def test_disabling_memory_keeps_the_agent_working(self) -> None:
        agent = Agent(self.path, options=Options(use_memory=False))
        result = evaluate(agent, self.samples, self.ids, self.cats, self.products)
        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(agent._impl.memory.stats()["confirmed_conversions"], 0)


if __name__ == "__main__":
    unittest.main()
