"""Frame-free parsing, and the paraphrase harness that justified building it.

The competition specification reserves the right to reword the customer's
language on the private split. Measured, that was the single largest risk to
this agent: rewording only the *frames*, while leaving product attributes
verbatim, cost more than half the score. These tests cover the two mechanisms
that fixed it and the harness that found it.
"""

from __future__ import annotations

import random
import unittest

from tests.fixtures import subset_catalog

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate
from src.agent import Options
from src.catalog import Catalog
from src.lexical import Retriever
from src.parse import find_category, parse_turn
from tools.paraphrase import LEVELS, ParaphrasingAgent, paraphrase


class TestFrameFreeCategory(unittest.TestCase):
    KNOWN = frozenset({"jewelry necklaces", "accessories belts", "watches wrist watches",
                       "tops & tees tanks & camis"})

    def test_finds_the_category_regardless_of_surrounding_words(self) -> None:
        for text in (
            "I'm looking for Jewelry Necklaces. A key requirement is: alloy.",
            "Shopping for Jewelry Necklaces. Must-have: alloy.",
            "Can you help me find Jewelry Necklaces?",
            "any Jewelry Necklaces going?",
            "I want to buy Jewelry Necklaces, ideally something simple",
        ):
            self.assertEqual(find_category(text, self.KNOWN), "Jewelry Necklaces", msg=text)

    def test_prefers_the_longest_match(self) -> None:
        known = frozenset({"belts", "accessories belts"})
        self.assertEqual(find_category("I need Accessories Belts", known), "Accessories Belts")

    def test_returns_none_when_no_category_is_present(self) -> None:
        self.assertIsNone(find_category("What matters there is leather and 100% Leather.",
                                        self.KNOWN))
        self.assertIsNone(find_category("", self.KNOWN))
        self.assertIsNone(find_category("anything", None))


class TestSpanMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _ = subset_catalog()
        cls.catalog = Catalog(path)
        cls.retriever = Retriever(cls.catalog)
        cls.doc = next(d for d in range(cls.catalog.size)
                       if any(len(k.split()) >= 3 for k in cls.catalog.card_keys[d]))
        cls.phrase = next(k for k in cls.catalog.card_keys[cls.doc] if len(k.split()) >= 3)

    def test_recovers_a_constraint_from_any_sentence_shape(self) -> None:
        for template in ("For that, what matters is: {p}.",
                         "What matters there is {p}.",
                         "Must-have: {p}",
                         "honestly it just needs {p} and that's it",
                         "{p}"):
            found = self.retriever.match_spans(template.format(p=self.phrase))
            keys = {s.strip(" .,;:").casefold() for s in found}
            self.assertIn(self.phrase, keys, msg=template)

    def test_returns_nothing_for_text_with_no_catalog_constraint(self) -> None:
        self.assertEqual(self.retriever.match_spans("zzqq wibble frobnicate"), [])

    def test_empty_input_is_safe(self) -> None:
        self.assertEqual(self.retriever.match_spans(""), [])
        self.assertEqual(self.retriever.match_spans("   "), [])

    def test_respects_its_budget(self) -> None:
        """Long input must not turn into a quadratic blow-up."""
        found = self.retriever.match_spans(" ".join(["leather"] * 400), max_spans=50)
        self.assertLessEqual(len(found), 50)


class TestFrameGating(unittest.TestCase):
    KNOWN = frozenset({"accessories belts"})

    def test_every_known_frame_is_recognised(self) -> None:
        cases = {
            "I'm looking for Accessories Belts, but I'm still exploring.": "opener",
            "For that, what matters is: leather.": "matters",
            "Actually, ignore my earlier preference. What I need is: leather.": "override",
            "I don't have an additional preference for color.": "exhausted",
            "I don't have a preference for size; please use your judgment.": "no_preference",
            "Those options are not quite right yet. Ask me about one specific attribute.": "nudge",
        }
        for text, frame in cases.items():
            self.assertEqual(parse_turn(text, self.KNOWN).matched_frame, frame, msg=text)

    def test_reworded_text_is_not_claimed_as_a_frame(self) -> None:
        """This is what lets the fallback scan run without fighting the parser."""
        for text in ("Shopping for Accessories Belts. Must-have: leather.",
                     "What matters there is leather.",
                     "Scratch that. What I actually want is leather."):
            self.assertIsNone(parse_turn(text, self.KNOWN).matched_frame, msg=text)

    def test_a_recognised_browsing_opener_reports_no_constraints(self) -> None:
        """Trusting that empty list is what keeps the clean score at 0.95398."""
        obs = parse_turn("I'm looking for Accessories Belts, but I'm still exploring.",
                         self.KNOWN, first_turn=True)
        self.assertEqual(obs.matched_frame, "opener")
        self.assertEqual(obs.phrases, [])


class TestParaphraseHarness(unittest.TestCase):
    MESSAGE = "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy."

    def test_none_is_the_identity(self) -> None:
        self.assertEqual(paraphrase(self.MESSAGE, "none", random.Random(0)), self.MESSAGE)

    def test_every_level_rewrites_the_frame(self) -> None:
        for level in ("light", "medium", "heavy"):
            out = paraphrase(self.MESSAGE, level, random.Random(3))
            self.assertNotEqual(out, self.MESSAGE, msg=level)
            self.assertNotIn("I'm looking for", out, msg=level)

    def test_light_keeps_the_product_attribute_verbatim(self) -> None:
        """The point of the light level: only the frame moves."""
        out = paraphrase(self.MESSAGE, "light", random.Random(3))
        self.assertIn("Material:alloy", out)

    def test_light_keeps_the_category_verbatim(self) -> None:
        out = paraphrase(self.MESSAGE, "light", random.Random(3))
        self.assertIn("Jewelry Necklaces", out)

    def test_is_deterministic_given_a_seed(self) -> None:
        a = paraphrase(self.MESSAGE, "heavy", random.Random(11))
        b = paraphrase(self.MESSAGE, "heavy", random.Random(11))
        self.assertEqual(a, b)

    def test_rejects_an_unknown_level(self) -> None:
        with self.assertRaises(ValueError):
            ParaphrasingAgent(object(), level="nonsense")
        self.assertIn("heavy", LEVELS)


class TestTheHarnessIsAnExperiment(unittest.TestCase):
    """Two properties without which the robustness table means nothing.

    Both were false. The wrapper reseeded per session from ``session_id``, and
    the organizer's evaluator names every session ``f"public_{uuid.uuid4().hex}"``
    — freshly, every run. So ``--seed`` did nothing, two runs of the same code
    disagreed by up to 0.017 of composite score, and the hardened and unhardened
    arms were each handed *different customer text*, which is precisely the
    variable the comparison exists to hold fixed.
    """

    class Recorder:
        """Stands in for an Agent and keeps what it was told."""

        def __init__(self) -> None:
            self.heard: list = []

        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
            self.heard.append(user_message)
            return {"message": "", "ask_attribute": "other", "recommendations": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    MESSAGES = [
        "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.",
        "For that, what matters is: Buckle closure; Imported.",
        "I don't have an additional preference for material.",
    ]

    def drive(self, session_ids: list) -> list:
        """Run three sessions through a fresh wrapper under given session ids."""
        inner = self.Recorder()
        wrapper = ParaphrasingAgent(inner, level="heavy", seed=20260830)
        for session_id in session_ids:
            wrapper.reset(session_id, {})
            for turn, message in enumerate(self.MESSAGES, start=1):
                wrapper.respond(session_id, message, turn, 10)
        return inner.heard

    def test_the_wording_does_not_depend_on_the_session_id(self) -> None:
        """The evaluator's ids are uuid4 and differ every run, so anything keyed
        on them is not seeded — it is random."""
        first = self.drive(["public_a", "public_b", "public_c"])
        second = self.drive(["wholly", "different", "names"])
        self.assertEqual(first, second)

    def test_two_runs_of_the_same_arm_are_identical(self) -> None:
        ids = ["public_%s" % i for i in range(3)]
        self.assertEqual(self.drive(ids), self.drive(ids))

    def test_both_arms_are_handed_the_same_words(self) -> None:
        """A stress test that varies the text between arms is not measuring the
        agent, it is measuring the rewriter."""
        hardened = self.drive(["s1", "s2", "s3"])
        unhardened = self.drive(["t1", "t2", "t3"])
        self.assertEqual(hardened, unhardened)

    def test_the_seed_actually_selects_the_experiment(self) -> None:
        """The flip side: a different seed has to give different text, or the
        knob is decorative in the other direction."""
        inner_a, inner_b = self.Recorder(), self.Recorder()
        for inner, seed in ((inner_a, 1), (inner_b, 2)):
            wrapper = ParaphrasingAgent(inner, level="heavy", seed=seed)
            wrapper.reset("s", {})
            for turn, message in enumerate(self.MESSAGES, start=1):
                wrapper.respond("s", message, turn, 10)
        self.assertNotEqual(inner_a.heard, inner_b.heard)


class TestHardeningPaysOff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.samples = subset_catalog()
        cls.ids, cls.cats, cls.products = catalog_index(cls.path)

    def _score(self, level: str, **overrides) -> float:
        agent = Agent(self.path, options=Options(**overrides))
        driver = agent if level == "none" else ParaphrasingAgent(agent, level=level)
        return evaluate(driver, self.samples, self.ids, self.cats,
                        self.products)["recommended_technical_score"]

    def test_span_recovery_helps_under_paraphrase(self) -> None:
        self.assertGreater(self._score("light"),
                           self._score("light", use_span_recovery=False))

    def test_span_recovery_does_not_touch_the_clean_path(self) -> None:
        """It only fires when no frame matched, so clean scores must be identical."""
        self.assertEqual(self._score("none"), self._score("none", use_span_recovery=False))


if __name__ == "__main__":
    unittest.main()
