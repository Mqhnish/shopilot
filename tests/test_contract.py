"""Conformance with docs/agent_api_contract.json, and refusal to ever crash.

The evaluator scores a raised exception exactly the same as a wrong answer: the
whole session counts as a miss. So "never raise" is not defensive style here, it
is worth up to a full point of hit rate. These tests attack the agent with the
inputs the contract says cannot happen, and require a schema-valid response for
every one of them.
"""

from __future__ import annotations

import json
import unittest

from tests.fixtures import ROOT, subset_catalog

from agent import Agent

CONTRACT = json.loads((ROOT / "docs" / "agent_api_contract.json").read_text(encoding="utf-8"))
ALLOWED_ATTRIBUTES = set(CONTRACT["turn_response"]["properties"]["ask_attribute"]["enum"])


class ContractMixin:
    def assert_valid_response(self, response: object, catalog_ids: set) -> None:
        self.assertIsInstance(response, dict)
        self.assertEqual(
            set(response) - {"message", "ask_attribute", "recommendations", "usage"},
            set(),
            msg="additionalProperties is false in the contract",
        )
        for required in ("message", "ask_attribute", "recommendations"):
            self.assertIn(required, response)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)

        recommendations = response["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 100)
        seen = set()
        for item in recommendations:
            self.assertIsInstance(item, dict)
            self.assertEqual(set(item) - {"parent_asin", "score"}, set())
            asin = item["parent_asin"]
            self.assertIsInstance(asin, str)
            self.assertTrue(asin)
            self.assertIn(asin, catalog_ids, msg="identifiers must exist in the frozen catalog")
            self.assertNotIn(asin, seen, msg="duplicate identifier")
            seen.add(asin)

        usage = response.get("usage")
        if usage is not None:
            self.assertIsInstance(usage, dict)
            self.assertEqual(set(usage), {"prompt_tokens", "completion_tokens"})
            for value in usage.values():
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)


class TestContract(unittest.TestCase, ContractMixin):
    @classmethod
    def setUpClass(cls) -> None:
        path, cls.samples = subset_catalog()
        cls.agent = Agent(path)
        with open(path, encoding="utf-8") as handle:
            cls.catalog_ids = {str(json.loads(line)["parent_asin"]) for line in handle if line.strip()}

    def test_normal_session_is_schema_valid(self) -> None:
        sample = self.samples[0]
        self.agent.reset("s1", sample["user_profile"])
        message = "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy."
        for turn in range(1, 11):
            response = self.agent.respond("s1", message, turn, 10)
            self.assert_valid_response(response, self.catalog_ids)
            message = "For that, what matters is: Imported; Pull On closure."

    def test_never_exceeds_requested_top_k(self) -> None:
        self.agent.reset("s2", self.samples[0]["user_profile"])
        for top_k in (1, 3, 10):
            response = self.agent.respond("s2", "I'm looking for Watches Wrist Watches.", 1, top_k)
            self.assertLessEqual(len(response["recommendations"]), top_k)

    def test_respond_without_reset_recovers(self) -> None:
        """The contract promises reset first; raising here would forfeit a session."""
        response = self.agent.respond("never-reset", "I'm looking for Bras Everyday Bras.", 1, 10)
        self.assert_valid_response(response, self.catalog_ids)

    def test_hostile_inputs_never_raise(self) -> None:
        hostile = [
            "", "   ", "\n\t", "?" * 5000, "I'm looking for " + "x" * 3000,
            "For that, what matters is: ", "For that, what matters is: ;;;;.",
            "I don't have a preference for ; please use your judgment.",
            "Actually, ignore my earlier preference. What I need is: .",
            "\\x00 nul-ish", "🙂🙂🙂", "SELECT * FROM products; DROP TABLE products;",
            "I'm looking for , but I'm still exploring.",
            "For that, what matters is: " + "; ".join(str(i) for i in range(60)) + ".",
        ]
        self.agent.reset("s3", {})
        for turn, message in enumerate(hostile, start=1):
            response = self.agent.respond("s3", message, min(turn, 10), 10)
            self.assert_valid_response(response, self.catalog_ids)

    def test_malformed_arguments_never_raise(self) -> None:
        self.agent.reset("s4", {"preference_tags": "not-a-list", "summary": None})
        for turn, top_k in ((1, 10), (0, 10), (-5, 10), (99, 10), ("x", 10), (3, 0), (3, None)):
            response = self.agent.respond("s4", "I'm looking for Jewelry Necklaces.", turn, top_k)
            self.assert_valid_response(response, self.catalog_ids)

    def test_reset_with_odd_profiles(self) -> None:
        for profile in (None, {}, {"preference_tags": [1, 2, 3]}, {"unexpected": object()}):
            self.agent.reset("s5", profile)
            response = self.agent.respond("s5", "I'm looking for Accessories Belts.", 1, 10)
            self.assert_valid_response(response, self.catalog_ids)

    def test_never_withholds_when_no_question_could_help(self) -> None:
        """The guard that stops the agent trickling one candidate per turn.

        Tested directly because it is a correctness property, not a tuning
        choice: if no attribute can teach us anything, there is no better turn
        to wait for and the full list must go out.
        """
        impl = self.agent._impl
        impl.reset("trim", {})
        impl.respond("trim", "I'm looking for Jewelry Necklaces.", 1, 10)
        state = impl._sessions["trim"]
        state.turn = 1
        docs = list(range(10))
        zero_gain = {name: 0.0 for name in ("other", "feature", "material")}
        wide_margin = {"margin": 0.0}
        self.assertEqual(
            impl._trim(state, docs, zero_gain, dict(wide_margin), 10), docs,
            msg="nothing left to learn: return everything",
        )
        self.assertEqual(
            impl._trim(state, docs, {}, dict(wide_margin), 10), docs,
            msg="no gains computed at all: return everything",
        )
        high_gain = {"other": 9.0}
        self.assertEqual(
            len(impl._trim(state, docs, high_gain, dict(wide_margin), 10)),
            impl.options.narrow_k,
            msg="a question can still help: hold back and ask it",
        )

    def test_sessions_are_isolated(self) -> None:
        self.agent.reset("a", {})
        self.agent.reset("b", {})
        self.agent.respond("a", "I'm looking for Jewelry Necklaces. A key requirement is: leather.", 1, 10)
        response_b = self.agent.respond("b", "I'm looking for Watches Wrist Watches, but I'm still exploring.", 1, 10)
        self.assert_valid_response(response_b, self.catalog_ids)
        # Session b must not inherit a's constraints.
        state_b = self.agent._impl._sessions["b"]
        self.assertEqual(state_b.disclosed_keys, set())

    def test_reset_clears_previous_session_state(self) -> None:
        self.agent.reset("c", {})
        self.agent.respond("c", "I'm looking for Jewelry Necklaces. A key requirement is: leather.", 1, 10)
        self.assertTrue(self.agent._impl._sessions["c"].disclosed_keys)
        self.agent.reset("c", {})
        self.assertEqual(self.agent._impl._sessions["c"].disclosed_keys, set())
        self.assertEqual(self.agent._impl._sessions["c"].shown, set())



class TestNeverAnEmptyTurn(unittest.TestCase):
    """A turn that returns nothing cannot hit, and the session spends it anyway.

    It never happens on the public set — every turn there names a category or a
    constraint — but the organizer reserves the right to reword the customer,
    and text the parser cannot read at all leaves nothing to rank. Sending the
    best guess available costs nothing an empty list does not already cost.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path, _samples = subset_catalog()
        cls.agent = Agent(path)

    def turn(self, session_id: str, messages: list) -> dict:
        self.agent.reset(session_id, {})
        response = {}
        for turn, message in enumerate(messages, start=1):
            response = self.agent.respond(session_id, message, turn, 10)
        return response

    def test_unreadable_text_still_returns_products(self) -> None:
        for name, message in (("empty", ""), ("whitespace", "   \t "),
                              ("emoji", "\U0001f457\U0001f460"), ("control", "\x07\x1b[31m"),
                              ("script", "<script>alert(1)</script>")):
            response = self.turn(f"lr-{name}", [message])
            self.assertTrue(response["recommendations"], msg=name)

    def test_the_guess_stays_inside_the_held_category(self) -> None:
        """With a category in hand the fallback is not a shot in the dark."""
        catalog = self.agent._impl.catalog
        opener = f"I'm looking for {catalog.coarse[0]}, but I'm still exploring."
        response = self.turn("lr-cat", [opener, "\U0001f457\U0001f460"])
        self.assertTrue(response["recommendations"])
        for item in response["recommendations"]:
            doc = catalog.index_of[item["parent_asin"]]
            self.assertEqual(catalog.coarse[doc], catalog.coarse[0])

    def test_the_fallback_never_repeats_a_rejected_product(self) -> None:
        shown = set()
        self.agent.reset("lr-rep", {})
        for turn in range(1, 6):
            response = self.agent.respond("lr-rep", "\U0001f457", turn, 10)
            asins = [item["parent_asin"] for item in response["recommendations"]]
            self.assertFalse(shown & set(asins), msg=f"turn {turn} re-offered a rejected item")
            shown.update(asins)

    def test_it_does_not_fire_on_a_readable_turn(self) -> None:
        """Insurance, not a crutch: the public set must never reach it."""
        catalog = self.agent._impl.catalog
        opener = f"I'm looking for {catalog.coarse[0]}, but I'm still exploring."
        self.turn("lr-off", [opener])
        state = self.agent._impl._sessions["lr-off"]
        self.assertFalse(state.last_trace.get("last_resort"))

if __name__ == "__main__":
    unittest.main()
