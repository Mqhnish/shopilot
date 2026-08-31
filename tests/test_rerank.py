"""The optional semantic reranking stage.

The remote path cannot be exercised against a live API here -- there are no
credentials in this environment and the submission must never require any -- so
these tests drive it through a stub client. What they verify is everything that
is actually ours: the request shape we send, the recovery of a valid ordering
from a malformed reply, the token accounting, and above all that every failure
mode degrades to the local ordering instead of costing a session.
"""

from __future__ import annotations

import unittest

from tests.fixtures import subset_catalog

from src.catalog import Catalog
from src.rerank import (ZERO_USAGE, ClaudeReranker, LocalReranker,
                        from_environment)
from src.state import SessionState


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.input_tokens = prompt
        self.output_tokens = completion


class _Response:
    def __init__(self, text: str, prompt: int = 100, completion: int = 20) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage(prompt, completion)


class _StubClient:
    """Records the request and returns a canned reply."""

    def __init__(self, reply: str = '{"ranking": [2, 1, 0]}', raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = []
        outer = self

        class _Messages:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if outer.raises:
                    raise RuntimeError("network unavailable")
                return _Response(outer.reply)

        self.messages = _Messages()


class TestLocalReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _ = subset_catalog()
        cls.catalog = Catalog(path)

    def test_is_identity_and_free(self) -> None:
        state = SessionState("s", {})
        docs = [4, 9, 1]
        out, usage = LocalReranker().rerank(state, docs, self.catalog)
        self.assertEqual(out, docs)
        self.assertEqual(usage, ZERO_USAGE)

    def test_default_from_environment_needs_no_network(self) -> None:
        reranker = from_environment("local")
        self.assertEqual(reranker.name, "local")
        self.assertFalse(reranker.requires_network)


class TestClaudeReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, _ = subset_catalog()
        cls.catalog = Catalog(path)

    def _state(self) -> SessionState:
        state = SessionState("s", {})
        state.category = self.catalog.coarse[0]
        state.add_phrase("100% Leather")
        return state

    def test_reorders_according_to_the_model(self) -> None:
        client = _StubClient('{"ranking": [2, 1, 0]}')
        out, usage = ClaudeReranker(client=client).rerank(self._state(), [7, 8, 9], self.catalog)
        self.assertEqual(out, [9, 8, 7])
        self.assertEqual(usage, {"prompt_tokens": 100, "completion_tokens": 20})

    def test_request_shape_matches_the_documented_api(self) -> None:
        client = _StubClient()
        ClaudeReranker(client=client, effort="low").rerank(self._state(), [1, 2, 3], self.catalog)
        sent = client.calls[0]
        self.assertEqual(sent["model"], "claude-opus-5")
        self.assertEqual(sent["output_config"]["effort"], "low")
        self.assertEqual(sent["output_config"]["format"]["type"], "json_schema")
        self.assertIn("ranking", sent["output_config"]["format"]["schema"]["properties"])
        self.assertEqual(sent["messages"][0]["role"], "user")
        self.assertIn("100% Leather", sent["messages"][0]["content"])

    def test_prompt_states_when_nothing_has_been_disclosed(self) -> None:
        client = _StubClient()
        ClaudeReranker(client=client).rerank(SessionState("s", {}), [1, 2, 3], self.catalog)
        self.assertIn("browsing", client.calls[0]["messages"][0]["content"])

    def test_only_the_window_is_reordered_and_the_tail_is_preserved(self) -> None:
        client = _StubClient('{"ranking": [1, 0]}')
        docs = [10, 11, 12, 13]
        out, _usage = ClaudeReranker(client=client, window=2).rerank(
            self._state(), docs, self.catalog
        )
        self.assertEqual(out, [11, 10, 12, 13])

    def test_malformed_reply_still_yields_a_full_permutation(self) -> None:
        for reply in ('{"ranking": [3, 3, 99, 1, -4]}', '{"ranking": []}',
                      '{"ranking": ["a", null]}', '{"other": 1}'):
            client = _StubClient(reply)
            out, _usage = ClaudeReranker(client=client).rerank(
                self._state(), [1, 2, 3], self.catalog
            )
            self.assertEqual(sorted(out), [1, 2, 3], msg=reply)

    def test_api_failure_falls_back_to_the_local_ordering(self) -> None:
        client = _StubClient(raises=True)
        docs = [5, 6, 7]
        out, usage = ClaudeReranker(client=client).rerank(self._state(), docs, self.catalog)
        self.assertEqual(out, docs)
        self.assertEqual(usage, ZERO_USAGE)

    def test_gives_up_after_repeated_failures(self) -> None:
        """A broken configuration should cost a few attempts, not one per turn."""
        client = _StubClient(raises=True)
        reranker = ClaudeReranker(client=client)
        for _ in range(5):
            reranker.rerank(self._state(), [1, 2, 3], self.catalog)
        self.assertTrue(reranker._disabled)
        self.assertEqual(len(client.calls), 3)

    def test_missing_sdk_or_credentials_disables_cleanly(self) -> None:
        reranker = ClaudeReranker()
        reranker._resolved = True
        reranker._client = None
        reranker._disabled = True
        out, usage = reranker.rerank(self._state(), [1, 2], self.catalog)
        self.assertEqual(out, [1, 2])
        self.assertEqual(usage, ZERO_USAGE)

    def test_single_candidate_needs_no_call(self) -> None:
        client = _StubClient()
        out, _usage = ClaudeReranker(client=client).rerank(self._state(), [3], self.catalog)
        self.assertEqual(out, [3])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
