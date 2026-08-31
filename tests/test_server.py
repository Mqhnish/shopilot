"""The demo server's HTTP surface.

``server.py`` is not the scored path, but a broken demo is still a broken
deliverable — the problem statement asks for a walkthrough that shows the
solution working end to end. These tests run against a real ``ThreadingHTTPServer``
on a loopback port with a small catalog, so they exercise the routing, the JSON
contract and the traversal guard rather than the handler functions in isolation.

Two properties matter more than the rest and are asserted directly:

* the page can only ever render what the agent actually did — every panel field
  is traced back to a live ``SessionState``;
* the server never serves a file from outside ``web/``.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from tests.fixtures import subset_catalog

import server as server_module


class ServerTestCase(unittest.TestCase):
    """One server, one small index, shared by every test in the class."""

    @classmethod
    def setUpClass(cls) -> None:
        catalog_path, cls.samples = subset_catalog()
        cls.demo = server_module.Demo(catalog_path, "data/public_set.jsonl")
        # The background thread builds the official starter over the *full*
        # catalog, which these tests neither need nor want to wait for.
        cls.demo.display.ready = True

        class Handler(server_module.Handler):
            demo = cls.demo  # noqa: read off the class by the request handler
            quiet = True  # noqa: read off the class; request logs are test noise

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    # -- helpers ------------------------------------------------------------

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url(path), timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.url(path),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def status_of(self, path: str) -> int:
        try:
            with urllib.request.urlopen(self.url(path), timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code


class TestStatic(ServerTestCase):
    def test_root_serves_the_page(self) -> None:
        with urllib.request.urlopen(self.url("/"), timeout=10) as response:
            body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("<title>Shopilot", body)
        self.assertIn("copilot.js", body)

    def test_assets_are_served(self) -> None:
        for asset in ("/copilot.css", "/copilot.js"):
            self.assertEqual(self.status_of(asset), 200, msg=asset)

    def test_traversal_is_refused(self) -> None:
        """Nothing outside web/ is reachable, however the path is spelled."""
        for attempt in (
            "/../agent.py",
            "/../../etc/passwd",
            "/..%2fagent.py",
            "/subdir/../../server.py",
        ):
            self.assertEqual(self.status_of(attempt), 404, msg=attempt)

    def test_unknown_api_route_is_404(self) -> None:
        self.assertEqual(self.status_of("/api/copilot/nope"), 404)


class TestMetadata(ServerTestCase):
    def test_health(self) -> None:
        health = self.get("/api/copilot/health")
        self.assertTrue(health["ok"])
        self.assertGreater(health["catalog_size"], 0)
        self.assertEqual(health["sessions_available"], 200)

    def test_suggestions_are_real_openers(self) -> None:
        examples = self.get("/api/copilot/suggestions")["examples"]
        self.assertTrue(examples)
        # The simulator opens every scenario with this stem, so a suggestion
        # that does not is a suggestion the agent was never scored against.
        for example in examples:
            self.assertTrue(example.startswith("I'm looking for"), msg=example)

    def test_sessions_list_matches_the_public_set(self) -> None:
        sessions = self.get("/api/copilot/sessions")["sessions"]
        self.assertEqual(len(sessions), 200)
        self.assertEqual(sessions[0]["sample_id"], "public_0001")
        self.assertIn(sessions[0]["scenario"],
                      {"buying", "browsing", "intent_override", "boundary"})

    def test_benchmark_reports_committed_artifacts(self) -> None:
        """The page must not be able to quote a number the repo cannot back."""
        benchmark = self.get("/api/copilot/benchmark")
        self.assertAlmostEqual(benchmark["ours"]["technical_score"], 0.953833, places=5)
        self.assertAlmostEqual(benchmark["baseline"]["technical_score"], 0.10671, places=5)
        self.assertEqual(
            set(benchmark["scenarios"]),
            {"buying", "browsing", "intent_override", "boundary"},
        )
        self.assertEqual(benchmark["ablation"][0]["arm"], "full system")


class TestConversation(ServerTestCase):
    def opener(self) -> str:
        return "I'm looking for Accessories Belts. A key requirement is: 100% Leather."

    def test_chat_returns_every_panel_field(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "t1"})
        data = self.post("/api/copilot/chat", {"session_id": "t1", "message": self.opener()})

        self.assertEqual(data["turn"], 1)
        self.assertEqual(data["turns_remaining"], 9)
        self.assertIn(data["track"], {"buy", "browse", "blend"})
        self.assertTrue(data["message"])
        self.assertTrue(data["results"])
        for field in ("constraints", "gains", "trace", "weights", "decision"):
            self.assertIn(field, data, msg=field)
        for field in ("pool", "in_bucket", "phrase_docs", "excluded", "margin", "returned"):
            self.assertIn(field, data["trace"], msg=field)
        self.assertEqual(data["decision"]["returned"], len(data["results"]))

    def test_results_are_hydrated_catalog_products(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "t2"})
        data = self.post("/api/copilot/chat", {"session_id": "t2", "message": self.opener()})
        for card in data["results"]:
            self.assertIn(card["parent_asin"], self.demo.catalog.index_of)
            self.assertTrue(card["title"])
            self.assertIsInstance(card["rank"], int)

    def test_turns_accumulate_and_constraints_persist(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "t3"})
        first = self.post("/api/copilot/chat", {"session_id": "t3", "message": self.opener()})
        second = self.post("/api/copilot/chat",
                           {"session_id": "t3", "message": "For that, what matters is: Buckle closure."})

        self.assertEqual(second["turn"], 2)
        texts = {p["text"] for p in second["constraints"]["phrases"]}
        self.assertIn("100% Leather", texts, msg="turn 1's constraint must survive turn 2")
        self.assertIn("Buckle closure", texts)
        self.assertGreaterEqual(
            len(second["constraints"]["phrases"]), len(first["constraints"]["phrases"])
        )
        self.assertGreater(second["constraints"]["ruled_out"], 0,
                           msg="turn 1's products are negative evidence on turn 2")

    def test_gains_explain_the_question_that_was_asked(self) -> None:
        """The 'next question' panel must show the winner, not an unrelated list."""
        self.post("/api/copilot/reset", {"session_id": "t4"})
        data = self.post("/api/copilot/chat", {"session_id": "t4", "message": self.opener()})
        if data["ask_attribute"] is not None and data["gains"]:
            best = max(data["gains"].values())
            self.assertAlmostEqual(data["gains"][data["ask_attribute"]], best, places=6)

    def test_empty_message_is_rejected_not_crashed(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "t5"})
        self.assertEqual(self.status_of("/api/copilot/health"), 200)
        request = urllib.request.Request(
            self.url("/api/copilot/chat"),
            data=json.dumps({"session_id": "t5", "message": "   "}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 400)

    def test_chat_without_reset_self_heals(self) -> None:
        """The agent recovers an unknown session rather than forfeiting it."""
        data = self.post("/api/copilot/chat",
                         {"session_id": "never-reset", "message": self.opener()})
        self.assertTrue(data["results"])

    def test_baseline_reports_its_state_honestly(self) -> None:
        data = self.post("/api/copilot/baseline",
                         {"session_id": "t6", "message": self.opener()})
        self.assertIn("ready", data)
        if not data["ready"]:
            self.assertTrue(data["note"], msg="an unready baseline must say why")


class TestDecision(ServerTestCase):
    def test_probe_turns_are_explained(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "d1"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "d1", "message": "I'm looking for Accessories Belts, but I'm still exploring."})
        decision = data["decision"]
        self.assertIn(decision["mode"], {"probe", "full"})
        self.assertTrue(decision["headline"])
        self.assertTrue(decision["why"])
        if decision["mode"] == "probe":
            self.assertEqual(len(data["results"]), self.demo.agent.options.narrow_k)

    def test_late_turns_always_send_the_full_list(self) -> None:
        """From LATE_TURN the escape hatch must fire, and be named as the reason."""
        self.post("/api/copilot/reset", {"session_id": "d2"})
        data = {}
        for _ in range(self.demo.agent.options.late_turn):
            data = self.post("/api/copilot/chat",
                             {"session_id": "d2", "message": "Those options are not quite right yet."})
        self.assertEqual(data["decision"]["mode"], "full")
        self.assertFalse(data["trace"]["trimmed"])


class TestReplay(ServerTestCase):
    def test_replay_drives_the_official_simulator(self) -> None:
        sample_id = self.samples[0]["sample_id"]
        run = self.post("/api/copilot/replay", {"sample_id": sample_id})

        self.assertEqual(run["sample_id"], sample_id)
        self.assertTrue(run["turns"])
        self.assertLessEqual(len(run["turns"]), 10)
        self.assertEqual(
            run["target"]["parent_asin"],
            self.samples[0]["ground_truth"]["parent_asin"],
        )
        # The customer's first line is the simulator's, not ours.
        self.assertTrue(run["turns"][0]["customer"].startswith("I'm looking for"))
        for turn in run["turns"]:
            self.assertTrue(turn["message"])
            self.assertIn("decision", turn)

    def test_replay_verdict_agrees_with_its_own_turns(self) -> None:
        """A reported hit must be visible in the turn it claims."""
        run = self.post("/api/copilot/replay", {"sample_id": self.samples[0]["sample_id"]})
        if run["hit"]:
            final = run["turns"][-1]
            self.assertEqual(final["turn"], run["hit_turn"])
            self.assertEqual(final["target_rank"], run["hit_rank"])
            self.assertAlmostEqual(run["reciprocal_rank"], 1.0 / run["hit_rank"], places=3)
        else:
            self.assertIsNone(run["hit_turn"])
            self.assertEqual(run["reciprocal_rank"], 0.0)

    def test_unknown_sample_is_404(self) -> None:
        request = urllib.request.Request(
            self.url("/api/copilot/replay"),
            data=json.dumps({"sample_id": "nope_9999"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 404)


class TestIsolation(unittest.TestCase):
    """The demo must stay out of the scored path."""

    def test_src_never_imports_the_server(self) -> None:
        from tests.fixtures import ROOT

        for path in sorted((ROOT / "src").glob("*.py")) + [ROOT / "agent.py"]:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("import server", source, msg=str(path))
            self.assertNotIn("from server", source, msg=str(path))


if __name__ == "__main__":
    unittest.main()
