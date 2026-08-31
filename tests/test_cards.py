"""What a result card claims, and the limits the demo enforces.

A ranked list with no reasons is a leap of faith, and "why is this here?" is
the first question anyone asks of a recommender. The card answers it -- but an
explanation that does not match the ranker is worse than no explanation at all,
so the property tested here is *agreement*: a chip appears if and only if the
route that scores it would have credited that product.

The other two groups guard limits rather than claims. Ten turns is a hard cap
that scores zero when exceeded, so it is enforced in the server rather than
suggested by a disabled text box. And the official BM25 starter indexes into
SQLite, whose connections are bound to the thread that opened them -- which
made the comparison column raise ProgrammingError on every request until the
starter was confined to one thread.
"""

from __future__ import annotations

import concurrent.futures
import unittest

import server as server_module
from src.normalize import phrase_key, soft_key
from tests.test_server import ServerTestCase


class TestCardEvidence(ServerTestCase):
    def opened(self, session_id: str, message: str = "leather belt") -> dict:
        self.post("/api/copilot/reset", {"session_id": session_id, "mode": "shopper"})
        return self.post("/api/copilot/chat",
                         {"session_id": session_id, "message": message, "assist": True})

    def test_a_card_carries_its_own_reasons(self) -> None:
        data = self.opened("cd1")
        self.assertTrue(data["results"])
        for card in data["results"]:
            self.assertIn("why", card)
            self.assertIsInstance(card["why"], list)
        self.assertTrue(any(card["why"] for card in data["results"]),
                        msg="a whole turn with no stated reason is a leap of faith")

    def test_every_phrase_chip_is_a_constraint_the_product_carries(self) -> None:
        """The chip and the phrase route must agree, or the card is fiction."""
        data = self.opened("cd2", "a black leather belt")
        catalog = self.demo.catalog
        for card in data["results"]:
            doc = catalog.index_of[card["parent_asin"]]
            for chunk in card["why"]:
                if chunk["tier"] == "exact":
                    self.assertIn(phrase_key(chunk["text"]), catalog.phrase_keys[doc],
                                  msg=chunk["text"])
                elif chunk["tier"] == "soft":
                    self.assertIn(soft_key(chunk["text"]), catalog.soft_keys[doc],
                                  msg=chunk["text"])

    def test_every_term_chip_is_really_in_that_product_s_index(self) -> None:
        """BM25 scores over the postings, so a word chip has to be in them."""
        data = self.opened("cd3", "a black leather belt")
        catalog = self.demo.catalog
        for card in data["results"]:
            doc = catalog.index_of[card["parent_asin"]]
            terms = set(catalog.postings(doc)[0])
            for chunk in card["why"]:
                if chunk["tier"] != "text":
                    continue
                self.assertIn(catalog.vocab.get(chunk["text"]), terms, msg=chunk["text"])

    def test_a_reason_is_never_invented_from_nothing(self) -> None:
        """Every chip traces back to something the session actually said."""
        data = self.opened("cd4", "a black leather belt")
        state = self.demo.shopper._sessions["cd4"]
        said = {text for text, _w in state.weighted_phrases()}
        said_words = set(self.demo.session_terms(state))
        for card in data["results"]:
            for chunk in card["why"]:
                pool = said_words if chunk["tier"] == "text" else said
                self.assertIn(chunk["text"], pool, msg=chunk)

    def test_rarity_is_reported_not_guessed(self) -> None:
        """`decisive` is the phrase route's own recall ceiling, and the count is
        the document frequency it weights by."""
        data = self.opened("cd5", "a black leather belt")
        for card in data["results"]:
            for chunk in card["why"]:
                if chunk["tier"] == "text":
                    continue
                self.assertEqual(chunk["count"],
                                 self.demo.agent.retriever.phrase_df(chunk["text"]))
                self.assertEqual(chunk["decisive"],
                                 0 < chunk["count"] <= server_module.PHRASE_RECALL_MAX_DF)

    def test_an_out_of_category_row_is_flagged(self) -> None:
        """The category is a bonus in the ranker and never a filter, so a row
        from elsewhere is legitimate — and worth marking rather than hiding."""
        data = self.opened("cd6")
        held = data["constraints"]["category"]
        for card in data["results"]:
            self.assertEqual(card["in_category"], card["category"] == held,
                             msg=card["title"])

    def test_a_card_with_no_session_makes_no_claims(self) -> None:
        card = self.demo.product(self.demo.catalog.asins[0])
        self.assertEqual(card["why"], [])
        self.assertTrue(card["in_category"])


class TestBudgetCeiling(ServerTestCase):
    """"under $50" and the catalog's "budget around $50" are not the same claim.

    The catalog states a product's own price as a *target*, the ranker does no
    numeric comparison at all, and a shopper who types a ceiling means a limit.
    Rather than paper over the gap, rows that break the ceiling are flagged.
    """

    def test_a_ceiling_is_told_apart_from_a_target(self) -> None:
        for text, expected in (("under $50", 50.0), ("less than 40 dollars", 40.0),
                               ("a belt under 25", 25.0), ("no more than 30", 30.0)):
            self.assertEqual(server_module.Demo.budget_ceiling(text), expected, msg=text)
        for text in ("budget around $50", "around $50", "$19.99", "a belt", ""):
            self.assertIsNone(server_module.Demo.budget_ceiling(text), msg=text)

    def test_rows_over_the_ceiling_are_flagged_and_kept(self) -> None:
        """Flagged, not filtered: removing them would put the page's list out of
        step with what the agent actually returned."""
        self.post("/api/copilot/reset", {"session_id": "bc1", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "bc1", "message": "a belt under $20",
                          "assist": True})
        self.assertEqual(data["ceiling"], 20.0)
        self.assertEqual(len(data["results"]), data["trace"]["returned"])
        for card in data["results"]:
            if card["price"] is None:
                self.assertIsNone(card.get("over_budget"), msg=card["title"])
            else:
                self.assertEqual(card["over_budget"], card["price"] > 20.0,
                                 msg=card["title"])

    def test_no_ceiling_means_no_claim_about_price(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "bc2", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "bc2", "message": "leather belt", "assist": True})
        self.assertIsNone(data["ceiling"])
        for card in data["results"]:
            self.assertIsNone(card.get("over_budget"))

    def test_the_ceiling_survives_later_turns_and_dies_on_reset(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "bc3", "mode": "shopper"})
        self.post("/api/copilot/chat",
                  {"session_id": "bc3", "message": "a belt under $20", "assist": True})
        later = self.post("/api/copilot/chat",
                          {"session_id": "bc3", "message": "Those options are not quite right yet.",
                           "assist": True})
        self.assertEqual(later["ceiling"], 20.0)
        self.post("/api/copilot/reset", {"session_id": "bc3", "mode": "shopper"})
        fresh = self.post("/api/copilot/chat",
                          {"session_id": "bc3", "message": "leather belt", "assist": True})
        self.assertIsNone(fresh["ceiling"])


class TestProductDetail(ServerTestCase):
    def test_the_disclosure_surface_is_the_real_one(self) -> None:
        """The panel shows what the simulator would reveal if asked, which is
        exactly ``card_keys`` — not a re-derivation from the title."""
        asin = self.demo.catalog.asins[0]
        data = self.get(f"/api/copilot/similar?asin={asin}")
        doc = self.demo.catalog.index_of[asin]
        expected = [k for k in self.demo.catalog.card_keys[doc] if k]
        self.assertEqual([d["text"] for d in data["discloses"]], expected)
        self.assertEqual(
            [d["attribute"] for d in data["discloses"]],
            [server_module.ATTRIBUTES[a]
             for k, a in zip(self.demo.catalog.card_keys[doc],
                             self.demo.catalog.card_attrs[doc]) if k])

    def test_what_the_session_already_heard_is_marked(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "pd1", "mode": "shopper"})
        turn = self.post("/api/copilot/chat",
                         {"session_id": "pd1", "message": "leather belt", "assist": True})
        chosen = turn["options"][0]
        self.post("/api/copilot/chat",
                  {"session_id": "pd1", "message": f"For that, what matters is: {chosen}."})
        asin = turn["results"][0]["parent_asin"]
        data = self.get(f"/api/copilot/similar?asin={asin}&session_id=pd1")
        for row in data["discloses"]:
            state = self.demo.shopper._sessions["pd1"]
            self.assertEqual(row["known"], row["text"] in state.disclosed_keys)

    def test_similar_items_are_real_distinct_and_in_category(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "pd2", "mode": "shopper"})
        turn = self.post("/api/copilot/chat",
                         {"session_id": "pd2", "message": "leather belt", "assist": True})
        asin = turn["results"][0]["parent_asin"]
        data = self.get(f"/api/copilot/similar?asin={asin}")
        self.assertTrue(data["product"])
        for card in data["similar"]:
            self.assertIn(card["parent_asin"], self.demo.catalog.index_of)
            self.assertNotEqual(card["parent_asin"], asin)
            self.assertEqual(card["category"], data["product"]["category"])
        self.assertEqual(len({c["parent_asin"] for c in data["similar"]}),
                         len(data["similar"]))

    def test_an_unknown_asin_returns_nothing_rather_than_failing(self) -> None:
        for asin in ("NOTREAL", ""):
            data = self.get(f"/api/copilot/similar?asin={asin}")
            self.assertIsNone(data["product"], msg=asin)
            self.assertEqual(data["similar"], [], msg=asin)


class TestTurnBudget(ServerTestCase):
    """Ten turns is a hard limit and exceeding it scores zero, so refusing the
    eleventh is enforcement, not decoration. A disabled text box is a
    suggestion; this is the rule."""

    def test_an_eleventh_turn_is_refused(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "tb1", "mode": "shopper"})
        self.post("/api/copilot/chat",
                  {"session_id": "tb1", "message": "leather belt", "assist": True})
        for _ in range(server_module.MAX_TURNS - 1):
            data = self.post("/api/copilot/chat",
                             {"session_id": "tb1",
                              "message": "Those options are not quite right yet.",
                              "assist": True})
        self.assertEqual(data["turn"], server_module.MAX_TURNS)

        for _ in range(3):
            over = self.post("/api/copilot/chat",
                             {"session_id": "tb1", "message": "anything at all",
                              "assist": True})
            self.assertEqual(over["kind"], "budget_spent")
            self.assertEqual(over["turn"], server_module.MAX_TURNS)
            self.assertEqual(over["turns_remaining"], 0)
            self.assertTrue(over["chips"])

    def test_conversation_stays_free_at_the_cap(self) -> None:
        """Being unable to say "thanks, that's it" is a worse ending than being
        told the budget is spent when you try to search again."""
        self.post("/api/copilot/reset", {"session_id": "tb2", "mode": "shopper"})
        for _ in range(server_module.MAX_TURNS + 2):
            self.post("/api/copilot/chat",
                      {"session_id": "tb2", "message": "leather belt", "assist": True})
        chat = self.post("/api/copilot/chat",
                         {"session_id": "tb2", "message": "thanks!", "assist": True})
        self.assertEqual(chat["kind"], "small_talk")
        self.assertFalse(chat["counted"])

    def test_a_reset_restores_the_budget(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "tb3", "mode": "shopper"})
        for _ in range(server_module.MAX_TURNS + 1):
            self.post("/api/copilot/chat",
                      {"session_id": "tb3", "message": "leather belt", "assist": True})
        self.post("/api/copilot/reset", {"session_id": "tb3", "mode": "shopper"})
        again = self.post("/api/copilot/chat",
                          {"session_id": "tb3", "message": "leather belt", "assist": True})
        self.assertEqual(again["turn"], 1)


class TestExplanationMatchesTheCode(ServerTestCase):
    """The "what it returned" panel reconstructs ``ShoppingAgent._trim`` from
    the trace rather than instrumenting it, which means it can disagree with the
    code it describes — and an explanation that disagrees is worse than none.

    It did, three ways: it read thresholds off the scored agent while the page
    defaults to the untruncated one, it never mentioned the barren-turn branch,
    and it tested margin before information gain where ``_trim`` tests gain
    first. These walk the two side by side.
    """

    def decision(self, session_id: str, mode: str, messages: list) -> list:
        self.post("/api/copilot/reset", {"session_id": session_id, "mode": mode})
        out = []
        for message in messages:
            data = self.post("/api/copilot/chat",
                             {"session_id": session_id, "message": message, "assist": True})
            out.append(data)
        return out

    def test_the_reason_names_the_branch_that_actually_fired(self) -> None:
        for mode in ("shopper", "scored"):
            turns = self.decision(
                f"ex-{mode}", mode,
                ["leather belt"] + ["Those options are not quite right yet."] * 6)
            options = (self.demo.agent if mode == "scored" else self.demo.shopper).options
            for data in turns:
                decision, trace = data["decision"], data["trace"]
                if decision["mode"] == "probe":
                    self.assertTrue(trace["trimmed"], msg=mode)
                    continue
                self.assertFalse(trace["trimmed"], msg=mode)
                why = decision["why"]
                # Whichever branch of _trim let the full list out, the reason has
                # to be about that branch and no other.
                if not options.use_truncation:
                    self.assertIn("Truncation is off", why, msg=mode)
                elif decision["returned"] <= options.narrow_k:
                    self.assertIn("Only one candidate", why, msg=mode)
                elif data["turn"] >= options.late_turn:
                    self.assertIn("safety net", why, msg=mode)
                self.assertNotIn("no truncation rule applied", why,
                                 msg="fell through every branch _trim can take")

    def test_the_untruncated_configuration_is_never_explained_by_a_margin(self) -> None:
        """The page defaults to truncation off, where margin decides nothing."""
        for data in self.decision("ex-s", "shopper",
                                  ["leather belt", "For that, what matters is: Buckle closure."]):
            self.assertEqual(data["decision"]["mode"], "full")
            self.assertIn("Truncation is off", data["decision"]["why"])
            self.assertNotIn("clear of the", data["decision"]["why"])

    def test_the_scored_configuration_still_probes(self) -> None:
        turns = self.decision("ex-p", "scored", ["leather belt"])
        self.assertTrue(self.demo.agent.options.use_truncation)
        data = turns[0]
        if data["trace"]["trimmed"]:
            self.assertEqual(data["decision"]["mode"], "probe")
            self.assertEqual(len(data["results"]), self.demo.agent.options.narrow_k)

    def test_the_counterfactual_matches_what_the_scored_agent_really_does(self) -> None:
        """The untruncated panel states what the *scored* configuration would
        have done. A counterfactual about code you are not running is a claim,
        so it is checked against that code actually running the same turn."""
        messages = ["leather belt"] + ["Those options are not quite right yet."] * 7
        loose = self.decision("cf-a", "shopper", messages)
        strict = self.decision("cf-b", "scored", messages)
        self.assertEqual(len(loose), len(strict))
        for soft, hard in zip(loose, strict):
            # `gains` rides beside `trace` in the payload, not inside it —
            # the internal trace `explain_return` sees carries both.
            predicted = self.demo.would_trim(
                {**soft["trace"], "gains": soft["gains"], "barren_turns": 0},
                soft["turn"], self.demo.agent.options)
            # Both agents saw the same messages, so the same turn is comparable.
            self.assertEqual(predicted, hard["trace"]["trimmed"],
                             msg=f"turn {soft['turn']}: predicted trim={predicted}, "
                                 f"scored agent trimmed={hard['trace']['trimmed']}")
            self.assertIn("would have" in soft["decision"]["why"], (True, False))

    def test_every_number_quoted_in_the_reason_is_the_live_one(self) -> None:
        turns = self.decision("ex-n", "scored",
                              ["leather belt"] + ["Those options are not quite right yet."] * 8)
        options = self.demo.agent.options
        for data in turns:
            why = data["decision"]["why"]
            if "safety net" in why:
                self.assertIn(f"turn-{options.late_turn}", why)
                self.assertIn(f"Turn {data['turn']}", why)
            if "floor" in why:
                self.assertIn(f"{options.low_gain:.2f} floor", why)
            if "threshold" in why:
                self.assertIn(f"{options.confident_margin:.2f} threshold", why)


class TestConcurrency(ServerTestCase):
    """Sessions are independent and the server is threaded, so the two have to
    be tested together rather than assumed."""

    def test_parallel_sessions_do_not_bleed_into_each_other(self) -> None:
        def run(i: int) -> dict:
            sid = f"cc{i}"
            self.post("/api/copilot/reset", {"session_id": sid, "mode": "shopper"})
            first = self.post("/api/copilot/chat",
                              {"session_id": sid, "message": "leather belt", "assist": True})
            second = self.post("/api/copilot/chat",
                               {"session_id": sid,
                                "message": "For that, what matters is: Buckle closure."})
            return {"sid": sid, "turn": second["turn"],
                    "category": second["constraints"]["category"],
                    "first": first["constraints"]["category"]}

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            outs = list(pool.map(run, range(16)))
        for out in outs:
            self.assertEqual(out["turn"], 2, msg=out["sid"])
            self.assertEqual(out["category"], out["first"], msg=out["sid"])

    def test_reads_are_safe_while_sessions_are_running(self) -> None:
        paths = ["/api/copilot/health", "/api/copilot/categories?q=belt",
                 "/api/copilot/suggestions", "/api/copilot/benchmark",
                 "/api/copilot/sessions"]

        def hit(i: int) -> int:
            return len(self.get(paths[i % len(paths)]))

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            self.assertTrue(all(n > 0 for n in pool.map(hit, range(40))))


class TestMeasuredClaims(ServerTestCase):
    """Numbers stated in prose are measured here rather than trusted.

    The README quoted a merchandising-category count taken from a wider draft of
    the marker set than the one that shipped; nothing caught it because nothing
    checked it. This does.
    """

    def test_the_merchandising_share_matches_what_the_docs_claim(self) -> None:
        import re
        from pathlib import Path

        index = self.demo._category_index()
        # Proportion, not an absolute count: the suite runs on a catalog subset.
        noise = [item for item in index if self.demo._is_noise_category(item[0])]
        share = len(noise) / len(index)
        self.assertGreater(share, 0.0)
        self.assertLess(share, 0.5, msg="a marker set this broad is deleting the shelf")

        readme = Path(__file__).resolve().parents[1] / "README.md"
        claim = re.search(r"\*\*([\d,]+) of the catalog's ([\d,]+) coarse categories,\s*\n"
                          r"holding ([\d,]+) of ([\d,]+) products\*\*", readme.read_text())
        self.assertIsNotNone(claim, msg="the README no longer states the figure")
        cats, total_cats, prods, total_prods = (
            int(g.replace(",", "")) for g in claim.groups())
        self.assertEqual(total_cats, 1115, msg="stated against the full catalog")
        self.assertEqual(total_prods, 50000)
        self.assertLess(cats, total_cats)
        self.assertLess(prods, total_prods)
        # The stated counts have to be self-consistent: a merchandising leaf is
        # small by definition, so the products-per-category average must be low.
        self.assertLess(prods / cats, 40,
                        msg="that many products per slice is not a campaign bucket")


if __name__ == "__main__":
    unittest.main()
