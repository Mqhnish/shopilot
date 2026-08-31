"""The conversational surface: small talk, suggestions, and refinements.

The scored task is a four-frame protocol between an agent and a simulator. A
person at the demo is not a simulator, and the difference shows up in three
places, each of which is tested here:

* they say hello. Answering that with ten belts is not a shopping agent, it is
  a search box wearing a chat skin -- and worse, it burns one of the ten turns
  the problem statement scores you on;
* they type an occasion ("something warm for winter") rather than a catalog
  category, which shares no token with anything the catalog holds;
* mid-session they need the *vocabulary the pool can still disclose*, not more
  categories -- the agent locks its category for the life of a session, so a
  category picker after turn one only invites a topic change nobody asked for.

The invariant underneath all three: none of it may leak into the scored path.
``assist`` is the switch, the evaluator never sets it, and the four customer
frames are asserted below to be untouched by every one of these features.
"""

from __future__ import annotations

import unittest
import urllib.parse

import server as server_module
from tests.test_server import ServerTestCase


class TestSmallTalk(ServerTestCase):
    def test_greetings_and_farewells_are_recognised(self) -> None:
        cases = {
            "hi": "greeting", "Hello!": "greeting", "hey there": "greeting",
            "good morning": "greeting", "yo": "greeting",
            "bye": "farewell", "see you": "farewell", "that's all": "farewell",
            "thanks": "thanks", "thank you so much": "thanks", "ty": "thanks",
            "what can you do": "capability", "help": "capability",
            "how does this work": "capability", "what is this": "capability",
            "who are you": "identity", "are you a bot": "identity",
            "what's your name": "identity", "are you chatgpt": "identity",
            "how are you": "wellbeing", "ok": "affirm", "got it": "affirm",
            "tell me a joke": "off_topic", "2+2": "off_topic",
        }
        for text, expected in cases.items():
            self.assertEqual(server_module.Demo.classify_small_talk(text), expected,
                             msg=text)

    def test_the_scored_frames_are_never_small_talk(self) -> None:
        """The one property that keeps this out of the graded path."""
        for message in (
            "I'm looking for Accessories Belts, but I'm still exploring.",
            "I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
            "For that, what matters is: Buckle closure.",
            "Actually, ignore my earlier preference. What I need is: leather.",
            "I don't have an additional preference for other.",
            "Those options are not quite right yet.",
        ):
            self.assertIsNone(server_module.Demo.classify_small_talk(message), msg=message)

    def test_a_search_is_never_small_talk(self) -> None:
        for message in ("leather belt", "black", "no", "none of these",
                        "something warm for winter", "cheaper"):
            self.assertIsNone(server_module.Demo.classify_small_talk(message), msg=message)

    def test_a_greeting_costs_no_turn(self) -> None:
        """Ten turns is a hard limit and exceeding it scores zero, so a hello
        must not be able to spend one."""
        self.post("/api/copilot/reset", {"session_id": "sm1", "mode": "shopper"})
        hello = self.post("/api/copilot/chat",
                          {"session_id": "sm1", "message": "hi", "assist": True})
        self.assertEqual(hello["kind"], "small_talk")
        self.assertEqual(hello["turn"], 0)
        self.assertEqual(hello["turns_remaining"], 10)
        self.assertFalse(hello["counted"])
        self.assertTrue(hello["message"])
        self.assertTrue(hello["chips"], msg="a reply with nowhere to go is a dead end")

        first = self.post("/api/copilot/chat",
                          {"session_id": "sm1", "message": "leather belt", "assist": True})
        self.assertEqual(first["turn"], 1, msg="the hello must not have consumed a turn")

    def test_small_talk_never_disturbs_a_session_in_progress(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "sm2", "mode": "shopper"})
        before = self.post("/api/copilot/chat",
                           {"session_id": "sm2", "message": "leather belt", "assist": True})
        self.post("/api/copilot/chat",
                  {"session_id": "sm2", "message": "thanks!", "assist": True})
        after = self.post(
            "/api/copilot/chat",
            {"session_id": "sm2", "message": "For that, what matters is: Buckle closure."})
        self.assertEqual(after["turn"], 2, msg="the thanks sat between turns 1 and 2")
        self.assertEqual(after["constraints"]["category"], before["constraints"]["category"])

    def test_it_knows_its_own_name(self) -> None:
        """A shopping agent asked who it is should answer, not recite a feature
        list — and it should not claim to be a person."""
        self.post("/api/copilot/reset", {"session_id": "id1", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "id1", "message": "who are you", "assist": True})
        self.assertEqual(data["intent"], "identity")
        self.assertIn(server_module.NAME, data["message"])
        self.assertNotIn("I am a person", data["message"])

    def test_a_reply_mid_session_says_where_the_session_is(self) -> None:
        """"Hi" before a search and "hi" during one are different questions."""
        self.post("/api/copilot/reset", {"session_id": "sm3", "mode": "shopper"})
        cold = self.post("/api/copilot/chat",
                         {"session_id": "sm3", "message": "hello", "assist": True})
        opened = self.post("/api/copilot/chat",
                           {"session_id": "sm3", "message": "leather belt", "assist": True})
        warm = self.post("/api/copilot/chat",
                         {"session_id": "sm3", "message": "hello", "assist": True})
        self.assertNotEqual(cold["message"], warm["message"])
        self.assertIn(opened["constraints"]["category"], warm["message"])

    def test_the_scored_path_never_sees_small_talk(self) -> None:
        """Without ``assist`` the message reaches the agent whatever it says."""
        self.post("/api/copilot/reset", {"session_id": "sm4", "mode": "scored"})
        data = self.post("/api/copilot/chat", {"session_id": "sm4", "message": "hi"})
        self.assertNotIn("kind", data)
        self.assertEqual(data["turn"], 1)
        self.assertTrue(data["results"])


class TestHowPeopleActuallyTalk(ServerTestCase):
    """Three things a person does that the simulator never does, each of which
    was silently mishandled until a full conversation was driven end to end."""

    def test_a_change_of_subject_is_not_the_override_frame(self) -> None:
        """"Actually, ignore my earlier preference. What I need is: leather." is
        the frame. "actually I want sneakers instead" is a change of subject,
        and matching frames by their first word swallowed it whole — leaving
        the shopper refining necklaces after asking for shoes."""
        self.assertFalse(server_module.Demo.is_customer_frame(
            "actually I want sneakers instead"))
        self.assertTrue(server_module.Demo.is_customer_frame(
            "Actually, ignore my earlier preference. What I need is: leather."))

        self.post("/api/copilot/reset", {"session_id": "hp1", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "hp1", "message": "a gift", "assist": True})
        held = first["constraints"]["category"]
        noun = max(self.demo._words(self.demo._category_index()[0][0]), key=len)
        switched = self.post("/api/copilot/chat",
                             {"session_id": "hp1",
                              "message": f"actually I want {noun} instead", "assist": True})
        if switched["constraints"]["category"] == held:
            self.skipTest("that noun resolves to the category already held")
        self.assertIsNotNone(switched["assist"])
        self.assertEqual(switched["assist"]["switched_from"], held)

    def test_a_rejection_said_loosely_reaches_the_agent_as_a_rejection(self) -> None:
        """The parser reads exactly one phrasing; a person has twenty. Left
        alone, "none of these" was searched for the word "these"."""
        for text in ("none of these", "no", "not quite", "nope",
                     "show me something else", "try again", "nothing here"):
            self.assertEqual(server_module.Demo.as_dissatisfaction(text),
                             server_module.Demo.DISSATISFIED, msg=text)
        for text in ("leather belt", "a gift", "For that, what matters is: leather.",
                     "Those options are not quite right yet."):
            self.assertIsNone(server_module.Demo.as_dissatisfaction(text), msg=text)

    def test_a_loose_rejection_is_reported_not_applied_silently(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "hp2", "mode": "shopper"})
        self.post("/api/copilot/chat",
                  {"session_id": "hp2", "message": "leather belt", "assist": True})
        data = self.post("/api/copilot/chat",
                         {"session_id": "hp2", "message": "none of these", "assist": True})
        self.assertEqual(data["rewrote_as"], server_module.Demo.DISSATISFIED)
        self.assertEqual(data["unmatched"], [],
                         msg="it must not be searched for the word “these”")
        self.assertEqual(data["turn"], 2)

    def test_two_utterances_in_one_message_are_still_conversation(self) -> None:
        """"thanks, bye" is a goodbye. Patterns anchored at both ends match
        neither half, so it used to be searched for."""
        for text, expected in (("thanks, bye", "farewell"),
                               ("ok thanks", "thanks"),
                               ("hi there, how are you", "wellbeing"),
                               ("ok cool, see you", "farewell"),
                               ("hey what can you do", "capability")):
            self.assertEqual(server_module.Demo.classify_small_talk(text), expected,
                             msg=text)

    def test_a_greeting_glued_to_a_search_is_still_a_search(self) -> None:
        """Every word has to be accounted for, or the message is a query."""
        for text in ("thanks now show me boots", "cool belts", "hi leather belt",
                     "yes please show me more"):
            self.assertIsNone(server_module.Demo.classify_small_talk(text), msg=text)


class TestNoiseCategories(ServerTestCase):
    """Amazon's category tree carries campaign and housekeeping nodes beside
    real product types. They match word-for-word and are useless to shop in."""

    def categories(self, query: str) -> list:
        return self.get(f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]

    def test_the_markers_select_only_merchandising(self) -> None:
        for name in ("Shoes & Jewelry Westlake", "Shoes & Jewelry Top 50 by Product Type",
                     "Men's Watches Under $50", "Up to 30% off Shoes Handbags and More",
                     "Girls Sneakers (fs no puma)", "Swimwear TEST Women's Swimwear",
                     "Shoes & Jewelry MFN ONLY V2"):
            self.assertTrue(self.demo._is_noise_category(name), msg=name)

    def test_real_product_types_are_never_marked(self) -> None:
        for name in ("Accessories Belts", "Watches Wrist Watches", "Shoes Fashion Sneakers",
                     "Necklaces Pendant Necklaces", "Gloves & Mittens Cold Weather Gloves",
                     "Shoes & Jewelry", "Dresses Casual", "Women Wigs"):
            self.assertFalse(self.demo._is_noise_category(name), msg=name)

    def test_a_campaign_never_leads_the_list(self) -> None:
        for query in ("", "watch", "sneakers", "shoes", "dress"):
            listed = self.categories(query)
            if not listed:
                continue
            self.assertFalse(listed[0]["noise"], msg=f"{query} -> {listed[0]['category']}")

    def test_a_campaign_is_demoted_and_not_deleted(self) -> None:
        """Hiding a category that is genuinely the only match would be worse
        than showing it, so the demotion has to be a sort key, not a filter."""
        noise = next((name for name, _tokens, count in self.demo._category_index()
                      if self.demo._is_noise_category(name) and count > 0), None)
        if noise is None:
            self.skipTest("no merchandising leaf in this catalog subset")
        ranked = [name for name, _count in self.demo.rank_categories(noise)]
        self.assertIn(noise, ranked)

    def test_a_demographic_word_alone_cannot_place_a_category(self) -> None:
        """"mens watch" must not offer Men Jeans, which matches only "mens"."""
        for row in self.categories("mens watch"):
            self.assertTrue(
                self.demo._variants("watch") & self.demo._token_set(row["category"])
                or row["via"],
                msg=row["category"])


class TestScenarioHints(ServerTestCase):
    def categories(self, query: str) -> list:
        return self.get(f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]

    def test_an_occasion_reaches_the_catalog(self) -> None:
        """"winter" shares no token with "Gloves & Mittens Cold Weather Gloves"."""
        for query in ("something warm for winter", "a gift for an anniversary",
                      "what to wear to a wedding", "clothes for the gym"):
            listed = self.categories(query)
            self.assertTrue(listed, msg=query)
            self.assertFalse(listed[0]["noise"], msg=query)

    def test_an_inferred_match_says_so(self) -> None:
        """A leap the shopper did not ask for has to be visible as a leap."""
        listed = self.categories("a gift for an anniversary")
        self.assertTrue(listed)
        self.assertEqual(listed[0]["kind"], "related")
        self.assertIn(listed[0]["via"], ("gift", "anniversary"))

    def test_a_named_category_always_outranks_an_inferred_one(self) -> None:
        """The hint breaks ties; it never overrules a word actually typed.

        Asserted as an ordering invariant rather than against one expected
        category, so it holds on the 2,500-product test subset as well as on
        the full catalog.
        """
        for query in ("warm winter gloves", "a gift watch", "gym sneakers",
                      "wedding dress", "beach sandals"):
            # Compared within a tier: a three-product direct match is still
            # ranked below a real category, which is the size rule doing its
            # job and not the hint overruling anything.
            for tier in (False, True):
                kinds = [row["kind"] for row in self.categories(query)
                         if row["thin"] is tier and not row["noise"]]
                if "match" not in kinds or "related" not in kinds:
                    continue
                self.assertLess(kinds.index("match"), kinds.index("related"),
                                msg=f"{query} (thin={tier})")

    def test_hints_cannot_conjure_a_catalog_that_does_not_exist(self) -> None:
        for query in ("headphone", "laptop", "bluetooth", "under 100 dollars", "xyzzy"):
            self.assertEqual(self.categories(query), [], msg=query)


class TestOpeners(ServerTestCase):
    def test_the_dropdown_sends_exactly_what_it_previews(self) -> None:
        """The row's opener and the rewrite must be the same string, or the
        page is promising one search and running another."""
        for query in ("leather belt", "gold earrings", "a gift for an anniversary",
                      "black leather belt under $40"):
            rows = self.get(
                f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]
            rewrite = self.demo.assist(query)
            self.assertTrue(rows, msg=query)
            self.assertIsNotNone(rewrite, msg=query)
            self.assertEqual(rows[0]["opener"], rewrite["message"], msg=query)

    def test_a_typed_price_becomes_the_catalog_s_own_budget_phrase(self) -> None:
        """A bare "40" left in the requirement matches nothing and reads as a
        feature; the catalog writes prices as "budget around $24.99"."""
        for query, expected in (("under $40", "budget around $40"),
                                ("under 40 dollars", "budget around $40"),
                                ("$19.99", "budget around $19.99"),
                                ("budget of 60", "budget around $60")):
            self.assertEqual(self.demo.budget_in(query), expected, msg=query)
        for query in ("a belt", "100% leather", ""):
            self.assertIsNone(self.demo.budget_in(query), msg=query)

    def test_the_requirement_never_carries_a_bare_numeral(self) -> None:
        opener = self.demo.opener_for("Accessories Belts", "black leather belt under 50")
        self.assertIn("black leather", opener["requirement"])
        self.assertIn("budget around $50", opener["requirement"])
        self.assertNotIn(" 50", opener["requirement"].replace("$50", ""))

    def test_an_occasion_word_is_not_echoed_back_as_a_requirement(self) -> None:
        """"gift" produced the jewellery; repeating it as "a key requirement is:
        gift" would be nonsense.

        Two occasion words can imply the same category, so this also pins the
        de-duplication: crediting only the first source left "anniversary"
        stranded in the requirement.
        """
        for query in ("a gift for an anniversary", "something warm for winter",
                      "clothes for the gym"):
            for row in self.get(
                    f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]:
                if row["kind"] != "related" or not row["requirement"]:
                    continue
                self.assertNotIn(row["via"], row["requirement"].casefold(),
                                 msg=f"{query} -> {row['category']}")

    def test_every_previewed_opener_is_a_frame_the_parser_reads(self) -> None:
        for query in ("", "belt", "gift", "warm winter gloves"):
            for row in self.get(
                    f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]:
                self.assertTrue(row["opener"].startswith("I'm looking for "), msg=row)
                self.assertIn(row["category"], row["opener"], msg=row)


class TestRefinementPanel(ServerTestCase):
    def open_session(self, session_id: str) -> dict:
        self.post("/api/copilot/reset", {"session_id": session_id, "mode": "shopper"})
        return self.post("/api/copilot/chat",
                         {"session_id": session_id, "message": "leather belt", "assist": True})

    def refinements(self, session_id: str, query: str = "") -> dict:
        return self.get(f"/api/copilot/refinements?session_id={session_id}"
                        f"&q={urllib.parse.quote(query)}")

    def test_nothing_is_offered_before_a_category_is_held(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "rf0", "mode": "shopper"})
        data = self.refinements("rf0")
        self.assertFalse(data["ready"])
        self.assertEqual(data["values"], [])

    def test_every_offered_value_is_really_disclosable(self) -> None:
        """A suggestion the catalog cannot disclose teaches the agent nothing."""
        self.open_session("rf1")
        data = self.refinements("rf1")
        self.assertTrue(data["ready"])
        self.assertTrue(data["values"])
        _key, pool = self.demo.catalog.bucket_for(data["category"])
        available = set()
        for doc in pool[:4000]:
            available.update(self.demo.catalog.card_keys[doc])
        for row in data["values"]:
            self.assertIn(row["value"], available, msg=row["value"])
            self.assertTrue(row["say"].startswith("For that, what matters is: "))

    def test_answers_to_the_live_question_come_first(self) -> None:
        opened = self.open_session("rf2")
        data = self.refinements("rf2")
        asked = opened["ask_attribute"]
        self.assertEqual(data["asked"], asked)
        if asked and asked != "other" and data["values"]:
            self.assertEqual(data["values"][0]["attribute"], asked)

    def test_nothing_already_said_is_offered_back(self) -> None:
        """``disclosed_keys`` only records exact card-key matches, so a stated
        "black leather" leaves the key "leather" looking undisclosed."""
        self.post("/api/copilot/reset", {"session_id": "rf3", "mode": "shopper"})
        self.post("/api/copilot/chat",
                  {"session_id": "rf3", "message": "a black leather belt", "assist": True})
        for row in self.refinements("rf3")["values"]:
            self.assertNotIn(row["value"].casefold(), ("leather", "black"))

    def test_the_query_filters_the_offer(self) -> None:
        self.open_session("rf4")
        everything = self.refinements("rf4")["values"]
        if not everything:
            self.skipTest("nothing disclosable in this catalog subset")
        needle = everything[0]["value"][:4]
        for row in self.refinements("rf4", needle)["values"]:
            self.assertIn(needle.casefold(), row["value"].casefold())

    def test_the_standing_actions_are_frames_the_parser_reads(self) -> None:
        self.open_session("rf5")
        data = self.refinements("rf5")
        says = [a["say"] for a in data["actions"] if a.get("say")]
        self.assertIn("Those options are not quite right yet.", says)
        for say in says:
            self.assertTrue(
                say.startswith(("Those options", "I don't have an additional preference")),
                msg=say)


class TestFramesSurviveAssist(ServerTestCase):
    """The page sends ``assist`` on every turn, including the ones where the
    customer is answering the agent's own question. Every assist-layer feature
    therefore has to agree on what a frame is, and originally only the rewrite
    did -- which cost a whole session, silently, the moment anyone clicked a
    follow-up chip."""

    def test_the_frames_are_recognised(self) -> None:
        for message in (
            "I'm looking for Accessories Belts, but I'm still exploring.",
            "For that, what matters is: color: gold.",
            "Actually, ignore my earlier preference. What I need is: leather.",
            "I don't have an additional preference for other.",
            "Those options are not quite right yet.",
        ):
            self.assertTrue(server_module.Demo.is_customer_frame(message), msg=message)
        for message in ("leather belt", "for men", "a gift", "", "brown"):
            self.assertFalse(server_module.Demo.is_customer_frame(message), msg=message)

    def test_answering_a_question_never_restarts_the_session(self) -> None:
        """The content words of "For that, what matters is: color: gold." name a
        different category than the one being refined. Read as a search, that is
        a change of subject; read as a frame, it is an answer."""
        self.post("/api/copilot/reset", {"session_id": "fr1", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "fr1", "message": "a gift", "assist": True})
        held = first["constraints"]["category"]
        self.assertTrue(held)
        for reply in ("For that, what matters is: color: gold.",
                      "I don't have an additional preference for material.",
                      "Those options are not quite right yet."):
            data = self.post("/api/copilot/chat",
                             {"session_id": "fr1", "message": reply, "assist": True})
            self.assertIsNone(data["assist"], msg=reply)
            self.assertEqual(data["constraints"]["category"], held, msg=reply)
            self.assertTrue(data["constraints"]["category_exact"], msg=reply)
            self.assertEqual(data["unmatched"], [], msg=reply)

    def test_a_disclosure_advances_the_turn_counter(self) -> None:
        """A session silently reset reads as turn 1 forever, which is exactly
        how this went unnoticed."""
        self.post("/api/copilot/reset", {"session_id": "fr2", "mode": "shopper"})
        self.post("/api/copilot/chat",
                  {"session_id": "fr2", "message": "leather belt", "assist": True})
        second = self.post("/api/copilot/chat",
                           {"session_id": "fr2",
                            "message": "For that, what matters is: Buckle closure.",
                            "assist": True})
        self.assertEqual(second["turn"], 2)

    def test_clicking_an_offered_answer_narrows_the_search(self) -> None:
        """End to end through the exact strings the page sends."""
        self.post("/api/copilot/reset", {"session_id": "fr3", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "fr3", "message": "leather belt", "assist": True})
        offered = self.get("/api/copilot/refinements?session_id=fr3&q=")["values"]
        if not offered:
            self.skipTest("nothing disclosable in this catalog subset")
        second = self.post("/api/copilot/chat",
                           {"session_id": "fr3", "message": offered[0]["say"], "assist": True})
        self.assertEqual(second["turn"], 2)
        self.assertGreater(len(second["constraints"]["phrases"]),
                           len(first["constraints"]["phrases"]))


class TestSuggestions(ServerTestCase):
    def test_both_shapes_are_offered(self) -> None:
        data = self.get("/api/copilot/suggestions")
        self.assertTrue(data["examples"])
        self.assertTrue(data["natural"])

    def test_no_human_phrasing_dead_ends(self) -> None:
        """An empty state that hands you an unanswerable query is worse than an
        empty state with no chips at all."""
        for item in self.get("/api/copilot/suggestions")["natural"]:
            ranked = self.demo.rank_categories_detailed(item["text"])
            self.assertTrue(ranked, msg=item["text"])
            self.assertEqual(ranked[0]["category"], item["resolves_to"], msg=item["text"])
            self.assertFalse(self.demo._is_noise_category(item["resolves_to"]), msg=item)

    def test_the_scored_examples_stay_in_the_evaluator_s_frame(self) -> None:
        for text in self.get("/api/copilot/suggestions")["examples"]:
            self.assertTrue(text.startswith("I'm looking for "), msg=text)
            self.assertIsNone(self.demo.assist(text), msg=text)


if __name__ == "__main__":
    unittest.main()
