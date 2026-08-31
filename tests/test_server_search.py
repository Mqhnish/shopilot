"""The search surface that makes the demo usable by a person.

The agent is scored against a simulator that always names a coarse category on
turn 1. A human typing into a search box does not, and without a category the
candidate pool is all 50,000 products and the ranking is close to meaningless.
The demo therefore resolves a category first, offers concrete answers to the
question it asks, and returns a full list rather than the single-product probe
that the scored configuration uses.

None of that touches ``src/``. These tests pin the three properties that make
it honest rather than merely helpful:

* the typeahead and the rewrite can never disagree, so the page cannot show one
  category and search another;
* every offered follow-up is a constraint the catalog can really disclose;
* a message already in a customer frame is passed through untouched, so the
  scored path is reachable exactly as the evaluator drives it.
"""

from __future__ import annotations

import unittest
import urllib.parse

from tests.test_server import ServerTestCase


class TestTypeahead(ServerTestCase):
    def categories(self, query: str) -> list:
        return self.get(f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]

    def test_multiword_queries_match(self) -> None:
        # Substring matching alone fails here: "leather belt" does not appear
        # inside "Accessories Belts".
        names = [c["category"] for c in self.categories("leather belt")]
        self.assertTrue(names)
        self.assertEqual(names[0], "Accessories Belts")

    def test_plurals_match_in_both_directions(self) -> None:
        for query in ("necklace", "necklaces", "watch", "watches"):
            self.assertTrue(self.categories(query), msg=query)

    def test_promotional_leaves_are_demoted(self) -> None:
        # A five-product "Men's Watches Under $50" matches "mens watch" twice
        # over and is useless to search inside; the 1,034-product watch
        # category is what was meant.
        names = [c["category"] for c in self.categories("mens watch")]
        self.assertTrue(names)
        self.assertNotIn("Under $50", names[0])
        self.assertGreaterEqual(
            self.categories("mens watch")[0]["count"],
            self.demo.MIN_ASSIST_POOL,
        )

    def test_demographic_words_do_not_win_alone(self) -> None:
        """A query must resolve on its noun, not on "womens".

        Asserted against whatever the indexed catalog actually holds, so this
        does not quietly become a test about the size of the test fixture.
        """
        noun, expected = self.distinctive_category()
        names = [c["category"] for c in self.categories(f"womens {noun}")]
        self.assertTrue(names)
        self.assertEqual(names[0], expected)

    def distinctive_category(self) -> tuple:
        """A big category plus a word that appears only in its own name.

        Merchandising slices are skipped when picking the fixture: "Shoes &
        Jewelry Westlake" is 1,136 products and owns the word "westlake"
        outright, but it is a campaign bucket that the matcher deliberately
        demotes below every real product type, so it cannot be the expected
        answer to anything.
        """
        index = self.demo._category_index()
        big = sorted(index, key=lambda item: -item[2])
        for name, _tokens, count in big:
            if count < self.demo.MIN_ASSIST_POOL or self.demo._is_noise_category(name):
                continue
            for word in self.demo._words(name):
                if word in self.demo._WEAK or word in self.demo._STOP or len(word) < 4:
                    continue
                owners = [
                    other for other, tokens, size in index
                    if size >= self.demo.MIN_ASSIST_POOL
                    and not self.demo._is_noise_category(other)
                    and self.demo._variants(word) & tokens
                ]
                if owners == [name]:
                    return word, name
        self.skipTest("no unambiguous category in this catalog subset")

    def test_products_the_catalog_does_not_carry_match_nothing(self) -> None:
        for query in ("headphone", "laptop", "xyzzy"):
            self.assertEqual(self.categories(query), [], msg=query)

    def test_empty_query_lists_the_largest_categories(self) -> None:
        listed = self.categories("")
        self.assertTrue(listed)
        counts = [c["count"] for c in listed]
        self.assertEqual(counts, sorted(counts, reverse=True))


class TestRewrite(ServerTestCase):
    def test_rewrite_agrees_with_the_list_it_shows(self) -> None:
        """A typeahead that disagrees with the rewrite would be a lie."""
        for query in ("leather belt", "mens watch", "gold earrings", "warm winter gloves"):
            listed = self.get(
                f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]
            rewrite = self.demo.assist(query)
            self.assertIsNotNone(rewrite, msg=query)
            self.assertEqual(rewrite["category"], listed[0]["category"], msg=query)

    def test_rewrite_is_a_frame_the_parser_recognises(self) -> None:
        rewrite = self.demo.assist("a black leather belt under $50")
        self.assertIsNotNone(rewrite)
        self.assertTrue(rewrite["message"].startswith("I'm looking for "))
        self.assertIn(rewrite["category"], rewrite["message"])

    def test_customer_frames_are_never_rewritten(self) -> None:
        """The scored path must stay reachable exactly as the evaluator drives it."""
        for message in (
            "I'm looking for Accessories Belts. A key requirement is: 100% Leather.",
            "For that, what matters is: Buckle closure.",
            "Actually, ignore my earlier preference. What I need is: leather.",
            "I don't have an additional preference for other.",
        ):
            self.assertIsNone(self.demo.assist(message), msg=message)

    def test_unresolvable_text_is_left_alone(self) -> None:
        for message in ("headphone", "laptop", ""):
            self.assertIsNone(self.demo.assist(message), msg=message)

    def test_refinements_do_not_restart_the_search(self) -> None:
        """"for men" after a belt search is a refinement, not a new search."""
        self.post("/api/copilot/reset", {"session_id": "r1", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "r1", "message": "leather belt", "assist": True})
        self.assertIsNotNone(first["assist"])
        second = self.post("/api/copilot/chat",
                           {"session_id": "r1", "message": "for men", "assist": True})
        self.assertIsNone(second["assist"], msg="a follow-up must not become a new opener")
        self.assertEqual(second["constraints"]["category"], first["constraints"]["category"])
        # Whether "for men" yields a new constraint depends on the catalog, but
        # the session must carry the ones it already had either way.
        self.assertGreaterEqual(len(second["constraints"]["phrases"]),
                                len(first["constraints"]["phrases"]))


class TestModes(ServerTestCase):
    def test_shopper_mode_returns_the_full_list(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "m1", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "m1", "message": "leather belt", "assist": True})
        self.assertEqual(data["mode"], "shopper")
        self.assertGreater(len(data["results"]), 1,
                           msg="a shopping experience has to offer options")
        self.assertFalse(data["trace"]["trimmed"])

    def test_scored_mode_is_unchanged_and_still_available(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "m2", "mode": "scored"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "m2", "message": "leather belt", "assist": True})
        self.assertEqual(data["mode"], "scored")
        self.assertTrue(self.demo.agent.options.use_truncation)
        self.assertFalse(self.demo.shopper.options.use_truncation)

    def test_both_modes_share_one_index(self) -> None:
        """The second configuration must cost no extra index build or memory."""
        self.assertIs(self.demo.agent.catalog, self.demo.shopper.catalog)
        self.assertIs(self.demo.agent.retriever, self.demo.shopper.retriever)

    def test_replay_always_uses_the_scored_configuration(self) -> None:
        run = self.post("/api/copilot/replay", {"sample_id": self.samples[0]["sample_id"]})
        self.assertEqual(run["turns"][0]["mode"], "scored")


class TestFollowUps(ServerTestCase):
    def test_offered_answers_are_really_disclosable(self) -> None:
        """Every chip must be a string the catalog can disclose, or clicking it
        teaches the agent nothing."""
        self.post("/api/copilot/reset", {"session_id": "f1", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "f1", "message": "leather belt", "assist": True})
        self.assertTrue(data["options"], msg="a question with no answers is a dead end")

        category = data["constraints"]["category"]
        key, pool = self.demo.catalog.bucket_for(category)
        self.assertTrue(pool, msg=key)
        available = set()
        for doc in pool[:4000]:
            available.update(self.demo.catalog.card_keys[doc])
        self.assertTrue(set(data["options"]) <= available,
                        msg=set(data["options"]) - available)

    def test_answers_never_repeat_what_was_already_disclosed(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "f2", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "f2", "message": "leather belt", "assist": True})
        chosen = first["options"][0]
        second = self.post("/api/copilot/chat",
                           {"session_id": "f2",
                            "message": f"For that, what matters is: {chosen}."})
        self.assertNotIn(chosen, second["options"])
        self.assertGreater(len(second["constraints"]["phrases"]),
                           len(first["constraints"]["phrases"]),
                           msg="clicking an answer has to actually narrow the search")

    def test_catalog_housekeeping_is_offered_last(self) -> None:
        """Amazon's `details` carry bookkeeping the simulator really would
        disclose and no shopper would ever volunteer. It sorts last rather than
        being hidden: a click still teaches the agent something, but "is
        discontinued by manufacturer: no" must not lead the list."""
        import server as server_module

        self.post("/api/copilot/reset", {"session_id": "hk1", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "hk1", "message": "leather belt", "assist": True})
        self.assertTrue(data["options"])
        flags = [bool(server_module._HOUSEKEEPING.match(v)) for v in data["options"]]
        self.assertEqual(flags, sorted(flags),
                         msg=f"housekeeping is not last: {data['options']}")
        self.assertFalse(flags[0], msg=f"housekeeping led the list: {data['options']}")

    def test_profile_tags_are_reported_only_where_they_land(self) -> None:
        profile = {"preference_tags": ["leather", "comfort"], "average_prior_rating": 4.5,
                   "purchase_frequency": "3-4 prior purchases",
                   "rating_style": "usually positive", "summary": "x"}
        self.post("/api/copilot/reset",
                  {"session_id": "f3", "profile": profile, "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "f3", "message": "leather belt", "assist": True})
        self.assertEqual(data["profile_tags"], ["leather", "comfort"])
        self.assertGreater(data["profile_weight"], 0.0)
        for card in data["results"]:
            text = f"{card.get('title') or ''} {card.get('blurb') or ''}".lower()
            for tag in card.get("matched_tags", []):
                self.assertIn(tag, text,
                              msg="a claimed match must be in the product's own text")


if __name__ == "__main__":
    unittest.main()


class TestTopicChange(ServerTestCase):
    """Refinements and topic changes look identical over HTTP and must not be
    treated the same. The agent locks its category slot for the life of a
    session by design, so a real change of subject has to start a new one."""

    def leading_category(self) -> str:
        index = self.demo._category_index()
        big = max(index, key=lambda item: item[2])
        return big[0]

    def test_a_constraint_word_refines(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "tc1", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "tc1", "message": "leather belt", "assist": True})
        for follow_up in ("brown", "for men", "cheaper"):
            data = self.post("/api/copilot/chat",
                             {"session_id": "tc1", "message": follow_up, "assist": True})
            self.assertIsNone(data["assist"], msg=follow_up)
            self.assertEqual(data["constraints"]["category"],
                             first["constraints"]["category"], msg=follow_up)
            self.assertGreater(data["turn"], 1, msg="a refinement continues the session")

    def test_naming_another_category_starts_over(self) -> None:
        target = self.leading_category()
        noun = max(self.demo._words(target), key=len)
        self.post("/api/copilot/reset", {"session_id": "tc2", "mode": "shopper"})
        first = self.post("/api/copilot/chat",
                          {"session_id": "tc2", "message": "leather belt", "assist": True})
        if first["constraints"]["category"] == target:
            self.skipTest("the opening query already selected the target category")

        data = self.post("/api/copilot/chat",
                         {"session_id": "tc2", "message": f"i want {noun}", "assist": True})
        self.assertIsNotNone(data["assist"], msg="a change of subject must re-resolve")
        self.assertEqual(data["assist"]["switched_from"], first["constraints"]["category"])
        self.assertNotEqual(data["constraints"]["category"],
                            first["constraints"]["category"])
        self.assertEqual(data["turn"], 1, msg="a new subject is a new session")

    def test_demographic_words_never_switch_on_their_own(self) -> None:
        """"for men" names dozens of categories and identifies none of them."""
        self.assertIsNone(self.demo.topic_change("for men", "Accessories Belts"))
        self.assertIsNone(self.demo.topic_change("for women", "Accessories Belts"))

    def test_restating_the_held_category_is_not_a_change(self) -> None:
        self.assertIsNone(self.demo.topic_change("belts", "Accessories Belts"))
        self.assertIsNone(self.demo.topic_change("a wider belt", "Accessories Belts"))

    def test_unknown_words_are_constraints_not_subjects(self) -> None:
        for message in ("brown", "cheaper", "something wider", "headphone"):
            self.assertIsNone(self.demo.topic_change(message, "Accessories Belts"),
                              msg=message)


class TestOutOfCatalogue(ServerTestCase):
    """The catalog is Clothing, Shoes & Jewelry and nothing else, and the
    scored task can never ask for anything outside it -- the hidden target is
    always a real catalog product. A person at a search box has no such
    guarantee, so the demo has to fail visibly rather than quietly answer a
    question it cannot answer."""

    def categories(self, query: str) -> list:
        return self.get(f"/api/copilot/categories?q={urllib.parse.quote(query)}")["categories"]

    def test_electronics_resolve_to_nothing(self) -> None:
        for query in ("phone", "mobile phone", "headphone", "bluetooth", "wireless"):
            self.assertEqual(self.categories(query), [], msg=query)
            self.assertIsNone(self.demo.assist(query), msg=query)

    def test_a_price_phrase_is_not_a_category(self) -> None:
        """"under 100 dollars" names no product, and must not fall through to
        the largest category on the shelf."""
        self.assertEqual(self.categories("under 100 dollars"), [])
        self.assertIsNone(self.demo.assist("under 100 dollars"))

    def test_the_session_never_claims_a_category_it_lacks(self) -> None:
        self.post("/api/copilot/reset", {"session_id": "oc1", "mode": "shopper"})
        for turn, message in enumerate(("headphone", "wireless", "bluetooth"), start=1):
            data = self.post("/api/copilot/chat",
                             {"session_id": "oc1", "message": message, "assist": True})
            self.assertFalse(data["constraints"]["category_exact"], msg=message)
            self.assertEqual(data["constraints"]["category_pool"], 0, msg=message)
            self.assertEqual(data["turn"], turn, msg="an unresolved query is not a topic change")

    def test_an_unresolved_session_still_answers_safely(self) -> None:
        """Degrading to text matching is fine; raising or returning nothing is not."""
        self.post("/api/copilot/reset", {"session_id": "oc2", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "oc2", "message": "headphone", "assist": True})
        self.assertTrue(data["results"])
        self.assertTrue(data["message"])


class TestSpelling(ServerTestCase):
    def test_misspellings_are_repaired(self) -> None:
        for typo, expected in (("snekaers", "sneakers"), ("neckalce", "necklaces"),
                               ("braclet", "bracelets"), ("sunglases", "sunglasses")):
            fixes = self.demo.corrections(typo)
            if expected in self.demo._category_vocab():
                self.assertEqual(fixes.get(typo), expected, msg=typo)

    def test_words_the_catalog_really_uses_are_never_corrected(self) -> None:
        """Edit distance alone would turn "dollars" into "dolls" and send the
        shopper to baby-doll lingerie. Document frequency is what stops it."""
        for word in ("dollars", "bluetooth", "wireless", "phone"):
            self.assertEqual(self.demo.corrections(word), {}, msg=word)

    def test_listing_typos_do_not_count_as_real_words(self) -> None:
        """Amazon listings contain their own misspellings, so mere presence in
        the catalog text is not enough -- it has to be common."""
        vocab, df = self.demo.catalog.vocab, self.demo.catalog.df
        for word in ("neckalce", "braclet"):
            index = vocab.get(word)
            if index is not None:
                self.assertLess(df[index], self.demo.REAL_WORD_DF, msg=word)
                self.assertFalse(self.demo._is_real_word(word), msg=word)


class TestSimilar(ServerTestCase):
    """"More like this" reuses the agent's own vector route rather than adding
    a second retrieval system, and must never recommend the item you are
    already looking at."""

    def a_product(self) -> str:
        self.post("/api/copilot/reset", {"session_id": "sim0", "mode": "shopper"})
        data = self.post("/api/copilot/chat",
                         {"session_id": "sim0", "message": "leather belt", "assist": True})
        return data["results"][0]["parent_asin"]

    def test_similar_items_are_real_and_distinct(self) -> None:
        asin = self.a_product()
        found = self.get(f"/api/copilot/similar?asin={asin}")["similar"]
        self.assertTrue(found)
        for card in found:
            self.assertIn(card["parent_asin"], self.demo.catalog.index_of)
            self.assertNotEqual(card["parent_asin"], asin,
                                msg="an item is not similar to itself")
            self.assertTrue(card["title"])
        self.assertEqual(len({c["parent_asin"] for c in found}), len(found))

    def test_similar_items_share_the_category(self) -> None:
        asin = self.a_product()
        doc = self.demo.catalog.index_of[asin]
        category = self.demo.catalog.coarse[doc]
        for card in self.get(f"/api/copilot/similar?asin={asin}")["similar"]:
            self.assertEqual(card["category"], category)

    def test_an_unknown_asin_returns_nothing_rather_than_failing(self) -> None:
        self.assertEqual(self.get("/api/copilot/similar?asin=NOTREAL")["similar"], [])
        self.assertEqual(self.get("/api/copilot/similar?asin=")["similar"], [])
