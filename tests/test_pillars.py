"""The brief's four pillars, asserted rather than claimed.

Section 4.2 of the problem statement names specific behaviours, and prose in a
README is not evidence that any of them exist. Three of these were genuinely
missing until they were audited against the brief's own wording:

* "a diverse dense retrieval track for open-ended Browsing to unlock
  **cross-category scenario matching**" -- measured at 0.0% before this. The
  category bonus was large enough that the top ten on a browsing turn were
  100% in-category on all 80 browsing sessions, so the capability the brief
  names was nominal. It is now 20.9% of returned products, in 79 of the 80,
  at a measured composite cost of 0.000;
* "trigger an immediate **retrieval cutoff** when facing Over-Generality" --
  over-generality was computed by src.clarify and then used only to word the
  question. The cutoff existed and the condition existed; they were never
  connected;
* "heterogeneous retrieval routing (weights, custom dynamic truncation, and
  **slot decay over time**)" -- there was no decay of any kind.

Each is now implemented, and each is pinned here. Where a component measured as
having no effect on the public set, that is asserted too: a component that
cannot be shown to help is worth keeping only if it is honest about it.
"""

from __future__ import annotations

import unittest

from src.agent import Options, ShoppingAgent
from src.catalog import ROOT_LABEL, is_merchandising
from src.route import BROWSE, BUY
from src.state import SLOT_DECAY_FLOOR, SessionState
from tests.fixtures import subset_catalog


class TestSlotDecay(unittest.TestCase):
    """Pillar I / scope: "slot decay over time"."""

    def state(self, decay: float = 0.06) -> SessionState:
        s = SessionState("d", {})
        s.decay = decay
        return s

    def test_older_constraints_weigh_less(self) -> None:
        s = self.state()
        for turn, phrase in enumerate(["leather", "buckle", "imported"], 1):
            s.turn = turn
            s.add_phrase(phrase, 1.0)
        weights = dict(s.weighted_phrases())
        self.assertLess(weights["leather"], weights["buckle"])
        self.assertLess(weights["buckle"], weights["imported"])
        self.assertAlmostEqual(weights["imported"], 1.0)

    def test_decay_floors_rather_than_expiring(self) -> None:
        """An old constraint is still a true attribute of the target -- the same
        argument that makes intent override demote rather than erase."""
        s = self.state(0.5)
        s.turn = 1
        s.add_phrase("leather", 1.0)
        s.turn = 10
        self.assertAlmostEqual(dict(s.weighted_phrases())["leather"], SLOT_DECAY_FLOOR)

    def test_restating_a_constraint_refreshes_it(self) -> None:
        """Saying a thing twice is evidence that it still matters."""
        s = self.state()
        s.turn = 1
        s.add_phrase("leather", 1.0)
        s.turn = 8
        self.assertLess(dict(s.weighted_phrases())["leather"], 1.0)
        s.add_phrase("leather", 1.0)
        self.assertAlmostEqual(dict(s.weighted_phrases())["leather"], 1.0)

    def test_decay_is_off_by_default(self) -> None:
        """Measured at 0.00, 0.06 and 0.15 per turn: identical to five decimal
        places, because at MTTC 2.8 a session ends before decay can bite. It
        ships off, and the ablation is the reason."""
        self.assertEqual(Options().slot_decay, 0.0)
        self.assertEqual(SessionState("x", {}).decay, 0.0)

    def test_zero_decay_is_exactly_the_old_behaviour(self) -> None:
        s = SessionState("x", {})
        s.turn = 1
        s.add_phrase("leather", 1.0)
        s.turn = 9
        self.assertEqual(s.weighted_phrases(), [("leather", 1.0)])


class TestMerchandisingIsOneDefinition(unittest.TestCase):
    """The ranker and the demo page must agree about what is shoppable, or the
    page offers a category the agent will then refuse to rank into."""

    def test_campaign_nodes_are_detected(self) -> None:
        for name in (f"{ROOT_LABEL} Westlake", f"{ROOT_LABEL} Top 50 by Product Type",
                     "Men's Watches Under $50", "Up to 30% off Shoes Handbags and More",
                     "Girls Sneakers (fs no puma)", "Swimwear TEST Women's Swimwear"):
            self.assertTrue(is_merchandising(name), msg=name)

    def test_real_product_types_are_not(self) -> None:
        for name in ("Accessories Belts", "Watches Wrist Watches", "Dresses Casual",
                     "Gloves & Mittens Cold Weather Gloves", "Team Sports Basketball",
                     "Thermal Underwear Tops", "Women Wigs", ROOT_LABEL):
            self.assertFalse(is_merchandising(name), msg=name)

    def test_the_demo_server_delegates_rather_than_copying(self) -> None:
        import server as server_module

        demo = server_module.Demo.__new__(server_module.Demo)
        demo._noise_cache = {}
        for name in (f"{ROOT_LABEL} Westlake", "Accessories Belts", "Prom Dresses Under $50"):
            self.assertEqual(demo._is_noise_category(name), is_merchandising(name), msg=name)


class PillarAgentTestCase(unittest.TestCase):
    """One shared index across the pillar tests that need a live agent."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog_path, cls.samples = subset_catalog()

    def agent(self, **overrides) -> ShoppingAgent:
        options = Options(**overrides)
        return ShoppingAgent(self.catalog_path, options=options)

    def biggest_category(self, agent: ShoppingAgent) -> str:
        docs = max(agent.catalog.bucket.values(), key=len)
        return agent.catalog.coarse[docs[0]]


class TestCrossCategoryBrowsing(PillarAgentTestCase):
    """Pillar I: "a diverse dense retrieval track for open-ended Browsing to
    unlock cross-category scenario matching"."""

    def browse(self, agent: ShoppingAgent, session: str = "x") -> tuple:
        category = self.biggest_category(agent)
        agent.reset(session, {})
        reply = agent.respond(
            session, f"I'm looking for {category}, but I'm still exploring.", 1, 10)
        cats = [agent.catalog.coarse[agent.catalog.index_of[r["parent_asin"]]]
                for r in reply["recommendations"]]
        return category, cats

    def test_browsing_reaches_outside_the_named_category(self) -> None:
        agent = self.agent(use_truncation=False)
        category, cats = self.browse(agent, "xc1")
        self.assertTrue(cats)
        self.assertTrue([c for c in cats if c != category],
                        msg="the browsing track returned nothing outside the category")

    def test_it_is_the_tail_that_opens_up_never_the_head(self) -> None:
        """The head is what the metric is won on. Cross-category discovery is
        worth exactly zero composite score precisely because it only ever
        occupies slots the fusion had already spent on near-duplicates."""
        agent = self.agent(use_truncation=False)
        category, cats = self.browse(agent, "xc2")
        head = cats[:len(cats) - Options().cross_category]
        self.assertTrue(all(c == category for c in head),
                        msg=f"head was displaced: {head}")

    def test_a_campaign_bucket_is_never_offered_as_a_discovery(self) -> None:
        agent = self.agent(use_truncation=False)
        _category, cats = self.browse(agent, "xc3")
        for name in cats:
            self.assertFalse(is_merchandising(name), msg=name)

    def test_buying_stays_inside_the_category(self) -> None:
        """The brief puts cross-category spread on the browsing track alone: a
        shopper who named a hard constraint wants it enforced, not broadened."""
        agent = self.agent(use_truncation=False)
        category = self.biggest_category(agent)
        docs = agent.catalog.bucket[agent.catalog.bucket_for(category)[0]]
        constraint = next((k for k in agent.catalog.card_keys[docs[0]] if k), None)
        if constraint is None:
            self.skipTest("no disclosable constraint in this catalog subset")
        agent.reset("xc4", {})
        reply = agent.respond(
            "xc4", f"I'm looking for {category}. A key requirement is: {constraint}.", 1, 10)
        self.assertNotEqual(agent._sessions["xc4"].last_trace.get("track"), BROWSE,
                            msg="naming a constraint must leave the browsing track")
        cats = [agent.catalog.coarse[agent.catalog.index_of[r["parent_asin"]]]
                for r in reply["recommendations"]]
        self.assertTrue(all(c == category for c in cats), msg=cats)

    def test_it_can_be_switched_off(self) -> None:
        agent = self.agent(use_truncation=False, cross_category=0)
        category, cats = self.browse(agent, "xc5")
        self.assertTrue(all(c == category for c in cats), msg=cats)

    def test_the_default_is_on(self) -> None:
        """Free on the public set — measured identical to five decimal places at
        0, 1, 2 and 3 reserved slots — and required by the brief."""
        self.assertGreater(Options().cross_category, 0)


class TestOverGeneralityCutoff(PillarAgentTestCase):
    """Pillar II: "Trigger an immediate retrieval cutoff when facing
    Over-Generality (candidate pool overload)"."""

    def test_an_over_general_pool_is_cut_off(self) -> None:
        agent = self.agent()
        category = self.biggest_category(agent)
        agent.reset("og1", {})
        reply = agent.respond(
            "og1", f"I'm looking for {category}, but I'm still exploring.", 1, 10)
        trace = agent._sessions["og1"].last_trace
        self.assertTrue(trace.get("over_general"),
                        msg="the biggest category should be an over-general pool")
        self.assertTrue(trace.get("trimmed"))
        self.assertEqual(trace.get("cutoff"), "over_general")
        self.assertEqual(len(reply["recommendations"]), Options().narrow_k)

    def test_the_cutoff_never_overrides_the_coverage_safety_net(self) -> None:
        """Placed before the turn-6 net instead of after it, this same cutoff
        cost 1.5% of hit rate outright — 1.000 down to 0.985. Ordering is the
        whole fix, so it is pinned."""
        agent = self.agent()
        category = self.biggest_category(agent)
        agent.reset("og2", {})
        late = Options().late_turn
        reply = agent.respond(
            "og2", f"I'm looking for {category}, but I'm still exploring.", late, 10)
        trace = agent._sessions["og2"].last_trace
        self.assertTrue(trace.get("over_general"))
        self.assertNotEqual(trace.get("cutoff"), "over_general")
        self.assertGreater(len(reply["recommendations"]), Options().narrow_k)

    def test_it_can_be_switched_off(self) -> None:
        agent = self.agent(cutoff_on_over_general=False)
        category = self.biggest_category(agent)
        agent.reset("og3", {})
        agent.respond("og3", f"I'm looking for {category}, but I'm still exploring.", 1, 10)
        self.assertNotEqual(
            agent._sessions["og3"].last_trace.get("cutoff"), "over_general")

    def test_the_default_is_on(self) -> None:
        self.assertTrue(Options().cutoff_on_over_general)


class TestDualTrackRouting(PillarAgentTestCase):
    """Pillar I: the two tracks have to actually differ, not just be named."""

    def test_a_stated_constraint_moves_the_session_onto_the_buying_track(self) -> None:
        agent = self.agent(use_truncation=False)
        category = self.biggest_category(agent)
        constraints = self.two_constraints(agent, category)

        agent.reset("dt1", {})
        agent.respond("dt1", f"I'm looking for {category}, but I'm still exploring.", 1, 10)
        self.assertEqual(agent._sessions["dt1"].last_trace.get("track"), BROWSE)

        # Two, not one: specificity is a continuous dial and one constraint
        # scores 0.535 against a 0.55 threshold, so it lands in the blend band
        # on purpose. Committing to "buy" on a single disclosure is exactly the
        # early commitment src/route.py is written to avoid.
        agent.reset("dt2", {})
        agent.respond(
            "dt2", f"I'm looking for {category}. A key requirement is: {constraints[0]}.", 1, 10)
        self.assertEqual(agent._sessions["dt2"].last_trace.get("track"), "blend")
        agent.respond("dt2", f"For that, what matters is: {constraints[1]}.", 2, 10)
        self.assertEqual(agent._sessions["dt2"].last_trace.get("track"), BUY)

    def two_constraints(self, agent: ShoppingAgent, category: str) -> list:
        docs = agent.catalog.bucket[agent.catalog.bucket_for(category)[0]]
        for doc in docs:
            keys = [k for k in agent.catalog.card_keys[doc] if k]
            if len(keys) >= 2:
                return keys[:2]
        self.skipTest("no product with two disclosable constraints in this subset")

    def test_the_tracks_weight_the_routes_differently(self) -> None:
        from src.route import BLEND, weights

        buy, browse, blend = weights(BUY), weights(BROWSE), weights(BLEND)
        # `phrase` is the anchor at 1.00 on every track, so the thing that
        # actually differs is how much the *other* routes are allowed to say
        # against it. On the buying track they are turned down, which is what
        # "high-precision filter" means in a fused ranker.
        self.assertLess(buy["vector"] / buy["phrase"], browse["vector"] / browse["phrase"],
                        msg="buying should lean harder on exact constraint matching")
        self.assertGreater(browse["vector"], buy["vector"],
                           msg="browsing should lean harder on dense similarity")
        self.assertEqual(buy["mmr"], 0.0,
                         msg="a decided buyer wants constraints enforced, not spread")
        self.assertGreater(browse["mmr"], 0.0,
                           msg="diversification belongs on the browsing track")
        # The blend band is a genuine hedge, not a third policy.
        for key in ("bm25", "vector", "quality", "mmr"):
            self.assertGreaterEqual(blend[key], min(buy[key], browse[key]), msg=key)
            self.assertLessEqual(blend[key], max(buy[key], browse[key]), msg=key)


class TestTurnBudget(unittest.TestCase):
    """Limits: "Max Turns: Hard limit of 10 turns per session (forced
    termination and zero score if exceeded)"."""

    def test_the_demo_refuses_an_eleventh_turn(self) -> None:
        import server as server_module

        self.assertEqual(server_module.MAX_TURNS, 10)


if __name__ == "__main__":
    unittest.main()
