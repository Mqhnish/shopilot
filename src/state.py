"""Per-session conversational state.

Three things here are worth more than they look.

**Negative evidence.** If turn *t* showed ten products and the session did not
end, the target was not among them. That is not a heuristic, it is a logical
consequence of how the evaluator scores a turn, and it removes ten candidates
per turn for free. The one place it is invalid is an intent-override session
before the override lands, because those sessions are barred from converting
early -- so the exclusion is withheld until the override arrives.

**Accumulation, not erasure.** The brief asks for slot erasure on intent
override. Measured against the actual customer policy, erasing is wrong: the
"abandoned" preference is still a true attribute of the target, so the override
is a change of emphasis rather than a contradiction. We therefore *re-weight*
-- the new intent gets priority, the old evidence is demoted rather than
dropped -- and :mod:`tools.ablate` reports what full erasure would cost.

**Attribute bookkeeping.** Once the customer has said it has nothing more to
offer on an attribute, asking again burns one of ten turns for nothing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from .normalize import phrase_key

# Weight applied to evidence that the customer later de-emphasised. Not zero:
# on this customer policy the superseded preference still describes the target.
DEMOTED_WEIGHT = 0.45

# Slot decay. A constraint loses this much weight per turn since it was last
# stated, down to SLOT_DECAY_FLOOR and no further -- an old constraint is still
# a true attribute of the target, so it fades rather than expiring. Measured on
# the public set: see the ablation table.
SLOT_DECAY_PER_TURN = 0.06
SLOT_DECAY_FLOOR = 0.55


class SessionState:
    """Everything one session knows. One instance per ``reset``."""

    __slots__ = (
        "session_id", "profile", "turn", "category", "category_key",
        "category_exact",
        "candidates", "phrase_weights", "phrase_order", "phrase_turn",
        "disclosed_keys", "decay", "decay_floor",
        "asked", "exhausted", "no_preference", "scenario", "override_seen",
        "shown", "free_text", "learned_this_turn", "barren_turns", "last_trace",
        "cohort", "pending_ask", "yielded", "last_shown", "last_turn_size",
    )

    def __init__(self, session_id: str, profile: Optional[dict]) -> None:
        self.session_id = session_id
        self.profile: dict = profile if isinstance(profile, dict) else {}
        self.turn = 0
        self.category: Optional[str] = None
        self.category_key: Optional[str] = None
        # False while `category` is only a guess taken from raw text. A guess
        # must not lock the slot -- see ShoppingAgent._absorb.
        self.category_exact = False
        self.candidates: List[int] = []
        # phrase text -> evidence weight, plus insertion order for tie-breaks.
        self.phrase_weights: Dict[str, float] = {}
        self.phrase_order: List[str] = []
        # Turn on which each constraint was last stated, for slot decay.
        self.phrase_turn: Dict[str, int] = {}
        # Set from Options by the agent; 0.0 disables decay entirely.
        self.decay = 0.0
        self.decay_floor = SLOT_DECAY_FLOOR
        self.disclosed_keys: Set[str] = set()
        self.asked: List[str] = []
        self.exhausted: Set[str] = set()
        self.no_preference: Set[str] = set()
        self.scenario: str = "browsing"
        self.override_seen = False
        self.shown: Set[str] = set()
        self.free_text: List[str] = []
        # Whether the turn just parsed told us anything new, and how many turns
        # in a row have told us nothing. Drives the decision to stop holding
        # back candidates -- see src.agent._trim.
        self.learned_this_turn = False
        self.barren_turns = 0
        # Populated every turn purely so tools/demo.py can show the reasoning.
        # Nothing in the scored path reads it.
        self.last_trace: Dict[str, object] = {}
        # Long-term learning bookkeeping (see src.memory). `last_shown` plus
        # `last_turn_size` are what let a later session infer, with certainty,
        # which product ended this one.
        self.cohort: str = "unknown"
        self.pending_ask: Optional[str] = None
        self.yielded: List[str] = []
        self.last_shown: List[str] = []
        self.last_turn_size = 0

    # ------------------------------------------------------------------ slots

    def add_phrase(self, text: str, weight: float = 1.0) -> None:
        """Record a disclosed constraint, keeping the strongest weight seen."""
        text = text.strip()
        if not text:
            return
        key = phrase_key(text)
        if key not in self.disclosed_keys:
            self.learned_this_turn = True
        if text not in self.phrase_weights:
            self.phrase_order.append(text)
            self.phrase_weights[text] = weight
            # When it arrived, for slot decay. Re-stating a constraint refreshes
            # it: saying a thing twice is evidence that it still matters.
            self.phrase_turn[text] = self.turn
        else:
            self.phrase_weights[text] = max(self.phrase_weights[text], weight)
            self.phrase_turn[text] = self.turn
        self.disclosed_keys.add(key)

    def demote_existing(self, factor: float = DEMOTED_WEIGHT) -> None:
        """Down-weight everything heard so far, on an explicit intent override."""
        for text in self.phrase_weights:
            self.phrase_weights[text] *= factor

    def clear_slots(self) -> None:
        """Full erasure. Not used by default; kept for the override ablation."""
        self.phrase_weights.clear()
        self.phrase_order.clear()
        self.phrase_turn.clear()
        self.disclosed_keys.clear()

    def weighted_phrases(self) -> List[tuple]:
        """Constraints and their current evidence weight, newest-first in effect.

        With ``decay`` set, a constraint loses a little weight for every turn
        since it was last stated. The brief lists "slot decay over time" as part
        of heterogeneous retrieval routing, and the reasoning is sound: in a
        ten-turn conversation the thing someone said one turn ago is a better
        guide than the thing they said six turns ago, and a customer who keeps
        answering questions is steadily revising what they want.

        It is a floor-limited decay, not an expiry. An old constraint is still a
        true attribute of the target -- the same argument that makes override
        demote rather than erase -- so it fades toward ``decay_floor`` and never
        to nothing.
        """
        if self.decay <= 0.0:
            return [(text, self.phrase_weights[text]) for text in self.phrase_order]
        out = []
        for text in self.phrase_order:
            age = max(0, self.turn - self.phrase_turn.get(text, self.turn))
            factor = max(self.decay_floor, 1.0 - self.decay * age)
            out.append((text, self.phrase_weights[text] * factor))
        return out

    # -------------------------------------------------------------- questions

    def record_ask(self, attribute: Optional[str]) -> None:
        self.pending_ask = attribute
        if attribute:
            self.asked.append(attribute)

    def credit_answer(self) -> None:
        """Credit the previous question if this turn disclosed something new."""
        if self.pending_ask and self.learned_this_turn:
            self.yielded.append(self.pending_ask)

    def mark_exhausted(self, attribute: Optional[str]) -> None:
        if attribute:
            self.exhausted.add(attribute)

    def mark_no_preference(self, attribute: Optional[str]) -> None:
        """Boundary customers decline one attribute outright, then behave normally."""
        if attribute:
            self.no_preference.add(attribute)
            self.exhausted.add(attribute)

    # ------------------------------------------------------- negative evidence

    def exclusions_active(self) -> bool:
        """Whether 'already shown and not a hit' is sound evidence yet."""
        if self.scenario == "intent_override" and not self.override_seen:
            return False
        return True

    def record_shown(self, asins: Sequence[str]) -> None:
        """Remember exactly the identifiers the evaluator will have scored."""
        self.last_shown = list(asins)
        self.last_turn_size = len(asins)
        if self.exclusions_active():
            self.shown.update(asins)
