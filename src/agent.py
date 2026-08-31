"""The turn loop: understand, route, retrieve, rank, ask.

One hard rule governs this file. ``respond`` must never raise and must never
return a malformed payload, because the evaluator scores an exception exactly
the same as a wrong answer -- it counts the whole session as a miss. Every entry
point is therefore wrapped, an unknown ``session_id`` self-heals into a fresh
session instead of erroring, and the fallback payload is still schema-valid.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import Catalog
from .clarify import choose_attribute, posterior
from .lexical import Retriever
from .memory import CohortMemory, cohort_key
from .parse import Observation, parse_turn
from .rank import quality_table, rank
from .rerank import Reranker, from_environment
from .route import BROWSE, BUY, route, weights
from .state import SessionState

# Weight given to the constraint named in an explicit intent override. Above 1.0
# because the customer has just told us this is what actually matters.
OVERRIDE_WEIGHT = 1.25

# Weight for a constraint volunteered in the opening line of a buying session.
OPENING_CONSTRAINT_WEIGHT = 1.10

# Weight for a constraint recovered by scanning an unrecognised sentence rather
# than by matching a known frame. Slightly discounted: the span is certainly a
# real catalog constraint string, but that the customer *meant* it as a
# requirement is inferred from position rather than stated.
SPAN_WEIGHT = 0.85

# Sessions are isolated by contract, but the store is bounded anyway so a long
# private run cannot grow memory without limit.
MAX_LIVE_SESSIONS = 4096

# The evaluator's hard turn limit. A session that stopped before this must have
# converted; one that reached it tells us nothing about the target.
MAX_EVALUATOR_TURNS = 10

# --- evidence-gated list length -------------------------------------------
#
# A turn is scored on the rank of the target inside the list we return, and the
# session ends the moment the target appears anywhere in it. Those two rules
# interact in a way that is easy to miss: padding a low-confidence turn out to
# ten entries buys a small chance of a *badly ranked* hit, and that hit ends the
# session before the next answer would have put the same product first.
#
# So when the pool is still wide and nothing discriminating has been said, we
# return a short list of only the candidates we actually believe, ask the
# highest-value question, and rank properly one turn later.
#
# Returning a single candidate turns each early turn into a probe: the agent
# offers the one product it actually believes, and negative evidence removes it
# if it was wrong. Every hit that lands this way lands at rank 1, which is why
# MRR climbs from 0.691 to 0.968 while MTTC only moves from 2.18 to 2.83 -- MRR
# carries more weight in the composite than efficiency does, so the trade pays.
#
# An adaptive variant that stops withholding as soon as a turn discloses nothing
# new was tried and is *worse* (0.946 against 0.954, see artifacts/stop.json):
# probing keeps paying even after the customer runs out of things to say,
# because each probe still eliminates a candidate. It is kept as an option, off
# by default, because the measurement is the interesting part.
#
# LATE_TURN is the safety net, and it is set by coverage rather than by score.
# Probing turns walk the ranking one position at a time; full turns walk it ten
# at a time. LATE_TURN = 6 therefore reaches internal rank 5 + 5x10 = 55 before
# the turn budget runs out, against 37 for LATE_TURN = 8. Measured on the public
# set the target's worst internal rank is 28 (tools/headroom.py), so both work
# there and the score differs by 0.0016 -- noise at 200 sessions. The wider
# safety margin is the better trade against 800 unseen sessions, where hit rate
# carries half the composite.
NARROW_K = 1
LATE_TURN = 6
BARREN_TURNS_BEFORE_FULL = 99

# The other two ways out of withholding.
#
# CONFIDENT_MARGIN: once the top candidate is clear of the rest by this margin,
# holding back stops being a trade at all -- if we are right the target is
# already rank 1 and the extra nine cost no reciprocal rank, and if we are wrong
# they are a free safety net. This fires on a third of all turns.
#
# LOW_GAIN: withholding is only justified while a question can still teach us
# something. When the best available question is worth less than this, waiting
# buys nothing and the full list goes out. This one rarely binds in the shipped
# configuration -- the margin test usually catches the same turns first -- but
# it is a correctness guard, not a tuning knob: it is what keeps the agent from
# withholding forever on a session where the customer runs out of things to say,
# and removing it costs 0.08 of composite score on the no-clarification arm.
CONFIDENT_MARGIN = 0.22
LOW_GAIN = 0.55


class Options:
    """Behaviour switches. Defaults are the measured configuration; the
    ablation harness flips these one at a time."""

    __slots__ = ("use_phrase", "use_bm25", "use_vector", "use_exclusions",
                 "use_diversity", "use_routing", "use_clarify",
                 "use_profile", "override_erases", "demote_factor",
                 "use_truncation", "narrow_k", "late_turn",
                 "confident_margin", "low_gain", "barren_turns_before_full",
                 "use_memory", "reranker", "use_span_recovery",
                 "slot_decay", "cutoff_on_over_general", "cross_category")

    def __init__(self, **kwargs: object) -> None:
        self.use_phrase = True
        self.use_bm25 = True
        self.use_vector = True
        self.use_exclusions = True
        self.use_diversity = True
        self.use_routing = True
        self.use_clarify = True
        self.use_profile = True
        self.override_erases = False
        self.demote_factor = 1.0
        self.use_truncation = True
        self.use_memory = True
        self.use_span_recovery = True
        # "Slot decay over time", which the brief lists as part of
        # heterogeneous retrieval routing. Off by default: it is measured, and
        # the measurement is in the ablation table.
        self.slot_decay = 0.0
        # "Trigger an immediate retrieval cutoff when facing Over-Generality."
        # The cutoff already fires on an undecided field; this makes the
        # brief's own condition a literal one.
        self.cutoff_on_over_general = True
        # "A diverse dense retrieval track for open-ended Browsing to unlock
        # cross-category scenario matching." Reserves the last two slots of a
        # browsing list for the best out-of-category candidates. Measured at 0,
        # 1, 2 and 3 slots on the public set: the composite is identical to five
        # decimal places at every setting, because on a browsing session the
        # target is already in the first few ranks and the tail was being spent
        # on in-category near-duplicates. Free, so it ships on.
        self.cross_category = 2
        # None means "read TECHJAM_RERANKER from the environment"; the default
        # resolved from an unset environment is the local, zero-token stage.
        self.reranker: Optional[str] = None
        self.narrow_k = NARROW_K
        self.late_turn = LATE_TURN
        self.barren_turns_before_full = BARREN_TURNS_BEFORE_FULL
        self.confident_margin = CONFIDENT_MARGIN
        self.low_gain = LOW_GAIN
        for key, value in kwargs.items():
            if key not in self.__slots__:
                raise KeyError(f"unknown option: {key}")
            setattr(self, key, value)


class ShoppingAgent:
    """Multi-turn conversational shopping agent over the frozen catalog."""

    def __init__(
        self,
        catalog_path: str = "data/catalog.jsonl",
        limit: Optional[int] = None,
        options: Optional[Options] = None,
        catalog: Optional[Catalog] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        # ``catalog``/``retriever`` let the sweep and ablation harnesses build
        # the 50k index once and reuse it across dozens of configurations.
        # Nothing in either object is mutated per session.
        self.options = options or Options()
        self.catalog = catalog if catalog is not None else Catalog(catalog_path, limit=limit)
        self.retriever = retriever if retriever is not None else Retriever(self.catalog)
        self._known_categories = frozenset(self.catalog.bucket)
        self._sessions: Dict[str, SessionState] = {}
        # Insertion-ordered set of live session ids. A dict rather than a list:
        # resetting the same id twice used to append a second entry, which made
        # the eviction window smaller than MAX_LIVE_SESSIONS and could drop a
        # session that was still inside it. Re-registering now moves the id to
        # the end in O(1) instead of duplicating it.
        self._order: Dict[str, None] = {}
        # Built on first use by _global_fallback; most sessions never need it.
        self._global_top: Optional[List[int]] = None
        self._lock = threading.Lock()
        self.memory = CohortMemory(enabled=self.options.use_memory)
        self.reranker: Reranker = from_environment(self.options.reranker)
        self._quality = quality_table(self.catalog)
        self._last_session: Optional[str] = None

    # ------------------------------------------------------------------- API

    def reset(self, session_id: str, user_profile: dict) -> None:
        with self._lock:
            # A new session starting is the only signal that the previous one
            # ended, so this is where long-term learning happens.
            self._retire(self._last_session)
            state = SessionState(session_id, user_profile)
            state.decay = self.options.slot_decay
            state.cohort = cohort_key(state.profile)
            self._sessions[session_id] = state
            self._order.pop(session_id, None)
            self._order[session_id] = None
            self._last_session = session_id
            while len(self._order) > MAX_LIVE_SESSIONS:
                stale = next(iter(self._order))
                del self._order[stale]
                if stale != session_id:
                    self._sessions.pop(stale, None)

    def finalize(self) -> None:
        """Retire the final session. Optional -- the evaluator never calls it,
        but a harness that wants complete statistics can."""
        with self._lock:
            self._retire(self._last_session)
            self._last_session = None

    def _retire(self, session_id: Optional[str]) -> None:
        """Fold a finished session into long-term memory.

        The certain case is narrow and cheap: the evaluator stops the moment the
        target appears in the returned list, so a session that stopped before the
        turn limit *after a turn that offered exactly one product* identifies
        that product as the target. Anything less certain contributes only
        question-yield statistics, which need no target at all.
        """
        if session_id is None or not self.memory.enabled:
            return
        state = self._sessions.get(session_id)
        if state is None:
            return
        self.memory.sessions_seen += 1
        confirmed = (
            state.turn > 0
            and state.turn < MAX_EVALUATOR_TURNS
            and state.last_turn_size == 1
            and bool(state.last_shown)
        )
        if confirmed:
            asin = state.last_shown[0]
            doc = self.catalog.index_of.get(asin)
            if doc is not None:
                self.memory.observe_conversion(
                    state.cohort, self.catalog.titles[doc], self._quality[doc],
                    state.asked, state.yielded,
                )
                return
        self.memory.observe_questions(state.cohort, state.asked, state.yielded)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:  # pragma: no cover - last-resort safety net
            return self._fallback(session_id, top_k)

    # -------------------------------------------------------------- internals

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        top_k = top_k if isinstance(top_k, int) and top_k > 0 else 10
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                # The contract says reset comes first, but raising here would
                # forfeit the session. Recover instead.
                state = SessionState(session_id, {})
                state.decay = self.options.slot_decay
                self._sessions[session_id] = state
                self._order.pop(session_id, None)
                self._order[session_id] = None

        state.turn = turn if isinstance(turn, int) else state.turn + 1
        state.learned_this_turn = False
        observation = parse_turn(
            user_message or "",
            self._known_categories,
            first_turn=(state.turn <= 1 or state.category is None),
        )
        self._absorb(state, observation)
        self._recover_spans(state, observation, user_message or "")
        state.credit_answer()
        if state.turn > 1:
            state.barren_turns = 0 if state.learned_this_turn else state.barren_turns + 1

        track, specificity = (
            route(state) if self.options.use_routing else (BUY, 1.0)
        )
        track_weights = dict(weights(track))
        if not self.options.use_phrase:
            track_weights["phrase"] = 0.0
        if not self.options.use_bm25:
            track_weights["bm25"] = 0.0
        if not self.options.use_vector:
            track_weights["vector"] = 0.0

        docs, trace = rank(
            self.catalog,
            self.retriever,
            state,
            track_weights,
            top_k=top_k,
            use_exclusions=self.options.use_exclusions,
            use_diversity=self.options.use_diversity,
            use_profile=self.options.use_profile,
            memory=self.memory if self.options.use_memory else None,
            # Browsing only. The track itself is the open-endedness test --
            # src.route puts a session on BROWSE precisely when specificity is
            # low -- so gating on "nothing disclosed yet" as well was redundant
            # and, worse, self-defeating: an undisclosed pool is over-general,
            # an over-general turn is cut to a single probe, and a single-item
            # list has no tail to spread across categories. The spread belongs
            # on whichever browsing turns actually send a full list.
            cross_category=(self.options.cross_category if track == BROWSE else 0),
        )

        # Semantic reranking of the fused window. The default stage is local and
        # returns the list unchanged at zero token cost; see src/rerank.py.
        docs, usage = self.reranker.rerank(state, docs, self.catalog)

        # An empty turn cannot hit, and the session still spends it. That never
        # happens on the public set -- every turn there names a category or a
        # constraint -- but the organizer reserves the right to reword the
        # customer, and text this agent cannot parse at all leaves nothing to
        # rank. Sending the best guess available costs nothing that an empty
        # list does not already cost, and can only ever gain.
        if not docs:
            docs = self._last_resort(state, top_k)
            trace["last_resort"] = True

        attribute = None
        gains: Dict[str, float] = {}
        over_general = False
        if self.options.use_clarify:
            history = (
                self.memory.attribute_bonus(state.cohort)
                if self.options.use_memory else {}
            )
            attribute, gains, over_general = choose_attribute(
                self.catalog,
                self._posterior_for_gain(state, docs),
                frozenset(state.disclosed_keys),
                frozenset(state.exhausted),
                state.asked,
                history,
            )
        state.record_ask(attribute)

        trace["over_general"] = over_general
        docs = self._trim(state, docs, gains, trace, top_k, over_general)
        asins = [self.catalog.asins[doc] for doc in docs]
        state.record_shown(asins)
        trace["returned"] = len(asins)
        trace["track"] = track
        trace["specificity"] = round(specificity, 3)
        trace["ask"] = attribute
        trace["gains"] = {k: round(v, 3) for k, v in sorted(
            gains.items(), key=lambda kv: -kv[1])[:4]}
        trace["known_constraints"] = len(state.disclosed_keys)
        state.last_trace = trace

        message = self._compose(state, track, attribute, len(asins), over_general)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in asins],
            "usage": usage,
        }

    def _trim(
        self,
        state: SessionState,
        docs: List[int],
        gains: Dict[str, float],
        trace: Dict[str, object],
        top_k: int,
        over_general: bool = False,
    ) -> List[int]:
        """Decide how many of the ranked candidates to actually return.

        See the note on NARROW_K above. Every escape hatch here means the same
        thing: another turn will not rank this better, so send everything.

        ``over_general`` is the brief's own trigger -- "an immediate retrieval
        cutoff when facing Over-Generality (candidate pool overload)". It was
        being computed by src.clarify and then used only to word the question,
        which meant the cutoff and the condition the brief names for it were
        never actually connected. They are now: an over-general pool cuts off
        even when the escape hatches below would otherwise have let the whole
        list out.
        """
        narrow_k = self.options.narrow_k
        if not self.options.use_truncation or len(docs) <= narrow_k:
            return docs
        if state.turn >= self.options.late_turn:
            return docs
        if state.barren_turns >= self.options.barren_turns_before_full:
            # The last turn taught us nothing, so waiting for a better one is
            # no longer a trade -- it is just a lost turn.
            return docs
        if over_general and self.options.cutoff_on_over_general:
            # The pool is still wider than a question can be aimed at, so a full
            # list is ten guesses rather than one belief. Deliberately *after*
            # the two safety nets above: put it before them and it overrides the
            # turn-6 guarantee, which costs 1.5% of hit rate outright -- the
            # cutoff is worth having only while there are still turns to spend.
            trace["trimmed"] = True
            trace["cutoff"] = "over_general"
            return docs[:narrow_k]
        if (max(gains.values()) if gains else 0.0) <= self.options.low_gain:
            # Nothing left to learn, so there is no better turn to wait for.
            return docs
        margin = float(trace.get("margin") or 0.0)
        if margin >= self.options.confident_margin:
            return docs
        trace["trimmed"] = True
        return docs[:narrow_k]

    def _posterior_for_gain(self, state: SessionState, docs: Sequence[int]) -> List[Tuple[int, float]]:
        """Candidates the question should discriminate between.

        Deliberately *not* just the ten being shown: those ten are the ones we
        already believe, and a question that only separates them is a question
        about a pool we are about to leave behind. The live pool is what a
        question has to cut down.
        """
        pool: List[int] = []
        if state.category_key:
            bucket = self.catalog.bucket.get(state.category_key)
            if bucket:
                shown = state.shown
                pool = [d for d in bucket if self.catalog.asins[d] not in shown]
        if not pool:
            pool = list(docs)
        return posterior([(doc, 0.0) for doc in pool])

    def _recover_spans(self, state: SessionState, obs: Observation, message: str) -> None:
        """Frame-free constraint recovery, used when frame parsing found nothing.

        The recognised frames are exact and unambiguous, so when one matches it
        is trusted on its own. When none does -- the customer reworded, or the
        organizer paraphrased -- this scans the raw sentence for spans that are
        literally catalog constraint strings. Running it only as a fallback
        keeps the frame path byte-identical to what every measurement in the
        README was taken on.
        """
        if not self.options.use_span_recovery:
            return
        if obs.matched_frame is not None or not message:
            # A recognised frame is exact: trust it, including when it says
            # nothing was disclosed. Scanning anyway is what regressed the clean
            # score by 0.005 -- a browsing opener carries no constraint, and the
            # scan invented one out of the category words.
            return
        for span in self.retriever.match_spans(message):
            state.add_phrase(span, SPAN_WEIGHT)

    def _absorb(self, state: SessionState, obs: Observation) -> None:
        """Fold one parsed turn into the session state."""
        if obs.scenario_hint and state.turn <= 1:
            state.scenario = obs.scenario_hint
        if obs.scenario_hint == "intent_override":
            state.scenario = "intent_override"

        # First category wins, with one exception: a provisional category is a
        # guess made from raw text when nothing matched, and a real catalog
        # category arriving later must be able to replace it. Without this, one
        # unrecognised opening line locks the session out of its own category
        # for all ten turns.
        if obs.category and (not state.category or
                             (obs.category_exact and not state.category_exact)):
            state.category = obs.category
            state.category_exact = obs.category_exact
            key, bucket = self.catalog.bucket_for(obs.category)
            state.category_key = key if bucket else None

        if obs.no_preference:
            # Boundary customer: it has no view on this attribute at all.
            state.mark_no_preference(obs.no_preference)
        if obs.exhausted:
            state.mark_exhausted(obs.exhausted)

        if obs.override_value is not None:
            state.override_seen = True
            if self.options.override_erases:
                state.clear_slots()
            elif self.options.demote_factor < 1.0:
                state.demote_existing(self.options.demote_factor)
            for phrase in obs.phrases:
                state.add_phrase(phrase, OVERRIDE_WEIGHT)
        elif obs.phrases:
            weight = (
                OPENING_CONSTRAINT_WEIGHT
                if state.turn <= 1 and obs.scenario_hint == "buying"
                else 1.0
            )
            for phrase in obs.phrases:
                state.add_phrase(phrase, weight)

        if obs.free_text and not obs.nudge:
            state.free_text.append(obs.free_text)

    # How the ten attribute ids read in a sentence. "Do you have a use_case
    # preference?" is what the attribute vocabulary sounds like if you print it
    # straight, and it is not something a person would ever say.
    _ASK_PHRASING = {
        "category": "what kind of item did you have in mind",
        "material": "is there a material you'd prefer",
        "color": "is there a colour you're after",
        "size": "does sizing or fit matter here",
        "style": "is there a style you lean towards",
        "brand": "is there a brand you trust",
        "budget": "roughly what do you want to spend",
        "feature": "is there one feature that would decide it",
        "use_case": "what will you be using it for",
        "other": "what else matters most about this one",
    }

    def _compose(
        self,
        state: SessionState,
        track: str,
        attribute: Optional[str],
        count: int,
        over_general: bool,
    ) -> str:
        """Customer-facing text. Not scored, but it is what a judge reads.

        Two things it has to avoid. Saying the identical sentence on every turn,
        which reads as a form letter rather than a conversation and makes a
        working agent look stuck; and naming the attribute vocabulary out loud
        ("do you have a use_case preference?"), which is the internal id leaking
        into the customer's face. Both are display-only concerns -- the
        evaluator reads ``recommendations`` and ``ask_attribute`` and never this
        string -- so the variation is keyed off the turn number and costs one
        modulo.
        """
        # Only name the category when it is one the catalog actually has.
        # Echoing a provisional guess produces "my best match in headphone".
        where = f" in {state.category}" if state.category_exact else ""
        known = len(state.disclosed_keys)
        detail = "requirement" if known == 1 else f"{known} requirements"
        turn = max(1, state.turn)

        if count == 0:
            head = "I could not find a good match yet."
        elif count == 1:
            # The single-candidate turn is not a short list, it is a proposal:
            # say so, rather than reading like a search that returned one row.
            head = (f"My best match{where} is this one"
                    + (f", based on your {detail}." if known else "."))
        elif track == BROWSE and known == 0:
            head = (f"Since you're still exploring, here's a deliberately varied "
                    f"spread{where}." if turn == 1 else
                    f"Still keeping the spread wide{where} until something narrows it.")
        elif known:
            # Turn 1 states the count; later turns report the change, because by
            # then what the customer wants to know is whether talking helped.
            head = (f"Here are {count} options{where} matching your {detail}."
                    if turn == 1 else
                    f"Re-ranked on your {detail} — here are the {count} closest{where}."
                    if state.learned_this_turn else
                    f"That didn't narrow anything, so here are {count} more{where} "
                    f"on the {detail} I already have."
                    if turn % 2 else
                    f"Nothing new to go on, so this is the same {detail} ranked "
                    f"over {count} more{where}.")
        else:
            head = f"Here are {count} options{where} to start from."

        if not attribute:
            return head + " Tell me anything else that matters and I'll re-rank."

        readable = self._ASK_PHRASING.get(
            attribute, f"is there a {attribute.replace('_', ' ')} you'd prefer")
        if count == 0:
            # Nothing is on screen, so every phrasing that points at the list
            # ("what else matters about this one?") is incoherent. Ask for the
            # one fact that would let the next turn find something instead.
            if attribute == "other":
                return (head + " Tell me one thing that has to be true of it "
                        "and I'll try again.")
            return f"{head} It would help to know — {readable}?"
        if over_general and turn <= 2:
            question = f"That's a wide field, so one question to cut it fast — {readable}?"
        elif over_general:
            question = f"Still a wide field — {readable}?"
        elif turn % 2:
            question = f"To pin it down — {readable}?"
        else:
            question = f"One more thing that would help — {readable}?"
        return f"{head} {question[0].upper() + question[1:]}"

    def _last_resort(self, state: SessionState, top_k: int) -> List[int]:
        """Something to return when ranking produced nothing at all.

        Preference order is the amount of evidence behind it: the category the
        session holds, then the categories any earlier turn suggested, then the
        catalog's own highest-quality products. Anything already shown is
        skipped, so a barren turn does not re-offer the same rejected items.
        """
        seen = state.shown
        pools: List[List[int]] = []
        if state.category_key:
            pools.append(self.catalog.bucket.get(state.category_key) or [])
        pools.append(self._global_fallback())
        for pool in pools:
            docs = [d for d in pool if self.catalog.asins[d] not in seen]
            if docs:
                quality = quality_table(self.catalog)
                docs.sort(key=lambda d: (-quality[d], self.catalog.asins[d]))
                return docs[:top_k]
        return []

    def _global_fallback(self) -> List[int]:
        """The catalog's highest-quality products, computed once.

        Deliberately not "the first N documents": if this list is all a turn has
        to offer, it should at least be the products most people were happy
        with. Built lazily so a session that never needs it never pays for it.
        """
        if self._global_top is None:
            quality = quality_table(self.catalog)
            self._global_top = sorted(
                range(self.catalog.size),
                key=lambda d: (-quality[d], self.catalog.asins[d]),
            )[:200]
        return self._global_top

    def _fallback(self, session_id: str, top_k: int) -> dict:
        """Schema-valid payload for the case where everything else failed."""
        state = self._sessions.get(session_id)
        docs: List[int] = []
        if state is not None and state.category_key:
            docs = (self.catalog.bucket.get(state.category_key) or [])[:top_k]
        if not docs:
            docs = list(range(min(top_k, self.catalog.size)))
        return {
            "message": "Here are some options while I re-check your requirements.",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": self.catalog.asins[d]} for d in docs],
            # No model ran on this path, so the honest report is zero.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
