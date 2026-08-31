"""Turn a customer utterance into structured observations.

The simulated customer speaks in a small number of fixed frames. Recognising
them exactly is worth a lot -- it is the difference between knowing that
``"Pull-On closure"`` is a *constraint the target product literally carries*
and treating it as six more words of bag-of-words noise.

Every parse is best-effort and additive. Nothing here can fail a turn: whatever
is not recognised still reaches retrieval as free text, so an unfamiliar
phrasing degrades the agent to plain hybrid search rather than breaking it.
That matters because the organizer reserves the right to paraphrase these
frames on the private split.
"""

from __future__ import annotations

import re
from typing import AbstractSet, List, Optional

_LOOKING_FOR = re.compile(r"^\s*i'?m looking for\s+(.*)$", re.I)
_EXPLORING = re.compile(r",?\s*but i'?m still exploring\.?\s*$", re.I)
_KEY_REQUIREMENT = re.compile(r"\.\s*a key requirement is:\s*(.*?)\.?\s*$", re.I)
_MATTERS = re.compile(r"^\s*for that,?\s*what matters is:\s*(.*?)\.?\s*$", re.I)
_NO_EXTRA = re.compile(r"^\s*i don'?t have an additional preference for\s+([a-z_]+)\.?\s*$", re.I)
_NO_PREF = re.compile(r"^\s*i don'?t have a preference for\s+([a-z_]+)\s*;", re.I)
_OVERRIDE = re.compile(
    r"^\s*actually,?\s*ignore my earlier preference\.\s*what i need is:\s*(.*?)\.?\s*$", re.I
)
_NUDGE = re.compile(
    r"ask me about one specific attribute|ask me about one thing|"
    r"what do you want to know|could you ask me about something specific", re.I)

# Signals an open-ended browse when the opening frame is unrecognised.
_VAGUE = re.compile(
    r"\b(exploring|browsing|undecided|not sure|no firm|nothing specific|"
    r"just (having a )?look|haven'?t narrowed)\b", re.I)

# Bounds the O(n^2) span enumeration below. Real disclosures never come close.
_MAX_SPLIT_PARTS = 12


class Observation:
    """What one customer turn told us."""

    __slots__ = ("free_text", "category", "category_exact", "phrases", "exhausted",
                 "no_preference", "override_value", "nudge", "scenario_hint",
                 "matched_frame")

    def __init__(self, free_text: str) -> None:
        self.free_text = free_text
        self.category: Optional[str] = None
        # Whether ``category`` is a string the catalog actually knows, or a
        # provisional guess taken from raw text because nothing else matched.
        # A guess is still useful as a query hint, but it must never lock the
        # slot: a real category arriving on a later turn has to be able to
        # replace it. See SessionState.category_exact.
        self.category_exact = False
        self.phrases: List[str] = []
        self.exhausted: Optional[str] = None
        self.no_preference: Optional[str] = None
        self.override_value: Optional[str] = None
        self.nudge = False
        self.scenario_hint: Optional[str] = None
        # Which of the customer's known frames this turn matched, if any. A
        # recognised frame is exact, so its (possibly empty) constraint list is
        # authoritative and no fallback scan should second-guess it.
        self.matched_frame: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Observation(frame={self.matched_frame!r}, category={self.category!r}, "
                f"phrases={self.phrases!r}, hint={self.scenario_hint!r})")


def _split_constraints(payload: str) -> List[str]:
    """Recover the individual constraint strings from a joined disclosure.

    The customer joins its disclosures with ``"; "``, but a single constraint
    may itself contain ``"; "`` -- fabric breakdowns like ``"Solids: 100%
    Cotton; Heathers: 60% Cotton, 40% Polyester"`` are one constraint, not two.
    The split is therefore ambiguous, so we emit every *contiguous span* of the
    parts. The true constraints are always among them, and because phrase
    lookup is exact, a wrong span matches nothing rather than something
    misleading.
    """
    payload = payload.strip()
    if not payload:
        return []
    parts = [part.strip() for part in payload.split("; ") if part.strip()]
    if len(parts) <= 1:
        return [payload] if payload else []
    out = [payload]
    n = min(len(parts), _MAX_SPLIT_PARTS)
    for i in range(n):
        for j in range(i + 1, n + 1):
            span = "; ".join(parts[i:j])
            if span != payload:
                out.append(span)
    return out


def parse_turn(
    message: str,
    known_categories: Optional[AbstractSet[str]] = None,
    first_turn: bool = False,
) -> Observation:
    """Parse one customer message. Never raises."""
    obs = Observation(message or "")
    text = (message or "").strip()
    if not text:
        return obs

    if _NUDGE.search(text):
        obs.nudge = True
        obs.matched_frame = "nudge"

    match = _NO_PREF.match(text)
    if match:
        obs.no_preference = match.group(1).lower()
        obs.matched_frame = "no_preference"
        return obs

    match = _NO_EXTRA.match(text)
    if match:
        obs.exhausted = match.group(1).lower()
        obs.matched_frame = "exhausted"
        return obs

    match = _OVERRIDE.match(text)
    if match:
        obs.override_value = match.group(1).strip()
        obs.phrases = _split_constraints(match.group(1))
        obs.scenario_hint = "intent_override"
        obs.matched_frame = "override"
        return obs

    match = _MATTERS.match(text)
    if match:
        obs.phrases = _split_constraints(match.group(1))
        obs.matched_frame = "matters"
        return obs

    opener = _LOOKING_FOR.match(text)
    if opener:
        _parse_opener(obs, opener.group(1).strip(), known_categories)
        obs.matched_frame = "opener"
        return obs

    # No recognised frame. Fall back to finding the category anywhere in the
    # sentence; constraint spans are recovered separately and frame-free by
    # Retriever.match_spans, so nothing else is needed here.
    found = find_category(text, known_categories)
    if found:
        obs.category = found
        obs.category_exact = True
        obs.scenario_hint = obs.scenario_hint or (
            "browsing" if _VAGUE.search(text) else None
        )
    elif first_turn:
        # Nothing in the sentence is a catalog category. The raw text is still
        # the best query hint available, so keep it -- but only provisionally.
        obs.category = text.rstrip(".")
    return obs



def _is_known(category: Optional[str], known: Optional[AbstractSet[str]]) -> bool:
    """Whether ``category`` is a coarse category the catalog actually carries.

    The simulator's opening frame always names one, so this is True on every
    scored turn. It is False when a human types the same frame around something
    the catalog has never heard of -- and that is exactly the case where the
    slot must stay replaceable.
    """
    if not category or not known:
        return False
    return " ".join(category.split()).casefold() in known


def _parse_opener(obs: Observation, rest: str, known: Optional[AbstractSet[str]]) -> None:
    """Split an ``I'm looking for ...`` opener into category and constraint."""
    explore = _EXPLORING.search(rest)
    if explore:
        obs.category = rest[: explore.start()].strip().rstrip(",")
        obs.category_exact = _is_known(obs.category, known)
        obs.scenario_hint = "browsing"
        return

    requirement = _KEY_REQUIREMENT.search(rest)
    if requirement:
        obs.category = rest[: requirement.start()].strip()
        obs.category_exact = _is_known(obs.category, known)
        obs.phrases = _split_constraints(requirement.group(1))
        obs.scenario_hint = "buying"
        return

    # Intent-override opener: "I'm looking for {category}. {old_value}".
    # The category may itself contain a period, so prefer the longest known
    # category the remainder starts with over a naive split on the first dot.
    category = _longest_known_prefix(rest, known)
    if category is not None:
        obs.category = category
        obs.category_exact = True
        tail = rest[len(category):].lstrip(" .").strip()
        if tail:
            obs.phrases = _split_constraints(tail)
            obs.scenario_hint = "intent_override"
        return

    head, sep, tail = rest.partition(". ")
    obs.category = (head if sep else rest).strip().rstrip(".")
    tail = tail.strip()
    if tail:
        obs.phrases = _split_constraints(tail)
        obs.scenario_hint = "intent_override"


def _longest_known_prefix(rest: str, known: Optional[AbstractSet[str]]) -> Optional[str]:
    """Longest catalog category value that ``rest`` starts with, if any."""
    if not known:
        return None
    best: Optional[str] = None
    # Categories are at most a handful of words; bound the scan by word count.
    words = rest.split()
    for n in range(min(len(words), 12), 0, -1):
        prefix = " ".join(words[:n])
        stripped = prefix.rstrip(".,")
        if stripped.casefold() in known:
            best = stripped
            break
        if prefix.casefold() in known:
            best = prefix
            break
    return best


def find_category(text: str, known: Optional[AbstractSet[str]]) -> Optional[str]:
    """Longest catalog category occurring anywhere in ``text``.

    The opening frame is the one place the agent learns which slice of 50,000
    products it is working in, so the recovery of that string must not depend on
    the sentence around it. "I'm looking for Accessories Belts", "Shopping for
    Accessories Belts", and "any Accessories Belts going?" all have to work.
    """
    if not known or not text:
        return None
    words = text.replace(",", " , ").split()
    best: Optional[str] = None
    best_len = 0
    for start in range(len(words)):
        for end in range(min(start + 12, len(words)), start, -1):
            if end - start <= best_len:
                break
            span = " ".join(words[start:end]).strip(" .,;:!?")
            if span.casefold() in known:
                best, best_len = span, end - start
                break
    return best
