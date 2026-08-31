"""A paraphrasing stress test for the private-set risk.

The single biggest threat to this agent's score is stated plainly in the
competition specification: "If natural-language paraphrasing is added by the
organizer, it cannot decide correctness." The public customer speaks in eight
fixed frames and quotes product metadata verbatim, and the agent exploits both.
If either changes on the private split, how much does it actually lose?

Rather than guess, this module wraps the *agent* -- not the evaluator, which
stays untouched -- so the organizer's own ``evaluate()`` generates its normal
messages, this wrapper garbles them, and the agent underneath sees only garbled
text. Scoring is unchanged: hits are still exact identifier matches.

Three levels, each strictly harder:

``light``   the eight frames are reworded; constraint strings stay verbatim.
            Tests whether the *parser* is over-fitted to fixed phrasing.
``medium``  frames reworded and constraints lightly perturbed -- case,
            punctuation, list order, a dropped trailing clause.
``heavy``   constraints genuinely paraphrased: synonym substitution, dropped
            filler, reordered tokens. Exact-phrase matching cannot survive this,
            so it measures what the rest of the stack is worth.

Deterministic given a seed, and entirely local -- no model, no network. That is
the point: the hardening it drives can be verified by anyone who clones this.
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple

LEVELS = ("none", "light", "medium", "heavy")

# The evaluator's eight customer frames, matched so the constraint payload can
# be lifted out and re-dressed in different words.
_OPEN_REQUIREMENT = re.compile(
    r"^I'm looking for (?P<cat>.+?)\. A key requirement is: (?P<c>.+?)\.$")
_OPEN_EXPLORING = re.compile(r"^I'm looking for (?P<cat>.+?), but I'm still exploring\.$")
_OPEN_PLAIN = re.compile(r"^I'm looking for (?P<cat>.+?)\. (?P<c>.+)$")
_MATTERS = re.compile(r"^For that, what matters is: (?P<c>.+?)\.$")
_NO_EXTRA = re.compile(r"^I don't have an additional preference for (?P<a>[a-z_]+)\.$")
_NO_PREF = re.compile(
    r"^I don't have a preference for (?P<a>[a-z_]+); please use your judgment\.$")
_NUDGE = re.compile(r"^Those options are not quite right yet\..*$")
_OVERRIDE = re.compile(
    r"^Actually, ignore my earlier preference\. What I need is: (?P<c>.+?)\.$")

_OPEN_REQUIREMENT_FORMS = (
    "Hi — I want to buy {cat}. It has to have {c}.",
    "I need {cat}, and the one thing I can't compromise on is {c}.",
    "Shopping for {cat}. Must-have: {c}.",
    "Can you help me find {cat}? It needs {c}.",
)
_OPEN_EXPLORING_FORMS = (
    "I'm browsing {cat} at the moment, nothing specific in mind yet.",
    "Just having a look at {cat} really, no firm ideas.",
    "Show me some {cat} — I'm undecided.",
    "I'm in the market for {cat} but haven't narrowed it down.",
)
_OPEN_PLAIN_FORMS = (
    "I'm after {cat}. {c}",
    "Looking at {cat} — {c}",
    "I want {cat}. Something like: {c}",
    "{cat} please. {c}",
)
_MATTERS_FORMS = (
    "What matters there is {c}.",
    "For me it's {c}.",
    "Mainly {c}.",
    "I'd say {c}.",
    "That'd be {c}.",
)
_NO_EXTRA_FORMS = (
    "No particular {a} preference from me.",
    "Nothing more on {a}, sorry.",
    "I don't really have a view on {a}.",
    "Can't help you on {a}.",
)
_NO_PREF_FORMS = (
    "I don't mind about {a} — you choose.",
    "No preference on {a}, use your judgement.",
    "{a} is up to you honestly.",
    "Whatever you think for {a}.",
)
_NUDGE_FORMS = (
    "Not quite right. Could you ask me about something specific?",
    "Hmm, none of those. Ask me about one thing in particular.",
    "Those aren't it. What do you want to know?",
)
_OVERRIDE_FORMS = (
    "Actually, forget what I said before — what I really need is {c}.",
    "Change of plan: ignore that. The important thing is {c}.",
    "Scratch that. What I actually want is {c}.",
    "On reflection, never mind my earlier point. It has to be {c}.",
)

# Substitutions applied to constraint text at the `heavy` level. Chosen to be
# meaning-preserving, which is what a paraphrasing simulator would do.
_SYNONYMS = {
    "closure": "fastening", "imported": "shipped in", "lightweight": "light",
    "comfortable": "comfy", "adjustable": "adjusts", "durable": "hard-wearing",
    "breathable": "airy", "material": "fabric", "approximately": "about",
    "measures": "is about", "featuring": "with", "designed": "made",
    "perfect": "great", "quality": "well-made", "premium": "high-end",
    "sleeve": "arm", "waterproof": "water resistant", "handmade": "hand made",
}
_FILLER = frozenset({"the", "a", "an", "and", "with", "for", "of", "to", "is",
                     "are", "that", "this", "in", "on"})
_TOKEN = re.compile(r"[A-Za-z0-9%°\"'.]+")


def _perturb_constraint(text: str, level: str, rng: random.Random) -> str:
    """Degrade one constraint string according to ``level``."""
    if level == "light":
        return text
    if level == "medium":
        out = text
        if rng.random() < 0.5:
            out = out.lower()
        if rng.random() < 0.4:
            out = out.replace(":", " ").replace("-", " ")
        if "," in out and rng.random() < 0.4:
            parts = [p.strip() for p in out.split(",") if p.strip()]
            rng.shuffle(parts)
            out = ", ".join(parts)
        if rng.random() < 0.3:
            words = out.split()
            if len(words) > 6:
                out = " ".join(words[: max(4, int(len(words) * 0.75))])
        return " ".join(out.split())

    # heavy: synonym substitution, filler removal, mild reordering
    words = _TOKEN.findall(text)
    out: List[str] = []
    for word in words:
        low = word.lower().strip(".,;:")
        if low in _SYNONYMS and rng.random() < 0.8:
            out.append(_SYNONYMS[low])
            continue
        if low in _FILLER and rng.random() < 0.5:
            continue
        out.append(word.lower() if rng.random() < 0.6 else word)
    if len(out) > 5 and rng.random() < 0.35:
        cut = rng.randrange(1, len(out) - 1)
        out = out[cut:] + out[:cut]
    if len(out) > 8 and rng.random() < 0.4:
        out = out[: int(len(out) * 0.7)]
    return " ".join(out) or text


def _join(parts: List[str], rng: random.Random) -> str:
    if len(parts) == 1:
        return parts[0]
    if rng.random() < 0.5:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def paraphrase(message: str, level: str, rng: random.Random) -> str:
    """Reword one customer message. ``none`` returns it untouched."""
    if level == "none" or not message:
        return message
    text = message.strip()

    match = _OPEN_REQUIREMENT.match(text)
    if match:
        c = _perturb_constraint(match.group("c"), level, rng)
        return rng.choice(_OPEN_REQUIREMENT_FORMS).format(cat=match.group("cat"), c=c)

    match = _OPEN_EXPLORING.match(text)
    if match:
        return rng.choice(_OPEN_EXPLORING_FORMS).format(cat=match.group("cat"))

    match = _OVERRIDE.match(text)
    if match:
        c = _perturb_constraint(match.group("c"), level, rng)
        return rng.choice(_OVERRIDE_FORMS).format(c=c)

    match = _MATTERS.match(text)
    if match:
        parts = [_perturb_constraint(p.strip(), level, rng)
                 for p in match.group("c").split("; ") if p.strip()]
        return rng.choice(_MATTERS_FORMS).format(c=_join(parts, rng))

    match = _NO_PREF.match(text)
    if match:
        return rng.choice(_NO_PREF_FORMS).format(a=match.group("a"))

    match = _NO_EXTRA.match(text)
    if match:
        return rng.choice(_NO_EXTRA_FORMS).format(a=match.group("a"))

    if _NUDGE.match(text):
        return rng.choice(_NUDGE_FORMS)

    match = _OPEN_PLAIN.match(text)
    if match:
        c = _perturb_constraint(match.group("c"), level, rng)
        return rng.choice(_OPEN_PLAIN_FORMS).format(cat=match.group("cat"), c=c)
    return text


class ParaphrasingAgent:
    """Wraps an Agent so it only ever sees reworded customer text.

    The evaluator is untouched and does the scoring exactly as it always does;
    only what reaches the agent is degraded.
    """

    def __init__(self, inner: object, level: str = "light", seed: int = 20260830) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        self.inner = inner
        self.level = level
        self.seed = seed
        self._rng = random.Random(seed)
        self._session_index = 0
        self.samples: List[Tuple[str, str]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Reseed per session, from the session's *position*, not its name.

        This used to seed on ``session_id``, which looks stable and is not: the
        organizer's evaluator names every session ``f"public_{uuid.uuid4().hex}"``,
        freshly, on every run. So the ``--seed`` flag did nothing, every run drew
        a different set of rewordings, and -- much worse -- the hardened and
        unhardened arms were each handed *different customer text*, which is
        exactly the variable the experiment is supposed to hold fixed. Two
        consecutive runs of `make robust` on identical code disagreed by up to
        0.017 of composite score.

        The evaluator walks the dataset in file order, so the n-th reset is
        always the n-th sample. Seeding on that is reproducible across runs and
        identical across arms, which is what makes the comparison a comparison.
        """
        self._session_index += 1
        self._rng = random.Random(f"{self.seed}:{self._session_index}")
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        reworded = paraphrase(user_message, self.level, self._rng)
        if len(self.samples) < 40:
            self.samples.append((user_message, reworded))
        return self.inner.respond(session_id, reworded, turn, top_k)
