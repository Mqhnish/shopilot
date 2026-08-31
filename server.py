"""Local demo server for the Shopping Copilot walkthrough.

    python3 server.py                      # -> http://127.0.0.1:8000/
    python3 server.py --port 8080 --limit 5000

**This is not the scored path.** The graded artifact is ``agent.py`` plus
``src/``; the organizer evaluates headlessly and the problem statement puts
UI work out of scope. This file exists for the demo video and for watching the
agent's own reasoning while you talk to it. It imports the agent, never
modifies it, and nothing under ``src/`` imports anything from here — deleting
this file and ``web/`` leaves every number in the README unchanged.

Standard library only, like the rest of the project: no framework, no build
step, no install. Two slow objects are built once in a background thread so the
page is interactive immediately:

* the *official* weak BM25 starter (``starter/agent.py``), so the side-by-side
  comparison is against the real baseline rather than a re-implementation;
* a display index (store name and a one-line blurb per product), which the
  scored agent has no reason to carry and therefore does not.

Both degrade cleanly: until they are ready the endpoints say so, and the page
renders without them.
"""

from __future__ import annotations

import argparse
import difflib
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import Options, ShoppingAgent  # noqa: E402
from src.attributes import ATTRIBUTE_ID, ATTRIBUTES, classify_id  # noqa: E402
from src.catalog import is_merchandising  # noqa: E402
from src.lexical import query_vector  # noqa: E402
from src.normalize import content_tokens, phrase_key, soft_key  # noqa: E402
from src.parse import parse_turn  # noqa: E402
from src.rank import PHRASE_RECALL_MAX_DF  # noqa: E402
from src.route import weights as fusion_weights  # noqa: E402
from src.state import SessionState  # noqa: E402

# What the agent calls itself when someone asks. Display only -- nothing in the
# scored path has a name, or needs one.
NAME = "Shopilot"

API = "/api/copilot"
WEB = ROOT / "web"

# The evaluator's constants, restated so a running demo does not depend on the
# organizer package being importable. The replay endpoint imports the real
# evaluator and would fail loudly if these ever disagreed.
TOP_K = 10
MAX_TURNS = 10

# src.rank.W_PROFILE_TAG, restated for display. Imported rather than guessed so
# the page cannot quote a weight the ranker does not use.
from src.rank import W_PROFILE_TAG as PROFILE_TAG_WEIGHT  # noqa: E402

# Profiles offered in the UI are taken from the public set rather than invented,
# so the personalisation panel shows the exact shape the evaluator supplies.
NEUTRAL_PROFILE = {
    "average_prior_rating": 4.2,
    "preference_tags": [],
    "purchase_frequency": "unknown",
    "rating_style": "mixed",
    "summary": "No prior signal; the agent has only what you say in the conversation.",
}


# --------------------------------------------------------------------------- language

# Campaign and housekeeping nodes in the category tree. The definition lives in
# src/catalog.py because it is a fact about the catalog, and both the ranker's
# cross-category discovery and this page's typeahead read the same one -- so the
# list a shopper is offered can never disagree with what the agent will rank.
# Words that describe an *occasion* rather than a product, mapped onto the
# catalog's own vocabulary. A shopper types "something warm for winter"; the
# catalog has "Gloves & Mittens Cold Weather Gloves". Neither BM25 nor the
# category matcher can bridge that on its own, because the two share no token.
#
# This is a demo-layer affordance and it is never applied silently: a category
# reached this way is returned tagged with the word that reached it, and the
# page labels it "via winter" so the shopper can see the leap that was made.
# Nothing under src/ consults this map -- the scored evaluator always names a
# real category on turn 1, so the scored path never needs it.
SCENARIO_HINTS: Dict[str, Tuple[str, ...]] = {
    # occasions
    "gift": ("necklaces", "earrings", "bracelets", "watches", "jewelry"),
    "gifts": ("necklaces", "earrings", "bracelets", "watches", "jewelry"),
    "present": ("necklaces", "earrings", "bracelets", "watches"),
    "anniversary": ("necklaces", "earrings", "rings", "watches"),
    "birthday": ("necklaces", "bracelets", "watches", "charms"),
    "wedding": ("dresses", "formal", "suits", "pumps", "earrings"),
    "bridal": ("dresses", "formal", "earrings", "necklaces"),
    "prom": ("dresses", "formal", "pumps", "earrings"),
    "party": ("dresses", "pumps", "jewelry"),
    "graduation": ("dresses", "suits", "watches"),
    "halloween": ("costumes", "cosplay", "wigs"),
    "christmas": ("sweaters", "pajamas", "necklaces", "socks"),
    # weather and season
    "winter": ("coats", "sweaters", "gloves", "scarves", "beanies", "boots"),
    "cold": ("coats", "sweaters", "gloves", "scarves", "beanies"),
    "warm": ("sweaters", "coats", "gloves", "scarves", "hoodies"),
    "snow": ("boots", "coats", "gloves", "beanies"),
    "rain": ("boots", "jackets", "raincoats"),
    "rainy": ("boots", "jackets", "raincoats"),
    "summer": ("shorts", "sandals", "swimwear", "sunglasses", "tanks"),
    "beach": ("swimwear", "bikinis", "sandals", "sunglasses"),
    "pool": ("swimwear", "bikinis", "sandals"),
    "spring": ("dresses", "cardigans", "sneakers"),
    "autumn": ("sweaters", "boots", "cardigans"),
    "fall": ("sweaters", "boots", "cardigans"),
    # activity and setting
    "gym": ("bras", "leggings", "shorts", "sneakers", "athletic"),
    "workout": ("bras", "leggings", "shorts", "sneakers", "athletic"),
    "training": ("bras", "leggings", "shorts", "sneakers", "athletic"),
    "yoga": ("leggings", "bras", "athletic"),
    "run": ("running", "sneakers", "socks", "shorts"),
    "running": ("running", "sneakers", "socks", "athletic"),
    "jogging": ("running", "sneakers", "socks"),
    "hiking": ("hiking", "boots", "socks", "outdoor"),
    "camping": ("hiking", "boots", "jackets", "outdoor"),
    "office": ("blouses", "shirts", "pants", "loafers", "pumps"),
    "work": ("blouses", "shirts", "pants", "loafers", "boots"),
    "interview": ("suits", "shirts", "blouses", "loafers", "pumps"),
    "school": ("backpacks", "sneakers", "hoodies"),
    "travel": ("luggage", "backpacks", "sneakers", "wallets"),
    "vacation": ("swimwear", "sandals", "sunglasses", "shorts"),
    "sleep": ("pajamas", "sleepwear", "robes", "slippers"),
    "sleeping": ("pajamas", "sleepwear", "robes", "slippers"),
    "lounging": ("robes", "sleepwear", "slippers", "hoodies"),
    "wedding": ("dresses", "formal", "suits", "pumps"),
    "casual": ("jeans", "shirts", "sneakers", "hoodies"),
    "formal": ("suits", "dresses", "formal", "pumps"),
    "hot": ("shorts", "tanks", "sandals"),
    "holiday": ("swimwear", "sandals", "sunglasses", "luggage"),
    "holidays": ("swimwear", "sandals", "sunglasses", "luggage"),
    "trip": ("luggage", "backpacks", "sneakers", "sunglasses"),
    "commute": ("backpacks", "sneakers", "jackets"),
    "date": ("dresses", "pumps", "shirts"),
    "festival": ("boots", "sunglasses", "shorts"),
}

# Catalog bookkeeping that the simulator can disclose and no shopper would ever
# volunteer. Offered last rather than hidden: the simulator really does say
# these, so a click still teaches the agent something -- but "is discontinued by
# manufacturer: no" should not be the first thing a person is asked to pick.
_HOUSEKEEPING = re.compile(
    r"^\s*(?:is discontinued|date first available|asin\b|item model number|"
    r"manufacturer\b|package dimensions|product dimensions|best sellers rank|"
    r"customer reviews|item weight|country of origin|batteries|"
    r"domestic shipping|international shipping|department\b|upc\b|global trade)",
    re.IGNORECASE)

# What a hint contributes relative to a word the shopper actually typed. Below
# 1.0 on purpose: a real category match always outranks an inferred one, so
# "winter gloves" resolves on "gloves" and the hint only breaks the tie.
HINT_WEIGHT = 0.7

# Utterances that are conversation rather than search. The scored simulator
# never produces any of these -- it speaks four fixed frames, none of which can
# match here -- so this only ever fires for a person at the demo, and it is
# checked only when the caller asks for assistance.
SMALL_TALK = (
    ("greeting", re.compile(
        r"^(?:hi+|hey+|hello+|hiya|howdy|yo|sup|what'?s up|wassup|"
        r"good\s+(?:morning|afternoon|evening|day)|greetings|hola|namaste|"
        r"bonjour|hei|halo)\b[\s!.,?~]*(?:there|copilot|bot|again)?[\s!.,?~]*$",
        re.IGNORECASE)),
    ("farewell", re.compile(
        r"^(?:bye+|goodbye|good\s?night|see\s?(?:ya|you)|cya|later|ciao|adios|"
        r"exit|quit|stop|i'?m done|that'?s all|thats all|nothing else)"
        r"\b[\s!.,?~]*(?:now|then|for now)?[\s!.,?~]*$", re.IGNORECASE)),
    ("thanks", re.compile(
        r"^(?:thanks?|thank\s?you|thx|tysm|ty|cheers|appreciate\s?(?:it|that)|"
        r"much appreciated|nice one|perfect thanks)"
        r"\b[\s!.,?~]*(?:a lot|so much|very much|mate|man|dude|friend)?[\s!.,?~]*$",
        re.IGNORECASE)),
    ("identity", re.compile(
        r"^(?:who are you|what are you|what(?:'?s| is) your name|"
        r"your name|do you have a name|are you (?:a )?(?:bot|robot|human|real|"
        r"an ai|ai|chatgpt|gpt|claude|siri|alexa))\b[\s!.,?~]*$", re.IGNORECASE)),
    ("capability", re.compile(
        r"^(?:help|help me|what can you do|what do you do|what are you for|"
        r"how does this work|how do(?:es)? (?:this|it) work|how do i use (?:this|you)|"
        r"what is this|what'?s this|what do you sell|what do you have|"
        r"what can i (?:ask|search|buy))"
        r"\b[\s!.,?~]*$", re.IGNORECASE)),
    ("wellbeing", re.compile(
        r"^(?:how are you|how'?re you|how are things|how'?s it going|hows it going|"
        r"how you doing|how'?s your day|you good|you ok)\b[\s!.,?~]*(?:today|doing)?"
        r"[\s!.,?~]*$", re.IGNORECASE)),
    ("affirm", re.compile(
        r"^(?:ok(?:ay)?|k|kk|cool|nice|great|awesome|sweet|lovely|neat|"
        r"got it|gotcha|understood|sure|alright|all right|fine|yep|yup|yeah|yes|"
        r"mhm|indeed|makes sense|good stuff)\b[\s!.,?~]*$", re.IGNORECASE)),
    ("off_topic", re.compile(
        r"^(?:tell me a joke|say something funny|what'?s the weather|weather|"
        r"what time is it|what'?s the time|sing (?:me )?a song|write (?:me )?a poem|"
        r"who won .*|what'?s \d+\s*[-+*/x]\s*\d+|\d+\s*[-+*/x]\s*\d+)"
        r"[\s!.,?~]*$", re.IGNORECASE)),
)


# --------------------------------------------------------------------------- data


class DisplayIndex:
    """Store name and a short blurb per product, for the cards only.

    The scored agent never needs either, so it does not parse them. This is a
    second pass over the catalog and costs about fifteen seconds, which is why
    it runs off the request path.
    """

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        self.blurb: Dict[str, str] = {}
        self.ready = False

    def build(self, path: Path, limit: Optional[int] = None) -> None:
        try:
            with path.open(encoding="utf-8") as handle:
                for count, line in enumerate(handle):
                    if limit is not None and count >= limit:
                        break
                    try:
                        product = json.loads(line)
                    except ValueError:
                        continue
                    asin = str(product.get("parent_asin") or "")
                    if not asin:
                        continue
                    store = product.get("store")
                    if isinstance(store, str) and store.strip():
                        self.store[asin] = store.strip()[:60]
                    features = product.get("features")
                    if isinstance(features, list):
                        for value in features:
                            text = " ".join(str(value).split())
                            # The first feature line is usually boilerplate
                            # ("Imported"); the first *sentence-shaped* one is
                            # the line a shopper would actually read.
                            if len(text) > 30:
                                self.blurb[asin] = text[:150]
                                break
        except OSError:
            pass
        self.ready = True


class Demo:
    """Everything the HTTP layer needs, built once and shared across threads.

    ``ShoppingAgent`` is safe to share: the catalog and the retriever are
    read-only after construction and per-session state lives behind the agent's
    own lock. Sessions here are keyed by the browser's session id exactly as
    the evaluator keys them.
    """

    def __init__(self, catalog: str, dataset: str, limit: Optional[int] = None) -> None:
        self.catalog_path = Path(catalog)
        self.dataset_path = Path(dataset)
        self.limit = limit

        started = time.time()
        # The scored configuration, exactly as the evaluator runs it.
        self.agent = ShoppingAgent(str(self.catalog_path), limit=limit, options=Options())
        # The same agent with list truncation off. Withholding nine results to
        # probe with one is worth +0.070 of composite score (artifacts/ablate.json)
        # and is a bad shopping experience: a person wants options and a reason,
        # not one item at a time. So the page defaults to this and offers the
        # scored behaviour as a toggle, rather than pretending the two are the
        # same thing. Both share one catalog and one retriever, so the second
        # agent costs no extra index build and no extra memory.
        self.shopper = ShoppingAgent(
            str(self.catalog_path),
            options=Options(use_truncation=False),
            catalog=self.agent.catalog,
            retriever=self.agent.retriever,
        )
        self.index_seconds = round(time.time() - started, 2)
        self.catalog = self.agent.catalog

        self.samples: List[dict] = []
        if self.dataset_path.exists():
            with self.dataset_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        self.turns: Dict[str, int] = {}
        self.modes: Dict[str, str] = {}
        # How many small-talk exchanges this session has had, so a second
        # "hi" does not get the same sentence back as the first one.
        self.chatter: Dict[str, int] = {}
        self._noise_cache: Dict[str, bool] = {}
        self._signature_cache: Dict[str, dict] = {}
        # A price ceiling the shopper typed, remembered for the session so
        # later turns keep flagging rows that break it.
        self.ceilings: Dict[str, float] = {}
        self.lock = threading.Lock()

        self.display = DisplayIndex()
        self.baseline: Optional[Any] = None
        self.baseline_error: Optional[str] = None
        # Exactly one worker: see _build_extras. It is also what serialises the
        # starter's single mutable session store, which is not concurrency-safe
        # either.
        self.baseline_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="baseline")
        threading.Thread(target=self._build_extras, daemon=True).start()

    # -- background construction -------------------------------------------

    def _build_extras(self) -> None:
        self.display.build(self.catalog_path, self.limit)
        # The first keystroke in an empty box lists the biggest categories, and
        # computing eight shelf signatures cold costs ~80ms. Doing it here means
        # the dropdown is never the slow thing.
        for name, _tokens, _count in sorted(
                self._category_index(), key=lambda item: -item[2])[:24]:
            self.category_signature(name)
        # The official starter is built *inside* its own single worker thread,
        # not merely on this one. It indexes into an in-memory SQLite database,
        # and sqlite3 connections are bound to the thread that opened them --
        # so a starter constructed here and queried from a request handler
        # raises ProgrammingError on every single call. That is an organizer
        # file and stays unmodified, so the confinement lives here: build and
        # query both happen on `self.baseline_pool`'s one thread.
        self.baseline_pool.submit(self._build_baseline)

    def _build_baseline(self) -> None:
        try:
            from starter.agent import Agent as StarterAgent

            self.baseline = StarterAgent(str(self.catalog_path))
        except Exception as error:  # pragma: no cover - environment dependent
            # A missing or broken starter must not take the page down; the
            # comparison column simply stays empty and says why.
            self.baseline_error = f"{type(error).__name__}: {error}"

    # -- product hydration --------------------------------------------------

    def product(self, asin: str, rank: Optional[int] = None,
                state: Optional[SessionState] = None,
                ceiling: Optional[float] = None) -> dict:
        doc = self.catalog.index_of.get(asin)
        card: Dict[str, Any] = {"parent_asin": asin, "rank": rank}
        if doc is None:
            card["title"] = asin
            return card
        price = self.catalog.prices[doc]
        category = self.catalog.coarse[doc]
        card.update(
            title=self.catalog.titles[doc],
            price=None if price is None or price <= 0 else round(float(price), 2),
            rating=round(float(self.catalog.ratings[doc]), 2),
            rating_count=int(self.catalog.rating_counts[doc]),
            category=category,
            store=self.display.store.get(asin),
            blurb=self.display.blurb.get(asin),
            # The category is a large bonus in the ranker and never a filter, so
            # an out-of-category row is legitimate and worth marking rather than
            # hiding: it means the text evidence outweighed the category.
            in_category=(state is None or not state.category_exact
                         or category == state.category),
            why=self.evidence_for(doc, state),
        )
        if ceiling is not None and card["price"] is not None:
            card["over_budget"] = card["price"] > ceiling
        return card

    def cards(self, asins: List[str], state: Optional[SessionState] = None,
              ceiling: Optional[float] = None) -> List[dict]:
        return [self.product(asin, rank, state, ceiling)
                for rank, asin in enumerate(asins, start=1)]

    @staticmethod
    def tag_cards(cards: List[dict], tags: List[str]) -> List[dict]:
        """Record which profile tags each card's own text actually contains.

        The aggregate profile enters ranking at weight 0.18 against 1.00 for a
        stated constraint, which is deliberately too weak to overrule anything
        you say -- and therefore invisible unless it is pointed at. This is the
        pointing: it shows where the personalisation signal really landed
        instead of claiming an effect the numbers do not support.
        """
        if not tags:
            return cards
        for card in cards:
            haystack = f"{card.get('title') or ''} {card.get('blurb') or ''}".lower()
            card["matched_tags"] = [tag for tag in tags if tag in haystack]
        return cards

    def evidence_for(self, doc: int, state: Optional[SessionState]) -> List[dict]:
        """Which of the session's stated constraints this product literally carries.

        A ranked list with no reasons is a leap of faith, and it is the thing a
        judge asks about first. This is not a re-derivation or an approximation:
        it is the same two-tier lookup ``src.lexical.phrase_hits`` scores with --
        the evaluator's own cleaning for the exact tier, punctuation stripped for
        the soft tier -- so a chip appears on a card if and only if the phrase
        route credited that product for that phrase.

        ``decisive`` carries the rarity that route weights by. "Imported" is
        carried by thousands of products and separates nothing; a phrase held by
        a handful is most of the reason this row is where it is, and the card
        should not present the two as equal.
        """
        if state is None:
            return []
        exact = self.catalog.phrase_keys[doc]
        soft = self.catalog.soft_keys[doc]
        index = self.agent.retriever
        out: List[dict] = []
        # Tokens already accounted for by a whole-phrase match, so the same
        # evidence is not reported twice in two different shapes.
        matched: set = set()
        for text, weight in state.weighted_phrases():
            if weight <= 0.0:
                continue
            key = phrase_key(text)
            tier = "exact" if key in exact else None
            if tier is None:
                key = soft_key(text)
                tier = "soft" if key in soft else None
            if tier is None:
                continue
            df = index.phrase_df(text)
            matched.update(content_tokens(text))
            out.append({
                "text": text,
                "tier": tier,
                "count": df,
                # Rare enough that carrying it is most of the reason this row
                # ranks where it does. The threshold is the phrase route's own
                # recall ceiling: above it a phrase is a property, not an
                # identifier.
                "decisive": bool(df and df <= PHRASE_RECALL_MAX_DF),
                # A demoted phrase is one the customer overrode; it still
                # describes the target, but it is no longer what they asked for.
                "weight": round(weight, 2),
            })

        # The phrase route only fires on verbatim constraint strings. Most of
        # what a person types is not one -- "black leather" is scored by BM25
        # over the product's own text, and a card with no chips at all would
        # imply the row arrived for no reason. So the words that really are in
        # this product's index are reported too, as a weaker class: read off
        # the same postings BM25 reads, not re-derived from the title.
        terms, _tfs = self.catalog.postings(doc)
        present = set(terms)
        for token in self.session_terms(state):
            if token in matched:
                continue
            term_id = self.catalog.vocab.get(token)
            if term_id is None or term_id not in present:
                continue
            out.append({
                "text": token,
                "tier": "text",
                "count": int(self.catalog.df[term_id]),
                "decisive": False,
                "weight": None,
            })
        return out

    # Words that survive tokenisation but say nothing about a product. They
    # arrive inside constraints the system itself composed -- "budget around
    # $40" contributes "around" -- and a card captioned "around" is worse than
    # a card captioned nothing, because it claims a reason that is not one.
    _EMPTY_TERMS = frozenset("""around about under over less more than most
        budget price priced cost costs dollar dollars usd very quite really
        item items product products thing things good best nice new""".split())

    @classmethod
    def session_terms(cls, state: SessionState) -> List[str]:
        """Content words the session has actually said, in disclosure order."""
        seen: List[str] = []
        for text, weight in state.weighted_phrases():
            if weight <= 0.0:
                continue
            for token in content_tokens(text):
                if len(token) > 2 and token not in cls._EMPTY_TERMS and token not in seen:
                    seen.append(token)
        return seen

    # -- one turn -----------------------------------------------------------

    def agent_for(self, session_id: str):
        return self.agent if self.modes.get(session_id) == "scored" else self.shopper

    def reset(self, session_id: str, profile: Optional[dict], mode: str = "shopper") -> None:
        profile = profile if isinstance(profile, dict) else dict(NEUTRAL_PROFILE)
        with self.lock:
            self.modes[session_id] = "scored" if mode == "scored" else "shopper"
        self.agent_for(session_id).reset(session_id, profile)
        with self.lock:
            self.turns[session_id] = 0
            self.chatter.pop(session_id, None)
            self.ceilings.pop(session_id, None)
            # Bound the demo's own bookkeeping the way the agent bounds its
            # session store; a long-lived server should not grow without limit.
            if len(self.turns) > 512:
                for stale in list(self.turns)[:256]:
                    if stale != session_id:
                        self.turns.pop(stale, None)
                        self.modes.pop(stale, None)
                        self.chatter.pop(stale, None)


    # -- "more like this" ----------------------------------------------------

    def similar(self, asin: str, limit: int = 3,
                session_id: str = "") -> dict:
        """Everything worth knowing about one product, on hover.

        Four similar titles was the wrong payload: in a category of near
        duplicates they read as four more of the same row, and the one thing a
        shopper cannot see from a ranked list -- *what this specific product is*
        -- was still missing.

        So this is a detail card. The interesting part is ``discloses``: the
        exact constraint strings the simulator would reveal about this product
        if the agent asked, taken from ``card_keys``. It is the disclosure
        surface the whole system is built around, and hovering a row is the one
        place a person can actually look at it.

        ``similar`` is unchanged in spirit -- ``Retriever.cosine`` over a query
        vector built from the product's own title, the same TF-IDF space the
        browsing track ranks in, restricted to the product's own category so
        "similar" means *another version of this thing* rather than *another
        item that happens to share adjectives*.
        """
        doc = self.catalog.index_of.get(asin)
        if doc is None:
            return {"of": asin, "product": None, "discloses": [], "similar": []}

        state = self.agent_for(session_id)._sessions.get(session_id) if session_id else None
        category = self.catalog.coarse[doc]
        pool = self.catalog.bucket.get(self.catalog.bucket_for(category)[0]) or []

        ranked: List[tuple] = []
        if len(pool) >= 2:
            query = query_vector([self.catalog.titles[doc]])
            scores = self.retriever_cosine(query, pool)
            ranked = sorted(
                ((other, value) for other, value in scores.items() if other != doc),
                key=lambda item: (-item[1], self.catalog.asins[item[0]]),
            )[:limit]

        # What the customer simulator would say about this product, and which
        # question would elicit each part. Marked against what this session has
        # already heard, so the panel shows what is left to learn rather than
        # repeating the conversation back.
        heard = state.disclosed_keys if state is not None else set()
        discloses = [
            {
                "text": key,
                "attribute": ATTRIBUTES[attr],
                "known": key in heard,
                "count": self.agent.retriever.phrase_df(key),
            }
            for key, attr in zip(self.catalog.card_keys[doc], self.catalog.card_attrs[doc])
            if key
        ]
        return {
            "of": asin,
            "product": self.product(asin, None, state),
            "pool": len(pool),
            "discloses": discloses,
            "similar": [
                {**self.product(self.catalog.asins[other], rank),
                 "closeness": round(float(score), 3)}
                for rank, (other, score) in enumerate(ranked, start=1)
            ],
        }

    def retriever_cosine(self, query: dict, pool: List[int]) -> Dict[int, float]:
        return self.agent.retriever.cosine(query, pool)

    # -- turning a typed phrase into a message the agent was scored on -------

    _STOP = frozenset("""a an and the for with of in on to my me i im i'm looking
        want need some please that this under over about any something
        dollar dollars usd bucks price priced cost costs cheap cheaper
        budget around less than most
        wife husband mom mum dad mother father son daughter sister brother
        girlfriend boyfriend partner friend him her them someone myself
        clothes clothing apparel outfit outfits wear stuff things items
        pair set piece nice good best great new
        actually instead rather now then just really maybe perhaps
        show find get give lets let hey okay ok well hmm""".split())

    # A category smaller than this is a promotional leaf of the Amazon tree
    # rather than something worth searching inside. Both the typeahead and the
    # rewrite demote them; neither hides a category that is genuinely the only
    # match, because the fallback is to leave the query alone and say so.
    MIN_ASSIST_POOL = 25

    # A constraint carried by more than this many of the 50,000 products is a
    # property of the catalog, not a description of one shelf. "Imported" sits
    # at 13,846 and characterises nothing.
    SIGNATURE_MAX_DF = 4000

    # How many of the 50,000 products must use a word before it counts as one
    # the shopper meant rather than a misspelling. Measured: real words used in
    # queries here sit at 27 ("bluetooth") and above, listing typos at 12 and
    # below, so anything in this range separates them.
    REAL_WORD_DF = 25

    # Demographic words appear in dozens of category names and almost never
    # identify the *product*. "womens sunglasses" matching "Women Shoes" as
    # strongly as "Sunglasses & Eyewear Accessories Sunglasses" is how a search
    # box ends up confidently wrong, so they count for half a word.
    _WEAK = frozenset("""men mens man woman women womens girl girls boy boys
        kid kids baby babies unisex adult junior ladies""".split())

    @staticmethod
    def _variants(word: str) -> set:
        """A word and its plausible singular/plural forms.

        Matching on one canonical stem does not work here: any rule that turns
        "watches" into "watch" also turns "necklaces" into "necklac", which then
        fails to match a typed "necklace". Emitting the small set of variants and
        intersecting is both simpler and correct in both directions.
        """
        forms = {word}
        if word.endswith("es") and len(word) > 4:
            forms.add(word[:-2])
        if word.endswith("s") and len(word) > 3:
            forms.add(word[:-1])
        else:
            forms.add(word + "s")
        return forms

    @staticmethod
    def _words(text: str) -> List[str]:
        return "".join(c if c.isalnum() else " " for c in str(text).lower()).split()

    @staticmethod
    def _joined_words(text: str) -> List[str]:
        """Words with intra-word punctuation closed up rather than split.

        "T-Shirts" splits into "t" and "shirts", so a shopper typing "tshirt"
        matches nothing and the search silently stays where it was. Indexing the
        closed-up form too fixes that, and the same rule covers "V-Neck" and
        "Jiu-Jitsu".
        """
        cleaned = str(text)
        for mark in ("-", "'", "’", ".", "&", "/"):
            cleaned = cleaned.replace(mark, "")
        return "".join(c if c.isalnum() else " " for c in cleaned.lower()).split()

    def _token_set(self, text: str) -> set:
        out: set = set()
        for raw in self._words(text):
            out |= self._variants(raw)
        for raw in self._joined_words(text):
            out |= self._variants(raw)
        return out

    def _category_index(self) -> List[tuple]:
        """(name, word-variant set, product count) for every coarse category."""
        if not hasattr(self, "_category_cache"):
            self._category_cache = [
                (self.catalog.coarse[docs[0]],
                 self._token_set(self.catalog.coarse[docs[0]]),
                 len(docs))
                for docs in self.catalog.bucket.values()
            ]
        return self._category_cache

    def _is_noise_category(self, name: str) -> bool:
        """Whether this coarse label is a merchandising slice, not a product type.

        Two independent markers, both measured rather than guessed:

        * the label starts with ``ROOT_LABEL`` -- the product's category path
          never reached a product-type node, so the leaf is a campaign or an
          internal bucket ("Westlake", "MFN ONLY V2", "Top 50 by Product Type");
        * the label carries a merchandising marker -- a price, a percentage, a
          parenthetical, or a campaign word.

        Between them they select 189 of the catalog's 1,115 coarse categories.
        These are demoted in every ranking, never removed: if one is genuinely
        the only match for what was typed, showing it beats showing nothing.
        """
        if name in self._noise_cache:
            return self._noise_cache[name]
        verdict = is_merchandising(name)
        self._noise_cache[name] = verdict
        return verdict

    def _hints_for(self, words: Sequence[str]) -> Dict[str, str]:
        """Catalog words implied by occasion words in the query, as {word: source}.

        "warm for winter" carries no catalog vocabulary at all; "winter" implies
        it. Only words the shopper did not already type are emitted, so a hint
        can add a route but never re-score one that already matched directly.
        """
        typed = set()
        for word in words:
            typed |= self._variants(word)
        hints: Dict[str, str] = {}
        for word in words:
            for target in SCENARIO_HINTS.get(word, ()):
                if target in typed or target in hints:
                    continue
                hints[target] = word
        return hints

    # The exact phrasing the customer uses to say "these are not right". A
    # person says it a dozen other ways, and the parser reads only this one.
    DISSATISFIED = "Those options are not quite right yet."

    _DISSATISFIED_RE = re.compile(
        r"^(?:no(?:ne|pe)?|nah|not (?:really|quite|it|these|those|right|what i)|"
        r"none of (?:these|those|them)|nothing (?:here|matches|works|good)|"
        r"these (?:are|aren'?t|don'?t)|those (?:are|aren'?t|don'?t)|"
        r"try again|show me (?:something )?(?:else|different|more)|"
        r"something else|different ones?|keep looking|next|other options?)"
        r"\b.{0,40}$", re.IGNORECASE)

    @staticmethod
    def is_customer_frame(message: str) -> bool:
        """Whether the parser reads this message as one of the customer's frames.

        Asked of ``src.parse`` rather than guessed from prefixes, because the
        only definition that cannot drift is the parser's own. Every part of the
        assist layer consults this, and originally only the rewrite did — which
        cost a whole session, silently, the moment anyone clicked a follow-up
        chip: ``"For that, what matters is: color: gold."`` has content words
        ("colour", "gold") that name a different category than the one being
        refined, so the topic-change detector read a deliberate answer as a
        change of subject and started over.

        Prefix matching was the first fix and it was too greedy in the other
        direction: the override frame begins "Actually, ignore my earlier
        preference…", so matching on the bare word "actually" swallowed
        *"actually I want sneakers instead"* — a real change of subject — and
        left the shopper refining necklaces.
        """
        text = " ".join(str(message or "").split())
        if not text:
            return False
        if text.casefold() == Demo.DISSATISFIED.casefold():
            return True
        return parse_turn(text).matched_frame is not None

    @classmethod
    def as_dissatisfaction(cls, message: str) -> Optional[str]:
        """The customer's own "not quite right" line, if that is what was meant.

        The parser recognises exactly one phrasing of dissatisfaction and a
        person has twenty. Left alone, "none of these" is read as a *search* for
        the word "these" — which is how the page came to answer it with
        "No category here for “these”." Mapping it onto the frame the parser
        reads is the difference between the agent hearing a rejection and
        hearing noise.
        """
        text = " ".join(str(message or "").split())
        if not text or len(text) > 48 or cls.is_customer_frame(text):
            return None
        return cls.DISSATISFIED if cls._DISSATISFIED_RE.match(text) else None

    def assist(self, message: str) -> Optional[dict]:
        """Rewrite free text into the opening frame the agent is scored against.

        The evaluator names a coarse category on turn 1 of every scenario, so the
        agent is built to be told one. A person typing into a search box is not,
        and without a category the candidate pool is all 50,000 products and the
        ranking is close to meaningless — which is exactly what "a black leather
        belt under $50" produces.

        So the *demo* resolves the category first: match the typed words against
        the catalog's real category names, then re-form the sentence. The
        rewrite is returned to the caller and shown on the page rather than
        applied silently — a search box that quietly changes your query is worse
        than one that fails. Nothing in ``src/`` is involved, and a message that
        already names a category is passed through untouched.
        """
        text = " ".join(str(message or "").split())
        if not text:
            return None
        # Already a customer frame: leave it exactly as the parser expects it.
        if self.is_customer_frame(text):
            return None
        if not self._query_words(text):
            return None
        ranked = self.rank_categories_detailed(text)
        if not ranked:
            return None
        best, pool = ranked[0]["category"], ranked[0]["count"]
        opener = self.opener_for(best, text)
        return {
            "category": best,
            "pool": pool,
            "message": opener["message"],
            "track": opener["track"],
            # A stated ceiling, which the catalog's own "budget around $X"
            # cannot express and the ranker cannot compare against.
            "ceiling": self.budget_ceiling(text),
            # Set when the category was reached through an occasion word rather
            # than named outright, so the page can show the leap it made.
            "via": ranked[0]["via"],
            "original": text,
            "corrections": self.corrections(text),
        }

    # -- conversation that is not a search ----------------------------------

    @staticmethod
    def classify_small_talk(message: str) -> Optional[str]:
        """Which kind of non-search utterance this is, or None.

        A shopping agent that answers "hi" with ten belts is not a shopping
        agent, it is a search box with a chat skin. The scored simulator speaks
        four fixed frames and can never land here, so this only ever fires for a
        person -- and it is consulted only when the caller asks for assistance,
        which the evaluator never does.
        """
        text = " ".join(str(message or "").split())
        if not text or len(text) > 64:
            return None
        for kind, pattern in SMALL_TALK:
            if pattern.match(text):
                return kind

        # "thanks, bye" and "ok cool see you" are several utterances in one, and
        # patterns anchored at both ends match none of them. Consume small-talk
        # phrases greedily from the front; the whole message counts only if
        # every word is accounted for, so "thanks, now show me boots" stays a
        # search and "cool belts" stays a search.
        words = [w for w in re.split(r"[\s,;.!?]+", text) if w]
        if len(words) < 2:
            return None
        kinds: List[str] = []
        cursor = 0
        while cursor < len(words):
            for span in range(min(5, len(words) - cursor), 0, -1):
                phrase = " ".join(words[cursor:cursor + span])
                found = next((k for k, pattern in SMALL_TALK if pattern.match(phrase)), None)
                if found is not None:
                    kinds.append(found)
                    cursor += span
                    break
            else:
                return None
        # Answer the part that decides how the exchange ends, not the first one:
        # "thanks, bye" is a goodbye, and "ok, who are you" is a question.
        for preferred in ("identity", "capability", "wellbeing", "farewell",
                          "off_topic", "thanks", "greeting"):
            if preferred in kinds:
                return preferred
        return kinds[-1]

    def small_talk(self, session_id: str, message: str) -> Optional[dict]:
        """A conversational reply, with somewhere to go next.

        Deliberately *not* a turn: it does not touch the 10-turn budget, does
        not reach the agent, and does not disturb a session in progress. That is
        the whole point -- saying "thanks" halfway through a search should not
        cost you a retrieval round, and the brief is explicit that exceeding ten
        turns is a zero.

        The replies are state-aware, because the useful part of answering "hi"
        is what you say *after* hello: before a search that is what the catalog
        holds, and mid-search it is where the conversation has got to.
        """
        kind = self.classify_small_talk(message)
        if kind is None:
            return None

        with self.lock:
            turn = self.turns.get(session_id, 0)
            seen = self.chatter.get(session_id, 0)
            self.chatter[session_id] = seen + 1

        state = self.agent_for(session_id)._sessions.get(session_id)
        category = state.category if state is not None and state.category_exact else None
        known = len(state.disclosed_keys) if state is not None else 0
        where = self._progress_line(category, known, turn)

        text, chips = self._small_talk_reply(kind, where, category, turn, seen)
        return {
            "kind": "small_talk",
            "intent": kind,
            "message": text,
            "chips": chips,
            "turn": turn,
            "turns_remaining": max(0, MAX_TURNS - turn),
            # A conversational turn is free. Saying so on the page is what stops
            # someone from being afraid to talk to it.
            "counted": False,
        }

    def _progress_line(self, category: Optional[str], known: int, turn: int) -> str:
        if not category or turn <= 0:
            return ""
        detail = "no constraints yet" if not known else (
            "one constraint" if known == 1 else f"{known} constraints")
        return f"{turn} turn{'s' if turn != 1 else ''} into {category}, {detail}"

    def _small_talk_reply(self, kind: str, where: str, category: Optional[str],
                          turn: int, seen: int) -> Tuple[str, List[dict]]:
        """The words. Varied by how often this session has done it, so a second
        "hi" does not come back identical to the first."""
        keep = [{"label": "keep going", "say": "Those options are not quite right yet."}]
        fresh = [{"label": "start a new search", "action": "reset"}]
        openers = self.opening_chips()

        if kind == "greeting":
            if category:
                return (f"Hey again. We're {where} — say anything else that matters "
                        f"and I'll re-rank, or start over.", keep + fresh)
            first = [
                f"Hi, I'm {NAME}. I search 50,000 Clothing, Shoes & Jewelry "
                "products — name a category and I'll narrow it down with you.",
                "Hello. Tell me roughly what you're after and I'll ask one good "
                "question a turn until we've found it.",
                "Hey. Start me anywhere — a garment, a shoe, an occasion — and "
                "I'll take it from there.",
            ]
            return first[seen % len(first)], openers

        if kind == "farewell":
            if category:
                return (f"Right — {where}. The session stays here if you come back; "
                        "otherwise start a fresh one whenever.", fresh)
            return (f"Take care. {NAME} will be here — 50,000 products, ten "
                    "turns, whenever you want them.", openers)

        if kind == "thanks":
            if category:
                return ("Any time. Still on " + str(category) +
                        " if you want to push further.", keep + fresh)
            return ("Any time. Name a category whenever you're ready.", openers)

        if kind == "identity":
            return (
                f"{NAME} — a shopping agent built for TechJam Track 4, running on "
                "50,000 frozen Amazon products. Not a person, and no model call "
                "behind me either: the ranking is pure Python over an in-memory "
                "index, which is why answers come back in milliseconds.",
                openers if not category else keep + fresh,
            )

        if kind == "capability":
            return (
                f"I'm {NAME} — a shopping agent over a frozen 50,000-product Amazon "
                "catalog, Clothing, Shoes & Jewelry only, so no electronics. Tell me "
                "what you're after and each turn I'll rank the catalog, then ask the "
                "one question that splits the remaining candidates most sharply. Ten "
                "turns per session; chatting like this doesn't spend any of them.",
                openers if not category else keep + fresh,
            )

        if kind == "wellbeing":
            base = ("Running fine — a full catalog indexed and nothing to do but "
                    "find you something.")
            return (base + (f" We're {where}." if where else " What are you after?"),
                    keep + fresh if category else openers)

        if kind == "affirm":
            if category:
                return ("Good. Tell me the next thing that matters, or say what's "
                        "wrong with these and I'll re-rank.", keep + fresh)
            return ("Whenever you're ready — name a category and we'll start.", openers)

        # off_topic
        return (
            "That one's outside my shelf, I'm afraid — I only know this catalog's "
            "clothing, shoes and jewellery. Happy to find you something in it.",
            keep + fresh if category else openers,
        )

    def opening_chips(self, limit: int = 3) -> List[dict]:
        """Somewhere to go, taken from the catalog rather than written by hand.

        ``suggestions`` already refuses to offer a phrasing this catalog cannot
        answer, so reusing it means a conversational reply can never end with a
        button that dead-ends.
        """
        if not hasattr(self, "_chip_cache"):
            self._chip_cache = [
                {"label": item["text"], "say": item["text"]}
                for item in self.suggestions()["natural"]
            ]
        return self._chip_cache[:limit]

    def refinements(self, session_id: str, query: str = "", limit: int = 8) -> dict:
        """What to say next, once a category is locked.

        A category typeahead is the right tool on turn 1 and the wrong one after
        it: the agent fixes its category for the life of a session, so offering
        more categories mid-search invites a topic change nobody asked for. What
        a shopper actually needs then is the vocabulary the *live candidates*
        can still disclose — which the agent already computes, and which this
        just filters by whatever is in the box.

        Every value returned is a real constraint from ``card_keys``, so
        clicking one teaches the agent something rather than adding noise.
        """
        state = self.agent_for(session_id)._sessions.get(session_id)
        if state is None or not state.category_exact:
            return {"ready": False, "values": [], "actions": []}

        asked = state.asked[-1] if state.asked else None

        # Answers to the question actually on screen come first; everything the
        # pool can still disclose follows. Both lists are real ``card_keys``, so
        # either way a click teaches the agent something.
        primary = self.option_values(state, asked, limit=32) if asked else []
        rest = [v for v in self.option_values(state, "other", limit=64) if v not in primary]

        values = primary + rest
        needle = " ".join(str(query or "").split()).casefold()
        if needle:
            values = [v for v in values if needle in v.casefold()]
        rows = [
            {
                "value": value,
                "say": f"For that, what matters is: {value}.",
                "attribute": ATTRIBUTES[self.catalog_attribute(value)],
            }
            for value in values[:limit]
        ]
        # The three moves that are always available and that a person never
        # guesses the wording of. Each is a frame src/parse.py reads exactly.
        actions = [
            {"label": "none of these are right",
             "say": "Those options are not quite right yet."},
        ]
        if asked:
            actions.append({
                "label": f"no {str(asked).replace('_', ' ')} preference",
                "say": f"I don't have an additional preference for {asked}.",
            })
        actions.append({"label": "start a new search", "action": "reset"})
        return {
            "ready": True,
            "category": state.category,
            "asked": asked,
            "values": rows,
            "actions": actions,
        }

    @staticmethod
    def catalog_attribute(value: str) -> int:
        return classify_id(value)

    def chat(self, session_id: str, message: str, assist: bool = False) -> dict:
        # Conversation before search. Checked first and only under ``assist``,
        # so the scored path -- which never sets it, and whose four frames
        # cannot match anyway -- reaches the agent byte for byte as before.
        if assist:
            chat = self.small_talk(session_id, message)
            if chat is not None:
                return chat

        agent = self.agent_for(session_id)
        held = agent._sessions.get(session_id)
        held_category = held.category if held is not None and held.category_exact else None

        # Only the opening turn is a search. Once a category is held, a short
        # follow-up ("for men", "brown") refines it, and rewriting that into a
        # fresh opener throws the conversation away. A genuine topic change is
        # the exception, and has to start over -- the agent locks its category
        # slot for the life of a session by design.
        # A recognised frame is an answer to the agent's own question, so none
        # of the assist layer applies to it -- not the rewrite, not the
        # topic-change detector, not the unmatched-word warning.
        rejected = self.as_dissatisfaction(message) if assist else None
        if rejected:
            message = rejected
        assist = assist and not rejected and not self.is_customer_frame(message)

        switched_to = self.topic_change(message, held_category) if assist else None
        abandoned = held_category if switched_to else None
        if switched_to:
            profile = dict(held.profile) if held is not None else None
            mode = self.modes.get(session_id, "shopper")
            self.reset(session_id, profile, mode)
            held_category = None

        rewrite = self.assist(message) if assist and not held_category else None
        if rewrite:
            message = rewrite["message"]
            rewrite["switched_from"] = abandoned
            with self.lock:
                if rewrite.get("ceiling") is not None:
                    self.ceilings[session_id] = rewrite["ceiling"]
                else:
                    self.ceilings.pop(session_id, None)

        # A word that clearly names a product but matches no category is the
        # one case the agent cannot act on and cannot report: it just keeps
        # answering about the category it already had. Say so instead.
        unmatched = (
            self.unmatched_nouns(message, held_category)
            if assist and held_category and not rewrite
            else []
        )
        with self.lock:
            turn = self.turns.get(session_id, 0) + 1
            if turn > MAX_TURNS:
                # The brief caps a session at ten turns and scores zero above
                # it, so an eleventh turn is not a degraded answer -- it is an
                # invalid one. The demo refuses it here rather than relying on
                # the page to disable a text box, which is a suggestion and not
                # an enforcement. Conversation stays free: small talk is
                # intercepted before this point and never reaches it.
                self.turns[session_id] = MAX_TURNS
                spent = True
            else:
                self.turns[session_id] = turn
                spent = False
        if spent:
            return {
                "kind": "budget_spent",
                "turn": MAX_TURNS,
                "turns_remaining": 0,
                "message": (
                    f"That's all {MAX_TURNS} turns. The evaluator scores a session "
                    "zero past this point, so I'd rather stop than answer an "
                    "eleventh time — start a new one and I'll keep everything I "
                    "learned about the catalog, just not about this search."
                ),
                "chips": [{"label": "start a new search", "action": "reset"}],
                "counted": False,
            }

        started = time.perf_counter()
        response = agent.respond(session_id, message, turn, TOP_K)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        state = agent._sessions.get(session_id)
        asins = [item["parent_asin"] for item in response.get("recommendations", [])]
        tags = [str(t).lower() for t in (state.profile.get("preference_tags") or [])] if state else []
        payload = {
            "turn": turn,
            "turns_remaining": max(0, MAX_TURNS - turn),
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "results": self.tag_cards(
                self.cards(asins, state, self.ceilings.get(session_id)), tags),
            "ceiling": self.ceilings.get(session_id),
            "profile_tags": tags,
            "profile_weight": PROFILE_TAG_WEIGHT,
            "usage": response.get("usage", {}),
            "latency_ms": elapsed_ms,
            "assist": rewrite,
            # Set when a loose rejection was mapped onto the customer's own
            # phrasing. Shown, never applied silently.
            "rewrote_as": rejected,
            "unmatched": unmatched,
            "mode": self.modes.get(session_id, "shopper"),
        }
        payload.update(self.snapshot(state, turn, session_id))
        payload["options"] = self.option_values(state, response.get("ask_attribute"))
        return payload

    def option_values(self, state: Optional[SessionState], attribute: Optional[str],
                      limit: int = 6) -> List[str]:
        """Concrete answers to the question the agent just asked.

        "Do you have a material preference?" is a worse question than the same
        question with *leather, suede, alloy* underneath it, and the values are
        not guesswork: ``card_attrs`` records which attribute each of a
        product's disclosable constraints belongs to, so these are the exact
        strings the live candidates would actually offer. Clicking one sends the
        simulator's own disclosure frame, which the parser reads precisely.
        """
        if not attribute or state is None:
            return []
        # "other" matches a constraint of any type -- that is exactly why it so
        # often wins the information-gain calculation -- so it takes the whole
        # disclosure surface rather than one attribute's slice.
        wanted = None if attribute == "other" else ATTRIBUTE_ID.get(attribute)
        if wanted is None and attribute != "other":
            return []
        docs = self.catalog.bucket.get(state.category_key) or []
        if not docs:
            return []
        seen = state.disclosed_keys
        counts: Dict[str, int] = {}
        for doc in docs[:4000]:
            for key, attr in zip(self.catalog.card_keys[doc], self.catalog.card_attrs[doc]):
                if key and key not in seen and (wanted is None or attr == wanted):
                    counts[key] = counts.get(key, 0) + 1
        # A value held by nearly everything separates nothing, and one held by a
        # single product is noise; the middle of the distribution is what a
        # useful option list is made of. Amazon's `details` also carry catalog
        # housekeeping -- "is discontinued by manufacturer: no", "date first
        # available" -- which the simulator really would disclose and no shopper
        # would ever volunteer, so they sort last rather than being removed: a
        # click still teaches the agent something, it just should not be the
        # first thing offered.
        floor = max(2, len(docs) // 200)
        ranked = sorted(
            ((k, n) for k, n in counts.items() if n >= floor),
            key=lambda kv: (bool(_HOUSEKEEPING.match(kv[0])), -kv[1], kv[0]),
        )
        return self.still_open(state, [k for k, _n in ranked])[:limit]

    @staticmethod
    def still_open(state: SessionState, values: List[str]) -> List[str]:
        """Drop values this session has effectively already stated.

        ``disclosed_keys`` only records constraints that matched a card key
        exactly, so after "a black leather belt" the keys "leather" and
        "color: black" both still look undisclosed -- and the page offered them
        straight back as things to tell it. Offering someone their own words as
        a suggestion is the one thing a suggestion list must never do, so the
        check is against the phrases the session actually holds.
        """
        stated = [text.casefold() for text, weight in state.weighted_phrases()
                  if weight > 0.0]
        return [
            value for value in values
            if not any(value.casefold() in said or said in value.casefold()
                       for said in stated)
        ]

    def snapshot(self, state: Optional[SessionState], turn: int,
                 session_id: str = "") -> dict:
        """The agent's own reasoning for this turn, in one JSON object.

        Everything here is read straight off the live session — `last_trace` is
        the same dict `tools/demo.py` prints — so the panels cannot drift from
        what the agent actually did.
        """
        if state is None:
            return {"track": "blend", "specificity": 0.0, "trace": {}, "constraints": {}}
        trace = dict(state.last_trace or {})
        track = str(trace.get("track") or "blend")
        gains = trace.get("gains") or {}
        return {
            "track": track,
            "specificity": trace.get("specificity", 0.0),
            "scenario": state.scenario,
            "override_seen": bool(state.override_seen),
            "constraints": {
                "category": state.category,
                "category_exact": bool(state.category_exact),
                "category_pool": len(self.catalog.bucket.get(state.category_key) or [])
                if state.category_key else 0,
                "phrases": [
                    {"text": text, "weight": round(weight, 2)}
                    for text, weight in state.weighted_phrases()
                ],
                "exhausted": sorted(state.exhausted),
                "no_preference": sorted(state.no_preference),
                "ruled_out": len(state.shown),
                "asked": list(state.asked),
            },
            "gains": gains,
            "trace": {
                "pool": trace.get("pool", 0),
                "in_bucket": trace.get("in_bucket", 0),
                "phrase_docs": trace.get("phrase_docs", 0),
                "excluded": trace.get("excluded", 0),
                "top_score": trace.get("top_score", 0.0),
                "margin": trace.get("margin", 0.0),
                "returned": trace.get("returned", 0),
                "trimmed": bool(trace.get("trimmed")),
                "memory": bool(trace.get("memory")),
                "known_constraints": trace.get("known_constraints", 0),
            },
            "weights": fusion_weights(track),
            "decision": self.explain_return(
                {**trace, "barren_turns": state.barren_turns}, turn, session_id),
        }

    def explain_return(self, trace: dict, turn: int, session_id: str = "") -> dict:
        """Why this turn returned the number of products it did.

        The single highest-leverage decision in the system, and invisible in a
        plain results list — so it gets its own panel. Reconstructed from the
        trace rather than instrumented into ``_trim``, which stays on the hot
        path and is not going to grow a display dependency.

        Reconstruction means this can *disagree* with the code it describes, and
        an explanation that disagrees is worse than none. Three ways it did:

        * it read thresholds off the scored agent while the page defaults to the
          untruncated one, so every turn on the default configuration was
          explained by a margin that had nothing to do with it -- truncation was
          simply switched off;
        * ``_trim`` short-circuits on barren turns and this never mentioned it,
          so that case fell through to a bare "Full list.";
        * ``_trim`` tests information gain before margin and this tested margin
          first, so when both held it named the wrong one.

        The branches below are now in ``_trim``'s order, against the options of
        the agent that actually ran, and ``tests/test_cards.py`` walks the two
        side by side.
        """
        options = self.agent_for(session_id).options
        returned = int(trace.get("returned") or 0)
        margin = float(trace.get("margin") or 0.0)
        gains = trace.get("gains") or {}
        best_gain = max(gains.values()) if gains else 0.0
        barren = int(trace.get("barren_turns") or 0)

        if trace.get("trimmed"):
            return {
                "mode": "probe",
                "returned": returned,
                "headline": "Offering one product, not ten",
                "why": (
                    "Nothing has separated the field yet. A padded list can convert at "
                    "a bad rank and end the session before a better turn arrives, so the "
                    "agent shows the one product it believes, asks the highest-value "
                    "question, and ranks properly next turn."
                ),
            }

        # `_trim`'s escape hatches, in the order it tests them.
        if not options.use_truncation:
            # True but dull on its own, and it would be the same sentence on
            # every turn of the default configuration. The interesting quantity
            # is the counterfactual: run the same trace through the scored
            # agent's thresholds and say what it would have done here.
            scored = self.agent.options
            held = self.would_trim(trace, turn, scored)
            reason = (
                "Truncation is off in this configuration, so the full ranked list "
                "goes out every turn. "
                + (f"The scored setup would have shown {scored.narrow_k} here and held "
                   f"{max(0, returned - scored.narrow_k)} back to buy a sharper question — "
                   "worth 0.070 of composite score, and a worse experience."
                   if held else
                   "The scored setup would have sent the whole list here too.")
            )
        elif returned <= options.narrow_k:
            reason = "Only one candidate survived the filters, so there is nothing to hold back."
        elif turn >= options.late_turn:
            reason = (
                f"Turn {turn} is at or past the turn-{options.late_turn} safety net: from "
                "here the full list always goes out, leaving five full-width turns."
            )
        elif barren >= options.barren_turns_before_full:
            reason = (
                f"The last {barren} turn{'s' if barren != 1 else ''} taught the agent "
                "nothing, so waiting for a better one is no longer a trade — it is "
                "just a lost turn."
            )
        elif best_gain <= options.low_gain:
            reason = (
                f"The best remaining question is worth {best_gain:.2f} bits, below the "
                f"{options.low_gain:.2f} floor. No question can teach us more, so there is "
                "no better turn to wait for."
            )
        elif margin >= options.confident_margin:
            reason = (
                f"The top candidate leads the next by {margin:.2f}, clear of the "
                f"{options.confident_margin:.2f} threshold. If it is right the target is "
                "already rank 1; if it is wrong the other nine are a free safety net."
            )
        else:
            # Unreachable while this mirrors _trim: one of the branches above
            # must have fired for the list to have gone out whole. Say so
            # rather than inventing a reason.
            reason = "Full list — no truncation rule applied to this turn."
        return {
            "mode": "full",
            "returned": returned,
            "headline": f"Returning {returned} products",
            "why": reason,
        }

    @staticmethod
    def would_trim(trace: dict, turn: int, options) -> bool:
        """Whether ``ShoppingAgent._trim`` would have held this turn back.

        The same predicate, in the same order, over the same trace -- which is
        the only honest way to state a counterfactual about code you are not
        running. ``tests/test_cards.py`` checks it against the scored agent's
        actual behaviour on the same turn rather than trusting the mirror.
        """
        returned = int(trace.get("returned") or 0)
        gains = trace.get("gains") or {}
        if not options.use_truncation or returned <= options.narrow_k:
            return False
        if turn >= options.late_turn:
            return False
        if int(trace.get("barren_turns") or 0) >= options.barren_turns_before_full:
            return False
        if (max(gains.values()) if gains else 0.0) <= options.low_gain:
            return False
        return float(trace.get("margin") or 0.0) < options.confident_margin

    # -- the official starter, for comparison -------------------------------

    def baseline_chat(self, session_id: str, message: str) -> dict:
        if self.baseline is None:
            return {
                "ready": False,
                "results": [],
                "note": self.baseline_error or "The official BM25 starter is still indexing.",
            }
        key = f"baseline::{session_id}"

        def run() -> List[str]:
            self.baseline.reset(key, {})
            response = self.baseline.respond(key, message, 1, TOP_K)
            return [item["parent_asin"] for item in response.get("recommendations", [])]

        try:
            # Marshalled onto the thread that owns the starter's SQLite handle.
            # The timeout is the demo's own guard: the comparison column is a
            # nicety, and a wedged baseline must not hold a request open.
            asins = self.baseline_pool.submit(run).result(timeout=20)
            return {"ready": True, "results": self.cards(asins)}
        except Exception as error:
            return {"ready": False, "results": [], "note": f"{type(error).__name__}: {error}"}

    # -- replaying a labelled session ---------------------------------------

    def replay(self, sample_id: str) -> dict:
        """Drive one *labelled* session through the organizer's own simulator.

        This is the honest version of a demo: the customer's replies come from
        ``evaluator.local_evaluator``, not from a script, and the hidden target
        is revealed only in the response envelope — never to the agent.
        """
        from evaluator.local_evaluator import (
            catalog_index, coarse_category, customer_reply, initial_message,
            materialize_hidden_fields,
        )

        sample = next((s for s in self.samples if s.get("sample_id") == sample_id), None)
        if sample is None:
            raise KeyError(sample_id)

        # The evaluator's own product map, needed for the intent card. Built
        # once and cached on the instance: it is a third pass over the catalog.
        if not hasattr(self, "_eval_index"):
            self._eval_index = catalog_index(str(self.catalog_path))
        _ids, categories, products = self._eval_index

        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}

        session_id = f"replay::{sample_id}::{time.time():.0f}"
        self.reset(session_id, sample.get("user_profile") or {}, mode="scored")

        disclosed: set = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        turns: List[dict] = []
        hit_turn: Optional[int] = None
        hit_rank: Optional[int] = None

        for turn in range(1, MAX_TURNS + 1):
            payload = self.chat(session_id, message)
            payload["customer"] = message
            asins = [item["parent_asin"] for item in payload["results"]]
            payload["target_rank"] = asins.index(target) + 1 if target in asins else None
            turns.append(payload)

            if override_applied and target in asins[:TOP_K]:
                hit_turn, hit_rank = turn, asins.index(target) + 1
                break
            if turn == MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", ""))
            else:
                message, boundary_used = customer_reply(
                    effective, payload.get("ask_attribute"), disclosed, boundary_used
                )

        return {
            "sample_id": sample_id,
            "scenario": sample["scenario_type"],
            "difficulty": sample.get("difficulty_bucket"),
            "profile": sample.get("user_profile", {}),
            "turns": turns,
            "target": {
                **self.product(target),
                "hard_constraints": list(card.get("hard_constraints", [])),
                "soft_preferences": list(card.get("soft_preferences", [])),
            },
            "hit": hit_turn is not None,
            "hit_turn": hit_turn,
            "hit_rank": hit_rank,
            "reciprocal_rank": round(1.0 / hit_rank, 3) if hit_rank else 0.0,
        }

    # -- static content for the page ---------------------------------------

    def suggestions(self) -> dict:
        """Two ways in, and they are not the same thing.

        ``examples`` are openers the customer simulator itself would produce —
        the shape the agent is scored against. The buying opener quotes a
        constraint the *target actually holds*, taken from ``card_keys``, so
        every one is a session the agent can really solve rather than a
        plausible-looking sentence about nothing in the catalog.

        ``natural`` is how a person would say it instead. These deliberately do
        not name a category: they exercise the assist layer, including the
        occasion words that only resolve through ``SCENARIO_HINTS``. Both are
        offered because the honest demo is the pair — this is what it was
        scored on, and this is what it does with a human sentence.
        """
        seen: Dict[str, str] = {}
        for sample in self.samples:
            scenario = sample["scenario_type"]
            if scenario in seen:
                continue
            target = str(sample["ground_truth"]["parent_asin"])
            doc = self.catalog.index_of.get(target)
            if doc is None:
                continue
            category = self.catalog.coarse[doc]
            card = self.catalog.card_keys[doc]
            if scenario in ("buying", "intent_override") and card and card[0]:
                seen[scenario] = f"I'm looking for {category}. A key requirement is: {card[0]}."
            else:
                seen[scenario] = f"I'm looking for {category}, but I'm still exploring."
        order = ("buying", "browsing", "intent_override", "boundary")

        # Only offer a human phrasing that this catalog can actually answer, so
        # the empty state can never hand someone a query that dead-ends.
        wanted = [
            ("a black leather belt", "a hard constraint on turn one"),
            ("something warm for winter", "an occasion, not a category"),
            ("a gift for an anniversary", "no product word at all"),
            ("comfortable shoes for standing all day", "a need, not a noun"),
            ("gold hoop earrings under $30", "two constraints and a price"),
            ("running shoes with arch support", "a category plus a feature"),
        ]
        natural = []
        for text, why in wanted:
            ranked = self.rank_categories_detailed(text)
            if not ranked or self._is_noise_category(ranked[0]["category"]):
                continue
            natural.append({
                "text": text,
                "why": why,
                "resolves_to": ranked[0]["category"],
                "via": ranked[0]["via"],
            })
        return {
            "examples": [seen[k] for k in order if k in seen],
            "natural": natural[:4],
        }

    def _category_vocab(self) -> List[str]:
        """Every distinct word appearing in a category name."""
        if not hasattr(self, "_vocab_cache"):
            vocab = set()
            for name, _tokens, count in self._category_index():
                if count >= self.MIN_ASSIST_POOL:
                    vocab.update(self._words(name))
            self._vocab_cache = sorted(vocab)
        return self._vocab_cache

    def _is_real_word(self, word: str) -> bool:
        """Whether the catalog's own product text uses this word in earnest.

        This is what separates a typo from a word the catalog simply does not
        sell. Edit distance cannot: "neckalce" -> "necklaces" (0.824) and
        "dollars" -> "dolls" (0.833) are indistinguishable by ratio, and one of
        them is a disaster -- it silently turned "under 100 dollars" into a
        search for baby-doll lingerie.

        Mere presence is not enough either, because Amazon listings are full of
        their own typos: "neckalce" really does appear, in 12 products out of
        50,000. Document frequency separates them cleanly -- misspellings sit in
        single or low double digits, words people mean sit in the hundreds or
        thousands -- so the test is whether the catalog uses the word *often*.
        """
        vocab, df = self.catalog.vocab, self.catalog.df
        for form in self._variants(word):
            index = vocab.get(form)
            if index is not None and df[index] >= self.REAL_WORD_DF:
                return True
        return False

    def _query_words(self, text: str) -> List[str]:
        """Usable words from a typed query, closed-up forms included."""
        seen: List[str] = []
        for word in list(self._words(text)) + list(self._joined_words(text)):
            if word in self._STOP or word.isdigit() or len(word) < 2:
                continue
            if word not in seen:
                seen.append(word)
        return seen

    def _accepted_forms(self, word: str) -> set:
        """The forms of ``word`` a category name may match, typos included.

        The problem statement guarantees pre-cleaned input, so the scored agent
        correctly spends nothing on spelling. A demo search box has no such
        guarantee: "snekaers" returning nothing reads as broken. Correction
        happens here, in the demo layer, against the closed category vocabulary
        and only for words the catalog does not otherwise use.
        """
        forms = self._variants(word)
        vocab = self._category_vocab()
        if forms & set(vocab):
            return forms
        # Short words are too easy to "correct" into something unrelated, and a
        # word the catalog really uses is not a misspelling of anything.
        if len(word) < 4 or self._is_real_word(word):
            return forms
        near = difflib.get_close_matches(word, vocab, n=3, cutoff=0.82)
        for match in near:
            forms |= self._variants(match)
        return forms

    def corrections(self, query: str) -> Dict[str, str]:
        """Spelling repairs this query needed, for display."""
        vocab = set(self._category_vocab())
        fixes: Dict[str, str] = {}
        for word in self._words(query):
            if (word in self._STOP or len(word) < 4
                    or self._variants(word) & vocab or self._is_real_word(word)):
                continue
            near = difflib.get_close_matches(word, self._category_vocab(), n=1, cutoff=0.82)
            if near and near[0] != word:
                fixes[word] = near[0]
        return fixes

    def topic_change(self, message: str, held: Optional[str]) -> Optional[str]:
        """The new category this message switches to, if it switches at all.

        Refinements and topic changes look identical at the HTTP layer and must
        not be treated the same. "for men" after a belt search narrows it;
        "i want football" abandons it. The distinguishing signal is whether the
        message names a *different* category on a word that identifies a
        product -- "men", "women", "kids" appear in dozens of category names and
        identify nothing, so they can never trigger a switch on their own, and a
        word matching no category at all ("brown", "wide") is a constraint.
        """
        if not held:
            return None
        strong = [
            word for word in self._query_words(message)
            if word not in self._WEAK and len(word) >= 4
        ]
        if not strong:
            return None
        held_tokens = self._token_set(held)
        # A word already accounted for by the current category is a restatement.
        novel = [w for w in strong if not (self._accepted_forms(w) & held_tokens)]
        if not novel:
            return None
        ranked = self.rank_categories(" ".join(novel))
        if not ranked:
            return None
        best, _count = ranked[0]
        if best == held:
            return None
        # Require the switch to rest on a word that names the new category,
        # not on incidental overlap.
        target_tokens = self._token_set(best)
        if not any(self._accepted_forms(w) & target_tokens for w in novel):
            return None
        return best

    def unmatched_nouns(self, message: str, held: Optional[str]) -> List[str]:
        """Substantial words in this turn that name no category at all.

        These are the ones worth reporting: "drone" is not a refinement of a
        belt search, it is a request the catalog cannot serve, and answering it
        silently with more belts is what makes the agent look broken.
        """
        held_tokens = self._token_set(held or "")
        out: List[str] = []
        for word in self._query_words(message):
            if word in self._WEAK or len(word) < 4:
                continue
            forms = self._accepted_forms(word)
            if forms & held_tokens:
                continue
            if any(forms & tokens for _n, tokens, _c in self._category_index()):
                continue
            # A word the catalog's product text uses is a constraint ("brown",
            # "waterproof"), not a failed product request.
            if self._is_real_word(word):
                continue
            out.append(word)
        return out

    def rank_categories_detailed(self, query: str) -> List[dict]:
        """Coarse categories matching ``query``, best first, with provenance.

        One matcher backs the typeahead, the query rewrite, and the topic-change
        detector, so the list a person is shown cannot disagree with the category
        the agent is then given. Whole-string substring matching is not enough on
        its own -- "leather belt" is not a substring of "Accessories Belts" -- so
        scoring is by matched words, with a substring hit as a strong bonus.

        Three things separate this from a plain word-overlap search:

        * merchandising slices are demoted below every real product type, so
          "mens watch" cannot resolve to a five-product "Under $50" campaign;
        * occasion words route through ``SCENARIO_HINTS``, so "something warm
          for winter" reaches the gloves and coats it meant;
        * every entry says *why* it is here, so a hinted match can be labelled
          on the page rather than passed off as a direct one.
        """
        words = self._query_words(query)
        needle = " ".join(str(query or "").split()).casefold()
        if not needle:
            # The empty typeahead legitimately lists the biggest categories --
            # the real ones. A shopper opening the box wants somewhere to start,
            # and "Shoes & Jewelry Westlake" is not a place to start.
            ranked = sorted(
                self._category_index(),
                key=lambda item: (self._is_noise_category(item[0]),
                                  item[2] < self.MIN_ASSIST_POOL, -item[2]),
            )
            return [{"category": name, "count": count, "kind": "popular", "via": None}
                    for name, _tokens, count in ranked]

        # Resolve each typed word once, not once per category.
        accepted = {word: self._accepted_forms(word) for word in set(words)}
        hints = self._hints_for(words)
        hint_forms = {word: self._variants(word) for word in hints}

        scored = []
        # A category matched only by "mens" is not a match for "mens watch": it
        # is the word doing no work at all, and it is how a search box ends up
        # offering Men Jeans for a watch query. Those are kept aside and used
        # only if nothing identified a product, so "womens" alone still lists
        # something rather than dropping to an empty box.
        weak_only = []
        for name, tokens, count in self._category_index():
            folded = name.casefold()
            direct = 0.0
            strong = 0.0
            for word in words:
                if not (accepted[word] & tokens):
                    continue
                if word in self._WEAK:
                    direct += 0.5
                else:
                    direct += 1.0
                    strong += 1.0
            hit_hints = [word for word, forms in hint_forms.items() if forms & tokens]
            inferred = HINT_WEIGHT * len(hit_hints)
            if needle in folded:
                direct += 2.0
                strong += 2.0
            if not direct and not inferred:
                # With no usable words there is nothing to rank on. Returning
                # the whole catalog by size is how "under 100 dollars" came
                # back as the largest category on the shelf.
                continue
            (scored if (strong or inferred) else weak_only).append((
                # Merchandising slices sort below everything real, then thin
                # ones below fat ones, and only then by how well the words fit.
                (not self._is_noise_category(name),
                 count >= self.MIN_ASSIST_POOL,
                 direct + inferred,
                 folded.startswith(needle),
                 count),
                name, count, direct, hit_hints,
            ))
        rows = scored or weak_only
        rows.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "category": name,
                "count": count,
                # A category nothing typed actually names is an inference, and
                # the page says so rather than presenting it as a match.
                "kind": "match" if direct else "related",
                "via": None if direct else hints[hit_hints[0]],
            }
            for _key, name, count, direct, hit_hints in rows
        ]

    def rank_categories(self, query: str) -> List[tuple]:
        """``rank_categories_detailed`` as (name, count) pairs, for callers that
        only need to know which category won."""
        return [(item["category"], item["count"])
                for item in self.rank_categories_detailed(query)]

    def consumed_by(self, category: str, query: str) -> set:
        """Word forms in ``query`` that ``category`` already accounts for.

        Used to work out what is left over after the category is named, which is
        what becomes the stated requirement. Occasion words that *reached* an
        inferred category count as consumed -- "gift" produced the jewelry, so
        repeating it back as "a key requirement is: gift" would be nonsense.
        """
        tokens = self._token_set(category)
        consumed = set(tokens) | self._STOP
        # Checked per typed word, not per hint target: "gift" and "anniversary"
        # both imply necklaces, and ``_hints_for`` deduplicates targets, so
        # crediting only the first source would leave the second one behind as
        # "a key requirement is: anniversary".
        for word in self._query_words(query):
            if any(self._variants(target) & tokens
                   for target in SCENARIO_HINTS.get(word, ())):
                consumed |= self._variants(word)
        return consumed

    def opener_for(self, category: str, query: str = "") -> dict:
        """The customer frame a chosen category turns into.

        The typeahead shows this before you commit and ``assist`` sends exactly
        it, so the preview and the search can never diverge. Whatever the
        category did not account for becomes the stated requirement, which is
        how "black" survives out of "a black leather belt".
        """
        consumed = self.consumed_by(category, query)
        fixes = self.corrections(query)
        parts: List[str] = []
        for word in str(query or "").split():
            plain = self._words(word)
            if not plain:
                continue
            if self._accepted_forms(plain[0]) & consumed:
                continue
            # A bare numeral is never a feature. It is either a price, which is
            # rebuilt below in the catalog's own phrasing, or a size, which the
            # agent will ask about properly.
            if plain[0].isdigit():
                continue
            parts.append(fixes.get(plain[0], word))
        remainder = " ".join(parts).strip(" ,.")

        budget = self.budget_in(query)
        clauses = [c for c in (remainder, budget) if c]
        if clauses:
            return {
                "message": f"I'm looking for {category}. A key requirement is: "
                           f"{'; '.join(clauses)}.",
                "requirement": "; ".join(clauses),
                "budget": budget,
                "track": "buying",
            }
        return {
            "message": f"I'm looking for {category}, but I'm still exploring.",
            "requirement": None,
            "budget": None,
            "track": "browsing",
        }

    # Phrasings that mean "at most", as opposed to "about". The catalog writes
    # a product's own price as "budget around $24.99" -- a *target*, not a
    # ceiling -- so a shopper who types "under $50" and the constraint the
    # agent is handed do not mean the same thing, and the ranker has no
    # numeric price comparison at all. Rather than paper over that, the demo
    # detects the ceiling and marks the rows that break it.
    _CEILING = re.compile(
        r"\b(?:under|below|less than|cheaper than|max|maximum|no more than|"
        r"up to|within|at most)\b", re.IGNORECASE)

    @classmethod
    def budget_ceiling(cls, query: str) -> Optional[float]:
        """The price ceiling this query states, if it states one."""
        if not cls._CEILING.search(str(query or "")):
            return None
        stated = cls.budget_in(query)
        if not stated:
            return None
        try:
            return float(stated.rsplit("$", 1)[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def budget_in(query: str) -> Optional[str]:
        """A typed price, in the phrasing the catalog itself uses, or None.

        The catalog records a product's price as ``budget around $24.99`` and
        the attribute classifier reads anything starting "budget" as a budget
        constraint. Emitting the same string keeps a typed price inside the
        vocabulary the rest of the system already speaks, instead of leaving a
        bare "50" in the requirement to be matched as if it were a feature.
        """
        text = " ".join(str(query or "").split())
        if not re.search(r"[$\d]", text):
            return None
        # Only a number that reads as money: a price marker, or a currency sign.
        match = re.search(
            r"(?:under|below|less than|up to|cheaper than|max|maximum|around|about|"
            r"no more than|within|budget of|budget)\s*\$?\s*(\d{1,5}(?:\.\d{1,2})?)"
            r"|\$\s*(\d{1,5}(?:\.\d{1,2})?)"
            r"|(\d{1,5}(?:\.\d{1,2})?)\s*(?:dollars|usd|bucks)",
            text, re.IGNORECASE)
        if not match:
            return None
        amount = next(g for g in match.groups() if g)
        return f"budget around ${amount}"

    def category_signature(self, name: str) -> dict:
        """What is actually on this shelf, for the dropdown to show.

        A category name and a product count answer "does this exist?" and
        nothing else. "Accessories Belts, 258" and "Belts Belt Buckles, 18" are
        equally plausible to someone who has not seen either shelf, and picking
        wrong costs a whole session, because the agent locks its category on
        turn one.

        So each row gets the two constraints its products most often carry --
        drawn from ``card_keys``, the same disclosure surface the agent
        questions against, so the preview is made of the exact strings the
        conversation will later trade in. Plus the price band and typical
        rating, which is what a shopper actually wants to know before
        committing.

        Cached per category and computed from a sample: a shelf's character is
        not a close-run thing, and this runs on every keystroke.
        """
        cached = self._signature_cache.get(name)
        if cached is not None:
            return cached

        docs = (self.catalog.bucket.get(self.catalog.bucket_for(name)[0]) or [])[:1200]
        counts: Dict[str, int] = {}
        prices: List[float] = []
        ratings: List[float] = []
        for doc in docs:
            for key in self.catalog.card_keys[doc]:
                # A constraint carried by nearly every product in the catalog
                # ("Imported") describes no shelf in particular.
                if key and self.agent.retriever.phrase_df(key) <= self.SIGNATURE_MAX_DF:
                    counts[key] = counts.get(key, 0) + 1
            price = self.catalog.prices[doc]
            if price:
                prices.append(float(price))
            if self.catalog.ratings[doc]:
                ratings.append(float(self.catalog.ratings[doc]))

        floor = max(2, len(docs) // 12)
        common = [k for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                  if n >= floor][:2]
        prices.sort()
        signature = {
            # Truncated here rather than in CSS: these are Amazon feature
            # strings and some of them are a paragraph long.
            "traits": [k if len(k) <= 26 else k[:25].rstrip() + "…" for k in common],
            "price_low": round(prices[len(prices) // 10], 2) if prices else None,
            "price_high": round(prices[-max(1, len(prices) // 10)], 2) if prices else None,
            "priced": len(prices),
            "rating": round(sum(ratings) / len(ratings), 1) if ratings else None,
        }
        self._signature_cache[name] = signature
        return signature

    def categories(self, query: str = "", limit: int = 8) -> dict:
        """The typeahead's data source.

        The catalog is 50,000 *Clothing, Shoes & Jewelry* products and nothing
        else, so an unguided search bar is a trap: "headphones" returns clothing
        and the page looks broken while behaving correctly. Offering only
        categories that exist turns the bar into something that guides you to a
        session the agent can actually solve.

        Each row carries the sentence it would send, so what the dropdown
        promises and what the agent is handed are the same string. Rows are
        returned in two groups -- named outright, and reached through an
        occasion word -- because a shopper should be able to see which is which.
        """
        typed = " ".join(str(query or "").split())
        chat = self.classify_small_talk(typed)
        ranked = self.rank_categories_detailed(typed)

        # Thin and promotional leaves are held back while there are real
        # categories to show. "sneakers" has five genuine matches; padding the
        # list with a two-product "Girls Sneakers (fs no puma)" makes the
        # catalog look broken. They come back the moment the real ones run out,
        # so a category that is genuinely the only match is never hidden.
        solid = [item for item in ranked
                 if item["count"] >= self.MIN_ASSIST_POOL
                 and not self._is_noise_category(item["category"])]
        if len(solid) >= 3:
            ranked = solid
        else:
            tail = [item for item in ranked if item not in solid]
            ranked = solid + tail[:max(0, 4 - len(solid))]

        rows = []
        for item in ranked[:limit]:
            # "hi" is a greeting that happens to prefix "Hiking Boots". Offering
            # the category as a did-you-mean is fine; quoting "a key requirement
            # is: hi" back at someone is not.
            opener = self.opener_for(item["category"], "" if chat else typed)
            rows.append({
                **item,
                "opener": opener["message"],
                "track": opener["track"],
                "requirement": opener["requirement"],
                "thin": item["count"] < self.MIN_ASSIST_POOL,
                "noise": self._is_noise_category(item["category"]),
                **self.category_signature(item["category"]),
            })
        return {
            "total": len(self.catalog.bucket),
            "query": typed,
            "categories": rows,
            "corrections": self.corrections(typed),
            # Set when the box holds a greeting rather than a search, so the
            # dropdown can say hello instead of offering "Hiking Boots" for "hi".
            "small_talk": chat,
            "note": self.scope_note(typed, rows),
        }

    def scope_note(self, query: str, rows: List[dict]) -> Optional[str]:
        """One line explaining an empty or weak dropdown, or None when it is fine.

        A dropdown that simply vanishes is the single most confusing thing a
        search box can do, and this box vanishes legitimately all the time --
        the catalog really does not carry headphones.
        """
        if not query:
            return None
        if not rows:
            words = self._query_words(query)
            if not words:
                return ("Nothing to search on yet — name a garment, a shoe, or a "
                        "piece of jewellery.")
            return (f"No category here for “{query}”. This catalog is 50,000 "
                    "Clothing, Shoes & Jewelry products — no electronics, no home, "
                    "no groceries.")
        if all(row["noise"] or row["thin"] for row in rows):
            return ("Only thin campaign slices match this. Try a plainer word — "
                    "“belt” rather than a brand or a price.")
        return None

    def catalogue_sessions(self) -> dict:
        return {
            "sessions": [
                {
                    "sample_id": s["sample_id"],
                    "scenario": s["scenario_type"],
                    "difficulty": s.get("difficulty_bucket"),
                }
                for s in self.samples
            ]
        }

    def benchmark(self) -> dict:
        """Measured results, read from the committed artifacts.

        Nothing is recomputed here. These are the same files ``make verify``
        writes and ``tools/check_readme.py`` checks the README against, so the
        page cannot quote a number the repository does not stand behind.
        """
        def read(name: str) -> Any:
            path = ROOT / name
            try:
                with path.open(encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, ValueError):
                return None

        results = read("artifacts/results.json") or {}
        overall = results.get("overall") or results
        return {
            "ours": {
                "hit_rate_at_10": overall.get("hit_rate_at_10"),
                "mrr": overall.get("mrr"),
                "mttc": overall.get("mttc"),
                "efficiency": overall.get("efficiency"),
                "technical_score": overall.get("recommended_technical_score"),
                "sample_count": overall.get("sample_count"),
                "tokens": (overall.get("reported_token_usage") or {}).get("total_tokens"),
            },
            "baseline": read("docs/baseline_results.json"),
            "scenarios": overall.get("scenario_metrics") or {},
            "ablation": read("artifacts/ablate.json") or [],
            "robustness": read("artifacts/robustness.json") or [],
        }

    def health(self) -> dict:
        return {
            "ok": True,
            "catalog": str(self.catalog_path),
            "catalog_size": self.catalog.size,
            "index_seconds": self.index_seconds,
            "sessions_available": len(self.samples),
            "display_ready": self.display.ready,
            "baseline_ready": self.baseline is not None,
            "baseline_error": self.baseline_error,
            "reranker": type(self.agent.reranker).__name__,
        }


# --------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    demo: Demo = None  # type: ignore[assignment]
    server_version = "ShoppingCopilotDemo/1.0"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A demo server edited while running should never serve a stale asset.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- routing ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def _route(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith(API):
                self._api(method, path[len(API):] or "/", parse_qs(parsed.query))
            elif method == "GET":
                self._static(path)
            else:
                self._json({"error": "not found"}, 404)
        except BrokenPipeError:  # pragma: no cover - browser navigated away
            pass
        except Exception as error:  # pragma: no cover - defensive
            traceback.print_exc()
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def _api(self, method: str, path: str, query: Dict[str, List[str]]) -> None:
        demo = self.demo
        if method == "GET":
            handlers: Dict[str, Callable[[], Any]] = {
                "/health": demo.health,
                "/suggestions": demo.suggestions,
                "/sessions": demo.catalogue_sessions,
                "/categories": lambda: demo.categories(query.get("q", [""])[0]),
                "/refinements": lambda: demo.refinements(
                    query.get("session_id", ["web"])[0], query.get("q", [""])[0]),
                "/similar": lambda: demo.similar(
                    query.get("asin", [""])[0],
                    session_id=query.get("session_id", [""])[0]),
                "/benchmark": demo.benchmark,
            }
            handler = handlers.get(path)
            if handler is None:
                self._json({"error": "not found"}, 404)
                return
            self._json(handler())
            return

        body = self._body()
        session_id = str(body.get("session_id") or query.get("session_id", [""])[0] or "web")
        message = str(body.get("message") or "")

        if path == "/reset":
            demo.reset(session_id, body.get("profile"), str(body.get("mode") or "shopper"))
            self._json({"ok": True, "session_id": session_id})
        elif path == "/chat":
            if not message.strip():
                self._json({"error": "message is required"}, 400)
                return
            self._json(demo.chat(session_id, message, assist=bool(body.get("assist"))))
        elif path == "/baseline":
            self._json(demo.baseline_chat(session_id, message))
        elif path == "/replay":
            try:
                self._json(demo.replay(str(body.get("sample_id") or "")))
            except KeyError:
                self._json({"error": "unknown sample_id"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            path = "/copilot.html"
        target = (WEB / path.lstrip("/")).resolve()
        # Path traversal guard: everything served must sit inside web/.
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        kind = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
            kind += "; charset=utf-8"
        self._send(200, target.read_bytes(), kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    # Loopback by default -- this indexes 50,000 products and holds session
    # state in memory, so it is a development server and should not be exposed
    # by accident. Every platform-as-a-service injects PORT and expects a bind
    # on 0.0.0.0, so both are read from the environment when set.
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--limit", type=int, default=None,
                        help="index only the first N products (faster startup, worse results)")
    args = parser.parse_args()

    if not Path(args.catalog).exists():
        raise SystemExit(
            f"catalog not found at {args.catalog}\n"
            "Run `make setup` first — it downloads and checksums the frozen catalog."
        )

    print(f"indexing {args.catalog} …")
    Handler.demo = Demo(args.catalog, args.dataset, args.limit)
    print(f"indexed {Handler.demo.catalog.size:,} products in {Handler.demo.index_seconds}s")
    print("the official BM25 starter and the display index are still loading in the background")
    print(f"\n  ->  http://{args.host}:{args.port}/\n")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
