<img src="docs/shopilot-mark.svg" alt="" width="76" align="left" hspace="14">

# Shopilot

**TikTok TechJam 2026 · Track 4 — Shopping Copilot: AI Conversational Search
and Recommendations.**

Built by **Mohnish Rawat** (lead) · **Advik Jain** · **Raghav Gupta** ·
**Pranav Gupta** · **Aarav Gupta** — see [Team](#team) for who did what.

<br clear="left">

A multi-turn conversational shopping agent for the TechJam 2026 Conversational
E-Commerce Search Challenge. It finds a hidden purchase target inside a frozen
50,000-product Amazon catalog by routing intent, retrieving over three
independent routes, and — the part that matters most — choosing each
clarifying question by expected information gain.

Scored by the organizer's own evaluator, unmodified, on the 200 public sessions:

| | Hit Rate@10 | MRR | MTTC | Efficiency | **TechnicalScore** |
|---|---|---|---|---|---|
| Official weak BM25 starter | 0.125 | 0.0680 | 9.81 | 0.119 | **0.10671** |
| **This agent** | **1.000** | **0.9684** | **2.835** | **0.8165** | **0.95383** |

Every public session is solved, at a median rank of 1, in a mean of 2.83 turns
out of a budget of 10. That is **8.9× the published baseline composite**.

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 1.000 | 0.9706 | 2.388 |
| Browsing | 80 | 1.000 | 0.9795 | 2.737 |
| Intent Override | 30 | 1.000 | 0.9450 | 3.967 |
| Boundary | 10 | 1.000 | 0.9333 | 3.800 |

**Zero tokens, zero cost, no network, no required dependencies.** The default
configuration runs on the Python 3.9 standard library alone: 14 s to index
50,000 products, then **39 ms per session** — 13.7 ms per turn — end to end. An optional Claude
reranking stage can be switched on; it is off by default and every number here
is from the offline configuration.

It also holds up when the customer stops speaking in the exact phrasings it was
built against — **0.815 under a paraphrase stress test that costs an unhardened
version 0.590**. That test ships, runs locally, and needs no model: `make robust`.

### Try it in about a minute

```bash
make setup     # fetch and checksum the frozen catalog (one time, ~60 MB)
make eval      # score the agent with the organizer's evaluator
make serve     # the browser walkthrough at http://127.0.0.1:8000
```

No install step, no virtualenv, no API key. `make verify` runs the whole
check — lint, 261 tests, the official baseline reproduced, our score, and a
documentation check that every number below still matches a committed artifact.

---

## The brief's four pillars, and where each one lives

| Pillar | What the brief asks | Where it is, and what it measured |
|---|---|---|
| **I · Intent routing & hybrid pipeline** | Dual-track routing between a high-precision *buying* filter and a diverse *browsing* retrieval; multi-route retrieval → semantic ranking, in memory | `src/route.py` scores a continuous **specificity** and splits it at 0.30 / 0.55, so the two tracks are endpoints of one dial rather than a branch. `src/rank.py` fuses three routes — exact phrase, BM25, TF‑IDF cosine — each normalised to its own maximum, with per-track weights, MMR diversification on the browsing side only. The browsing track also runs an unconditional full-catalog dense recall arm and reserves its last two slots for the best out-of-category candidates, which is what makes **cross-category scenario matching** real rather than nominal: measured before, the top ten on a browsing turn were 100% in-category on all 80 browsing sessions; measured after, **20.9% of returned products come from outside the named category, in 79 of 80 sessions**, at a composite cost of 0.000. Entirely in memory, no vector DB, stdlib only. Ablated: removing the phrase route costs **0.085**. |
| **II · Multi-turn scenario evolution** | A state machine handling incremental slot accumulation and abrupt intent override; proactive clarification when the pool is over-general | `src/state.py` accumulates weighted constraints and never loses one to a bad parse; `src/parse.py` recognises the customer's frames *and* recovers constraints from free text when none matches. Over-generality is measured on the live posterior (`src/clarify.py`) and **triggers the retrieval cutoff literally**, as the brief words it: an over-general pool is answered with one believed product rather than ten guesses. It fires *after* the turn-6 safety net, deliberately — placed before it, the cutoff overrides the coverage guarantee and costs 1.5% of hit rate outright. A turn carries results **and** a question, so asking is never itself a cost. `src/state.py` also implements **slot decay over time** (the brief's own phrase): a constraint loses 0.06 of weight per turn since it was last stated, floored at 0.55 so an old constraint fades rather than expiring, and restating one refreshes it. Measured at 0.00, 0.06 and 0.15 per turn: **identical to five decimal places**, because at MTTC 2.8 a session ends before decay can bite. It ships off, and the measurement is the reason. On override the agent **demotes rather than erases**, which is a measured departure from the brief: erasing costs **0.009**, and the ablation is the argument. |
| **III · Self-evolution / dynamic context programming** | Runtime adaptation, personalized context distillation, adaptive orchestration that refines its own guidance logic | Two layers. Short term: the question is re-chosen every turn by **expected information gain** over the live posterior (`src/clarify.py`), so the workflow re-orchestrates itself from the evidence rather than from a script — remove it and the score falls **0.379**, more than everything else combined. Long term: `src/memory.py` distils conversions into safe cohort aggregates and feeds back term weights, quality affinity and per-attribute question yields, learning *across* sessions. Worth +0.0014 here, and reported as such rather than oversold. |
| **IV · Evaluation matrix** | Hit Rate@10, MRR, MTTC, efficiency | Scored by the organizer's evaluator, unmodified: **1.000 / 0.9684 / 2.835 / 0.8165 → 0.95383**. Plus an 18-arm ablation, a three-level paraphrase stress test, a reproduced official baseline, and 261 tests — every number in this file traced to a committed artifact by `make docs`. |

---

## Setup

Python 3.9 or newer. **Nothing to install** — the agent runs on the standard
library alone.

```bash
make setup     # downloads + SHA256-verifies the frozen catalog (~19 MB, ~30 s)
make verify    # lint, 261 tests, official baseline, our score, doc check
make robust    # score under three levels of customer paraphrase
```

`make setup` is the only step that needs network, and it is needed once. The
60 MB `data/catalog.jsonl` is the organizer's frozen artifact and is not
committed to this repository; the script fetches it from the participant-kit
release, checks it against the organizer's published `SHA256SUMS`, decompresses
it, and confirms the documented 50,000 rows. If you would rather do it by hand:

```bash
curl -LO https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
curl -LO https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS
shasum -a 256 -c SHA256SUMS
gunzip -c catalog.jsonl.gz > data/catalog.jsonl
```

Everything else needed to reproduce the results is committed:
`data/public_set.jsonl` (the 200 labelled sessions), and `evaluator/`,
`starter/` and `docs/` — the organizer's files, included **unmodified** so the
scoring path here is provably the official one. Only the catalog is fetched.

**To score this agent in your own harness**, you need `agent.py` and `src/`
only. Both are pure standard library, so there is no install step, no API key,
no network call and no model download on the scored path.

## Reproducing every number in this README

```bash
make verify
```

That runs the linter, the test suite, reproduces the official baseline, scores
the agent, and checks that every number quoted in this README still matches a
committed artifact. Individually:

```bash
python3 -m unittest discover -s tests -t .                    # 261 tests, ~82s
python3 tools/run_eval.py --agent baseline                    # 0.10671, matches docs/baseline_results.json
python3 tools/run_eval.py                                     # 0.95383
python3 tools/sweep.py --mode ablate                          # the ablation table below
python3 tools/headroom.py                                     # ranking safety margin
python3 tools/robustness.py                                   # paraphrase matrix
python3 tools/demo.py --sample public_0005 --reveal           # one narrated session
python3 server.py                                             # the browser walkthrough
python3 tools/check_readme.py                                 # README vs artifacts/
```

Everything is deterministic: same catalog in, same numbers out.

---

## The browser walkthrough

```bash
make serve      # -> http://127.0.0.1:8000
```

Standard library only, like everything else here: no framework, no build step,
nothing to install. It indexes the catalog once (~35 s) and is then interactive.

**This is not the scored path.** The graded artifact is `agent.py` plus `src/`;
the organizer scores headlessly and the problem statement puts UI work out of
scope. `server.py` imports the agent and never modifies it, nothing under `src/`
imports anything from it, and deleting `server.py` and `web/` leaves every
number above unchanged — `tests/test_server.py` asserts that last part.

What it is for is the two things a score cannot show: what the agent is
*thinking*, and what it is like to actually use.

**Every panel is the agent's own trace.** The routing meter is `specificity`
from `src/route.py` against that file's real 0.30/0.55 thresholds. The bar chart
is expected information gain per attribute from `src/clarify.py`, with the
chosen question highlighted — watch an attribute drop to zero the moment the
customer says it has nothing more to offer. The retrieval panel is the live
candidate pool, phrase hits, negative-evidence exclusions and top-two margin.
Nothing is mocked; if a field is missing the panel says so rather than inventing
a value.

**Replay a scored session.** Pick any of the 200 labelled public sessions and
watch it run against the organizer's own customer simulator from
`evaluator/local_evaluator.py`. The hidden target and its intent card are
revealed to *you* afterwards, never to the agent, and the verdict reproduces the
evaluator exactly — `public_0002` converts on turn 6 at rank 4 here and in
`artifacts/results.json` alike. This is the honest version of a demo: the
customer's replies are the real ones.

### Hosting it

The walkthrough needs a persistent process — it indexes 50,000 products into
memory and then keeps multi-turn session state there. That rules out a
serverless host: consecutive turns would land on different instances and the
conversation would reset every message.

`make static` records it instead. It drives a running server and captures every
payload the page would have fetched — six conversations turn by turn, 24
replayed scored sessions, the benchmark tables — into a 0.38 MB bundle that
`dist/` serves with no backend at all. Every turn in it is a real response the
agent really gave, trace included. What it cannot do is answer a sentence nobody
recorded, so the page offers the recorded conversations and says why.

### Four things the demo does that the scored agent does not

All four are in the demo layer only, and all four exist because the metric and a
human want different things.

**It returns the whole list by default.** Withholding nine results to probe with
one is worth **+0.070** of composite score (the truncation row of the ablation
table) and is a bad experience for a person: you get one item at a time with no
way to compare. So the page defaults to a second configuration —
`Options(use_truncation=False)`, which still scores **0.88384** — and offers the
scored behaviour as a toggle rather than pretending the two are the same thing.
Both configurations share one catalog and one retriever, so the second agent
costs no extra index build and no extra memory.

**It resolves a category before searching.** The evaluator names a coarse
category on turn 1 of every scenario, so the agent is built to be told one. A
person typing into a search box is not, and without a category the candidate
pool is all 50,000 products and the ranking is close to meaningless — which is
exactly what *"a black leather belt under $50"* produced. The demo therefore
matches the typed words against the catalog's real category names first, and
**shows the rewrite it made**. A search box that quietly changes your query is
worse than one that fails. A message already in a customer frame is passed
through untouched, so the scored path stays reachable exactly as the evaluator
drives it, and text the catalog cannot serve at all ("headphones") is not
rewritten — the page says so instead.

The same matcher backs the category typeahead, so the list you are shown can
never disagree with the category the agent is then given, and it tolerates
misspellings — `snekaers`, `neckalce`, `leathr belt` all resolve, and the repair
is shown rather than applied silently. The competition explicitly guarantees
pre-cleaned input, so the scored agent correctly spends nothing on spelling;
a demo search box has no such guarantee, and correction against a closed
few-thousand-word category vocabulary is cheap enough to be safe.

Two failure modes of that matcher were measured and fixed rather than argued
about. Amazon's category tree carries campaign and housekeeping nodes beside real
product types — `Shoes & Jewelry Westlake` (1,136 products, an unclassified
bucket), `Men's Watches Under $50`, `Swimwear TEST Women's Swimwear`,
`Girls Sneakers (fs no puma)`. They are indistinguishable to a word matcher and
useless to search inside, and `"mens watch"` matched a five-product campaign
twice over. Two markers select them — a coarse label that never reached a
product-type node, and a price, percentage, parenthetical or campaign word in the
name — and between them they take **189 of the catalog's 1,115 coarse categories,
holding 2,570 of 50,000 products**, without touching a single garment type. They
are demoted below everything real, never deleted: if one is genuinely the only
match, showing it beats showing nothing. Separately, a category matched *only* by
a demographic word is held back — `"mens watch"` should not offer Men Jeans.

**It understands occasions, and says when it has.** `"something warm for winter"`
shares no token with `Gloves & Mittens Cold Weather Gloves`; neither BM25 nor a
category matcher can bridge that, and before this the query returned two
one-product campaign slices. A small map from occasion words to catalog
vocabulary (`winter → coats, sweaters, gloves, scarves, beanies, boots`) supplies
the missing route at **0.7 of the weight of a word you actually typed**, so a
named category always outranks an inferred one — `"warm winter gloves"` still
resolves on *gloves*. A category reached this way is returned tagged with the
word that reached it and the dropdown labels it *via "winter"*, because a leap
the shopper did not ask for should be visible as a leap. It cannot conjure a
catalog that does not exist: `"headphones"`, `"laptop"` and `"under 100 dollars"`
still resolve to nothing, and the page says why.

**It answers a greeting instead of searching for one.** `"hi"` is a greeting that
happens to prefix *Hiking Boots*, and returning ten hiking boots for it is not a
shopping agent, it is a search box wearing a chat skin. Greetings, thanks,
farewells, "what can you do", "how are you" and off-topic asks are recognised and
answered conversationally, **and none of them spends a turn** — the brief caps a
session at ten and scores zero above that, so saying thanks halfway through a
search must not cost a retrieval round. The replies are state-aware: before a
search, "hi" explains what the catalog holds; during one it reports where the
conversation has got to (*"3 turns into Accessories Belts, 2 constraints"*), and
every reply ends with somewhere to go. None of it can reach the scored path — it
is gated on the demo's `assist` flag, which the evaluator never sets, and the
customer's four frames are asserted not to match any of it.

**The suggestion panel changes job mid-session.** A category picker is the right
tool on turn 1 and the wrong one after it: the agent fixes its category for the
life of a session by design, so offering more categories later only invites a
topic change nobody asked for. Once a category is locked the panel switches to
the constraints the *live candidate pool* can still disclose — sourced from
`card_keys`, ordered so that answers to the question currently on screen come
first, and filtered against what the session has already said. Type a different
product into it and it offers the switch openly (*"start over in Shoes Fashion
Sneakers — drops the current search"*) rather than silently ranking belts for
someone who typed *sneakers*. Every row shows the exact sentence it will send
before you commit to it, and both the row and the rewrite are generated by one
function, so the preview and the search cannot diverge.

Follow-up chips under each answer come from the same place, so every offered
answer is one the agent can really learn from — clicking one sends the
simulator's own disclosure frame, which the parser reads precisely. Nothing the
session has already said is offered back: `disclosed_keys` records only exact
card-key matches, so after "a black leather belt" the keys *leather* and
*color: black* both still look undisclosed, and the page was suggesting a
shopper their own words.

**Every result says why it is there.** A ranked list with no reasons is a leap
of faith, and "why is this one here?" is the first thing anyone asks of a
recommender. Each card carries the constraints the product literally holds,
established by the *same two-tier lookup* `src/lexical.py` scores with — the
evaluator's own cleaning for the exact tier, punctuation stripped for the soft
tier — so a chip is on a card if and only if the phrase route credited that
product for that phrase. The chip is drawn by rarity, because that is what the
route weights by: a constraint held by nine products is most of the reason a row
ranks where it does, one held by fourteen thousand ("Imported") is nearly none.
Words that are not whole constraints are reported separately and more quietly,
read off the same postings BM25 scores over. Evidence that *every* row on the
turn carries is factored out and stated once above the list, because a fact
about all ten rows is not a fact about any one of them. And a row outside the
category you named is flagged rather than hidden — the category is a large bonus
in the ranker and never a filter, so an out-of-category row means the text
evidence outweighed it, which is worth seeing.

**Hovering a row shows what the simulator would disclose about it.** The
disclosure surface is the object the whole system is built around — `card_keys`,
the ordered pool the customer draws its answers from — and a hover panel is the
one place a person can actually look at it, with the attribute that would elicit
each entry and a tick against the ones this session has already heard.

**The ten-turn cap is enforced, not suggested.** The brief scores a session zero
past ten turns, so an eleventh is not a degraded answer, it is an invalid one.
The server refuses it. Conversation stays free either way: greetings and thanks
never reach the agent and never spend a turn, so being unable to say "that's it,
thanks" is not the ending you get.

---

## How it works

A turn flows through five stages. Each module imports only from the ones above
it, so the dependency graph is a line, not a web.

```
customer utterance
      |
  parse.py      recognise the frame, recover the constraint strings verbatim
      |
  state.py      accumulate slots, absorb overrides, record negative evidence
      |
  route.py      Buying / Browsing / Blend, from an evidence-based specificity score
      |
  lexical.py    three retrieval routes: exact phrase, BM25, TF-IDF cosine
  rank.py       normalise, fuse with per-track weights, diversify, truncate
      |
  clarify.py    pick the next question by expected information gain
      |
recommendations + ask_attribute
```

### I. Intent routing and a hybrid pipeline

Routing is driven by a **specificity score** — how much the customer has
actually pinned down — rather than by keyword spotting, so it survives
rephrasing. Below 0.30 the session is Browsing, above 0.55 it is Buying, and in
between it is deliberately left as Blend: committing early is how a session
either over-filters a browser down to nothing or hands a decided buyer a
diverse spread it has to wade through.

The track selects fusion weights. Buying leans on exact constraint matching and
suppresses diversity. Browsing leans on cosine similarity, applies a stronger
quality prior, and diversifies with MMR.

Three retrieval routes run in memory over flat integer-indexed arrays:

- **Exact phrase.** An inverted index over the exact strings a customer could
  quote about a product, scored by inverse document frequency. A phrase held by
  one product in fifty thousand is conclusive; one held by fourteen thousand
  ("Imported") is nearly worthless, and IDF separates them with no
  hand-maintained stoplist.
- **BM25.** Okapi BM25 over a weighted bag of words spanning title, categories,
  features, details, store and description. This is the route that survives
  paraphrase, because it needs word overlap rather than string identity.
- **Vector.** Cosine similarity in TF-IDF space. A sparse vector-space model
  rather than neural embeddings — official scoring may run without network
  access, so shipping or downloading weights was not an option. Cosine
  normalises away document length, so it ranks short precise titles very
  differently from BM25 and genuinely contributes to the fusion.

**Semantic reranking** is the last stage of the pipeline, and it is a seam
rather than a hard dependency, because two organizer documents pull against each
other here. The specification names "Multi-Route Retrieval → LLM Semantic
Ranking" as the intended shape. `docs/submission_rules.md` warns that official
scoring may run with network access disabled, and requires every submission to
state whether it needs network and to describe its offline fallback.

So the stage is real and pluggable, and the default implementation is local:

- **`LocalReranker`** (default) keeps the fused ordering. Not a stub — the fused
  score is already a semantic ranking over three routes, and it puts the target
  first in 85.5% of sessions at zero token cost.
- **`ClaudeReranker`** (opt-in via `TECHJAM_RERANKER=claude`) reorders the top 20
  candidates with Claude through the official Anthropic SDK, using structured
  outputs so the reply is guaranteed-parseable JSON, and reports its real token
  counts in `usage`.

Every failure mode of the remote path — SDK absent, credentials unresolvable,
timeout, rate limit, malformed reply, a hallucinated index — degrades to the
local ordering, and three consecutive failures disable it for the process. A
reranker only *reorders* candidates already retrieved, so falling back costs
ranking quality and nothing else; it can never drop a candidate or fail a turn.
`tests/test_rerank.py` drives all of that through a stub client.

**Honest caveat:** the remote path has no live-API test. There are no
credentials in this environment and the submission must never require any, so
what is verified is the request shape, the permutation recovery, the token
accounting and every fallback — not a real round trip.

### II. Dialog strategy: multi-turn scenario evolution

State accumulates incrementally and absorbs intent overrides. Two decisions
here are worth more than they look, and both came out of reading the customer
policy rather than guessing:

**Negative evidence.** If a turn showed ten products and the session did not
end, the target was not among them. That is not a heuristic — it follows from
how a turn is scored — and it removes candidates for free. The one place it is
invalid is an Intent Override session before the override lands, because those
sessions are barred from converting early; the exclusion is withheld until then.

**Accumulation beats erasure.** The brief asks for slot erasure on intent
override. Measured against the actual customer, erasing is *wrong*: the
"abandoned" preference is still a true attribute of the target, so an override
is a change of emphasis, not a contradiction. The agent re-weights instead —
the new intent gets priority, old evidence is demoted rather than dropped.
Full erasure costs **0.009** of composite score (table below). This is the one
place the implementation deliberately departs from the brief's wording, and the
ablation is the argument.

The shipped default demotes by a factor of 1.0 — that is, pure accumulation.
Demoting to 0.45 measures **+0.001**, very slightly better. That is inside the
noise band for 200 sessions, so it is not treated as a win and the principled
setting is kept; `Options(demote_factor=0.45)` reproduces it. Reporting a row
that mildly contradicts the default seemed better than quietly adopting a
tuning artefact.

**Over-generality.** The clarification gate reports when the live pool is too
wide to resolve by ranking alone, and the question policy responds to it.

### III. Self-evolution: dynamic context programming

The question policy is the single highest-value component in the system, and it
is a real calculation rather than an ordered list of questions to try.

The customer does not "answer the question asked". It discloses its undisclosed
constraints **whose type matches the attribute named**, at most two at a time.
So the value of a question is entirely determined by how finely it splits the
candidates still in play — and that is computable, because for any candidate
product the agent can predict which constraints a given attribute would elicit:

```
IG(a) = H(C) − E_r[ H(C | reply r to a) ]
```

over the posterior on the live candidate set. Two useful behaviours fall out of
the arithmetic instead of being coded in. Attributes the customer has already
exhausted score exactly zero, so they are never asked twice. And the ranking of
questions tracks the catalog: `other` usually wins because it matches any
constraint type, but when a material fills both leading card slots, `feature`
wins instead because the descriptive strings it elicits cut the pool far more
finely. Neither is hardcoded.

**Long-term cohort memory** is the other half of this pillar, and two facts
about the setup decide its shape.

*There are cohorts, not users.* The agent gets a safe aggregate profile and
never an identifier — but those aggregates repeat. Across the 200 public
sessions there are only 75 distinct profile signatures, and **156 of the 200
share a signature with another session**. So there is real repeated structure to
learn from, and learning it requires no identity at all.

*The agent is never told whether it succeeded.* No callback, no reward, no
label — a session simply stops. But the evaluator ends a session the instant the
target appears in the returned list, and the truncation policy above means that
list is usually a *single* product. So "the session ended right after we offered
exactly one product" identifies that product as the target **with certainty**.
That is the supervision signal, and it costs nothing: it falls out of a design
decision made for an entirely different reason. `tests/test_memory.py` checks
every inferred conversion against real ground truth and requires zero errors.

What accumulates per cohort is deliberately weak and bounded: a quality *band*
(a Gaussian bump around the cohort's revealed level, not "higher is better" — a
cohort that buys mid-range should not be pushed to the top-rated item), a
vocabulary from converged titles, and which questions actually paid. It enters
ranking capped at 0.05 against ~1.0 for a stated constraint, and enters question
selection only as a tie-break after the measured information gain, so it can
never overrule something the shopper just said. Over a 200-session run it builds
75 cohorts from 43 certain conversions and is worth **+0.0014**.

Short-term personalisation uses only the safe aggregate profile — no
identifiers, timestamps or raw history, none of which the agent is given. Preference tags are
weighted at 0.18 against 1.0 for a stated constraint: enough to break ties,
never enough to override something the customer actually said. Measured, that
weighting is worth −0.0002 of composite score, which is to say nothing at all.
It is kept because the deliberately small weight is the point — a personalisation
layer that *could* override a stated constraint would be a worse product even if
it scored better, and this one demonstrably cannot.

### IV. Efficiency: what to return, not just what to rank

The highest-leverage decision in the whole system is **how many
recommendations to return**, and it is not obvious.

A turn is scored on the target's rank inside the returned list, and the session
ends the moment the target appears anywhere in it. Those two rules interact:
padding a low-confidence turn out to ten entries buys a small chance of a
*badly ranked* hit — and that hit ends the session before the next answer would
have put the same product first.

So while the pool is still wide and nothing discriminating has been said, the
agent returns the single candidate it actually believes, asks the
highest-value question, and ranks properly one turn later. Every hit that
lands this way lands at rank 1. MRR rises from 0.691 to 0.966 while MTTC only
moves from 2.18 to 2.86 — and MRR carries more weight in the composite (0.30)
than efficiency does (0.20), so the trade pays.

It is also the more honest interface: the agent shows what it believes rather
than padding a list to fill a quota, which is how a good human shop assistant
behaves. Three escape hatches stop it becoming a gamble:

- **From turn 6** the full ten always go out, so five full-width turns remain.
- **A clear margin at the top** sends the full ten immediately. This fires on a
  third of all turns at no cost: if we are right the target is already rank 1,
  and if we are wrong the extra nine are a free safety net.
- **Nothing left to learn** sends the full ten. Withholding is only justified
  while a question can still teach us something; if no attribute can, there is
  no better turn to wait for. This is a correctness guard rather than a tuning
  knob — without it, a session where the customer stops disclosing degenerates
  into trickling one candidate per turn until the budget runs out, which costs
  0.08 of composite score on the no-clarification arm.

`use_truncation=False` restores plain always-return-ten behaviour and still
scores **0.884**, if a reviewer prefers to see it that way.

---

## What actually drives the score

Each component switched off, everything else held fixed, all 200 sessions:

| Component removed | HR@10 | MRR | MTTC | Score | Δ |
|---|---|---|---|---|---|
| *(full system)* | 1.000 | 0.9684 | 2.835 | **0.95383** | — |
| Clarification (question policy) | 0.730 | 0.3267 | 5.420 | 0.57460 | **-0.379** |
| Exact phrase route | 0.955 | 0.8389 | 4.000 | 0.86918 | **-0.085** |
| List truncation | 1.000 | 0.6898 | 2.175 | 0.88343 | **-0.070** |
| Negative evidence | 0.985 | 0.9047 | 3.075 | 0.92242 | **-0.031** |
| Intent override erases slots | 1.000 | 0.9455 | 2.960 | 0.94444 | -0.009 |
| Long-term cohort memory | 1.000 | 0.9658 | 2.855 | 0.95263 | -0.001 |
| Profile personalisation | 1.000 | 0.9683 | 2.845 | 0.95358 | -0.000 |
| Slot decay, 0.15/turn *(added)* | 1.000 | 0.9684 | 2.840 | 0.95373 | -0.000 |
| Dual-track routing | 1.000 | 0.9684 | 2.835 | 0.95383 | 0.000 |
| Diversity (MMR) | 1.000 | 0.9684 | 2.835 | 0.95383 | 0.000 |
| Frame-free span recovery | 1.000 | 0.9684 | 2.835 | 0.95383 | 0.000 |
| Cross-category browsing spread | 1.000 | 0.9684 | 2.835 | 0.95383 | 0.000 |
| Slot decay, 0.06/turn *(added)* | 1.000 | 0.9684 | 2.835 | 0.95383 | 0.000 |
| Over-generality retrieval cutoff | 1.000 | 0.9683 | 2.825 | 0.95398 | 0.000 |
| Intent override demotes to 0.45 | 1.000 | 0.9729 | 2.850 | 0.95488 | +0.001 |
| BM25 route | 1.000 | 0.9713 | 2.810 | 0.95518 | +0.001 |
| Vector route | 1.000 | 0.9677 | 2.685 | 0.95662 | +0.003 |

**Read the bottom rows honestly: BM25, the vector route, dual-track routing, MMR,
profile personalisation and span recovery do not earn their keep on the public
set.** Removing BM25 or the vector route very slightly *improves* the score, and
span recovery is neutral to five decimal places.

Span recovery is the clearest case of a component that is worthless here and
decisive elsewhere: it is by construction inert whenever the customer speaks in a
recognised frame, which on the public set is every single turn. Under paraphrase
it is worth **+0.21** (next section). A component that scores 0.000 on the
visible data and rescues a fifth of the composite on plausible unseen data is
exactly the kind of thing an ablation table alone would tell you to delete. They are kept anyway, and
the reason is the second row of the table.

When the exact phrase route is switched off — which is precisely what
paraphrased customer text would do to it — the remaining classical IR stack
still delivers **0.870**, eight times the baseline. That is what those routes
are for. The competition specification explicitly reserves the right to
paraphrase the customer's wording, so a system that scored 0.954 with verbatim
matching and collapsed without it would be a bad bet against 800 unseen
sessions. The ~0.003 they cost is insurance, priced and paid deliberately.

MMR is a related case. It measurably reorders the browsing track (verified in
`tests/test_retrieval.py`) but moves the composite by exactly zero, because the
metric only cares about one hidden target and cannot see whether the other nine
results were ten colourways of the same shirt. It is kept because that
difference is real to a user even though it is invisible to the scorer.

### Robustness under paraphrase — the biggest real risk

The competition specification reserves the right to reword the customer:
*"If natural-language paraphrasing is added by the organizer, it cannot decide
correctness."* Since this agent leans hard on the customer quoting product
metadata verbatim, that is the largest threat to its private-set score — so
rather than argue about it, `tools/paraphrase.py` builds a local, seeded,
model-free stress test. It wraps the *agent*, not the evaluator, so the
organizer's `evaluate()` generates its normal messages, the wrapper garbles them,
and scoring stays exact-identifier matching.

Three levels: **light** rewords the eight customer frames but leaves product
attributes verbatim; **medium** also perturbs case, punctuation and list order;
**heavy** genuinely paraphrases the attributes themselves with synonyms and
dropped filler.

| Customer text | Unhardened | **Hardened** | Recovered |
|---|---|---|---|
| verbatim (as scored) | — | **0.95383** | — |
| light paraphrase | 0.59011 | **0.81451** | +0.224 |
| medium paraphrase | 0.59401 | **0.81822** | +0.224 |
| heavy paraphrase | 0.59531 | **0.73857** | +0.143 |

> **This table was not reproducible until it was audited, and the comparison it
> makes was not a fair one.** The wrapper reseeded per session from the
> `session_id` — which looks stable and is not: the organizer's evaluator names
> every session `f"public_{uuid.uuid4().hex}"`, freshly, on every run. So the
> `--seed` flag did nothing, two consecutive runs of `make robust` on identical
> code disagreed by up to **0.017** of composite score, and worse, each arm was
> handed *different customer text* — the one variable the experiment exists to
> hold fixed. Seeding on the session's position in the dataset instead makes the
> run reproducible and hands both arms the same words; `tests/test_robustness.py`
> now asserts both properties. The numbers above are from the fixed harness, and
> the hardening looks slightly *better* under a fair comparison than it did under
> an unfair one.

**The finding that drove this was not the one we expected.** Rewording only the
*frames* — leaving every product attribute verbatim — cost as much as
paraphrasing the attributes too. Before hardening, `light` scored **0.490**. The
damage was almost entirely the parser: eight hard-coded regexes that recognised
the customer's exact sentence shapes and recovered nothing when those shapes
moved. The constraint text was still sitting there in the message, unread.

Two fixes, both frame-independent:

- **Category anywhere.** `find_category` scans for the longest catalog category
  occurring anywhere in the sentence, instead of expecting it after a fixed
  prefix. Under heavy paraphrase this recovers the category in **200/200**
  sessions and puts the target in the candidate pool **200/200** times.
- **Span recovery.** When no frame is recognised, `Retriever.match_spans`
  enumerates the contiguous word spans of the message and keeps the ones that
  are literally constraint strings in the catalog. Over-generation is free —
  a wrong split matches nothing — so "must-have: Buckle closure", "what matters
  there is Buckle closure" and "it needs a Buckle closure" all yield the same
  span without anyone enumerating phrasings.

Span recovery runs **only when no frame matched**, which is why it is exactly
neutral on the clean path (0.95383 with and without, see the ablation table).
An earlier version ran it whenever no constraints had been parsed, and that cost
0.005: a browsing opener legitimately carries no constraint, and the scan
invented one out of the category words.

What remains at `heavy` is irreducible rather than fixable. The category is
recovered every time and the target is always in the pool; the 23 remaining
misses are pure ranking, because when the customer's words no longer match any
catalog text there is nothing left but semantic overlap. we would also expect the
realistic private-set risk to sit nearer `light` than `heavy` — paraphrasing the
*product attributes* out of recognisability would break the organizer's own
baseline too.

Reproduce with `make robust`.

### Safety margin

`tools/headroom.py` replays every session and records the target's position in
the agent's full internal ranking:

| Target reaches internal rank ≤ | 1 | 2 | 3 | 5 | 10 | 20 | 37 |
|---|---|---|---|---|---|---|---|
| Share of sessions | 85.5% | 91.5% | 94.0% | 96.0% | 98.5% | 99.0% | 100% |

Worst case across all 200 sessions is rank 28. The turn budget reaches internal
rank 55 under the shipped configuration, so there is genuine headroom rather
than a coincidence. This measurement is why `LATE_TURN` is 6 and not 8: 8 scores
0.0016 higher on the public set (noise at n=200) but only reaches rank 37, and
hit rate carries half the composite on 800 sessions nobody has seen.

---

## Design decisions worth defending

**Reimplementing four evaluator functions.** Category coarsening, constraint
cleaning, value flattening and constraint classification are reimplemented in
`src/` so the submission is self-contained at run time and never imports
organizer files. Reimplementation is only safe while the copies stay identical,
so `tests/test_replicas.py` differential-tests all four against
`evaluator.local_evaluator` over thousands of real products, plus a check that
our predicted intent card matches the evaluator's exactly. If the organizer
changes the evaluator, those tests fail loudly instead of the agent quietly
losing accuracy.

**Never raising.** The evaluator scores a raised exception exactly like a wrong
answer: the whole session counts as a miss. So `respond` is fully wrapped, an
unknown `session_id` self-heals into a fresh session rather than erroring, and
the last-resort fallback is still schema-valid. `tests/test_contract.py` attacks
it with empty strings, 5,000-character messages, malformed turn numbers,
`None` profiles and SQL-injection-shaped input, and requires a contract-valid
response every time.

**Rarity gates recall, not just ranking.** A disclosed phrase may nominate
candidates only if at most 500 products carry it. That ceiling is not arbitrary:
the largest coarse category in the catalog holds **1,354** products, so a phrase
less selective than a whole category cannot narrow anything by definition. Above
it a phrase is a property, not an identifier — letting "Buckle closure" nominate
candidates replaced a 258-product category with an 18,000-product haystack.
Gating it left the score identical and made evaluation **5× faster**
(33 s → 6.6 s).

**Flat arrays over dicts.** Term postings are three flat `array` buffers with
per-document offsets rather than 50,000 Python dicts, which is the difference
between ~50 MB and several hundred.

**A bug the browser found that 200 sessions could not.** When no frame matched
and no catalog category appeared in the sentence, the opening turn used the
*whole message* as the category — and `_absorb` then locked that slot for the
rest of the session, so a real category arriving later could never replace it.
On the public set this line is unreachable: every turn matches a frame. Typing
"headphone" into the demo reaches it immediately, and the agent then reported
"my best match in headphone" and could not recover for ten turns. The fix marks
a category as *exact* only when the catalog actually has it, keeps the raw text
as a provisional query hint, and lets a real category displace it. The scored
result is unchanged to five decimal places — 0.95383 before and after, since the
path never runs on the public set — and the paraphrase matrix moved by less than
0.011 in either direction, inside the noise band for 200 sessions. It is worth
recording because the demo was supposedly out of scope, and building it found a
latent correctness bug that would only ever have shown up on paraphrased private
sessions.

**Three more the browser found, and one the audit did.** The official BM25
starter indexes into an in-memory SQLite database. `sqlite3` connections are
bound to the thread that opened them, and the demo server is threaded — so the
comparison column raised `ProgrammingError` on *every* request and printed the
traceback where results should have been. The starter is an organizer file and
stays unmodified, so the confinement lives in `server.py`: it is built and
queried on a single-worker executor. Separately, a message in one of the
customer's own frames was being run through the topic-change detector, whose
content words ("colour", "gold") name a different category than the one being
refined — so clicking a follow-up chip silently threw the session away and
restarted it at turn 1. One definition of "this is a frame" now gates the whole
assist layer. And the hover panel was positioned `absolute` inside a container
with `overflow-y: auto`, which clipped it on every row below the fold.

The audit one is the most interesting, because nothing was broken and everything
was wrong. The "what it returned" panel *reconstructs* `_trim` from the trace
rather than instrumenting it, and reconstruction can disagree with the code it
describes. It did, three ways: it read thresholds off the scored agent while the
page defaults to the untruncated one, so every turn on the default configuration
was explained by a confidence margin that had decided nothing; it never
mentioned the barren-turn branch; and it tested margin before information gain
where `_trim` tests gain first, so when both held it named the wrong one. The
branches are now in `_trim`'s order against the options of the agent that
actually ran — and on the untruncated configuration the panel states the
counterfactual instead, which `tests/test_cards.py` verifies by running the
scored agent over the same turns and comparing.

**Three more that only a real conversation could find.** Driving the demo end
to end the way a person would — *hi · a gift for my wife under $50 · [pick a
chip] · actually I want sneakers instead · none of these · thanks, bye* — broke
it three ways. `"actually I want sneakers instead"` was swallowed by the
frame guard, because the override frame begins "Actually, ignore my earlier
preference…" and the guard matched on the bare word; the shopper asked for shoes
and kept refining necklaces. `is_customer_frame` now asks `src.parse` what a
frame is, which is the only definition that cannot drift. `"none of these"` was
*searched for the word "these"* and answered "No category here for “these”" —
the parser recognises exactly one phrasing of rejection and a person has twenty,
so loose rejections are now mapped onto the customer's own line and the rewrite
is shown, never applied silently. And `"thanks, bye"` was treated as a query,
because both small-talk patterns are anchored at each end and neither half is
the whole string; the classifier now consumes small-talk phrases greedily and
requires every word to be accounted for, so `"thanks, now show me boots"` is
still a search.

**A number in this README that was simply wrong.** The paragraph above about
merchandising categories quoted "279 of 1,115, holding 2,879 products". The real
figures are **189** and **2,570**: the numbers came from a wider draft of the
marker set than the one that shipped, and nothing checked them because they were
counted from the catalog rather than read from an artifact. `make docs` now
re-counts them, along with the largest-category figure, and fails on drift. Prose
that quotes measurements is code that can be wrong.

**Two bugs worth recording.** An early version rebuilt a set inside a dict
comprehension's condition, costing 92% of total runtime; profiling found it, and
the one-line fix made the system 12× faster with identical output. Separately,
instrumenting the shipped configuration suggested the "nothing left to learn"
guard never fired, so it was deleted as dead code — which silently cost 0.08 on
the no-clarification arm, because the guard only binds on paths the default
configuration does not take. It is restored, and now has a direct test rather
than an inference from a profile.

---

## Cost, latency and resource disclosure

| | Default (what every number above uses) | With `TECHJAM_RERANKER=claude` |
|---|---|---|
| Model | none | `claude-opus-5`, effort `low` |
| Network access required | **none** | yes, for the rerank call only |
| API keys / credentials | none | your own, via the SDK's normal resolution |
| Token usage | **0** prompt, **0** completion | ~700 prompt / ~80 completion per reranked turn |
| Estimated cost | **$0.00** | ~$0.005 per session at list price |
| Dependencies | Python 3.9+ stdlib only | `pip install anthropic` |
| Index build | 14.3 s, once per process, all 50,000 products | unchanged |
| Latency | **39 ms per session**, 13.7 ms per turn on the scored mix, single-threaded | + one API round trip per reranked turn |
| Worst-case turn | 27 ms on an all-browsing workload, where the cross-category dense recall arm runs every turn | unchanged |
| Peak memory | **403 MB** resident for the agent and its index, measured with `resource.getrusage` against a 10 MB interpreter baseline | unchanged |

**Where the time goes, and what was done about it.** Profiling the scored run
puts 56% of total runtime in full-catalog BM25. The cross-category dense recall
arm added to the browsing track calls it on every browsing turn, which cost 26%
of total runtime when it was first written with the same 600-document budget the
thin-pool rescue path uses. The two arms want different budgets: the rescue may
be the *only* source of candidates, while the browsing arm only has to fill the
two tail slots the ranker reserves. Splitting them — 600 for the rescue, 120 for
the browsing arm — returned the run to **7.8 s for 200 sessions** with an
identical composite score to five decimal places, and cost 0.2 percentage points
of cross-category reach (21.1% to 20.9%).

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `TECHJAM_CATALOG` | `data/catalog.jsonl` | catalog path when the harness does not pass one |
| `TECHJAM_RERANKER` | `local` | `claude` enables the LLM reranking stage |
| `TECHJAM_RERANK_MODEL` | `claude-opus-5` | model id for that stage |
| `TECHJAM_RERANK_EFFORT` | `low` | `low` … `max` |
| `TECHJAM_RERANK_TIMEOUT` | `8` | seconds before falling back to local |

**The offline fallback is the default path**, which is the unusual direction for
this disclosure to run. With the reranker off there is nothing that can fail:
the agent behaves identically with and without network. With it on, any failure
— missing SDK, unresolvable credentials, timeout, rate limit, malformed reply —
falls back to exactly the default behaviour, and three consecutive failures
disable the stage for the process.

---

## Is the score fitted to the 200 public sessions?

`make crossval` answers this rather than asserting it. Every configuration in
the documented tuning grid (36 of them) is evaluated once over all
200 sessions and its per-session records kept, so a fold's score is arithmetic
rather than another run. For each fold the best configuration is chosen on the
other four and scored on the held-out one.

| | |
|---|---|
| Cross-validated estimate | **0.95317** ± 0.01158 (sd across 5 folds) |
| Best in-sample | 0.95407 |
| **Optimism (overfitting)** | **+0.00090** |
| Shipped constants, in-sample | 0.95263 |

**The constants are not meaningfully overfit.** Choosing them on four fifths of
the data and scoring on the fifth costs 0.00090 — about a
thousandth of composite, and two orders of magnitude smaller than the fold
spread.

**The fold standard deviation of ±0.0116 is the more useful number.**
It is this benchmark's empirical noise floor at n=200, and it is what licenses
the claim made throughout this file that differences below roughly 0.01 are not
resolvable here. Every ablation row inside that band — BM25, the vector route,
routing, MMR, span recovery, slot decay, the over-generality cutoff,
cross-category browsing — should be read as "no measured effect", not as a win
or a loss. It is measured, not asserted.

**Four of five folds preferred `late_turn=7` over the shipped `6`**, and in the
full configuration `late_turn=8` scores 0.95519 against 0.95383. That
is a real ordering on the public set and it is deliberately not taken. The whole
spread across `late_turn` 5–8 is 0.0026, a fifth of the fold noise, while the
constant itself controls something the noise cannot see: probing turns walk the
ranking one position at a time and full turns walk it ten, so `late_turn=6`
reaches internal rank 55 before the budget runs out against 37 for `8`. Against
800 unseen sessions where hit rate carries half the composite, the wider safety
margin is worth more than a difference this benchmark cannot resolve. Tuning to
the fifth decimal of 200 sessions is how a submission wins the public set and
loses the private one.

**The score is not order-invariant.** Long-term cohort memory accumulates across
sessions, so the sequence they are evaluated in changes the result. Measured over
4 orderings: **0.95368 to 0.95443, a spread of
0.00075**. That is small, and inside the fold noise, but a figure
quoted to five decimal places should say so rather than imply a point estimate.
The headline **0.95383** is the as-shipped ordering.

---

## Limitations, and what we would do next

**The public set is 200 sessions and the private set is 800.** Differences
below roughly 0.01 of composite score are not resolvable at this sample size.
The ablation rows for BM25, vector, routing and MMR all sit inside that band and
should be read as "no measured effect", not as wins or losses. The rows for
clarification, phrase matching, truncation and negative evidence are far outside
it and are real.

**The strongest single signal may not survive paraphrase — now measured, not
assumed.** Customer disclosures are verbatim substrings of product metadata,
which is what makes exact phrase matching so effective, and the specification
reserves the right to change that. `tools/paraphrase.py` and `make robust`
quantify it: **0.815 under light paraphrase against 0.954 verbatim**. That is a
real loss and we would not pretend otherwise. What we would not now claim is that
the system *collapses* — it degrades, and the fusion weights were re-checked in
the regime they exist to protect rather than only in the clean one. The residual
gap is ranking quality on text that no longer matches any catalog string, which
no amount of parsing can recover.

**Question selection assumes it can model the customer.** Expected information
gain is computed against a predicted disclosure policy. `choose_attribute` falls
back to the most productive untried attribute when that model says nothing is
left to learn, precisely because the model can be wrong — but a genuinely
different policy would degrade question quality before anything else.

**Gender and other dropped facets.** The opening category is the last two
segments of the product's category path, so `["…", "Men", "Accessories",
"Belts"]` arrives as "Accessories Belts" and the gender is simply gone. Session
`public_0002` is a men's belt whose four constraints are all generic
("leather", "100% Leather", "Imported", "Buckle closure"); it converts at turn 6,
rank 4, having spent several turns on women's belts. Mining `details.Department`
as a soft prior would likely fix that class of session, and we would test it next.

**The LLM reranking stage is unverified against a live API.** It is implemented,
schema-correct, and fully covered by stub-driven tests, but this environment has
no credentials and the submission must never require any, so no real round trip
has been made. Before relying on it in a scored run we would want one live
session and a measured comparison — and we would still ship with it off, because
the offline configuration is the one every number here comes from.

**Long-term memory is real but small.** It is worth +0.0014, which is inside the
noise band above. It works — 43 certain conversions, 75 cohorts, every inference
verified correct against ground truth — but with hit rate already at 1.000 and
MRR at 0.968 there is very little room for a weak prior to matter. Its value
would show on a harder catalog or a longer run, and we have not demonstrated that.

**Not attempted.** Neural embeddings, which the offline constraint rules out,
and any per-user (as opposed to per-cohort) modelling, which the anonymised
profile rules out by design.

**Deliberate deviation from the brief, restated plainly.** Pillar II asks for
slot erasure on intent override and this agent re-weights instead, because
erasure measures 0.009 worse against the actual customer policy. `Options(
override_erases=True)` restores the literal behaviour and `tools/sweep.py
--mode ablate` prints what it costs.

---

## Project layout

```
agent.py                 the scored artifact: exports Agent per the contract
src/
  normalize.py           text canonicalisation and phrase keys
  attributes.py          attribute vocabulary and the customer's classifier
  catalog.py             the frozen catalog, parsed once into flat arrays
  lexical.py             exact-phrase, BM25 and TF-IDF cosine routes
  parse.py               customer utterance -> structured observation
  state.py               slots, overrides, negative evidence
  route.py               dual-track routing
  rank.py                fusion, quality prior, MMR, candidate pool
  clarify.py             expected-information-gain question selection
  rerank.py              semantic reranking seam: local default, Claude opt-in
  memory.py              long-term cohort memory across sessions
  agent.py               the turn loop
server.py                demo server for the walkthrough -- NOT the scored path
web/                     the walkthrough page (html/css/js, no build step)
dist/                    the same page baked static by `make static`, for a
                         host that cannot run a persistent process
tools/
  run_eval.py            drive the official evaluator against our agent
  sweep.py               ablations and parameter sweeps
  demo.py                narrate one multi-turn session
  headroom.py            ranking safety-margin measurement
  paraphrase.py          customer-paraphrase stress harness (local, seeded)
  robustness.py          the hardened/unhardened paraphrase matrix
  setup_data.py          fetch + checksum the frozen catalog
  check_readme.py        every README number must match a committed artifact
  probe.py               inspect what the customer actually discloses
  analyze_signal.py      how much each signal narrows the catalog
  lint.py                stdlib hygiene check (no third-party linter available)
tests/                   261 tests: differential, contract, unit, robustness,
                         test_pillars.py maps 1:1 onto the brief's four pillars,
                         end-to-end, the demo server's HTTP surface, the
                         conversational layer (small talk, occasion routing,
                         category quality, mid-session suggestions), and the
                         claims the cards make about the ranker
evaluator/ starter/ docs/ data/   organizer files, unchanged
artifacts/               committed results behind every number above
```

An earlier prototype of this project — built against a self-generated synthetic
catalog and a self-written user simulator, before the official participant kit
was incorporated — is kept locally under `legacy/` but excluded from this
repository. Its metrics are not comparable to anything here (it measured HR@5
against its own simulator, not HR@10 against the organizer's) and nothing in the
current system depends on it.

## Deliverables

### Tools, APIs, libraries and data

The brief asks the written description to state these explicitly, so they are
here in one place rather than scattered through the prose.

| | |
|---|---|
| **Development tools** | VS Code, git, macOS/Linux terminal. No notebook, no build step, no bundler — the demo page is hand-written HTML/CSS/JS served by `server.py`. |
| **Languages** | Python 3.9+ (agent, server, tooling); vanilla JavaScript, CSS and HTML (walkthrough page only). |
| **Libraries and frameworks** | **None required.** The scored agent, the evaluator harness and the demo server run on the Python standard library alone — no numpy, no scikit-learn, no PyTorch, no Hugging Face, no web framework. BM25, TF‑IDF cosine, MMR diversification, the softmax posterior and the information-gain calculation are all implemented directly in `src/`. The only optional dependency is `anthropic>=1.0`, used exclusively by the off-by-default LLM reranking stage in `src/rerank.py`. |
| **APIs used** | **None in the scored path.** No model API is called, no network request is made, and reported token usage is a measured zero. `src/rerank.py` can call the Anthropic Messages API when `TECHJAM_RERANKER=claude` is set; it is off by default, degrades to the local stage if the package or credentials are absent, and no number in this README comes from it. |
| **Datasets and assets** | The organizer's frozen competition kit only: a 50,000-product catalog and 200 labelled public sessions derived from **Amazon Reviews 2023** (`Clothing_Shoes_and_Jewelry`), verified by SHA256 in `tools/setup_data.py`. No external data, no scraped data, no hand-labelled data, no pretrained weights. The catalog is read-only; nothing here mutates it or injects identifiers. See `DATA_ATTRIBUTION.md`. |
| **Fonts** | Bricolage Grotesque, Inter and JetBrains Mono via Google Fonts, on the walkthrough page only. |

### Team

| Member | Role | Area |
|---|---|---|
| **Mohnish Rawat** | Team lead · frontend | Product direction, the browser walkthrough (`web/`) — conversation view, result cards, the suggestion panel and the live state panels |
| **Advik Jain** | Full-stack | The demo server's HTTP surface (`server.py`) and the query-assist layer that turns human phrasing into the frames the agent is scored on |
| **Raghav Gupta** | Backend | Retrieval and ranking (`src/rank.py`, `src/lexical.py`) — the three fused routes, per-track weights and diversification |
| **Pranav Gupta** | Backend | Dialog state and the question policy (`src/state.py`, `src/clarify.py`, `src/route.py`) — intent routing, slot accumulation and expected information gain |
| **Aarav Gupta** | Full-stack | Evaluation and tooling (`tools/`, `tests/`) — the ablation and cross-validation harnesses, the 261-test suite and the documentation check |

### Required artefacts

- **Agent entry file** — `agent.py`, exporting `Agent` per `docs/agent_api_contract.json`.
- **Setup and reproduction** — above; `make verify` runs the lot.
- **Report** — this file: architecture, model choice, cost, limitations.
- **Demonstrated multi-turn session** — two ways. `python3 tools/demo.py
  --sample public_0002 --reveal` narrates a hard Intent Override session in the
  terminal, showing the routing decision, candidate pool, expected information
  gain per attribute, and what the agent chose to return and why. `make serve`
  does the same in a browser, live, and can replay any of the 200 labelled
  sessions against the organizer's own customer simulator.
- **Network requirement** — none. Runs fully offline.

## Licence

Code in this repository is MIT licensed — see `LICENSE`. The catalog and
evaluation sessions are the organizer's frozen competition data and are governed
by their own terms; `evaluator/`, `starter/` and the organizer's files under
`docs/` are reproduced unmodified and remain theirs.

## Data attribution

Derived from the Amazon Reviews 2023 dataset
(https://amazon-reviews-2023.github.io/) as frozen and distributed by the
organizer. See `DATA_ATTRIBUTION.md`. The catalog is treated as strictly
read-only; nothing in this repository mutates it or injects identifiers.
