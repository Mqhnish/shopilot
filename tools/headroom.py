"""How much safety margin does the ranking actually have?

The agent withholds candidates early, so the question that matters for
robustness is not "did it hit" but "how far down our own ranking was the target
when the session ended". If the target is reliably in the top handful, the
narrow early lists are cheap. If it is often at rank 30, they are a gamble.

This replays every public session against the real evaluator machinery and
records the target's position in the agent's *full* internal ranking each turn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
)
from src.agent import Options, ShoppingAgent  # noqa: E402
from src.catalog import Catalog  # noqa: E402
from src.lexical import Retriever  # noqa: E402
from src.rank import rank  # noqa: E402
from src.route import route, weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/headroom.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _ids, categories, products = catalog_index(args.catalog)
    catalog = Catalog(args.catalog)
    retriever = Retriever(catalog)
    agent = ShoppingAgent(args.catalog, options=Options(),
                          catalog=catalog, retriever=retriever)

    best_ranks = []
    per_turn = {}
    unreachable = []
    for sample in samples:
        session_id = sample["sample_id"]
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed, boundary_used = set(), False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

        best = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            state = agent._sessions[session_id]
            # Re-rank without truncation to read the agent's true ordering.
            track, _ = route(state)
            # Diversity is off here: MMR is quadratic in the list length and we
            # want the raw relevance ordering, not the presented one.
            docs, _trace = rank(catalog, retriever, state, dict(weights(track)),
                                top_k=200, use_exclusions=False, use_diversity=False)
            order = [catalog.asins[d] for d in docs]
            position = order.index(target) + 1 if target in order else None
            if position is not None:
                per_turn.setdefault(turn, []).append(position)
                best = position if best is None else min(best, position)
            shown = [r["parent_asin"] for r in response["recommendations"]]
            if override_applied and target in shown[:TOP_K]:
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
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
        if best is None:
            unreachable.append(sample["sample_id"])
        else:
            best_ranks.append(best)

    hist = Counter(min(r, 51) for r in best_ranks)
    print(f"sessions where the target ever entered the top 200: {len(best_ranks)}/{len(samples)}")
    if unreachable:
        print(f"never ranked in top 200: {unreachable}")
    print(f"best internal rank: median={statistics.median(best_ranks)} "
          f"mean={statistics.fmean(best_ranks):.2f} max={max(best_ranks)}")
    for bound in (1, 2, 3, 5, 10, 20, 37, 50):
        share = sum(1 for r in best_ranks if r <= bound) / len(best_ranks)
        print(f"  target reaches rank <= {bound:<3d}: {share:6.1%}")
    Path(args.output).write_text(json.dumps({
        "best_ranks": best_ranks,
        "unreachable": unreachable,
        "histogram": {str(k): v for k, v in sorted(hist.items())},
        "per_turn_median": {str(t): statistics.median(v) for t, v in sorted(per_turn.items())},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
