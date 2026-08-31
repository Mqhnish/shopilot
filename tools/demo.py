"""Walk one multi-turn session, showing the agent's reasoning at every turn.

This is the "one demonstrated multi-turn session" deliverable, and the script to
record the demo video from. It drives the *organizer's* customer simulator, so
the customer's replies are the real ones, not a scripted mock -- the hidden
target and its intent card are never shown to the agent.

    python3 tools/demo.py --sample public_0002
    python3 tools/demo.py --scenario browsing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
)
from src.agent import Options, ShoppingAgent  # noqa: E402

RULE = "=" * 78


def show(text: str, width: int = 74, indent: str = "    ") -> str:
    text = " ".join(str(text).split())
    return indent + (text if len(text) <= width else text[: width - 1] + "…")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample", default="", help="sample_id, e.g. public_0002")
    parser.add_argument("--scenario", default="", help="buying | browsing | intent_override | boundary")
    parser.add_argument("--reveal", action="store_true", help="print the hidden target up front")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.sample:
        samples = [s for s in samples if s["sample_id"] == args.sample]
    elif args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    if not samples:
        raise SystemExit("no session matched that filter")
    sample = samples[0]

    print("loading the frozen 50,000-product catalog…")
    _ids, categories, products = catalog_index(args.catalog)
    agent = ShoppingAgent(args.catalog, options=Options())

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print(RULE)
    print(f"SESSION {sample['sample_id']}   scenario={sample['scenario_type']}   "
          f"difficulty={sample['difficulty_bucket']}")
    print(f"profile: {sample['user_profile']['summary']}")
    if args.reveal:
        print(f"\nHIDDEN TARGET (never shown to the agent): {target}")
        print(show(products[target]["title"]))
        for value in card["hard_constraints"]:
            print(show(f"hard: {value}"))
        for value in card["soft_preferences"]:
            print(show(f"soft: {value}"))
    print(RULE)

    agent.reset(sample["sample_id"], sample["user_profile"])
    disclosed, boundary_used = set(), False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- turn {turn} " + "-" * 62)
        print(f"  CUSTOMER  {show(message, indent='')}")
        response = agent.respond(sample["sample_id"], message, turn, TOP_K)
        trace = agent._sessions[sample["sample_id"]].last_trace

        print(f"  ROUTE     track={trace.get('track')} "
              f"specificity={trace.get('specificity')} "
              f"constraints_known={trace.get('known_constraints')}")
        print(f"  RETRIEVE  pool={trace.get('pool')} in_category={trace.get('in_bucket')} "
              f"phrase_matches={trace.get('phrase_docs')} excluded={trace.get('excluded')}")
        print(f"  ASK       {trace.get('ask')}   expected_gain={trace.get('gains')}")
        print(f"  RETURN    {trace.get('returned')} of top-{TOP_K}"
              + ("  (holding back while questioning still pays)" if trace.get("trimmed") else ""))
        print(f"  AGENT     {show(response['message'], indent='')}")
        for rank_, item in enumerate(response["recommendations"][:5], start=1):
            asin = item["parent_asin"]
            marker = "  <-- TARGET" if asin == target else ""
            print(show(f"{rank_}. {asin}  {products[asin]['title']}"[:92] + marker))

        shown = [item["parent_asin"] for item in response["recommendations"]]
        if override_applied and target in shown[:TOP_K]:
            rank_ = shown.index(target) + 1
            print(f"\n{RULE}\nCONVERTED on turn {turn} at rank {rank_}  "
                  f"(reciprocal rank {1 / rank_:.3f})\n{RULE}")
            return
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
    print(f"\n{RULE}\nno conversion within {MAX_TURNS} turns\n{RULE}")


if __name__ == "__main__":
    main()
