"""Reconstruct exactly what an agent sees, using the official evaluator's own
message-generation functions. Read-only analysis; no scoring logic here."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator.local_evaluator import (  # noqa: E402
    load_jsonl, catalog_index, coarse_category, intent_card,
    materialize_hidden_fields, initial_message, classify_constraint,
)

def main() -> None:
    samples = load_jsonl("data/public_set.jsonl")
    _ids, categories, products = catalog_index("data/catalog.jsonl")
    attr_hist, n_hard, n_soft = Counter(), Counter(), Counter()
    for s in samples[:8]:
        target = s["ground_truth"]["parent_asin"]
        card, behavior = materialize_hidden_fields(s, products)
        cat = coarse_category(categories.get(target, []))
        disclosed: set[str] = set()
        msg = initial_message({**s, "intent_card": card, "behavior": behavior}, cat, disclosed)
        print("=" * 100)
        print(f"{s['sample_id']}  scenario={s['scenario_type']}  difficulty={s['difficulty_bucket']}")
        print(f"TARGET   {target} :: {products[target]['title'][:90]}")
        print(f"CATEGORY {cat!r}")
        print(f"TURN-1   {msg!r}")
        for k in ("hard_constraints", "soft_preferences"):
            for v in card[k]:
                print(f"  {k[:4]:4s} [{classify_constraint(v):9s}] {v!r}")
        if behavior.get("override"):
            print(f"  OVERRIDE turn={behavior['override']['turn']} msg={behavior['override']['message']!r}")
    for s in samples:
        card, _ = materialize_hidden_fields(s, products)
        n_hard[len(card["hard_constraints"])] += 1
        n_soft[len(card["soft_preferences"])] += 1
        for k in ("hard_constraints", "soft_preferences"):
            for v in card[k]:
                attr_hist[classify_constraint(v)] += 1
    print("=" * 100)
    print("attribute distribution over all 200 targets:", dict(attr_hist.most_common()))
    print("hard_constraints count:", dict(n_hard), " soft_preferences count:", dict(n_soft))

if __name__ == "__main__":
    main()
