"""Cross-validate the tuned constants, and price the noise floor.

`make eval` reports a score computed on the same 200 sessions the constants were
chosen on. That is an in-sample number. The private split is 800 unseen sessions
with different users *and* different target products, so the question this tool
answers is: how much of the headline figure is fitted to these 200?

Method. Every configuration in the grid is evaluated once over all 200 sessions
and its per-session records kept, so a fold's score is arithmetic rather than
another run -- 36 configurations cost 36 evaluations, not 36 x 5. For each fold
the best configuration is chosen on the other four and scored on the held-out
one; the mean of the five held-out scores is the generalisation estimate and the
gap to the in-sample best is the optimism.

The standard deviation across folds is the more useful output of the two. It is
the empirical noise floor of this benchmark, and it is what licenses the claim
made throughout the README that differences below roughly 0.01 are not
resolvable at this sample size.

Long-term memory is disabled here on purpose. With it on, a session's outcome
depends on the sessions before it, so a fold's score is not well defined
independently of the rest; `--order` measures that separately.

    python3 tools/crossval.py               # 5-fold CV over the tuning grid
    python3 tools/crossval.py --order       # how much session order alone moves it
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, evaluate, load_jsonl, metric_summary,
)
from src.agent import Options  # noqa: E402
from src.catalog import Catalog  # noqa: E402
from src.lexical import Retriever  # noqa: E402

# The three constants the README documents as swept. Everything else is either
# a correctness guard or was set by an argument rather than by search.
GRID = [
    {"narrow_k": k, "late_turn": t, "confident_margin": m}
    for k in (1, 2, 3)
    for t in (4, 5, 6, 7)
    for m in (0.15, 0.22, 0.35)
]

SHIPPED = {"narrow_k": Options().narrow_k,
           "late_turn": Options().late_turn,
           "confident_margin": Options().confident_margin}


def score_of(records: list) -> float:
    """The evaluator's own composite, over an arbitrary subset of sessions."""
    summary = metric_summary(records)
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    return (0.50 * summary["hit_rate_at_10"]
            + 0.30 * summary["mrr"]
            + 0.20 * efficiency)


def sweep(catalog_path: str, samples: list, folds: int, seed: int) -> dict:
    catalog_ids, categories, products = catalog_index(catalog_path)
    catalog = Catalog(catalog_path)
    retriever = Retriever(catalog)

    per_config: dict = {}
    started = time.time()
    for index, config in enumerate(GRID, 1):
        agent = Agent(catalog_path, options=Options(use_memory=False, **config),
                      catalog=catalog, retriever=retriever)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        per_config[json.dumps(config, sort_keys=True)] = result["sessions"]
        print(f"  [{index:2d}/{len(GRID)}] {config} -> "
              f"{result['recommended_technical_score']:.5f}  "
              f"({time.time() - started:.0f}s)", flush=True)
        del agent
        gc.collect()

    by_id = {key: {r["sample_id"]: r for r in rows} for key, rows in per_config.items()}
    keys = list(per_config)
    sample_ids = [r["sample_id"] for r in per_config[keys[0]]]

    in_sample_key = max(keys, key=lambda k: score_of(list(by_id[k].values())))
    in_sample = score_of(list(by_id[in_sample_key].values()))
    shipped_key = json.dumps(SHIPPED, sort_keys=True)
    shipped = score_of(list(by_id[shipped_key].values())) if shipped_key in by_id else None

    order = list(sample_ids)
    random.Random(seed).shuffle(order)
    parts = [order[i::folds] for i in range(folds)]

    held_out, picks = [], []
    for index, fold in enumerate(parts):
        train = [s for j, other in enumerate(parts) if j != index for s in other]
        best = max(keys, key=lambda k: score_of([by_id[k][s] for s in train]))
        held_out.append(score_of([by_id[best][s] for s in fold]))
        picks.append(json.loads(best))
        print(f"  fold {index + 1}: chose {json.loads(best)} -> "
              f"held-out {held_out[-1]:.5f}")

    mean = sum(held_out) / len(held_out)
    sd = (sum((x - mean) ** 2 for x in held_out) / (len(held_out) - 1)) ** 0.5
    return {
        "folds": folds,
        "seed": seed,
        "grid_size": len(GRID),
        "shipped_constants": SHIPPED,
        "shipped_in_sample": shipped,
        "best_in_sample_constants": json.loads(in_sample_key),
        "best_in_sample": round(in_sample, 6),
        "cross_validated": round(mean, 6),
        "fold_sd": round(sd, 6),
        "optimism": round(in_sample - mean, 6),
        "held_out_scores": [round(x, 6) for x in held_out],
        "per_fold_choice": picks,
    }


def order_sensitivity(catalog_path: str, samples: list, seeds: tuple) -> dict:
    """How much the score moves on session order alone.

    Long-term memory accumulates across sessions, so the shipped configuration
    is not order-invariant. The spread is small, but a headline score quoted to
    five decimal places should say so rather than imply a point estimate.
    """
    catalog_ids, categories, products = catalog_index(catalog_path)
    scores = []
    for seed in seeds:
        ordered = list(samples)
        if seed is not None:
            random.Random(seed).shuffle(ordered)
        agent = Agent(catalog_path)
        result = evaluate(agent, ordered, catalog_ids, categories, products)
        scores.append(result["recommended_technical_score"])
        label = "as shipped" if seed is None else f"shuffled seed={seed}"
        print(f"  {label:22s} {scores[-1]:.5f}", flush=True)
        del agent
        gc.collect()
    return {
        "orders": len(scores),
        "scores": scores,
        "spread": round(max(scores) - min(scores), 6),
        "min": min(scores),
        "max": max(scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--order", action="store_true",
                        help="measure order sensitivity instead of running the CV")
    parser.add_argument("--output", default="artifacts/crossval.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.order:
        print(f"order sensitivity over {len(samples)} sessions\n")
        payload = {"order_sensitivity": order_sensitivity(
            args.catalog, samples, (None, 1, 2, 3))}
        print(f"\n  spread on order alone: {payload['order_sensitivity']['spread']:.5f}")
    else:
        print(f"{len(GRID)} configurations x {len(samples)} sessions, "
              f"{args.folds}-fold CV\n")
        payload = sweep(args.catalog, samples, args.folds, args.seed)
        print(f"\n  cross-validated  {payload['cross_validated']:.5f} "
              f"+/- {payload['fold_sd']:.5f} (sd across folds)")
        print(f"  in-sample best   {payload['best_in_sample']:.5f}")
        print(f"  optimism         {payload['optimism']:+.5f}")

    out = ROOT / args.output
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    existing.update(payload)
    out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
