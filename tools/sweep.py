"""Sweep or ablate configurations against the official evaluator.

The 50k index is built once and shared by every configuration, so a twenty-arm
sweep costs one index build rather than twenty.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.agent import Options  # noqa: E402
from src.catalog import Catalog  # noqa: E402
from src.lexical import Retriever  # noqa: E402


def run(arms, catalog_path: str, dataset: str, limit: int = 0):
    samples = load_jsonl(dataset)
    if limit:
        samples = samples[:limit]
    catalog_ids, categories, products = catalog_index(catalog_path)
    started = time.time()
    catalog = Catalog(catalog_path)
    retriever = Retriever(catalog)
    print(f"index built in {time.time() - started:.1f}s; {len(arms)} arms x {len(samples)} sessions\n")

    rows = []
    for name, overrides in arms:
        agent = Agent(catalog_path, options=Options(**overrides),
                      catalog=catalog, retriever=retriever)
        started = time.time()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        rows.append({
            "arm": name,
            "overrides": overrides,
            "hit_rate_at_10": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "efficiency": result["efficiency"],
            "technical_score": result["recommended_technical_score"],
            "seconds": round(time.time() - started, 1),
            "scenario": {k: v["hit_rate_at_10"] for k, v in result["scenario_metrics"].items()},
        })
        r = rows[-1]
        print(f"{name:34s} HR={r['hit_rate_at_10']:.3f} MRR={r['mrr']:.4f} "
              f"MTTC={r['mttc']:.3f} SCORE={r['technical_score']:.5f}  ({r['seconds']}s)")
    return rows


ABLATIONS = [
    ("full system", {}),
    ("- exact phrase route", {"use_phrase": False}),
    ("- BM25 route", {"use_bm25": False}),
    ("- vector route", {"use_vector": False}),
    ("- negative evidence", {"use_exclusions": False}),
    ("- dual-track routing", {"use_routing": False}),
    ("- clarification", {"use_clarify": False}),
    ("- diversity (MMR)", {"use_diversity": False}),
    ("- profile personalisation", {"use_profile": False}),
    ("- long-term cohort memory", {"use_memory": False}),
    ("- frame-free span recovery", {"use_span_recovery": False}),
    ("- list truncation", {"use_truncation": False}),
    ("override erases slots", {"override_erases": True}),
    ("override demotes 0.45", {"demote_factor": 0.45}),
    # The three components added to close gaps against the brief's own wording.
    # Each is measured here rather than asserted in prose.
    ("- over-generality cutoff", {"cutoff_on_over_general": False}),
    ("- cross-category browsing", {"cross_category": 0}),
    ("+ slot decay 0.06/turn", {"slot_decay": 0.06}),
    ("+ slot decay 0.15/turn", {"slot_decay": 0.15}),
]

SWEEP = (
    [(f"narrow_k={k}", {"narrow_k": k}) for k in (1, 2, 3, 4, 5)]
    + [(f"late_turn={t}", {"late_turn": t}) for t in (2, 3, 4, 5, 6)]
    + [(f"confident_margin={m}", {"confident_margin": m}) for m in (0.05, 0.15, 0.22, 0.35, 1e9)]
)

# narrow_k and late_turn are not independent: how short the early lists are
# changes how many turns it is worth keeping them short for.
GRID = [
    (f"k={k},late={t}", {"narrow_k": k, "late_turn": t})
    for k in (1, 2, 3)
    for t in (3, 4, 5, 6, 7, 8, 10)
]

# Does the adaptive "stop withholding once a turn teaches us nothing" rule
# remove the need for a tight turn cap?
STOP = [
    ("adaptive barren=1, late=10", {"barren_turns_before_full": 1, "late_turn": 10}),
    ("adaptive barren=1, late=8", {"barren_turns_before_full": 1, "late_turn": 8}),
    ("adaptive barren=2, late=10", {"barren_turns_before_full": 2, "late_turn": 10}),
    ("adaptive off, late=8", {"barren_turns_before_full": 99, "late_turn": 8}),
    ("adaptive off, late=10", {"barren_turns_before_full": 99, "late_turn": 10}),
    ("adaptive off, late=4", {"barren_turns_before_full": 99, "late_turn": 4}),
]


ROBUSTNESS = [
    ("hardened", {}),
    ("unhardened (no span recovery)", {"use_span_recovery": False}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ablate", choices=["ablate", "sweep", "grid", "stop"])
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    arms = {"ablate": ABLATIONS, "sweep": SWEEP, "grid": GRID, "stop": STOP}[args.mode]
    rows = run(arms, args.catalog, args.dataset, args.limit)
    out = args.output or f"artifacts/{args.mode}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
