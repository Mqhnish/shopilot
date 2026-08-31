"""Score the agent under customer paraphrase, hardened and unhardened.

Produces the matrix quoted in README.md > Robustness. The evaluator is never
modified; only the text reaching the agent is degraded (see tools/paraphrase.py),
and hits are still exact identifier matches.
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
from tools.paraphrase import LEVELS, ParaphrasingAgent  # noqa: E402

ARMS = (
    ("hardened", {}),
    ("unhardened", {"use_span_recovery": False}),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/robustness.json")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.time()
    catalog = Catalog(args.catalog)
    retriever = Retriever(catalog)
    print(f"index built in {time.time() - started:.1f}s\n")

    rows = []
    print(f"{'paraphrase':<11}{'arm':<12}{'HR@10':>7}{'MRR':>9}{'MTTC':>8}{'SCORE':>10}")
    for level in LEVELS:
        for arm, overrides in ARMS:
            if level == "none" and arm == "unhardened":
                continue  # span recovery never fires on recognised frames
            agent = Agent(args.catalog, options=Options(**overrides),
                          catalog=catalog, retriever=retriever)
            driver = agent if level == "none" else ParaphrasingAgent(
                agent, level=level, seed=args.seed)
            result = evaluate(driver, samples, catalog_ids, categories, products)
            rows.append({
                "paraphrase": level,
                "arm": arm,
                "hit_rate_at_10": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
                "technical_score": result["recommended_technical_score"],
            })
            r = rows[-1]
            print(f"{level:<11}{arm:<12}{r['hit_rate_at_10']:7.3f}{r['mrr']:9.4f}"
                  f"{r['mttc']:8.3f}{r['technical_score']:10.5f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
