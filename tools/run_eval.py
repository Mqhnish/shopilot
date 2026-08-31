"""Score our Agent with the organizer's evaluator, unmodified.

The evaluator module is imported and driven directly rather than edited or
copied, so what runs here is exactly what runs officially.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from tools.paraphrase import LEVELS, ParaphrasingAgent  # noqa: E402


def build_agent(name: str, catalog: str, options_json: str = ""):
    if name == "baseline":
        from starter.agent import Agent as StarterAgent
        return StarterAgent(catalog)
    from agent import Agent, Options
    options = Options(**json.loads(options_json)) if options_json else None
    return Agent(catalog, options=options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official evaluator on our agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="artifacts/results.json")
    parser.add_argument("--agent", default="ours", choices=["ours", "baseline"])
    parser.add_argument("--options", default="", help="JSON dict of src.agent.Options overrides")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N sessions")
    parser.add_argument("--paraphrase", default="none", choices=list(LEVELS),
                        help="reword the customer's messages before the agent sees them")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--show-samples", type=int, default=0,
                        help="print N before/after message pairs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    started = time.time()
    agent = build_agent(args.agent, args.catalog, args.options)
    index_seconds = time.time() - started
    wrapper = None
    if args.paraphrase != "none":
        wrapper = ParaphrasingAgent(agent, level=args.paraphrase, seed=args.seed)
        agent = wrapper

    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    eval_seconds = time.time() - started

    result["paraphrase"] = {"level": args.paraphrase, "seed": args.seed}
    result["timing"] = {
        "index_build_seconds": round(index_seconds, 2),
        "evaluation_seconds": round(eval_seconds, 2),
        "sessions": len(samples),
        "seconds_per_session": round(eval_seconds / max(len(samples), 1), 4),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if wrapper is not None and args.show_samples:
        print(f"--- customer text the agent actually saw ({args.paraphrase}) ---")
        for before, after in wrapper.samples[: args.show_samples]:
            print(f"  was: {before}\n  saw: {after}\n")
    if not args.quiet:
        summary = {k: v for k, v in result.items() if k != "sessions"}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
