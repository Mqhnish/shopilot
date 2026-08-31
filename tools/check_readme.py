"""Check that every headline number in README.md matches a committed artifact.

Written after a number in the README was updated by hand and drifted from the
run it claimed to describe. Documentation that quotes measurements is code that
can be wrong, so it gets a test too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> object:
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def check_catalog_claims(readme: str, problems: list) -> None:
    """Re-count the claims the README makes about the catalog itself.

    Artifacts cover the measured scores. These are the other kind of number --
    counted from the frozen catalog by prose written at some point in the past,
    with nothing keeping it honest. One of them had already drifted: the
    merchandising-category figure was quoted from a wider draft of the marker
    set than the one that shipped, overstating it by nearly half.

    Skipped, not failed, when the catalog is absent: `make docs` has to stay
    runnable on a clean checkout.
    """
    import re

    if not (ROOT / "data" / "catalog.jsonl").exists():
        print("  (catalog absent — skipping the counted claims; run `make setup`)")
        return
    sys.path.insert(0, str(ROOT))
    from src.catalog import Catalog, is_merchandising  # noqa: E402

    catalog = Catalog(str(ROOT / "data" / "catalog.jsonl"))
    sizes = {catalog.coarse[docs[0]]: len(docs) for docs in catalog.bucket.values()}
    noise = [n for n in sizes if is_merchandising(n)]

    counted = {
        "merchandising categories": len(noise),
        "coarse categories": len(sizes),
        "merchandising products": sum(sizes[n] for n in noise),
        "catalog size": catalog.size,
    }
    claim = re.search(
        r"\*\*([\d,]+) of the catalog's ([\d,]+) coarse categories,\s*\n"
        r"holding ([\d,]+) of ([\d,]+) products\*\*", readme)
    if claim is None:
        problems.append("catalog claim: the merchandising sentence is gone from README")
        return
    stated = [int(g.replace(",", "")) for g in claim.groups()]
    for (label, actual), value in zip(counted.items(), stated):
        if actual != value:
            problems.append(f"{label}: README says {value:,}, the catalog has {actual:,}")

    largest = max(sizes.values())
    if f"{largest:,}" not in readme:
        problems.append(f"largest category: README does not contain {largest:,}")


def check_crossval_claims(readme: str, problems: list) -> None:
    """The cross-validation figures are measurements, so they get checked too.

    They are the numbers that license every "inside the noise band" judgment in
    this file, which makes them the last ones that should be allowed to drift.
    """
    path = ROOT / "artifacts" / "crossval.json"
    if not path.exists():
        print("  (artifacts/crossval.json absent — skipping; run `make crossval`)")
        return
    with path.open(encoding="utf-8") as handle:
        cv = json.load(handle)

    wanted = {
        "cross-validated estimate": f"{cv['cross_validated']:.5f}",
        "fold standard deviation": f"{cv['fold_sd']:.5f}",
        "best in-sample": f"{cv['best_in_sample']:.5f}",
        "optimism": f"{cv['optimism']:+.5f}",
    }
    order = cv.get("order_sensitivity")
    if order:
        wanted["order spread"] = f"{order['spread']:.5f}"
        wanted["order minimum"] = f"{order['min']:.5f}"
        wanted["order maximum"] = f"{order['max']:.5f}"
    for label, value in wanted.items():
        if value not in readme:
            problems.append(f"crossval {label}: README does not contain {value}")


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = load("results.json")
    baseline = load("baseline_repro.json")
    ablate = {row["arm"]: row for row in load("ablate.json")}
    robustness = load("robustness.json")
    problems: list = []

    def want(label: str, value: float, digits: int) -> None:
        text = f"{value:.{digits}f}"
        if text not in readme:
            problems.append(f"{label}: README does not contain {text}")

    want("agent hit rate", results["hit_rate_at_10"], 3)
    want("agent MRR", results["mrr"], 4)
    want("agent MTTC", results["mttc"], 3)
    want("agent efficiency", results["efficiency"], 4)
    want("agent score", results["recommended_technical_score"], 5)
    want("baseline score", baseline["recommended_technical_score"], 5)
    want("baseline MRR", baseline["mrr"], 4)

    for name, metrics in results["scenario_metrics"].items():
        want(f"{name} MRR", metrics["mrr"], 4)
        want(f"{name} MTTC", metrics["mttc"], 3)

    for arm, row in ablate.items():
        if arm == "full system":
            continue
        want(f"ablation '{arm}' score", row["technical_score"], 5)

    for row in robustness:
        want(f"robustness {row['paraphrase']}/{row['arm']} score",
             row["technical_score"], 5)

    check_catalog_claims(readme, problems)
    check_crossval_claims(readme, problems)

    ratio = results["recommended_technical_score"] / baseline["recommended_technical_score"]
    if f"{ratio:.1f}×" not in readme and f"{ratio:.1f}x" not in readme:
        problems.append(f"baseline multiple: README does not contain {ratio:.1f}x")

    for problem in problems:
        print(problem)
    print(f"\n{len(problems)} mismatches between README.md and artifacts/")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
