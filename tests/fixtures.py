"""Shared test fixtures.

Builds a small catalog on first use so the suite runs in seconds instead of
rebuilding the 50,000-product index for every test module. The subset always
contains the ground-truth targets of the public sessions it is paired with, so
end-to-end tests exercise real sessions rather than synthetic ones.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC_SET = ROOT / "data" / "public_set.jsonl"

_SUBSET: Tuple[str, List[dict]] = ()
_FULL_ROWS: List[dict] = []


def catalog_rows(limit: int = 0) -> List[dict]:
    """Raw catalog rows, cached across tests."""
    global _FULL_ROWS
    if not _FULL_ROWS:
        with CATALOG.open(encoding="utf-8") as handle:
            _FULL_ROWS = [json.loads(line) for line in handle if line.strip()]
    return _FULL_ROWS[:limit] if limit else _FULL_ROWS


def public_samples() -> List[dict]:
    with PUBLIC_SET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def subset_catalog(n_sessions: int = 24, filler: int = 2500) -> Tuple[str, List[dict]]:
    """A small catalog file plus the public sessions whose targets it contains."""
    global _SUBSET
    if _SUBSET:
        return _SUBSET
    rows = catalog_rows()
    samples = public_samples()[:n_sessions]
    wanted = {s["ground_truth"]["parent_asin"] for s in samples}
    by_asin = {str(r["parent_asin"]): r for r in rows}
    chosen = [by_asin[a] for a in wanted if a in by_asin]
    rng = random.Random(20260830)
    pool = [r for r in rows if str(r["parent_asin"]) not in wanted]
    chosen.extend(rng.sample(pool, min(filler, len(pool))))
    rng.shuffle(chosen)

    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in chosen:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    kept = {str(r["parent_asin"]) for r in chosen}
    samples = [s for s in samples if s["ground_truth"]["parent_asin"] in kept]
    _SUBSET = (handle.name, samples)
    return _SUBSET


def sampled_rows(n: int, seed: int = 7) -> List[dict]:
    """A deterministic sample of catalog rows for differential testing."""
    rows = catalog_rows()
    return random.Random(seed).sample(rows, min(n, len(rows)))
