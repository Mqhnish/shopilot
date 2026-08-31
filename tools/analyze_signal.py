"""Measure how much the turn-1 category and the disclosed constraint phrases
narrow the 50k catalog. Drives retrieval design."""
from __future__ import annotations
import statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator.local_evaluator import (  # noqa: E402
    load_jsonl, catalog_index, coarse_category, materialize_hidden_fields,
)

samples = load_jsonl("data/public_set.jsonl")
_ids, categories, products = catalog_index("data/catalog.jsonl")

# --- 1. coarse_category bucket sizes over the whole catalog ---
buckets: dict[str, list[str]] = defaultdict(list)
for asin, cats in categories.items():
    buckets[coarse_category(cats)].append(asin)
sizes = sorted((len(v) for v in buckets.values()))
print(f"distinct coarse_category values: {len(buckets)}")
print(f"bucket size  min={sizes[0]} median={statistics.median(sizes)} "
      f"mean={statistics.fmean(sizes):.1f} p90={sizes[int(.9*len(sizes))]} max={sizes[-1]}")

tgt_bucket = [len(buckets[coarse_category(categories[s['ground_truth']['parent_asin']])]) for s in samples]
tgt_bucket.sort()
print(f"bucket size AT THE 200 TARGETS: median={statistics.median(tgt_bucket)} "
      f"mean={statistics.fmean(tgt_bucket):.1f} p90={tgt_bucket[int(.9*len(tgt_bucket))]} max={tgt_bucket[-1]}")
print(f"  targets in a bucket of <=10: {sum(1 for n in tgt_bucket if n<=10)}/200"
      f"   <=50: {sum(1 for n in tgt_bucket if n<=50)}/200   <=200: {sum(1 for n in tgt_bucket if n<=200)}/200")

# --- 2. how discriminative is a verbatim constraint phrase across the catalog? ---
# Build an inverted index over the exact feature/detail strings of every product.
phrase_to_asins: dict[str, set[str]] = defaultdict(set)
for asin, p in products.items():
    vals = []
    for f in p.get("features") or []:
        vals.append(str(f))
    d = p.get("details") or {}
    if isinstance(d, dict):
        vals.extend(f"{k}: {v}" for k, v in d.items() if v not in (None, "", []))
    for v in vals:
        phrase_to_asins[" ".join(v.split()).strip(" -;,.\t\n")[:180].rstrip()].add(asin)

df = Counter()
for s in samples:
    card, _ = materialize_hidden_fields(s, products)
    for v in card["hard_constraints"] + card["soft_preferences"]:
        df[len(phrase_to_asins.get(v, ()))] += 1
tot = sum(df.values())
cum = 0
print(f"\nverbatim-phrase document frequency over all {tot} constraint strings:")
for n in sorted(df):
    cum += df[n]
    if n <= 5 or n in (10, 50, 100) or cum == tot:
        print(f"  df={n:<6} count={df[n]:<5} cumulative={cum}/{tot} ({cum/tot:.0%})")
    if cum / tot > 0.97:
        break
