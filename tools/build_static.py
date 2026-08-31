"""Bake the walkthrough into a static site.

The live walkthrough needs a persistent process: it indexes 50,000 products into
memory and then holds multi-turn session state there. A static host has neither,
and a serverless one has the second problem in a worse form -- consecutive turns
would land on different instances and the conversation would reset every message.

So this records the agent instead of hosting it. Every payload the page would
have fetched is captured from a *running* server and written into one JSON
bundle; `web/copilot.js` reads that bundle instead of the network when
`COPILOT_STATIC` is set. Nothing is fabricated: each turn in the bundle is the
real response the real agent gave, including its trace, its information-gain
figures and its chosen question.

What is lost is free text. A visitor picks from recorded conversations rather
than typing anything, and the page says so rather than pretending otherwise.

    python3 tools/build_static.py            # needs `make serve` running
    python3 tools/build_static.py --out dist
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

# Conversations worth recording. Each is an opening query plus the follow-ups a
# visitor can click through. They are chosen to cover the four things the agent
# does that a search box cannot: resolve an occasion, carry a constraint, change
# subject, and hold a conversation that costs no turn.
SCRIPTS = [
    {
        "id": "winter-wedding",
        "label": "something for a winter wedding",
        "why": "an occasion, not a category",
        "turns": ["something for a winter wedding",
                  "For that, what matters is: 100% polyester.",
                  "Those options are not quite right yet."],
    },
    {
        "id": "leather-belt",
        "label": "a black leather belt under $40",
        "why": "two constraints and a price ceiling",
        "turns": ["a black leather belt under $40",
                  "For that, what matters is: Buckle closure.",
                  "I don't have an additional preference for feature."],
    },
    {
        "id": "gift",
        "label": "a gift for my wife",
        "why": "no product word at all",
        "turns": ["a gift for my wife",
                  "For that, what matters is: color: gold.",
                  "Those options are not quite right yet."],
    },
    {
        "id": "switch",
        "label": "sneakers, then change your mind",
        "why": "an abrupt change of subject",
        "turns": ["comfortable running sneakers",
                  "actually I want a winter coat instead",
                  "For that, what matters is: 100% Polyester."],
    },
    {
        "id": "small-talk",
        "label": "just say hello",
        "why": "conversation that costs no turn",
        "turns": ["hi", "what can you do", "a gift for an anniversary", "thanks, bye"],
    },
    {
        "id": "out-of-scope",
        "label": "ask for something it does not sell",
        "why": "the catalog's boundary, stated honestly",
        "turns": ["wireless headphones", "ok then, a leather watch strap"],
    },
]

# Prefixes the typeahead is recorded for. Anything else falls back to the empty
# query, which legitimately lists the biggest categories.
PREFIXES = ["", "belt", "belts", "sneaker", "sneakers", "dress", "dresses",
            "watch", "watches", "necklace", "gift", "winter", "wedding",
            "boots", "coat", "shirt", "jeans", "bag", "ring", "earrings",
            "something for a winter wedding", "a black leather belt under $40",
            "a gift for my wife", "comfortable running sneakers"]

REPLAYS = 24  # labelled sessions to pre-run


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=120) as fh:
            return json.load(fh)

    def post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as fh:
            return json.load(fh)


def record(api: Api) -> dict:
    bundle: dict = {"scripts": [], "categories": {}, "replays": {}}

    print("  benchmark, suggestions, sessions")
    bundle["benchmark"] = api.get("/benchmark")
    bundle["suggestions"] = api.get("/suggestions")
    sessions = api.get("/sessions")["sessions"]
    bundle["sessions"] = sessions[:REPLAYS]

    print(f"  typeahead for {len(PREFIXES)} prefixes")
    for prefix in PREFIXES:
        bundle["categories"][prefix] = api.get(
            "/categories?q=" + urllib.parse.quote(prefix))

    for script in SCRIPTS:
        sid = f"static::{script['id']}"
        api.post("/reset", {"session_id": sid, "mode": "shopper"})
        turns = []
        for message in script["turns"]:
            data = api.post("/chat", {"session_id": sid, "message": message,
                                      "assist": True})
            base = None
            if data.get("kind") is None:
                try:
                    base = api.post("/baseline", {"session_id": sid, "message": message})
                except urllib.error.URLError:
                    base = None
            refinements = None
            if data.get("constraints", {}).get("category_exact"):
                refinements = api.get(f"/refinements?session_id={sid}&q=")
            turns.append({"said": message, "reply": data,
                          "baseline": base, "refinements": refinements})
        bundle["scripts"].append({**script, "recorded": turns})
        print(f"  script {script['id']:<16} {len(turns)} turns")

    print(f"  replaying {len(bundle['sessions'])} scored sessions")
    for row in bundle["sessions"]:
        bundle["replays"][row["sample_id"]] = api.post(
            "/replay", {"sample_id": row["sample_id"]})

    bundle["health"] = api.get("/health")
    return bundle


def build(out: Path, bundle: dict) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("copilot.html", "copilot.css", "copilot.js", "deck.html"):
        shutil.copy2(WEB / name, out / name)
    shutil.copy2(WEB / "copilot.html", out / "index.html")

    payload = json.dumps(bundle, separators=(",", ":"))
    (out / "demo-data.json").write_text(payload, encoding="utf-8")

    # Switch the page into static mode before copilot.js runs.
    for page in ("index.html", "copilot.html"):
        html = (out / page).read_text(encoding="utf-8")
        html = html.replace(
            '<script src="copilot.js"></script>',
            '<script>window.COPILOT_STATIC = "demo-data.json";</script>\n'
            '<script src="copilot.js"></script>')
        (out / page).write_text(html, encoding="utf-8")

    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "headers": [{
            "source": "/demo-data.json",
            "headers": [{"key": "Cache-Control",
                         "value": "public, max-age=3600"}],
        }],
    }, indent=2) + "\n", encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  wrote {out.relative_to(ROOT)}/  "
          f"({len(list(out.rglob('*')))} files, {size/1024/1024:.2f} MB)")
    print(f"  demo-data.json is {len(payload)/1024/1024:.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000/api/copilot")
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()

    api = Api(args.base)
    try:
        api.get("/health")
    except Exception as error:
        print(f"cannot reach {args.base}: {error}\n\nStart it first:\n"
              "    make serve", file=sys.stderr)
        return 1

    print(f"recording from {args.base}")
    build(ROOT / args.out, record(api))
    print("\n  preview:  python3 -m http.server -d dist 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
