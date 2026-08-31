# Convenience targets. Everything here runs on the standard library.
PY ?= python3
CATALOG ?= data/catalog.jsonl
DATASET ?= data/public_set.jsonl

.PHONY: help setup lint test eval baseline ablate grid demo serve headroom crossval static docs verify clean

help:
	@echo "make setup     download + checksum the frozen catalog (run this first)"
	@echo "make lint      unused imports and dead locals (stdlib ast, no deps)"
	@echo "make test      run the 261-test suite (~69s)"
	@echo "make eval      score our agent with the official evaluator"
	@echo "make baseline  score the organizer's weak BM25 starter, for comparison"
	@echo "make ablate    turn each component off in turn and rescore"
	@echo "make demo      walk one multi-turn session, showing the reasoning"
	@echo "make serve     open the browser walkthrough at http://127.0.0.1:8000"
	@echo "make robust    score under three levels of customer paraphrase"
	@echo "make headroom  measure how much ranking safety margin exists"
	@echo "make crossval  5-fold CV of the tuned constants + order sensitivity"
	@echo "make static    bake the walkthrough into dist/ for a static host"
	@echo "make docs      check every README number against artifacts/"
	@echo "make verify    lint + tests + baseline + eval + docs, the full check"

setup:
	$(PY) tools/setup_data.py

lint:
	$(PY) tools/lint.py

test:
	$(PY) -m unittest discover -s tests -t . -v

eval:
	$(PY) tools/run_eval.py --catalog $(CATALOG) --dataset $(DATASET) --output artifacts/results.json

baseline:
	$(PY) tools/run_eval.py --agent baseline --catalog $(CATALOG) --dataset $(DATASET) \
		--output artifacts/baseline_repro.json

ablate:
	$(PY) tools/sweep.py --mode ablate --catalog $(CATALOG) --dataset $(DATASET)

grid:
	$(PY) tools/sweep.py --mode grid --catalog $(CATALOG) --dataset $(DATASET)

demo:
	$(PY) tools/demo.py --sample public_0005 --reveal

# The browser walkthrough. Not the scored path -- see server.py.
serve:
	$(PY) server.py --catalog $(CATALOG) --dataset $(DATASET)

robust:
	$(PY) tools/robustness.py --catalog $(CATALOG) --dataset $(DATASET)

headroom:
	$(PY) tools/headroom.py --catalog $(CATALOG) --dataset $(DATASET)

# Is the headline score fitted to the 200 public sessions? Also prices the
# benchmark's own noise floor, which is what licenses the "below 0.01 is not
# resolvable" claim made throughout the README.
# Record the walkthrough into a static bundle. Needs `make serve` running in
# another shell: it drives the real agent and captures what it actually returns.
static:
	$(PY) tools/build_static.py

crossval:
	$(PY) tools/crossval.py --catalog $(CATALOG) --dataset $(DATASET)
	$(PY) tools/crossval.py --catalog $(CATALOG) --dataset $(DATASET) --order

verify: setup lint test baseline eval docs
	@echo
	@echo "verified: lint clean, suite green, baseline reproduced, agent scored,"
	@echo "          every README number traced to a committed artifact"

docs:
	$(PY) tools/check_readme.py

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
