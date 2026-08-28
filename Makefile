.PHONY: setup test ingest inspect clean eval

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	@echo "Now: cp .env.example .env and set SEC_USER_AGENT"

test:
	pytest -q

ingest:
	python scripts/01_ingest.py

ingest-one:
	python scripts/01_ingest.py --tickers AAPL --limit 1 -v

inspect:
	python scripts/02_inspect.py

parse:
	python scripts/03_parse.py

inspect-sections:
	python scripts/04_inspect_sections.py

# Regenerates every number in EVALUATION.md. Anything not reproducible by
# this target does not belong in the report.
eval:
	python scripts/09_eval_retrieval.py --use-filters
	python scripts/10_extract.py
	python scripts/13_analyze.py --dry-run
	python scripts/14_build_features.py
	python scripts/15_train_model.py

# Offline subset: no network, no API key, no model downloads. Same as CI.
ci:
	pytest -q --tb=short

# Reads only artifacts already on disk -- no LLM calls, no network.
demo:
	streamlit run app/main.py

clean:
	rm -rf data/db/*.duckdb __pycache__ .pytest_cache
