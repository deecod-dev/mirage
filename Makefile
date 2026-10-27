.PHONY: install test figures verify reproduce fetch-aegis fetch-traces paper

install:
	pip install -e '.[dev]'

## Unit tests and negative controls. Trace-level controls skip until traces are downloaded.
test:
	pytest tests/ -q

## Rebuild Figures 1-4 from results/*.json.
figures:
	python scripts/make_figures.py --results results --outdir figures

## Check every headline number in the paper against results/*.json.
verify:
	python scripts/verify_revision_numbers.py

## Everything that runs on CPU from the released results.
reproduce: test figures verify

fetch-aegis:
	bash scripts/fetch_aegis.sh

fetch-traces:
	bash scripts/fetch_traces.sh

paper:
	cd paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
