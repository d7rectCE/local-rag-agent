PY ?= python

.PHONY: install test demo demo-corpus serve ui

install:            ## editable install with dev extras (torch must be installed beforehand)
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

demo-corpus:        ## regenerate demo_corpus/ (executes the notebooks)
	$(PY) scripts/build_demo_corpus.py

demo:               ## API + indexing of demo_corpus + UI
	$(PY) -m rag_agent.cli demo

serve:
	$(PY) -m rag_agent.cli serve

ui:
	$(PY) -m rag_agent.cli ui
