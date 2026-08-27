PY ?= python3
export PYTHONPATH := src

.PHONY: all test lint clean topology main ablation sensitivity interaction detection frontier calibration

all: topology main ablation sensitivity interaction detection frontier calibration
	@echo "pipeline complete: results/ figures/ paper/tables/ paper/numbers.tex"

topology:    ; $(PY) scripts/01_topology.py
main:        ; $(PY) scripts/02_main.py
ablation:    ; $(PY) scripts/03_ablation.py
sensitivity: ; $(PY) scripts/04_sensitivity.py
interaction: ; $(PY) scripts/05_interaction.py
detection:   ; $(PY) scripts/06_detection.py
frontier:    ; $(PY) scripts/07_frontier.py
calibration: ; $(PY) scripts/08_calibration.py

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m compileall -q src scripts tests

clean:
	rm -rf results/*.csv results/*.json results/*.npz figures/*.pdf
	find . -name __pycache__ -type d -exec rm -rf {} +

paper: all
	cp figures/*.pdf paper/figures/
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
