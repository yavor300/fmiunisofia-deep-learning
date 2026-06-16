CONFIG ?= configs/default.yaml
CHECKPOINT ?=
RUN_NAME ?=
MAX_TRAIN_MASKS ?=
MAX_EVAL_MASKS ?=
VENV_PYTHON := $(wildcard .venv/bin/python)
VENV_PIP := $(wildcard .venv/bin/pip)
PYTHON ?= $(if $(VENV_PYTHON),.venv/bin/python,python)
PIP ?= $(if $(VENV_PIP),.venv/bin/pip,pip)
RUN_NAME_ARG := $(if $(RUN_NAME),--run-name $(RUN_NAME),)
BASELINE_LIMIT_ARGS := $(if $(MAX_TRAIN_MASKS),--max-train-masks $(MAX_TRAIN_MASKS),) $(if $(MAX_EVAL_MASKS),--max-eval-masks $(MAX_EVAL_MASKS),)

.PHONY: install eda baseline train eval report app test

install:
	$(PIP) install -r requirements.txt

eda:
	$(PYTHON) -m src.cityseg.eda.analyze_dataset --config configs/default.yaml

baseline:
	$(PYTHON) -m src.cityseg.models.majority_baseline --config configs/experiments/000_baseline_majority.yaml --split val $(RUN_NAME_ARG) $(BASELINE_LIMIT_ARGS)

train:
	$(PYTHON) -m src.cityseg.training.train --config $(CONFIG) $(RUN_NAME_ARG)

eval:
	$(PYTHON) -m src.cityseg.training.evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT) $(RUN_NAME_ARG)

report:
	$(PYTHON) -m src.cityseg.reporting.build_model_report --results reports/experiment_results.csv

app:
	streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
