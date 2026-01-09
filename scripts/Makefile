PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYV=$(VENV)/bin/python

.PHONY: venv install daily weekly build_timeline build_features train report

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

# Daily: ingest new raw files, rebuild features
daily: build_timeline build_features

# Weekly: train + report
weekly: train report

build_timeline:
	$(PYV) -m src.features.timeline

build_features:
	$(PYV) -m src.features.lag_features

train:
	$(PYV) -m src.models.train

report:
	$(PYV) -m src.reports.weekly_report
