PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYV=$(VENV)/bin/python

.PHONY: venv install daily weekly build_timeline build_features train report clean

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

daily: build_timeline build_features

weekly: train report

build_timeline:
	$(PYV) -m src.features.timeline

build_features:
	$(PYV) -m src.features.lag_features

train:
	$(PYV) -m src.models.train

report:
	$(PYV) -m src.reports.weekly_report

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
