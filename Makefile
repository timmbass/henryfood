.PHONY: install-deps install-dev run-sync dry-run test lint clean

# Defaults (override on CLI e.g. `make run-sync SHEET=... CREDS=...`)
SHEET ?=
CREDS ?= ~/.config/henryfood/henryfood-key.json
WORKSHEET ?= Sheet1
TARGET ?= scripts/data/raw/meals.json

install-deps:
	python -m pip install --upgrade pip
	python -m pip install gspread pandas oauth2client duckdb pyyaml pyarrow

install-dev: install-deps
	python -m pip install pytest flake8

run-sync:
	@if [ -z "$(SHEET)" ]; then echo "Provide SHEET=<sheet-url-or-id> or set SHEET env var"; exit 1; fi
	python scripts/tools/sync_google_sheet.py "$(SHEET)" --creds "$(CREDS)" --worksheet "$(WORKSHEET)" --target "$(TARGET)"

dry-run:
	@if [ -z "$(SHEET)" ]; then echo "Provide SHEET=<sheet-url-or-id> or set SHEET env var"; exit 1; fi
	python scripts/tools/sync_google_sheet.py "$(SHEET)" --creds "$(CREDS)" --worksheet "$(WORKSHEET)" --target "$(TARGET)" --dry-run

test:
	pytest -q

lint:
	flake8 .

clean:
	find . -type f -name '*.pyc' -delete || true
	rm -rf __pycache__ || true
