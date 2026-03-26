# HenryFood

Personal food-diary system built around two independent components:

| Component | Hardware | Purpose |
|-----------|----------|---------|
| **`app/`** | Raspberry Pi | Button-triggered voice capture → timestamped WAV files |
| **`scripts/`** | i5 laptop / desktop | Health analytics: CSV → features → model → weekly report |

Data flows one way: WAV files produced on the Pi are transcribed (future phase) and appended to `data/raw/meals.csv`; `scripts/` reads those CSVs at run time.

---

## Repository Layout

```
henryfood/
├── app/                  # Pi capture component
│   ├── cli.py            # Entry point — `python -m app.cli run`
│   ├── config.py         # AppConfig (Pydantic, validated at startup)
│   ├── models.py         # RecordingMetadata
│   ├── recorder.py       # Recorder ABC + SounddeviceRecorder
│   ├── gpio_button.py    # gpiozero button wrapper
│   ├── main.py           # CaptureWorkflow orchestrator
│   └── utils.py          # Logger setup, filesystem helpers
├── tests/                # Tests for app/ (mock sounddevice/gpiozero)
├── scripts/              # Analytics pipeline (DuckDB) — see scripts/README.md
├── data/raw/             # CSVs: meals, symptoms, sleep, stress
├── docs/
│   ├── specs/system_design.txt   # Original system specification
│   ├── archive/                  # Historical documents
│   └── SECURITY.md               # Threat model & hardening guide
├── archive/voice_prototype/      # Earlier voice-capture prototype (retired)
├── requirements.txt
└── README.md
```

---

## Component 1 — Pi Capture (`app/`)

### Hardware

- Raspberry Pi 3B+ / 4 / 5 running Raspberry Pi OS
- USB microphone
- Tactile button: GPIO 17 (BCM) → GND (internal pull-up enabled, no resistor needed)

### Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli run
```

Press and hold to record. Release to save. Ctrl-C to exit.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g` / `--gpio-pin` | `17` | BCM GPIO pin |
| `-r` / `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `-c` / `--channels` | `1` | Mono (1) or stereo (2) |
| `-m` / `--max-duration` | `30` | Auto-stop cap (seconds) |
| `--min-duration` | `0.5` | Discard clips shorter than this |
| `-o` / `--recordings-dir` | `recordings` | Output directory |
| `-l` / `--log-level` | `INFO` | Log level |

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Component 2 — Analytics Pipeline (`scripts/`)

See [`scripts/README.md`](scripts/README.md) for full usage.

```bash
cd scripts && make install && make daily   # ingest + features
cd scripts && make weekly                  # train + report
```

---

## Data Ingestion

There is no separate ingestion script. The Pi capture component appends entries directly to `data/raw/meals.csv` (and the other CSVs). The analytics pipeline reads those CSVs at run time.

---

## Security

See [`docs/SECURITY.md`](docs/SECURITY.md) for the full threat model and hardening guide.

Key controls implemented: parameterized SQL, path-traversal validation, file-size limits, local-only storage, no external API calls.

---

## License

[Add your license here]

## Disclaimer

Personal health tracking only. Not medical advice.
