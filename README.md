# HenryFood — Raspberry Pi Food Diary Voice Recorder

A button-triggered voice capture system for the HenryFood food diary, designed to run on a **Raspberry Pi**.

Press and hold a tactile push button wired to a GPIO pin — audio is captured while the button is held and saved as a timestamped WAV file when released.  A future phase will transcribe these recordings and convert them into structured food diary entries.

## Repository Layout

```
henryfood/
├── app/                  # Button recorder application (new)
│   ├── cli.py            # Typer CLI — `python -m app.cli run`
│   ├── config.py         # AppConfig (Pydantic model)
│   ├── models.py         # RecordingMetadata model
│   ├── utils.py          # Logger setup, filesystem helpers
│   ├── gpio_button.py    # gpiozero button wrapper
│   ├── recorder.py       # Recorder ABC + SounddeviceRecorder
│   └── main.py           # CaptureWorkflow orchestrator
├── tests/                # Tests for app/
├── scripts/              # Health analytics pipeline (DuckDB)
├── voice/                # Earlier voice-capture prototype
├── requirements.txt      # Python dependencies
└── README.md
```

## Hardware Requirements

- Raspberry Pi (3B+ / 4 / 5) running Raspberry Pi OS
- USB microphone or USB sound card with mic input
- Tactile push button wired between **GPIO 17** (BCM) and **GND**

### Wiring

| Button Pin | Pi Pin            |
|------------|-------------------|
| Leg A      | GPIO 17 (pin 11)  |
| Leg B      | GND (pin 9)       |

The code enables the internal pull-up resistor — no external resistor needed.

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Button Recorder

```bash
python -m app.cli run
```

Press and hold the button to record.  Release to save.  Ctrl-C to exit.

### 3. Test Audio (no button needed)

Verify the microphone works without GPIO hardware:

```bash
python -m app.cli test-audio --duration 5
```

## CLI Options

```
python -m app.cli run [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-g` / `--gpio-pin` | `17` | BCM GPIO pin number |
| `-r` / `--sample-rate` | `16000` | Audio sample rate in Hz |
| `-c` / `--channels` | `1` | Mono (1) or stereo (2) |
| `-m` / `--max-duration` | `30` | Auto-stop after this many seconds |
| `-o` / `--recordings-dir` | `recordings` | Directory for WAV files |
| `-l` / `--log-level` | `INFO` | Log level (DEBUG / INFO / …) |

## Configuration Model

All settings are validated at startup via Pydantic:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpio_pin` | int | 17 | BCM GPIO pin (0–27) |
| `sample_rate` | int | 16 000 | Sample rate in Hz |
| `channels` | int | 1 | Audio channels |
| `max_duration_seconds` | float | 30.0 | Hard recording cap |
| `recordings_dir` | Path | `recordings` | Output directory |
| `log_level` | str | `INFO` | Python log level |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests mock `sounddevice`, `soundfile`, and `gpiozero` so they run on any machine.

## Health Analytics Pipeline

The existing analytics pipeline lives in `scripts/`.  See `scripts/requirements.txt` and the `Makefile` for usage:

```bash
cd scripts && make install && make daily
```

## Security Features

This pipeline implements multiple security hardening measures:

### 1. Input Validation
- CSV file size limits (100MB max)
- String length limits (1000 chars per field)
- Numeric range validation (pain 0-10, sleep 0-24h, etc.)
- Safe timestamp parsing with error handling

### 2. SQL Injection Prevention
- Parameterized queries throughout
- Tag sanitization (alphanumeric only)
- Safe string construction for dynamic SQL
- No user input directly in SQL strings

### 3. Path Traversal Protection
- Path validation against base directory
- All paths resolved and checked
- Uses pathlib for safe path manipulation

### 4. Data Privacy
- Local-only storage (no external transmission)
- Read-only database access for reporting
- Secure file permissions (0600 for reports)
- No external API calls

### 5. Resource Limits
- Thread limits on database (max 4)
- File size limits on CSV input
- String length limits on all fields
- Data volume checks in training

### 6. Code Quality
- Type hints throughout
- Comprehensive error handling
- Logging for debugging
- Clean separation of concerns

## Security Best Practices for Users

1. **Keep data local**: Never commit sensitive health data to public repositories
2. **Backup regularly**: Use encrypted backups for the database
3. **Limit access**: Use proper file permissions (chmod 600 for sensitive files)
4. **Review inputs**: Validate any data before ingesting
5. **Monitor resources**: Check disk space and memory usage
6. **Update dependencies**: Keep Python packages up to date for security patches

## Automated Security Scanning

This repository includes automated security scanning:
- CodeQL analysis for vulnerability detection
- Dependency scanning for known CVEs
- Code review for security issues

## Privacy Considerations

This is a **personal health tracking system** designed for:
- Single-user, local operation
- No cloud storage or transmission
- No third-party integrations by default
- Complete user control over data

### HIPAA Compliance Notes

If using for clinical purposes:
- This tool does NOT provide HIPAA compliance out-of-box
- Requires additional safeguards (encryption at rest, audit logging, access controls)
- Consult with compliance experts before clinical use

## Future Hardening Roadmap

Potential enhancements for increased security:

1. **Encryption**
   - Database encryption at rest
   - Encrypted backups
   - Memory encryption for sensitive data

2. **Audit Logging**
   - Track all data access
   - Log all modifications
   - Tamper-evident logging

3. **Access Control**
   - User authentication
   - Role-based access
   - Session management

4. **Data Anonymization**
   - PII detection and removal
   - Differential privacy for reports
   - Secure multi-party computation for group analysis

5. **Secure Development**
   - Pre-commit hooks for security checks
   - Automated dependency updates
   - Continuous security monitoring

## Contributing

When contributing, please:
1. Run security scanners before submitting PRs
2. Follow secure coding practices
3. Add tests for new features
4. Document security implications of changes

## License

[Add your license here]

## Disclaimer

This tool is for personal health tracking and research purposes only. It is not intended as medical advice or diagnosis. Always consult healthcare professionals for medical decisions.
