# HenryFood Voice Capture

Button-triggered voice recording for the HenryFood food diary, designed to run on a **Raspberry Pi**.

Press and hold a tactile push button wired to a GPIO pin — audio is captured while the button is held and saved as a timestamped WAV file when released.  A future phase will transcribe these recordings and convert them into structured food diary entries.

## Hardware Requirements

- Raspberry Pi (3B+ / 4 / 5) running Raspberry Pi OS
- USB microphone or USB sound card with mic input
- Tactile push button wired between **GPIO 17** (BCM) and **GND**

### Wiring

| Button Pin | Pi Pin        |
|------------|---------------|
| Leg A      | GPIO 17 (pin 11) |
| Leg B      | GND (pin 9)      |

The code enables the internal pull-up resistor, so no external resistor is needed.

## Installation

```bash
cd voice
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Record (main loop)

```bash
python -m henryfood_voice record
```

Press and hold the button to record; release to save.  Press **Ctrl-C** to exit.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `recordings` | Directory for WAV files |
| `-r` / `--sample-rate` | `16000` | Sample rate in Hz |
| `-c` / `--channels` | `1` | Mono (1) or stereo (2) |
| `-g` / `--gpio-pin` | `17` | BCM GPIO pin number |
| `--min-duration` | `0.5` | Discard recordings shorter than this (seconds) |
| `-v` / `--verbose` | off | Debug logging |

### Test audio (no button required)

Record a short clip to verify the microphone works:

```bash
python -m henryfood_voice test-audio --duration 5
```

## Project Structure

```
voice/
├── requirements.txt
├── README.md
├── henryfood_voice/
│   ├── __init__.py
│   ├── __main__.py       # python -m entry point
│   ├── cli.py            # Typer CLI
│   ├── config.py         # VoiceConfig dataclass
│   ├── recorder.py       # Recorder ABC + SounddeviceRecorder
│   ├── button.py         # GPIO button wrapper (gpiozero)
│   └── workflow.py       # CaptureWorkflow orchestrator
└── tests/
    ├── test_config.py
    ├── test_recorder.py
    └── test_workflow.py
```

## Design Notes

- **Recorder abstraction** — `recorder.py` defines an abstract `Recorder` base class.  The default `SounddeviceRecorder` uses *sounddevice* + *soundfile*; swap the backend by implementing a new subclass.
- **Config dataclass** — `VoiceConfig` is an immutable `@dataclass(frozen=True)` with validation in `__post_init__`.
- **Logging** — all modules use the standard-library `logging` module.  Pass `-v` to the CLI for debug output.
- **Pathlib** — all file paths use `pathlib.Path`.
- **Thread safety** — the recorder uses a lock to guard start/stop transitions since GPIO callbacks run on background threads.

## Running Tests

```bash
pip install pytest
cd voice
python -m pytest tests/ -v
```

Tests mock `sounddevice`, `soundfile`, and `gpiozero` so they run on any machine — no Pi or audio hardware needed.

## Future Work

- Transcribe WAV files using a speech-to-text model (Whisper, etc.)
- Parse transcriptions into structured food diary entries via LLM
- Integrate with the existing HenryFood analytics pipeline
