"""Typer CLI entry points for the HenryFood voice recorder app."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import typer

from app.config import AppConfig

app = typer.Typer(help="HenryFood voice recorder")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s – %(message)s")


# ---------------------------------------------------------------------------
# run  (real GPIO mode, Raspberry Pi)
# ---------------------------------------------------------------------------


@app.command()
def run(
    output_dir: Path = typer.Option(Path("recordings"), help="Directory to save recordings"),
    sample_rate: int = typer.Option(16000, help="Sample rate in Hz"),
    max_duration: float = typer.Option(300.0, help="Max recording duration in seconds"),
    gpio_pin: int = typer.Option(17, help="GPIO pin for the physical button"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the recorder with a physical GPIO button (Raspberry Pi only)."""
    _setup_logging(verbose)

    from app.gpio_button import GpioButtonWrapper  # noqa: PLC0415 – GPIO import isolated
    from app.main import CaptureWorkflow  # noqa: PLC0415
    from app.recorder import SounddeviceRecorder  # noqa: PLC0415

    config = AppConfig(
        output_dir=output_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
        gpio_pin=gpio_pin,
    )

    button = GpioButtonWrapper(pin=config.gpio_pin)
    recorder = SounddeviceRecorder(config)
    workflow = CaptureWorkflow(button=button, recorder=recorder, config=config)

    logging.getLogger(__name__).info("GPIO mode – press the button to record.  Ctrl-C to exit.")

    stop_event = _make_stop_event()
    stop_event.wait()

    _maybe_stop_recorder(workflow, recorder)


# ---------------------------------------------------------------------------
# run-mock  (development mode, no GPIO hardware needed)
# ---------------------------------------------------------------------------


@app.command(name="run-mock")
def run_mock(
    output_dir: Path = typer.Option(Path("recordings"), help="Directory to save recordings"),
    sample_rate: int = typer.Option(16000, help="Sample rate in Hz"),
    max_duration: float = typer.Option(300.0, help="Max recording duration in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the recorder in mock mode (no GPIO hardware required).

    Controls:
      Enter     – start recording / stop and save
      q + Enter – quit
    """
    _setup_logging(verbose)

    from app.main import CaptureWorkflow  # noqa: PLC0415
    from app.mock_button import MockButton  # noqa: PLC0415
    from app.recorder import SounddeviceRecorder  # noqa: PLC0415

    config = AppConfig(
        output_dir=output_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
    )

    button = MockButton()
    recorder = SounddeviceRecorder(config)
    workflow = CaptureWorkflow(button=button, recorder=recorder, config=config)  # noqa: F841

    button.start(on_quit=lambda: None)
    button.wait()

    _maybe_stop_recorder(workflow, recorder)


# ---------------------------------------------------------------------------
# test-button  (verify GPIO wiring)
# ---------------------------------------------------------------------------


@app.command(name="test-button")
def test_button(
    gpio_pin: int = typer.Option(17, help="GPIO pin to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Print a message every time the GPIO button is pressed or released."""
    _setup_logging(verbose)

    from app.gpio_button import GpioButtonWrapper  # noqa: PLC0415

    button = GpioButtonWrapper(pin=gpio_pin)
    button.when_pressed = lambda: print("Button PRESSED", flush=True)  # noqa: T201
    button.when_released = lambda: print("Button RELEASED", flush=True)  # noqa: T201

    print(f"Listening on GPIO pin {gpio_pin} – Ctrl-C to exit.", flush=True)  # noqa: T201
    stop_event = _make_stop_event()
    stop_event.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stop_event() -> "threading.Event":
    """Return a threading.Event that is set when SIGINT/SIGTERM arrives."""
    import threading

    event = threading.Event()

    def _handler(sig, frame):  # noqa: ANN001
        event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return event


def _maybe_stop_recorder(workflow, recorder) -> None:  # noqa: ANN001
    """If a recording is in progress when exiting, stop it gracefully."""
    if workflow._recording:  # noqa: SLF001
        path = recorder.stop()
        if path:
            logging.getLogger(__name__).info("Final recording saved: %s", path)
