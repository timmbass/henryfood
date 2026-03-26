"""Typer CLI for the voice capture module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from henryfood_voice.config import VoiceConfig

app = typer.Typer(
    name="henryfood-voice",
    help="Button-triggered voice capture for the HenryFood diary.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def record(
    output_dir: Path = typer.Option(
        "recordings", "--output-dir", "-o", help="Directory for WAV files."
    ),
    sample_rate: int = typer.Option(
        16_000, "--sample-rate", "-r", help="Audio sample rate in Hz."
    ),
    channels: int = typer.Option(
        1, "--channels", "-c", help="Number of audio channels (1 or 2)."
    ),
    gpio_pin: int = typer.Option(
        17, "--gpio-pin", "-g", help="BCM GPIO pin for the button."
    ),
    min_duration: float = typer.Option(
        0.5, "--min-duration", help="Minimum recording length in seconds."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Start the button-triggered recording loop.

    Press and hold the tactile button to record; release to save.
    Ctrl-C to exit.
    """
    _setup_logging(verbose)

    config = VoiceConfig(
        output_dir=output_dir,
        sample_rate=sample_rate,
        channels=channels,
        gpio_pin=gpio_pin,
        min_duration=min_duration,
    )

    # Import here so missing gpiozero/sounddevice fails late, not at CLI parse time.
    from henryfood_voice.workflow import CaptureWorkflow  # noqa: PLC0415

    workflow = CaptureWorkflow(config)
    workflow.run()


@app.command()
def test_audio(
    output_dir: Path = typer.Option(
        "recordings", "--output-dir", "-o", help="Directory for WAV files."
    ),
    sample_rate: int = typer.Option(
        16_000, "--sample-rate", "-r", help="Audio sample rate in Hz."
    ),
    channels: int = typer.Option(
        1, "--channels", "-c", help="Number of audio channels (1 or 2)."
    ),
    duration: float = typer.Option(
        3.0, "--duration", "-d", help="Seconds to record."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Record a short test clip without needing a GPIO button."""
    import time

    _setup_logging(verbose)

    config = VoiceConfig(
        output_dir=output_dir,
        sample_rate=sample_rate,
        channels=channels,
    )

    from henryfood_voice.recorder import SounddeviceRecorder  # noqa: PLC0415

    recorder = SounddeviceRecorder(config)
    typer.echo(f"Recording for {duration}s …")
    recorder.start()
    time.sleep(duration)
    path = recorder.stop()
    if path is not None:
        typer.echo(f"Saved: {path}")
    else:
        typer.echo("Recording discarded (too short).")


def main() -> None:
    """Entry point (called from ``__main__`` or console-script)."""
    app()
