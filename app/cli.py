"""Typer CLI for the HenryFood button recorder.

Entry point
~~~~~~~~~~~
::

    python -m app.cli run [OPTIONS]

The ``run`` command starts the button-triggered recording loop.
Press and hold the GPIO button to record; release to save a WAV file.
Hit Ctrl-C to exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from app.config import AppConfig

app = typer.Typer(
    name="henryfood",
    help="Raspberry Pi button-triggered voice recorder for the HenryFood food diary.",
    add_completion=False,
)


@app.command()
def run(
    gpio_pin: int = typer.Option(17, "--gpio-pin", "-g", help="BCM GPIO pin for the button."),
    sample_rate: int = typer.Option(16_000, "--sample-rate", "-r", help="Audio sample rate in Hz."),
    channels: int = typer.Option(1, "--channels", "-c", help="Audio channels (1=mono, 2=stereo)."),
    max_duration: float = typer.Option(
        30.0, "--max-duration", "-m", help="Max recording length in seconds."
    ),
    recordings_dir: Path = typer.Option(
        "recordings", "--recordings-dir", "-o", help="Directory for saved WAV files."
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG/INFO/…)."),
) -> None:
    """Start the button-triggered recording loop.

    Press and hold the button to record.  Release the button to save.
    The recording auto-stops after --max-duration seconds.
    Ctrl-C to exit.
    """
    config = AppConfig(
        gpio_pin=gpio_pin,
        sample_rate=sample_rate,
        channels=channels,
        max_duration_seconds=max_duration,
        recordings_dir=recordings_dir,
        log_level=log_level,
    )

    # Set up logging before anything else
    from app.utils import setup_logging  # noqa: PLC0415

    setup_logging(config)

    # Late import so missing gpiozero / sounddevice errors appear *after*
    # config validation, not during CLI parsing.
    from app.main import CaptureWorkflow  # noqa: PLC0415

    workflow = CaptureWorkflow(config)
    workflow.run()


@app.command()
def test_audio(
    sample_rate: int = typer.Option(16_000, "--sample-rate", "-r", help="Audio sample rate in Hz."),
    channels: int = typer.Option(1, "--channels", "-c", help="Audio channels (1=mono, 2=stereo)."),
    duration: float = typer.Option(3.0, "--duration", "-d", help="Seconds to record."),
    recordings_dir: Path = typer.Option(
        "recordings", "--recordings-dir", "-o", help="Directory for saved WAV files."
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG/INFO/…)."),
) -> None:
    """Record a short test clip without needing a GPIO button.

    Useful for verifying that the microphone works before wiring up
    the physical button.
    """
    import time as _time  # noqa: PLC0415

    config = AppConfig(
        sample_rate=sample_rate,
        channels=channels,
        max_duration_seconds=duration,
        recordings_dir=recordings_dir,
        log_level=log_level,
    )

    from app.utils import setup_logging  # noqa: PLC0415

    setup_logging(config)

    from app.recorder import SounddeviceRecorder  # noqa: PLC0415

    recorder = SounddeviceRecorder(config)
    typer.echo(f"Recording for {duration:.1f} s — speak into the microphone …")
    recorder.start()
    _time.sleep(duration)
    meta = recorder.stop()
    if meta is not None:
        typer.echo(f"Saved: {meta.file_path}")
    else:
        typer.echo("Recording discarded (no audio captured).")


# Allow `python -m app.cli run`
if __name__ == "__main__":
    app()
