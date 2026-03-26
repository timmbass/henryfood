"""Application configuration.

Uses a Pydantic model so every setting is validated at startup.
All fields carry sensible defaults for a Raspberry Pi with a single
USB microphone and a tactile push-button on GPIO 17.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    """Top-level configuration for the button recorder.

    Attributes:
        gpio_pin: BCM pin number the tactile button is wired to (0-27).
        sample_rate: Audio sample rate in Hz (e.g. 16 000 for speech).
        channels: Number of audio channels (1 = mono, 2 = stereo).
        max_duration_seconds: Hard cap on a single recording in seconds.
            The recording stops automatically after this limit even if
            the button is still held.
        recordings_dir: Directory where WAV files are saved.
            Created automatically at startup if it does not exist.
        log_level: Python log level string.
    """

    gpio_pin: int = Field(default=17, ge=0, le=27, description="BCM GPIO pin for the button")
    sample_rate: int = Field(default=16_000, gt=0, description="Audio sample rate in Hz")
    channels: int = Field(default=1, ge=1, le=2, description="Audio channels (1=mono, 2=stereo)")
    max_duration_seconds: float = Field(
        default=30.0, gt=0, description="Maximum recording length in seconds"
    )
    recordings_dir: Path = Field(
        default=Path("recordings"), description="Directory for WAV files"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Python log level"
    )

    # Pydantic v2 config
    model_config = {"frozen": True}

    @field_validator("recordings_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        """Accept str or Path and normalise to Path."""
        return Path(v) if not isinstance(v, Path) else v
