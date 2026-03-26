"""Configuration for the voice capture module."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("recordings")
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_CHANNELS = 1
_DEFAULT_GPIO_PIN = 17
_MIN_RECORDING_SECONDS = 0.5


@dataclass(frozen=True)
class VoiceConfig:
    """Immutable configuration for a voice-capture session.

    Attributes:
        output_dir: Directory where WAV files are saved.
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels (1 = mono).
        gpio_pin: BCM pin number for the tactile button.
        min_duration: Minimum recording duration in seconds.
                      Recordings shorter than this are discarded.
    """

    output_dir: Path = field(default=_DEFAULT_OUTPUT_DIR)
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    channels: int = _DEFAULT_CHANNELS
    gpio_pin: int = _DEFAULT_GPIO_PIN
    min_duration: float = _MIN_RECORDING_SECONDS

    def __post_init__(self) -> None:
        # Validate fields on construction.
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if not 0 <= self.gpio_pin <= 27:
            raise ValueError(f"gpio_pin must be 0-27, got {self.gpio_pin}")
        if self.min_duration < 0:
            raise ValueError(f"min_duration must be non-negative, got {self.min_duration}")

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it does not exist and return it."""
        resolved = self.output_dir.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory ready: %s", resolved)
        return resolved
