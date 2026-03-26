"""Tests for AppConfig."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import AppConfig


def test_defaults():
    cfg = AppConfig()
    assert cfg.output_dir == Path("recordings")
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.max_duration == 300.0
    assert cfg.gpio_pin == 17


def test_custom_values():
    cfg = AppConfig(output_dir=Path("/tmp/audio"), sample_rate=44100, max_duration=60.0, gpio_pin=22)
    assert cfg.output_dir == Path("/tmp/audio")
    assert cfg.sample_rate == 44100
    assert cfg.max_duration == 60.0
    assert cfg.gpio_pin == 22


def test_frozen_model():
    from pydantic import ValidationError

    cfg = AppConfig()
    with pytest.raises(ValidationError):
        cfg.sample_rate = 8000  # type: ignore[misc]
