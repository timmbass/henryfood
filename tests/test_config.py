"""Tests for app.config."""

from pathlib import Path

import pytest

from app.config import AppConfig


class TestAppConfigDefaults:
    """Verify that the default configuration is sensible."""

    def test_default_values(self) -> None:
        cfg = AppConfig()
        assert cfg.gpio_pin == 17
        assert cfg.sample_rate == 16_000
        assert cfg.channels == 1
        assert cfg.max_duration_seconds == 30.0
        assert cfg.recordings_dir == Path("recordings")
        assert cfg.log_level == "INFO"

    def test_custom_values(self) -> None:
        cfg = AppConfig(
            gpio_pin=22,
            sample_rate=44_100,
            channels=2,
            max_duration_seconds=60.0,
            recordings_dir=Path("/tmp/test_recordings"),
            log_level="DEBUG",
        )
        assert cfg.gpio_pin == 22
        assert cfg.sample_rate == 44_100
        assert cfg.channels == 2
        assert cfg.max_duration_seconds == 60.0
        assert cfg.recordings_dir == Path("/tmp/test_recordings")
        assert cfg.log_level == "DEBUG"

    def test_frozen(self) -> None:
        cfg = AppConfig()
        with pytest.raises(Exception):  # pydantic raises ValidationError
            cfg.sample_rate = 8000  # type: ignore[misc]


class TestAppConfigValidation:
    """Verify that invalid values are rejected."""

    def test_negative_gpio_pin(self) -> None:
        with pytest.raises(Exception):
            AppConfig(gpio_pin=-1)

    def test_gpio_pin_too_high(self) -> None:
        with pytest.raises(Exception):
            AppConfig(gpio_pin=28)

    def test_zero_sample_rate(self) -> None:
        with pytest.raises(Exception):
            AppConfig(sample_rate=0)

    def test_invalid_channels(self) -> None:
        with pytest.raises(Exception):
            AppConfig(channels=3)

    def test_zero_max_duration(self) -> None:
        with pytest.raises(Exception):
            AppConfig(max_duration_seconds=0)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(Exception):
            AppConfig(log_level="VERBOSE")  # type: ignore[arg-type]

    def test_recordings_dir_string_coercion(self) -> None:
        cfg = AppConfig(recordings_dir="/tmp/voice")  # type: ignore[arg-type]
        assert isinstance(cfg.recordings_dir, Path)
        assert cfg.recordings_dir == Path("/tmp/voice")
