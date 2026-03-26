"""Tests for henryfood_voice.config."""

from pathlib import Path

import pytest

from henryfood_voice.config import VoiceConfig


class TestVoiceConfigDefaults:
    def test_default_values(self) -> None:
        cfg = VoiceConfig()
        assert cfg.output_dir == Path("recordings")
        assert cfg.sample_rate == 16_000
        assert cfg.channels == 1
        assert cfg.gpio_pin == 17
        assert cfg.min_duration == 0.5

    def test_custom_values(self) -> None:
        cfg = VoiceConfig(
            output_dir=Path("/tmp/test_voice"),
            sample_rate=44_100,
            channels=2,
            gpio_pin=22,
            min_duration=1.0,
        )
        assert cfg.sample_rate == 44_100
        assert cfg.channels == 2
        assert cfg.gpio_pin == 22
        assert cfg.min_duration == 1.0

    def test_frozen(self) -> None:
        cfg = VoiceConfig()
        with pytest.raises(AttributeError):
            cfg.sample_rate = 8000  # type: ignore[misc]


class TestVoiceConfigValidation:
    def test_invalid_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            VoiceConfig(sample_rate=0)

    def test_invalid_channels(self) -> None:
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            VoiceConfig(channels=3)

    def test_invalid_gpio_pin_low(self) -> None:
        with pytest.raises(ValueError, match="gpio_pin must be 0-27"):
            VoiceConfig(gpio_pin=-1)

    def test_invalid_gpio_pin_high(self) -> None:
        with pytest.raises(ValueError, match="gpio_pin must be 0-27"):
            VoiceConfig(gpio_pin=28)

    def test_negative_min_duration(self) -> None:
        with pytest.raises(ValueError, match="min_duration must be non-negative"):
            VoiceConfig(min_duration=-0.1)

    def test_zero_min_duration_ok(self) -> None:
        cfg = VoiceConfig(min_duration=0)
        assert cfg.min_duration == 0


class TestEnsureOutputDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "recordings"
        cfg = VoiceConfig(output_dir=target)
        result = cfg.ensure_output_dir()
        assert result.is_dir()
        assert result == target.resolve()

    def test_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "recordings"
        cfg = VoiceConfig(output_dir=target)
        cfg.ensure_output_dir()
        cfg.ensure_output_dir()  # should not raise
        assert target.is_dir()
