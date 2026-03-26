"""Tests for app.recorder."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import AppConfig


# ---------------------------------------------------------------------------
# Fixture: inject fake sounddevice / soundfile modules so that the lazy
# ``import sounddevice as sd`` and ``import soundfile as sf`` inside
# SounddeviceRecorder resolve without PortAudio installed.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_audio_libs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``sounddevice`` and ``soundfile`` importable as mock modules."""
    for name in ("sounddevice", "soundfile"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, MagicMock())


@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(recordings_dir=tmp_path / "recordings", sample_rate=16_000, channels=1)


class TestSounddeviceRecorder:
    """All tests mock sounddevice/soundfile so they run without audio hardware."""

    @patch("app.recorder.time")
    @patch("sounddevice.InputStream", create=True)
    @patch("soundfile.write", create=True)
    def test_start_stop_saves_wav(
        self,
        mock_sf_write: MagicMock,
        mock_input_stream: MagicMock,
        mock_time: MagicMock,
        config: AppConfig,
    ) -> None:
        from app.recorder import SounddeviceRecorder

        mock_time.monotonic.side_effect = [0.0, 2.0]
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        recorder = SounddeviceRecorder(config)
        recorder.start()
        assert recorder.is_recording
        mock_stream.start.assert_called_once()

        # Simulate audio frames via the callback
        callback = mock_input_stream.call_args[1]["callback"]
        callback(np.zeros((1600, 1), dtype="float32"), 1600, None, None)

        meta = recorder.stop()
        assert not recorder.is_recording
        assert meta is not None
        assert meta.file_path.suffix == ".wav"
        mock_sf_write.assert_called_once()

    def test_stop_without_start_returns_none(self, config: AppConfig) -> None:
        from app.recorder import SounddeviceRecorder

        recorder = SounddeviceRecorder(config)
        assert recorder.stop() is None

    @patch("sounddevice.InputStream", create=True)
    def test_double_start_is_safe(
        self, mock_input_stream: MagicMock, config: AppConfig
    ) -> None:
        from app.recorder import SounddeviceRecorder

        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(config)
        recorder.start()
        recorder.start()  # no-op
        assert mock_input_stream.call_count == 1

    @patch("app.recorder.time")
    @patch("sounddevice.InputStream", create=True)
    @patch("soundfile.write", create=True)
    def test_metadata_fields(
        self,
        _mock_sf: MagicMock,
        mock_input_stream: MagicMock,
        mock_time: MagicMock,
        config: AppConfig,
    ) -> None:
        from app.recorder import SounddeviceRecorder

        mock_time.monotonic.side_effect = [0.0, 3.5]
        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(config)
        recorder.start()
        recorder._frames.append(np.zeros((160, 1), dtype="float32"))
        meta = recorder.stop()
        assert meta is not None
        assert meta.sample_rate == 16_000
        assert meta.channels == 1
        assert meta.duration_seconds == 3.5
        assert meta.file_path.name.endswith(".wav")
        assert meta.transcription is None

    @patch("app.recorder.time")
    @patch("sounddevice.InputStream", create=True)
    @patch("soundfile.write", create=True)
    def test_filename_uses_timestamp_format(
        self,
        _mock_sf: MagicMock,
        mock_input_stream: MagicMock,
        mock_time: MagicMock,
        config: AppConfig,
    ) -> None:
        """Filenames follow YYYY-MM-DD_HH-MM-SS.wav convention."""
        import re

        from app.recorder import SounddeviceRecorder

        mock_time.monotonic.side_effect = [0.0, 1.0]
        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(config)
        recorder.start()
        recorder._frames.append(np.zeros((160, 1), dtype="float32"))
        meta = recorder.stop()
        assert meta is not None
        # e.g. 2026-03-26_15-40-10.wav
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.wav$", meta.file_path.name)
