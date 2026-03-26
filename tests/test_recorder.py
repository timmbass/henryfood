"""Tests for SounddeviceRecorder."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import threading

import pytest

from app.config import AppConfig
from app.recorder import SounddeviceRecorder


@pytest.fixture()
def config(tmp_path):
    return AppConfig(output_dir=tmp_path, sample_rate=16000, max_duration=10.0)


@pytest.fixture()
def recorder(config):
    return SounddeviceRecorder(config)


class TestSounddeviceRecorder:
    def test_stop_without_start_returns_none(self, recorder):
        result = recorder.stop()
        assert result is None

    def test_start_creates_stream(self, recorder, _mock_sounddevice):
        mock_sd = _mock_sounddevice["sd"]
        recorder.start()
        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()
        recorder._timer.cancel()  # clean up

    def test_double_start_is_ignored(self, recorder, _mock_sounddevice):
        mock_sd = _mock_sounddevice["sd"]
        recorder.start()
        recorder.start()
        assert mock_sd.InputStream.call_count == 1
        recorder._timer.cancel()

    def test_stop_saves_file_when_frames_exist(self, recorder, _mock_sounddevice, tmp_path):
        mock_sd = _mock_sounddevice["sd"]
        mock_sf = _mock_sounddevice["sf"]
        mock_np = _mock_sounddevice["np"]

        recorder.start()
        # Simulate a captured frame
        recorder._frames.append(b"frame1")
        path = recorder.stop()

        assert path is not None
        assert path.suffix == ".wav"
        mock_sf.write.assert_called_once()
        mock_np.concatenate.assert_called_once()

    def test_stop_returns_none_when_no_frames(self, recorder, _mock_sounddevice):
        recorder.start()
        recorder._frames = []
        result = recorder.stop()
        assert result is None

    def test_timer_cancelled_on_stop(self, recorder, _mock_sounddevice):
        recorder.start()
        recorder._frames.append(b"frame")
        recorder.stop()
        assert recorder._timer is None

    def test_filename_format(self, recorder, _mock_sounddevice):
        """Saved filename must match YYYY-MM-DD_HH-MM-SS.wav pattern."""
        import re

        recorder.start()
        recorder._frames.append(b"frame")
        path = recorder.stop()
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.wav", path.name)
