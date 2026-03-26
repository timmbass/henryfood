"""Tests for henryfood_voice.recorder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from henryfood_voice.config import VoiceConfig
from henryfood_voice.recorder import SounddeviceRecorder


@pytest.fixture()
def config(tmp_path: Path) -> VoiceConfig:
    return VoiceConfig(output_dir=tmp_path / "recordings", sample_rate=16_000, channels=1)


class TestSounddeviceRecorder:
    """Tests that mock out sounddevice so we can run without audio hardware."""

    @patch("henryfood_voice.recorder.time")
    @patch("sounddevice.InputStream")
    @patch("soundfile.write")
    def test_start_stop_saves_wav(
        self, mock_sf_write: MagicMock, mock_input_stream: MagicMock,
        mock_time: MagicMock, config: VoiceConfig,
    ) -> None:
        # Simulate 2 seconds of elapsed time.
        mock_time.monotonic.side_effect = [0.0, 2.0]
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        recorder = SounddeviceRecorder(config)

        # --- start ---
        recorder.start()
        assert recorder.is_recording
        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()

        # Simulate audio frames arriving via the callback.
        callback = mock_input_stream.call_args[1]["callback"]
        fake_audio = np.zeros((1600, 1), dtype="float32")
        callback(fake_audio, 1600, None, None)

        # --- stop ---
        path = recorder.stop()
        assert not recorder.is_recording
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert path is not None
        assert path.suffix == ".wav"
        mock_sf_write.assert_called_once()

    def test_stop_without_start_returns_none(self, config: VoiceConfig) -> None:
        recorder = SounddeviceRecorder(config)
        assert recorder.stop() is None

    @patch("sounddevice.InputStream")
    def test_double_start_is_safe(
        self, mock_input_stream: MagicMock, config: VoiceConfig
    ) -> None:
        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(config)
        recorder.start()
        recorder.start()  # second call should be a no-op
        assert mock_input_stream.call_count == 1

    @patch("sounddevice.InputStream")
    def test_short_recording_discarded(
        self, mock_input_stream: MagicMock, config: VoiceConfig
    ) -> None:
        """A recording shorter than min_duration should return None."""
        strict_config = VoiceConfig(
            output_dir=config.output_dir,
            sample_rate=config.sample_rate,
            channels=config.channels,
            min_duration=10.0,  # very long minimum
        )
        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(strict_config)
        recorder.start()
        path = recorder.stop()
        assert path is None

    @patch("henryfood_voice.recorder.time")
    @patch("sounddevice.InputStream")
    @patch("soundfile.write")
    def test_wav_filename_format(
        self, _mock_sf_write: MagicMock, mock_input_stream: MagicMock,
        mock_time: MagicMock, config: VoiceConfig,
    ) -> None:
        mock_time.monotonic.side_effect = [0.0, 2.0]
        mock_input_stream.return_value = MagicMock()
        recorder = SounddeviceRecorder(config)
        recorder.start()
        # inject a frame so save succeeds
        recorder._frames.append(np.zeros((160, 1), dtype="float32"))
        path = recorder.stop()
        assert path is not None
        assert path.name.startswith("recording_")
        assert path.name.endswith(".wav")
