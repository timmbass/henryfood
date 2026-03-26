"""Tests for henryfood_voice.workflow."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from henryfood_voice.config import VoiceConfig
from henryfood_voice.recorder import Recorder


class FakeRecorder(Recorder):
    """In-memory recorder for testing without audio hardware."""

    def __init__(self, save_path: Optional[Path] = None) -> None:
        self._recording = False
        self._save_path = save_path
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self._recording = True
        self.start_count += 1

    def stop(self) -> Optional[Path]:
        self._recording = False
        self.stop_count += 1
        return self._save_path

    @property
    def is_recording(self) -> bool:
        return self._recording


class TestCaptureWorkflow:
    @patch("henryfood_voice.workflow.Button")
    def test_on_press_starts_recording(
        self, mock_button_cls: MagicMock, tmp_path: Path
    ) -> None:
        from henryfood_voice.workflow import CaptureWorkflow

        config = VoiceConfig(output_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)

        wf._on_press()
        assert fake.is_recording
        assert fake.start_count == 1

    @patch("henryfood_voice.workflow.Button")
    def test_on_release_stops_recording(
        self, mock_button_cls: MagicMock, tmp_path: Path
    ) -> None:
        from henryfood_voice.workflow import CaptureWorkflow

        config = VoiceConfig(output_dir=tmp_path)
        save_path = tmp_path / "test.wav"
        fake = FakeRecorder(save_path=save_path)
        wf = CaptureWorkflow(config, recorder=fake)

        wf._on_press()
        wf._on_release()
        assert not fake.is_recording
        assert fake.stop_count == 1

    @patch("henryfood_voice.workflow.Button")
    def test_stop_sets_event(
        self, mock_button_cls: MagicMock, tmp_path: Path
    ) -> None:
        from henryfood_voice.workflow import CaptureWorkflow

        config = VoiceConfig(output_dir=tmp_path)
        wf = CaptureWorkflow(config, recorder=FakeRecorder())
        wf.stop()
        assert wf._stop_event.is_set()

    @patch("henryfood_voice.workflow.Button")
    def test_shutdown_stops_active_recording(
        self, mock_button_cls: MagicMock, tmp_path: Path
    ) -> None:
        from henryfood_voice.workflow import CaptureWorkflow

        config = VoiceConfig(output_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)

        wf._on_press()
        assert fake.is_recording
        wf._shutdown()
        assert not fake.is_recording
