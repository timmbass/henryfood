"""Tests for app.main (CaptureWorkflow)."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from app.config import AppConfig
from app.models import RecordingMetadata
from app.recorder import Recorder


class FakeRecorder(Recorder):
    """In-memory recorder stub for testing without audio hardware.

    Mimics the duplicate-prevention behaviour of :class:`SounddeviceRecorder`:
    ``start()`` is a no-op while already recording, and ``stop()`` returns
    ``None`` while not recording.
    """

    def __init__(self, result: Optional[RecordingMetadata] = None) -> None:
        self._recording = False
        self._result = result
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        if self._recording:
            return
        self._recording = True
        self.start_count += 1

    def stop(self) -> Optional[RecordingMetadata]:
        if not self._recording:
            return None
        self._recording = False
        self.stop_count += 1
        return self._result

    @property
    def is_recording(self) -> bool:
        return self._recording


class TestCaptureWorkflow:
    @patch("app.main.GpioButtonWrapper")
    def test_on_press_starts_recording(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)
        wf._on_press()
        assert fake.is_recording
        assert fake.start_count == 1

    @patch("app.main.GpioButtonWrapper")
    def test_on_release_stops_recording(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)
        wf._on_press()
        wf._on_release()
        assert not fake.is_recording
        assert fake.stop_count == 1

    @patch("app.main.GpioButtonWrapper")
    def test_stop_sets_event(self, _mock_btn: MagicMock, tmp_path: Path) -> None:
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        wf = CaptureWorkflow(config, recorder=FakeRecorder())
        wf.stop()
        assert wf._stop_event.is_set()

    @patch("app.main.GpioButtonWrapper")
    def test_shutdown_stops_active_recording(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)
        wf._on_press()
        assert fake.is_recording
        wf._shutdown()
        assert not fake.is_recording

    @patch("app.main.GpioButtonWrapper")
    def test_duplicate_press_ignored_while_recording(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """A second press while already recording should be a no-op."""
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)
        wf._on_press()
        wf._on_press()  # duplicate — should be ignored
        assert fake.is_recording
        assert fake.start_count == 1

    @patch("app.main.GpioButtonWrapper")
    def test_duplicate_release_ignored_while_not_recording(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """A release while not recording should be a no-op (no crash, no file)."""
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)
        wf._on_release()  # release without pressing first
        assert not fake.is_recording
        assert fake.stop_count == 0

    @patch("app.main.GpioButtonWrapper")
    def test_full_press_release_cycle(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """Two complete press-release cycles should each record independently."""
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path)
        fake = FakeRecorder()
        wf = CaptureWorkflow(config, recorder=fake)

        # First cycle
        wf._on_press()
        assert fake.is_recording
        wf._on_release()
        assert not fake.is_recording

        # Second cycle
        wf._on_press()
        assert fake.is_recording
        wf._on_release()
        assert not fake.is_recording

        assert fake.start_count == 2
        assert fake.stop_count == 2
