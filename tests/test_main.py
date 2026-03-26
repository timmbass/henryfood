"""Tests for app.main (CaptureWorkflow)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from app.config import AppConfig
from app.models import RecordingMetadata
from app.recorder import Recorder


def _make_metadata(tmp_path: Path) -> RecordingMetadata:
    return RecordingMetadata(
        file_path=tmp_path / "2024-01-01_00-00-00.wav",
        recorded_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        duration_seconds=2.0,
        sample_rate=16_000,
        channels=1,
    )


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

    # -- transfer integration ------------------------------------------------

    @patch("app.main.GpioButtonWrapper")
    def test_on_release_pushes_transfer_when_recording_saved(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """A successful recording should trigger transfer.push()."""
        from app.main import CaptureWorkflow

        meta = _make_metadata(tmp_path)
        fake = FakeRecorder(result=meta)
        mock_transfer = MagicMock()
        config = AppConfig(recordings_dir=tmp_path)
        wf = CaptureWorkflow(config, recorder=fake, transfer=mock_transfer)
        wf._on_press()
        wf._on_release()
        mock_transfer.push.assert_called_once_with(meta.file_path)

    @patch("app.main.GpioButtonWrapper")
    def test_on_release_no_transfer_when_recording_discarded(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """A discarded recording (too short / no audio) must not trigger transfer."""
        from app.main import CaptureWorkflow

        fake = FakeRecorder(result=None)
        mock_transfer = MagicMock()
        config = AppConfig(recordings_dir=tmp_path)
        wf = CaptureWorkflow(config, recorder=fake, transfer=mock_transfer)
        wf._on_press()
        wf._on_release()
        mock_transfer.push.assert_not_called()

    @patch("app.main.GpioButtonWrapper")
    def test_transfer_is_none_when_disabled_in_config(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """transfer_enabled=False → no WavTransfer is created."""
        from app.main import CaptureWorkflow

        config = AppConfig(recordings_dir=tmp_path, transfer_enabled=False)
        wf = CaptureWorkflow(config, recorder=FakeRecorder())
        assert wf._transfer is None

    @patch("app.main.GpioButtonWrapper")
    def test_run_calls_retry_pending_on_startup(
        self, _mock_btn: MagicMock, tmp_path: Path
    ) -> None:
        """run() should call retry_pending() before entering the event loop."""
        from app.main import CaptureWorkflow

        mock_transfer = MagicMock()
        config = AppConfig(recordings_dir=tmp_path)
        wf = CaptureWorkflow(config, recorder=FakeRecorder(), transfer=mock_transfer)
        wf._stop_event.set()  # return immediately from wait()
        wf.run()
        mock_transfer.retry_pending.assert_called_once()
