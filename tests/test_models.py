"""Tests for app.models."""

from datetime import datetime, timezone
from pathlib import Path

from app.models import RecordingMetadata


class TestRecordingMetadata:
    def test_create_metadata(self) -> None:
        meta = RecordingMetadata(
            file_path=Path("/tmp/recording_20260101T120000Z.wav"),
            recorded_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            duration_seconds=5.0,
            sample_rate=16_000,
            channels=1,
        )
        assert meta.file_path == Path("/tmp/recording_20260101T120000Z.wav")
        assert meta.duration_seconds == 5.0
        assert meta.transcription is None

    def test_transcription_placeholder(self) -> None:
        meta = RecordingMetadata(
            file_path=Path("/tmp/test.wav"),
            recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            duration_seconds=2.0,
            sample_rate=16_000,
            channels=1,
            transcription="hello world",
        )
        assert meta.transcription == "hello world"
