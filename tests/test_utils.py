"""Tests for app.utils."""

from pathlib import Path

from app.config import AppConfig
from app.utils import ensure_recordings_dir


class TestEnsureRecordingsDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "recordings"
        cfg = AppConfig(recordings_dir=target)
        result = ensure_recordings_dir(cfg)
        assert result.is_dir()
        assert result == target.resolve()

    def test_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "recordings"
        cfg = AppConfig(recordings_dir=target)
        ensure_recordings_dir(cfg)
        ensure_recordings_dir(cfg)  # should not raise
        assert target.is_dir()
