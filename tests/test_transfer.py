"""Tests for app.transfer (WavTransfer)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig
from app.transfer import WavTransfer


def _config(tmp_path: Path, *, transfer_enabled: bool = True) -> AppConfig:
    return AppConfig(recordings_dir=tmp_path, transfer_enabled=transfer_enabled)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


class TestManifest:
    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        assert t._read_manifest() == set()

    def test_add_appears_in_manifest(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()
        t._add_to_manifest(wav)
        assert wav.resolve() in t._read_manifest()

    def test_remove_clears_entry(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()
        t._add_to_manifest(wav)
        t._remove_from_manifest(wav)
        assert t._read_manifest() == set()

    def test_manifest_file_deleted_when_empty(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()
        t._add_to_manifest(wav)
        assert t._manifest.exists()
        t._remove_from_manifest(wav)
        assert not t._manifest.exists()

    def test_duplicate_add_is_idempotent(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()
        t._add_to_manifest(wav)
        t._add_to_manifest(wav)
        assert len(t._read_manifest()) == 1

    def test_remove_nonexistent_entry_is_safe(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "ghost.wav"
        # Should not raise even if the path was never added
        t._remove_from_manifest(wav)


# ---------------------------------------------------------------------------
# _do_push
# ---------------------------------------------------------------------------


class TestDoPush:
    def test_returns_true_when_transfer_disabled(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path, transfer_enabled=False))
        wav = tmp_path / "a.wav"
        wav.touch()
        assert t._do_push(wav) is True

    def test_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        assert t._do_push(tmp_path / "missing.wav") is False

    def test_success_returns_true_and_clears_manifest(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()
        # Seed the manifest so we verify it gets cleared on success.
        t._add_to_manifest(wav)

        ok_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=ok_result) as mock_run:
            assert t._do_push(wav) is True

        mock_run.assert_called_once()
        assert t._read_manifest() == set()

    def test_rsync_failure_returns_false_and_adds_to_manifest(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()

        fail_result = MagicMock(returncode=23, stderr="partial transfer")
        with patch("subprocess.run", return_value=fail_result):
            assert t._do_push(wav) is False

        assert wav.resolve() in t._read_manifest()

    def test_timeout_adds_to_manifest(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="rsync", timeout=60),
        ):
            assert t._do_push(wav) is False

        assert wav.resolve() in t._read_manifest()

    def test_os_error_adds_to_manifest(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()

        with patch("subprocess.run", side_effect=OSError("rsync not found")):
            assert t._do_push(wav) is False

        assert wav.resolve() in t._read_manifest()

    def test_rsync_command_includes_key_and_destination(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wav = tmp_path / "a.wav"
        wav.touch()

        ok_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=ok_result) as mock_run:
            t._do_push(wav)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "rsync"
        # destination contains user@host:
        destination = cmd[-1]
        assert "192.168.86.88" in destination
        assert "timbass" in destination
        # -e flag carries the ssh invocation
        ssh_arg = cmd[cmd.index("-e") + 1]
        assert "henryfood_i5" in ssh_arg


# ---------------------------------------------------------------------------
# retry_pending
# ---------------------------------------------------------------------------


class TestRetryPending:
    def test_noop_when_manifest_empty(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        # Should complete without error and call push zero times.
        with patch.object(t, "push") as mock_push:
            t.retry_pending()
        mock_push.assert_not_called()

    def test_calls_push_for_each_pending_entry(self, tmp_path: Path) -> None:
        t = WavTransfer(_config(tmp_path))
        wavs = [tmp_path / f"{i}.wav" for i in range(3)]
        for w in wavs:
            w.touch()
            t._add_to_manifest(w)

        with patch.object(t, "push") as mock_push:
            t.retry_pending()

        assert mock_push.call_count == 3
        pushed_paths = {call.args[0] for call in mock_push.call_args_list}
        assert pushed_paths == {w.resolve() for w in wavs}
