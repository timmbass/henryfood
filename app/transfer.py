"""WAV file transfer from the Pi to the i5 analytics machine.

Uses rsync over SSH so only changed/new files are sent, and the connection
reuses the pre-authorised ed25519 key — no passwords in the hot path.

Transfers are non-blocking: :meth:`WavTransfer.push` spawns a daemon
thread immediately so the GPIO button is ready for the next recording.

Failure handling
~~~~~~~~~~~~~~~~
If rsync exits non-zero (or times out / raises :exc:`OSError`), the path
is appended to a small manifest file
(``<recordings_dir>/pending_transfer.txt``).  On the next startup,
:meth:`WavTransfer.retry_pending` reads the manifest and retries each
entry so no recording is silently lost when the i5 is temporarily
unreachable.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
from pathlib import Path

from app.config import AppConfig

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "pending_transfer.txt"


class WavTransfer:
    """Pushes completed WAV files to the i5 via rsync over SSH."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # Manifest lives next to the WAV files for easy inspection.
        self._manifest: Path = config.recordings_dir.resolve() / _MANIFEST_NAME
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def push(self, wav_path: Path) -> None:
        """Enqueue *wav_path* for background transfer.

        Returns immediately — the actual rsync runs in a daemon thread.
        """
        t = threading.Thread(target=self._do_push, args=(wav_path,), daemon=True)
        t.start()

    def retry_pending(self) -> None:
        """Re-attempt every path recorded in the pending manifest.

        Called once at startup so recordings from previous sessions (when
        the i5 was unreachable) are eventually delivered.
        """
        pending = self._read_manifest()
        if not pending:
            return
        logger.info("Retrying %d pending transfer(s) from previous session", len(pending))
        for path in pending:
            self.push(path)

    # -- internals -----------------------------------------------------------

    def _do_push(self, wav_path: Path) -> bool:
        """Run rsync synchronously.  Returns ``True`` on success.

        On failure, the path is added to the pending manifest so it will
        be retried on the next startup.  On success it is removed.
        """
        if not self._config.transfer_enabled:
            return True

        if not wav_path.exists():
            logger.warning("Transfer skipped — file not found: %s", wav_path)
            self._remove_from_manifest(wav_path)
            return False

        destination = (
            f"{self._config.i5_user}@{self._config.i5_host}:"
            f"{self._config.i5_incoming_dir}/"
        )
        key = shlex.quote(str(self._config.ssh_key_path.expanduser()))
        cmd = [
            "rsync",
            "-az",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=accept-new",
            str(wav_path),
            destination,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        except subprocess.TimeoutExpired as exc:
            logger.error("Transfer timed out for %s: %s", wav_path.name, exc)
            self._add_to_manifest(wav_path)
            return False
        except OSError as exc:
            logger.error("Transfer failed for %s: %s", wav_path.name, exc)
            self._add_to_manifest(wav_path)
            return False

        if result.returncode == 0:
            logger.info("Transferred %s → %s", wav_path.name, self._config.i5_host)
            self._remove_from_manifest(wav_path)
            return True

        logger.error(
            "rsync exited %d for %s: %s",
            result.returncode,
            wav_path.name,
            result.stderr.strip(),
        )
        self._add_to_manifest(wav_path)
        return False

    # -- manifest helpers ----------------------------------------------------

    def _add_to_manifest(self, path: Path) -> None:
        with self._lock:
            paths = self._read_manifest_locked()
            paths.add(str(path.resolve()))
            self._write_manifest_locked(paths)

    def _remove_from_manifest(self, path: Path) -> None:
        with self._lock:
            paths = self._read_manifest_locked()
            paths.discard(str(path.resolve()))
            self._write_manifest_locked(paths)

    def _read_manifest(self) -> set[Path]:
        with self._lock:
            return {Path(s) for s in self._read_manifest_locked()}

    def _read_manifest_locked(self) -> set[str]:
        """Return manifest entries as strings.  Caller must hold ``self._lock``."""
        if not self._manifest.exists():
            return set()
        return {
            line.strip()
            for line in self._manifest.read_text().splitlines()
            if line.strip()
        }

    def _write_manifest_locked(self, paths: set[str]) -> None:
        """Persist *paths* to the manifest.  Caller must hold ``self._lock``."""
        if paths:
            self._manifest.parent.mkdir(parents=True, exist_ok=True)
            self._manifest.write_text("\n".join(sorted(paths)) + "\n")
        elif self._manifest.exists():
            self._manifest.unlink()
