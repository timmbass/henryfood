"""Abstract base class and concrete implementation for audio recording."""

from __future__ import annotations

import abc
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import AppConfig

logger = logging.getLogger(__name__)


class Recorder(abc.ABC):
    """Abstract base for audio recorders."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin recording audio."""

    @abc.abstractmethod
    def stop(self) -> Optional[Path]:
        """Stop recording and return the path to the saved file, or None if nothing was recorded."""


class SounddeviceRecorder(Recorder):
    """Concrete recorder that uses sounddevice and soundfile.

    Imports sounddevice and soundfile lazily so the module can be imported on
    machines without PortAudio installed (e.g. in tests).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._frames: list = []
        self._recording = False
        self._stream = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin recording.  No-op if already recording."""
        with self._lock:
            if self._recording:
                logger.warning("start() called while already recording – ignored")
                return
            self._frames = []
            self._recording = True

        import sounddevice as sd  # noqa: PLC0415 – lazy import intentional

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

        # Auto-stop after max_duration
        self._timer = threading.Timer(self._config.max_duration, self._auto_stop)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> Optional[Path]:
        """Stop recording and save the WAV file.  Returns the saved path."""
        with self._lock:
            if not self._recording:
                logger.warning("stop() called while not recording – ignored")
                return None
            self._recording = False

        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            logger.warning("No audio frames captured")
            return None

        return self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time, status) -> None:  # noqa: ANN001
        if status:
            logger.warning("sounddevice status: %s", status)
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

    def _auto_stop(self) -> None:
        logger.info("max_duration reached – auto-stopping")
        self.stop()

    def _save(self) -> Path:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self._config.output_dir / f"{timestamp}.wav"
        audio = np.concatenate(self._frames, axis=0)
        sf.write(str(path), audio, self._config.sample_rate)
        logger.info("Saved recording to %s", path)
        return path
