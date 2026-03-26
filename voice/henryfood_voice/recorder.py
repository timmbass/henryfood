"""Audio recording abstraction.

The public interface is :class:`Recorder` (an abstract base class).
The default concrete implementation, :class:`SounddeviceRecorder`, uses
*sounddevice* for capture and *soundfile* for WAV encoding.

Swapping the backend (e.g. to PyAudio) only requires implementing a new
:class:`Recorder` subclass — no other code needs to change.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from henryfood_voice.config import VoiceConfig

if TYPE_CHECKING:
    import sounddevice as sd
    import soundfile as sf

logger = logging.getLogger(__name__)


class Recorder(abc.ABC):
    """Abstract interface for audio recorders."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin capturing audio."""

    @abc.abstractmethod
    def stop(self) -> Optional[Path]:
        """Stop capturing and persist the audio.

        Returns the :class:`~pathlib.Path` to the saved WAV file, or
        ``None`` if the recording was too short / empty.
        """

    @property
    @abc.abstractmethod
    def is_recording(self) -> bool:
        """Whether the recorder is currently capturing."""


class SounddeviceRecorder(Recorder):
    """Concrete recorder backed by *sounddevice* + *soundfile*."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._frames: list[np.ndarray] = []
        self._stream: object | None = None
        self._recording = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        with self._lock:
            if self._recording:
                logger.warning("start() called while already recording — ignoring")
                return
            self._frames.clear()
            self._recording = True
            self._start_time = time.monotonic()
            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Recording started (rate=%d, ch=%d)",
                        self._config.sample_rate, self._config.channels)

    def stop(self) -> Optional[Path]:
        with self._lock:
            if not self._recording:
                logger.warning("stop() called while not recording — ignoring")
                return None
            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

        duration = time.monotonic() - (self._start_time or 0)
        if duration < self._config.min_duration:
            logger.info(
                "Recording discarded (%.2fs < min %.2fs)",
                duration, self._config.min_duration,
            )
            return None

        return self._save(duration)

    @property
    def is_recording(self) -> bool:
        return self._recording

    # -- internal helpers ----------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: object,
    ) -> None:
        if status:
            logger.warning("sounddevice status: %s", status)
        self._frames.append(indata.copy())

    def _save(self, duration: float) -> Path:
        import soundfile as sf  # noqa: PLC0415

        output_dir = self._config.ensure_output_dir()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"recording_{timestamp}.wav"
        path = output_dir / filename

        audio = np.concatenate(self._frames, axis=0)
        sf.write(str(path), audio, self._config.sample_rate)
        logger.info("Saved %s (%.1fs, %d samples)", path.name, duration, len(audio))
        return path
