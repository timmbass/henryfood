"""Audio recording abstraction.

Public interface
~~~~~~~~~~~~~~~~
:class:`Recorder` — an abstract base class that any recording backend
must implement.

Default backend
~~~~~~~~~~~~~~~
:class:`SounddeviceRecorder` — captures audio via *sounddevice* and
persists WAV files with *soundfile*.  Both libraries are imported
lazily inside methods so the abstract class can be imported on machines
without PortAudio (e.g. CI runners).

Swapping the backend only requires writing a new :class:`Recorder`
subclass — nothing else in the application changes.
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

from app.config import AppConfig
from app.models import RecordingMetadata

if TYPE_CHECKING:  # avoid PortAudio requirement at import time
    import sounddevice as sd
    import soundfile as sf

logger = logging.getLogger(__name__)


class Recorder(abc.ABC):
    """Abstract interface that every recording backend must satisfy."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin capturing audio from the microphone."""

    @abc.abstractmethod
    def stop(self) -> Optional[RecordingMetadata]:
        """Stop capturing and save the audio.

        Returns a :class:`RecordingMetadata` describing the saved file,
        or ``None`` if the recording was empty / too short.
        """

    @property
    @abc.abstractmethod
    def is_recording(self) -> bool:
        """``True`` while audio is being captured."""


class SounddeviceRecorder(Recorder):
    """Concrete recorder backed by *sounddevice* + *soundfile*.

    Thread-safe: :meth:`start` and :meth:`stop` guard shared state
    with a lock because GPIO callbacks run on background threads.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._frames: list[np.ndarray] = []
        self._stream: object | None = None
        self._recording = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._started_at: Optional[datetime] = None
        # Timer that enforces max_duration_seconds
        self._duration_timer: Optional[threading.Timer] = None

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd  # noqa: PLC0415 — lazy import

        with self._lock:
            if self._recording:
                logger.warning("start() called while already recording — ignoring")
                return

            self._frames.clear()
            self._recording = True
            self._start_time = time.monotonic()
            self._started_at = datetime.now(tz=timezone.utc)

            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()

            # Auto-stop after max_duration_seconds
            self._duration_timer = threading.Timer(
                self._config.max_duration_seconds,
                self._auto_stop,
            )
            self._duration_timer.daemon = True
            self._duration_timer.start()

            logger.info(
                "Recording started (rate=%d Hz, ch=%d, max=%.0f s)",
                self._config.sample_rate,
                self._config.channels,
                self._config.max_duration_seconds,
            )

    def stop(self) -> Optional[RecordingMetadata]:
        with self._lock:
            if not self._recording:
                logger.warning("stop() called while not recording — ignoring")
                return None

            self._recording = False

            # Cancel the auto-stop timer if it hasn't fired yet
            if self._duration_timer is not None:
                self._duration_timer.cancel()
                self._duration_timer = None

            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

        duration = time.monotonic() - (self._start_time or 0)
        if not self._frames:
            logger.info("Recording discarded (no audio frames captured)")
            return None

        return self._save(duration)

    @property
    def is_recording(self) -> bool:
        return self._recording

    # -- internal helpers ----------------------------------------------------

    def _auto_stop(self) -> None:
        """Called by the duration timer when max_duration_seconds elapses."""
        logger.info("Max duration reached — auto-stopping recording")
        self.stop()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: object,
    ) -> None:
        """Callback invoked by sounddevice for each audio block."""
        if status:
            logger.warning("sounddevice status: %s", status)
        self._frames.append(indata.copy())

    def _save(self, duration: float) -> RecordingMetadata:
        """Concatenate captured frames, write a WAV file, return metadata."""
        import soundfile as sf  # noqa: PLC0415 — lazy import

        # Ensure the output directory exists
        output_dir = self._config.recordings_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = (self._started_at or datetime.now(tz=timezone.utc)).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        filename = f"recording_{timestamp}.wav"
        path = output_dir / filename

        audio = np.concatenate(self._frames, axis=0)
        sf.write(str(path), audio, self._config.sample_rate)

        logger.info("Saved %s (%.1f s, %d samples)", path.name, duration, len(audio))

        return RecordingMetadata(
            file_path=path,
            recorded_at=self._started_at or datetime.now(tz=timezone.utc),
            duration_seconds=round(duration, 2),
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
        )
