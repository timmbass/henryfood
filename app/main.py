"""Main capture workflow — ties the button, recorder, and signal handling together.

Typical usage::

    from app.config import AppConfig
    from app.main import CaptureWorkflow

    config = AppConfig()
    workflow = CaptureWorkflow(config)
    workflow.run()          # blocks until Ctrl-C / SIGTERM
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Optional

from app.config import AppConfig
from app.gpio_button import GpioButtonWrapper
from app.recorder import Recorder, SounddeviceRecorder
from app.utils import ensure_recordings_dir

logger = logging.getLogger(__name__)


class CaptureWorkflow:
    """Orchestrates GPIO button events and audio recording.

    Press the button  → recording starts.
    Release the button → recording stops and the WAV file is saved.
    Ctrl-C / SIGTERM   → graceful shutdown.
    """

    def __init__(
        self,
        config: AppConfig,
        recorder: Optional[Recorder] = None,
    ) -> None:
        self._config = config
        self._recorder: Recorder = recorder or SounddeviceRecorder(config)
        self._button = GpioButtonWrapper(
            config,
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._stop_event = threading.Event()

    # -- public API ----------------------------------------------------------

    def run(self) -> None:
        """Block until SIGINT or SIGTERM is received."""
        ensure_recordings_dir(self._config)
        logger.info(
            "Voice capture ready on GPIO %d — press and hold the button to record.",
            self._config.gpio_pin,
        )

        # Register graceful-shutdown handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._stop_event.wait()
        self._shutdown()

    def stop(self) -> None:
        """Programmatically request the workflow to stop."""
        self._stop_event.set()

    # -- button callbacks ----------------------------------------------------

    def _on_press(self) -> None:
        logger.info("Button pressed — starting recording")
        self._recorder.start()

    def _on_release(self) -> None:
        logger.info("Button released — stopping recording")
        meta = self._recorder.stop()
        if meta is not None:
            logger.info("Recording saved: %s (%.1f s)", meta.file_path.name, meta.duration_seconds)
        else:
            logger.info("Recording discarded (empty or too short)")

    # -- internal helpers ----------------------------------------------------

    def _handle_signal(self, signum: int, _frame: object) -> None:
        logger.info("Received signal %d — shutting down", signum)
        self._stop_event.set()

    def _shutdown(self) -> None:
        if self._recorder.is_recording:
            self._recorder.stop()
        self._button.close()
        logger.info("Workflow shut down cleanly")
