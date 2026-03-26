"""Main capture workflow: button press → record → button release → save WAV."""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Optional

from henryfood_voice.button import Button
from henryfood_voice.config import VoiceConfig
from henryfood_voice.recorder import Recorder, SounddeviceRecorder

logger = logging.getLogger(__name__)


class CaptureWorkflow:
    """Orchestrates button events and audio recording.

    Typical usage::

        workflow = CaptureWorkflow(config)
        workflow.run()          # blocks until Ctrl-C
    """

    def __init__(
        self,
        config: VoiceConfig,
        recorder: Optional[Recorder] = None,
    ) -> None:
        self._config = config
        self._recorder: Recorder = recorder or SounddeviceRecorder(config)
        self._button = Button(
            config,
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._stop_event = threading.Event()

    # -- public API ----------------------------------------------------------

    def run(self) -> None:
        """Block until the process is interrupted (SIGINT / SIGTERM)."""
        self._config.ensure_output_dir()
        logger.info("Voice capture ready — press and hold the button to record.")

        # Graceful shutdown on common signals.
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._stop_event.wait()
        self._shutdown()

    def stop(self) -> None:
        """Programmatically request the workflow to stop."""
        self._stop_event.set()

    # -- callbacks -----------------------------------------------------------

    def _on_press(self) -> None:
        logger.info("Button pressed — starting recording")
        self._recorder.start()

    def _on_release(self) -> None:
        logger.info("Button released — stopping recording")
        path: Optional[Path] = self._recorder.stop()
        if path is not None:
            logger.info("Recording saved: %s", path)
        else:
            logger.info("Recording discarded (too short)")

    # -- internal helpers ----------------------------------------------------

    def _handle_signal(self, signum: int, _frame: object) -> None:
        logger.info("Received signal %d — shutting down", signum)
        self._stop_event.set()

    def _shutdown(self) -> None:
        if self._recorder.is_recording:
            self._recorder.stop()
        self._button.close()
        logger.info("Workflow shut down cleanly")
