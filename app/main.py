"""CaptureWorkflow – orchestrates button input and audio recording."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import AppConfig
from app.gpio_button import AbstractButton
from app.recorder import Recorder

logger = logging.getLogger(__name__)


class CaptureWorkflow:
    """Connects a button to a recorder.

    Press the button → recording starts.
    Press the button again → recording stops and the file is saved.

    Works with any concrete implementations of AbstractButton and Recorder,
    making it easy to swap GPIO hardware for a mock in development.
    """

    def __init__(self, button: AbstractButton, recorder: Recorder, config: AppConfig) -> None:
        self._button = button
        self._recorder = recorder
        self._config = config
        self._recording = False
        self._last_path: Optional[Path] = None

        self._button.when_pressed = self._on_press

    # ------------------------------------------------------------------
    # Button handler
    # ------------------------------------------------------------------

    def _on_press(self) -> None:
        if not self._recording:
            logger.info("Button pressed – starting recording")
            self._recording = True
            self._recorder.start()
        else:
            logger.info("Button pressed – stopping recording")
            self._recording = False
            path = self._recorder.stop()
            if path is not None:
                self._last_path = path
                logger.info("Recording saved: %s", path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_path(self) -> Optional[Path]:
        """Path of the most recently saved recording, or None."""
        return self._last_path
