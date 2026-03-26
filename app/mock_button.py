"""Mock button input for development / testing without Raspberry Pi hardware.

Usage
-----
Run via the CLI::

    python -m app.cli run-mock

Controls:
  Enter   – toggle recording (start if idle, stop & save if recording)
  q Enter – quit the loop

The mock button satisfies the same AbstractButton interface as GpioButtonWrapper,
so CaptureWorkflow works unchanged in either mode.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Optional

from app.gpio_button import AbstractButton

logger = logging.getLogger(__name__)


class MockButton(AbstractButton):
    """Simulates a physical button via stdin.

    The input loop runs in a background daemon thread.  Pressing Enter fires
    ``when_pressed``; pressing Enter again fires ``when_released``.  Typing
    ``q`` (then Enter) sets *running* to False and calls ``_on_quit`` if set.
    """

    def __init__(self) -> None:
        self._when_pressed: Optional[Callable[[], None]] = None
        self._when_released: Optional[Callable[[], None]] = None
        self._on_quit: Optional[Callable[[], None]] = None
        self._pressed = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # AbstractButton interface
    # ------------------------------------------------------------------

    @property
    def when_pressed(self) -> Optional[Callable[[], None]]:
        return self._when_pressed

    @when_pressed.setter
    def when_pressed(self, callback: Optional[Callable[[], None]]) -> None:
        self._when_pressed = callback

    @property
    def when_released(self) -> Optional[Callable[[], None]]:
        return self._when_released

    @when_released.setter
    def when_released(self, callback: Optional[Callable[[], None]]) -> None:
        self._when_released = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, on_quit: Optional[Callable[[], None]] = None) -> None:
        """Start the background stdin-reading thread."""
        self._on_quit = on_quit
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mock-button", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until the user quits (types q)."""
        if self._thread is not None:
            self._thread.join()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        print(  # noqa: T201
            "\nMock button mode – press Enter to start/stop recording, type 'q' to quit.\n",
            flush=True,
        )
        while self._running:
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                break

            if not self._running:
                break

            text = line.strip().lower()
            if text == "q":
                logger.info("Mock button: quit requested")
                self._running = False
                if self._on_quit is not None:
                    self._on_quit()
                break

            # Empty line (just Enter) – toggle press/release
            if self._pressed:
                logger.debug("Mock button: released")
                self._pressed = False
                if self._when_released is not None:
                    self._when_released()
            else:
                logger.debug("Mock button: pressed")
                self._pressed = True
                if self._when_pressed is not None:
                    self._when_pressed()
