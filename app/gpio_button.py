"""GPIO button abstraction.

Wraps *gpiozero.Button* behind a thin façade so the rest of the
application depends only on press/release callbacks, not on the
library directly.  The import is **not** lazy here because this
module is only imported when actually running on a Pi.

Two public classes:

:class:`GpioButtonWrapper`
    Low-level wrapper used by :class:`~app.main.CaptureWorkflow`.
    Accepts arbitrary press/release callbacks.

:class:`PiButtonListener`
    Higher-level listener with built-in logging and a blocking
    :meth:`run_forever` loop.  Used by the ``test-button`` CLI
    command and easy to mock in tests.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Callable, Optional

from gpiozero import Button as GpioButton

from app.config import AppConfig

logger = logging.getLogger(__name__)

# Default GPIO pin (BCM numbering) — matches AppConfig.gpio_pin default
DEFAULT_GPIO_PIN = 17


class GpioButtonWrapper:
    """Manages a physical tactile push-button on a Raspberry Pi GPIO pin.

    Parameters
    ----------
    config:
        Application config (provides *gpio_pin*).
    on_press:
        Callback fired (no args) when the button is pressed down.
    on_release:
        Callback fired (no args) when the button is released.
    """

    def __init__(
        self,
        config: AppConfig,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> None:
        self._pin = config.gpio_pin
        # pull_up=True  → GPIO reads HIGH when open, LOW when pressed to GND
        # bounce_time    → software de-bounce to ignore electrical noise
        self._button = GpioButton(self._pin, pull_up=True, bounce_time=0.05)

        if on_press is not None:
            self._button.when_pressed = on_press
        if on_release is not None:
            self._button.when_released = on_release

        logger.info("Button initialised on GPIO %d (pull-up, 50 ms debounce)", self._pin)

    # -- public helpers ------------------------------------------------------

    @property
    def is_pressed(self) -> bool:
        """Return ``True`` while the button is physically held down."""
        return self._button.is_pressed

    def close(self) -> None:
        """Release the underlying GPIO resource."""
        self._button.close()
        logger.info("Button on GPIO %d released", self._pin)


class PiButtonListener:
    """High-level button listener with logging and a blocking event loop.

    Wraps :class:`GpioButtonWrapper` and adds:

    * Default press/release callbacks that log state changes.
    * An optional pair of user-supplied callbacks executed *after* logging.
    * A :meth:`run_forever` method that blocks until Ctrl-C / SIGTERM.
    * A :meth:`stop` method for programmatic shutdown (useful in tests).

    Example
    -------
    ::

        from app.config import AppConfig
        from app.gpio_button import PiButtonListener

        listener = PiButtonListener(AppConfig())
        listener.run_forever()   # blocks until Ctrl-C

    Parameters
    ----------
    config:
        Application config (provides *gpio_pin*).
    on_press:
        Extra callback invoked when the button is pressed (after logging).
    on_release:
        Extra callback invoked when the button is released (after logging).
    """

    def __init__(
        self,
        config: AppConfig,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._user_on_press = on_press
        self._user_on_release = on_release
        self._stop_event = threading.Event()

        # Create the low-level wrapper with our logging callbacks
        self._wrapper = GpioButtonWrapper(
            config,
            on_press=self._handle_press,
            on_release=self._handle_release,
        )

    # -- public API ----------------------------------------------------------

    def run_forever(self) -> None:
        """Block until SIGINT (Ctrl-C) or SIGTERM is received.

        Registers signal handlers, then waits on an internal event.
        Call :meth:`stop` from another thread (or a signal handler)
        to unblock.
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "PiButtonListener ready on GPIO %d — press the button (Ctrl-C to quit)",
            self._config.gpio_pin,
        )
        self._stop_event.wait()
        self.close()

    def stop(self) -> None:
        """Unblock :meth:`run_forever` programmatically."""
        self._stop_event.set()

    def close(self) -> None:
        """Release GPIO resources."""
        self._wrapper.close()

    @property
    def is_pressed(self) -> bool:
        """``True`` while the button is physically held down."""
        return self._wrapper.is_pressed

    # -- internal callbacks --------------------------------------------------

    def _handle_press(self) -> None:
        """Invoked by gpiozero when the button is pressed down."""
        logger.info("Button PRESSED on GPIO %d", self._config.gpio_pin)
        if self._user_on_press is not None:
            self._user_on_press()

    def _handle_release(self) -> None:
        """Invoked by gpiozero when the button is released."""
        logger.info("Button RELEASED on GPIO %d", self._config.gpio_pin)
        if self._user_on_release is not None:
            self._user_on_release()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        """Graceful shutdown on SIGINT / SIGTERM."""
        logger.info("PiButtonListener received signal %d — stopping", signum)
        self._stop_event.set()
