"""GPIO button abstraction.

Wraps *gpiozero.Button* behind a thin façade so the rest of the
application depends only on press/release callbacks, not on the
library directly.  The import is **not** lazy here because this
module is only imported when actually running on a Pi.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gpiozero import Button as GpioButton

from app.config import AppConfig

logger = logging.getLogger(__name__)


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
