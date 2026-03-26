"""GPIO button abstraction for Raspberry Pi.

Uses *gpiozero* under the hood but exposes a thin wrapper so callers
only depend on callback registration, not on the underlying library.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from gpiozero import Button as GpioButton

from henryfood_voice.config import VoiceConfig

logger = logging.getLogger(__name__)


class Button:
    """Wraps a physical tactile push-button connected to a GPIO pin.

    Parameters
    ----------
    config:
        Voice configuration (provides ``gpio_pin``).
    on_press:
        Called (no args) when the button is pressed.
    on_release:
        Called (no args) when the button is released.
    """

    def __init__(
        self,
        config: VoiceConfig,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._btn = GpioButton(config.gpio_pin, pull_up=True, bounce_time=0.05)
        if on_press is not None:
            self._btn.when_pressed = on_press
        if on_release is not None:
            self._btn.when_released = on_release
        logger.info("Button initialised on GPIO %d", config.gpio_pin)

    @property
    def is_pressed(self) -> bool:
        """Return ``True`` while the button is physically held down."""
        return self._btn.is_pressed

    def close(self) -> None:
        """Release the underlying GPIO resource."""
        self._btn.close()
        logger.info("Button on GPIO %d closed", self._config.gpio_pin)
