"""Abstract button interface and GPIO-based implementation.

gpiozero is imported lazily inside GpioButtonWrapper so the rest of the
codebase can be used on non-Pi machines (e.g. in mock / dev mode) without
importing the GPIO library at all.
"""

from __future__ import annotations

import abc
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AbstractButton(abc.ABC):
    """Minimal interface for a push-button input source."""

    @property
    @abc.abstractmethod
    def when_pressed(self) -> Optional[Callable[[], None]]:
        """Callback invoked when the button is pressed."""

    @when_pressed.setter
    @abc.abstractmethod
    def when_pressed(self, callback: Optional[Callable[[], None]]) -> None: ...

    @property
    @abc.abstractmethod
    def when_released(self) -> Optional[Callable[[], None]]:
        """Callback invoked when the button is released."""

    @when_released.setter
    @abc.abstractmethod
    def when_released(self, callback: Optional[Callable[[], None]]) -> None: ...


class GpioButtonWrapper(AbstractButton):
    """Thin wrapper around gpiozero.Button that satisfies AbstractButton.

    gpiozero is imported lazily so importing this module is safe on machines
    without GPIO hardware – as long as GpioButtonWrapper is not *instantiated*.
    """

    def __init__(self, pin: int, pull_up: bool = True, bounce_time: Optional[float] = 0.05) -> None:
        import gpiozero  # noqa: PLC0415 – lazy import intentional

        self._btn = gpiozero.Button(pin, pull_up=pull_up, bounce_time=bounce_time)
        logger.debug("GPIO button initialised on pin %d", pin)

    @property
    def when_pressed(self) -> Optional[Callable[[], None]]:
        return self._btn.when_pressed

    @when_pressed.setter
    def when_pressed(self, callback: Optional[Callable[[], None]]) -> None:
        self._btn.when_pressed = callback

    @property
    def when_released(self) -> Optional[Callable[[], None]]:
        return self._btn.when_released

    @when_released.setter
    def when_released(self, callback: Optional[Callable[[], None]]) -> None:
        self._btn.when_released = callback
