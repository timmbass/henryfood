"""Shared test fixtures — mock hardware libraries that are unavailable in CI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_gpiozero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a mock ``gpiozero`` module so ``app.gpio_button`` can be imported.

    ``gpiozero`` requires a real Raspberry Pi (or pigpio daemon) at import
    time.  This fixture makes ``from gpiozero import Button`` succeed by
    placing a :class:`~unittest.mock.MagicMock` in :data:`sys.modules`.
    """
    if "gpiozero" not in sys.modules:
        monkeypatch.setitem(sys.modules, "gpiozero", MagicMock())
