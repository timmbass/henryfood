"""Shared pytest fixtures for the app test suite."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_gpiozero(monkeypatch):
    """Inject a MagicMock for gpiozero so tests run on non-Pi machines."""
    mock_gpiozero = MagicMock()
    monkeypatch.setitem(sys.modules, "gpiozero", mock_gpiozero)
    yield mock_gpiozero


@pytest.fixture(autouse=True)
def _mock_sounddevice(monkeypatch):
    """Inject MagicMocks for sounddevice / soundfile / numpy so PortAudio isn't needed."""
    mock_sd = MagicMock()
    mock_sf = MagicMock()
    mock_np = MagicMock()

    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    monkeypatch.setitem(sys.modules, "soundfile", mock_sf)
    monkeypatch.setitem(sys.modules, "numpy", mock_np)

    # numpy.concatenate must return something array-like for soundfile.write
    mock_np.concatenate.return_value = b"audio-data"

    yield {"sd": mock_sd, "sf": mock_sf, "np": mock_np}
