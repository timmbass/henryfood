"""Tests for GpioButtonWrapper."""

from __future__ import annotations

import pytest

from app.gpio_button import GpioButtonWrapper


class TestGpioButtonWrapper:
    def test_when_pressed_delegates_to_gpiozero(self, _mock_gpiozero):
        btn = GpioButtonWrapper(pin=17)
        callback = lambda: None  # noqa: E731
        btn.when_pressed = callback
        assert btn.when_pressed == callback

    def test_when_released_delegates_to_gpiozero(self, _mock_gpiozero):
        btn = GpioButtonWrapper(pin=17)
        callback = lambda: None  # noqa: E731
        btn.when_released = callback
        assert btn.when_released == callback

    def test_gpio_button_constructed_with_correct_pin(self, _mock_gpiozero):
        import gpiozero

        GpioButtonWrapper(pin=22)
        gpiozero.Button.assert_called_once_with(22, pull_up=True, bounce_time=0.05)
