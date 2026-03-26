"""Tests for app.gpio_button (GpioButtonWrapper and PiButtonListener)."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    """Provide a default AppConfig with a temporary recordings_dir."""
    return AppConfig(recordings_dir=tmp_path)


# ---------------------------------------------------------------------------
# GpioButtonWrapper
# ---------------------------------------------------------------------------


class TestGpioButtonWrapper:
    @patch("app.gpio_button.GpioButton")
    def test_initialises_with_pull_up(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import GpioButtonWrapper

        GpioButtonWrapper(config)
        mock_gpio_cls.assert_called_once_with(
            config.gpio_pin, pull_up=True, bounce_time=0.05
        )

    @patch("app.gpio_button.GpioButton")
    def test_registers_press_callback(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import GpioButtonWrapper

        cb = MagicMock()
        wrapper = GpioButtonWrapper(config, on_press=cb)
        assert mock_gpio_cls.return_value.when_pressed is cb

    @patch("app.gpio_button.GpioButton")
    def test_registers_release_callback(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import GpioButtonWrapper

        cb = MagicMock()
        wrapper = GpioButtonWrapper(config, on_release=cb)
        assert mock_gpio_cls.return_value.when_released is cb

    @patch("app.gpio_button.GpioButton")
    def test_close_releases_gpio(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import GpioButtonWrapper

        wrapper = GpioButtonWrapper(config)
        wrapper.close()
        mock_gpio_cls.return_value.close.assert_called_once()

    @patch("app.gpio_button.GpioButton")
    def test_is_pressed_delegates(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import GpioButtonWrapper

        mock_gpio_cls.return_value.is_pressed = True
        wrapper = GpioButtonWrapper(config)
        assert wrapper.is_pressed is True


# ---------------------------------------------------------------------------
# PiButtonListener
# ---------------------------------------------------------------------------


class TestPiButtonListener:
    @patch("app.gpio_button.GpioButton")
    def test_press_logs_and_calls_user_callback(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import PiButtonListener

        user_cb = MagicMock()
        listener = PiButtonListener(config, on_press=user_cb)

        # Simulate a button press by calling the internal handler
        listener._handle_press()
        user_cb.assert_called_once()

    @patch("app.gpio_button.GpioButton")
    def test_release_logs_and_calls_user_callback(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import PiButtonListener

        user_cb = MagicMock()
        listener = PiButtonListener(config, on_release=user_cb)

        listener._handle_release()
        user_cb.assert_called_once()

    @patch("app.gpio_button.GpioButton")
    def test_press_without_user_callback_does_not_raise(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import PiButtonListener

        listener = PiButtonListener(config)
        # Should not raise even with no user callback
        listener._handle_press()
        listener._handle_release()

    @patch("app.gpio_button.GpioButton")
    def test_stop_unblocks_run_forever(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        """Calling stop() should set the internal event so run_forever returns."""
        from app.gpio_button import PiButtonListener

        listener = PiButtonListener(config)
        listener.stop()
        assert listener._stop_event.is_set()

    @patch("app.gpio_button.GpioButton")
    def test_close_releases_gpio(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import PiButtonListener

        listener = PiButtonListener(config)
        listener.close()
        mock_gpio_cls.return_value.close.assert_called_once()

    @patch("app.gpio_button.GpioButton")
    def test_is_pressed_delegates(
        self, mock_gpio_cls: MagicMock, config: AppConfig
    ) -> None:
        from app.gpio_button import PiButtonListener

        mock_gpio_cls.return_value.is_pressed = False
        listener = PiButtonListener(config)
        assert listener.is_pressed is False
