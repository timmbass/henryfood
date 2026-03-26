"""Tests for CaptureWorkflow."""

from __future__ import annotations

from unittest.mock import MagicMock
from pathlib import Path

import pytest

from app.config import AppConfig
from app.gpio_button import AbstractButton
from app.main import CaptureWorkflow
from app.recorder import Recorder


class FakeButton(AbstractButton):
    """In-process fake button for workflow tests."""

    def __init__(self):
        self._when_pressed = None
        self._when_released = None

    @property
    def when_pressed(self):
        return self._when_pressed

    @when_pressed.setter
    def when_pressed(self, cb):
        self._when_pressed = cb

    @property
    def when_released(self):
        return self._when_released

    @when_released.setter
    def when_released(self, cb):
        self._when_released = cb

    def press(self):
        if self._when_pressed:
            self._when_pressed()


class FakeRecorder(Recorder):
    """In-process fake recorder."""

    def __init__(self, save_path: Path):
        self.started = 0
        self.stopped = 0
        self._save_path = save_path

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1
        return self._save_path


@pytest.fixture()
def config(tmp_path):
    return AppConfig(output_dir=tmp_path)


class TestCaptureWorkflow:
    def test_first_press_starts_recording(self, config, tmp_path):
        btn = FakeButton()
        rec = FakeRecorder(tmp_path / "out.wav")
        workflow = CaptureWorkflow(button=btn, recorder=rec, config=config)

        btn.press()

        assert rec.started == 1
        assert workflow._recording is True

    def test_second_press_stops_recording(self, config, tmp_path):
        btn = FakeButton()
        save_path = tmp_path / "out.wav"
        rec = FakeRecorder(save_path)
        workflow = CaptureWorkflow(button=btn, recorder=rec, config=config)

        btn.press()  # start
        btn.press()  # stop

        assert rec.stopped == 1
        assert workflow._recording is False
        assert workflow.last_path == save_path

    def test_multiple_cycles(self, config, tmp_path):
        btn = FakeButton()
        rec = FakeRecorder(tmp_path / "out.wav")
        workflow = CaptureWorkflow(button=btn, recorder=rec, config=config)

        for _ in range(3):
            btn.press()  # start
            btn.press()  # stop

        assert rec.started == 3
        assert rec.stopped == 3

    def test_last_path_none_before_any_recording(self, config, tmp_path):
        btn = FakeButton()
        rec = FakeRecorder(tmp_path / "out.wav")
        workflow = CaptureWorkflow(button=btn, recorder=rec, config=config)

        assert workflow.last_path is None
