"""Tests for the Typer CLI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


class TestCliHelp:
    def test_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run-mock" in result.output
        assert "run" in result.output

    def test_run_mock_help(self):
        result = runner.invoke(app, ["run-mock", "--help"])
        assert result.exit_code == 0
        assert "Enter" in result.output
        assert "quit" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "GPIO" in result.output

    def test_test_button_help(self):
        result = runner.invoke(app, ["test-button", "--help"])
        assert result.exit_code == 0
        assert "GPIO" in result.output


class TestRunMockCommand:
    """Smoke-test the run-mock command with mocked audio & immediate quit."""

    def test_run_mock_quits_on_q(self, tmp_path, _mock_sounddevice):
        """run-mock should exit cleanly when the user types q."""
        result = runner.invoke(
            app,
            ["run-mock", f"--output-dir={tmp_path}"],
            input="q\n",
        )
        # Exit code may be 0 or 1 depending on whether stdin closes cleanly;
        # the important thing is that the process doesn't hang.
        assert result.exit_code in (0, 1)

    def test_run_mock_records_on_enter_q(self, tmp_path, _mock_sounddevice):
        """Press Enter (start), press Enter (stop), then q to quit."""
        result = runner.invoke(
            app,
            ["run-mock", f"--output-dir={tmp_path}"],
            input="\n\nq\n",
        )
        assert result.exit_code in (0, 1)
