"""Shared utilities — logging setup and filesystem helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.config import AppConfig


def setup_logging(config: AppConfig) -> None:
    """Configure the root logger based on *config.log_level*.

    All output goes to stderr so stdout stays clean for piping.
    """
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,  # allow re-configuration if called more than once
    )
    logging.getLogger(__name__).debug("Logging initialised at %s", config.log_level)


def ensure_recordings_dir(config: AppConfig) -> Path:
    """Create *config.recordings_dir* if it does not exist.

    Returns the resolved absolute path.
    """
    resolved = config.recordings_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info("Recordings directory ready: %s", resolved)
    return resolved
