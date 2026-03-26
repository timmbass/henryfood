"""Data models for recordings.

These models describe *metadata* about captured audio files.
Transcription fields are stubbed out for a future phase.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RecordingMetadata(BaseModel):
    """Metadata for a single WAV recording.

    Attributes:
        file_path: Absolute path to the saved WAV file.
        recorded_at: UTC timestamp when the recording started.
        duration_seconds: Length of the recording in seconds.
        sample_rate: Sample rate the file was captured at.
        channels: Number of audio channels.
        transcription: Placeholder — will hold the speech-to-text
            output once the transcription phase is implemented.
    """

    file_path: Path
    recorded_at: datetime
    duration_seconds: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)
    transcription: Optional[str] = Field(
        default=None,
        description="Speech-to-text output (future phase)",
    )
