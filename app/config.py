from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel, frozen=True):
    """Application configuration for the voice recorder."""

    output_dir: Path = Field(default=Path("recordings"), description="Directory to save recordings")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    max_duration: float = Field(
        default=300.0,
        description="Maximum recording duration in seconds before auto-stop",
    )
    gpio_pin: int = Field(default=17, description="GPIO pin number for the physical button")
