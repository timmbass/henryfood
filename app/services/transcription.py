"""
Transcription service.

Provides an abstract STTProvider interface. The default implementation falls
back to typed text input if no microphone/STT is available.
Optional faster-whisper integration for real speech-to-text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from app.utils.logging import get_logger

logger = get_logger(__name__)


class STTProvider(ABC):
    """Abstract speech-to-text provider."""

    @abstractmethod
    def transcribe(self, prompt: str = "Tell me what you ate: ") -> str:
        """Transcribe audio or accept typed input and return raw text."""
        ...


class TypedInputProvider(STTProvider):
    """Fallback provider that reads typed input from the user."""

    def transcribe(self, prompt: str = "Tell me what you ate: ") -> str:
        """Prompt the user to type their food diary entry."""
        try:
            text = input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
        return text.strip()


class WhisperSTTProvider(STTProvider):
    """
    Placeholder for faster-whisper speech-to-text integration.
    
    To enable, install: pip install faster-whisper
    Then pass model_size='base' or similar to the constructor.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(self.model_size, device=self.device)
            logger.info("Whisper model loaded: %s on %s", self.model_size, self.device)
        except ImportError:
            logger.warning(
                "faster-whisper not installed. Falling back to typed input. "
                "Install with: pip install faster-whisper"
            )
            self._model = None

    def transcribe(self, prompt: str = "Tell me what you ate: ") -> str:
        """Attempt microphone transcription; fall back to typed input."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            return TypedInputProvider().transcribe(prompt)
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore

            print(f"\n🎤 {prompt}")
            print("Press ENTER to start recording, ENTER again to stop...")
            input()
            sample_rate = 16000
            print("Recording... (press ENTER to stop)")
            chunks = []
            import threading

            stop_event = threading.Event()

            def record():
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                    while not stop_event.is_set():
                        data, _ = stream.read(1024)
                        chunks.append(data)

            t = threading.Thread(target=record)
            t.start()
            input()
            stop_event.set()
            t.join()

            audio = np.concatenate(chunks, axis=0).flatten()
            segments, _ = self._model.transcribe(audio, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            print(f"Transcribed: {text}")
            return text
        except Exception as exc:  # noqa: BLE001
            logger.error("STT failed: %s — falling back to typed input", exc)
            return TypedInputProvider().transcribe(prompt)


def get_stt_provider(use_stt: bool = False, **kwargs) -> STTProvider:
    """Return the appropriate STT provider based on configuration."""
    if use_stt:
        return WhisperSTTProvider(**kwargs)
    return TypedInputProvider()
