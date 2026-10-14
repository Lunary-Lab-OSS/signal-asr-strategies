"""Abstract base class for all ASR strategies."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from ..config import ASRConfig

logger = logging.getLogger(__name__)


class ASRStrategy(ABC):
    """All ASR backends inherit from this class and implement the two abstract methods."""

    def __init__(self, config: ASRConfig, device: str, platform: str) -> None:
        """
        Args:
            config:   ASR configuration.
            device:   Compute device, e.g. "cpu", "cuda", "cuda:0"; each
                      backend validates the values it supports.
            platform: Operating system slug: "windows", "macos", "linux", "wsl2".
        """
        self.config = config
        self.device = device
        self.platform = platform
        self.model: Any | None = None
        self._load_lock = threading.Lock()

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize the backend.

        Implementations must set ``self.model`` to a non-None value on success.
        """

    @abstractmethod
    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        """Transcribe raw PCM audio.

        Args:
            audio_chunk: Raw audio bytes — 16kHz, mono, 16-bit signed PCM.
            language: Optional per-call language override (ISO code);
                backends that cannot honour it fall back to their config.

        Returns:
            Transcribed text, or empty string for valid silence/no input.

        Raises:
            Exception: Backend loading or decoding failed.
        """

    def _ensure_loaded(self) -> None:
        """Load the model once, even when several threads transcribe at once.

        Lazy loading from :meth:`transcribe` is serialised so concurrent first
        calls do not load the weights (or spawn a backend process) twice.
        """
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is None:
                self.load_model()

    def shutdown(self) -> None:
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def get_name(self) -> str:
        return self.__class__.__name__
