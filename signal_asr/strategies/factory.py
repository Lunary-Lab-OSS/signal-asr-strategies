"""ASR strategy factory — picks the right backend for the current platform."""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

from ..config import ASRConfig
from .base import ASRStrategy

logger = logging.getLogger(__name__)

# Canonical engine names and their accepted aliases.
_ENGINE_ALIASES: dict[str, str] = {
    "sherpa_onnx": "sherpa_onnx",
    "sherpa-onnx": "sherpa_onnx",
    "whisperkit": "whisperkit",
    "whisper_kit": "whisperkit",
    "whisper-kit": "whisperkit",
    "whisper": "whisper",
    "parakeet": "parakeet",
}


def canonical_engine(name: str | None) -> str | None:
    """Return the canonical engine name for ``name``, or None if unknown.

    Equivalent spellings (``SHERPA_ONNX``, ``sherpa-onnx``, ``sherpa_onnx``)
    normalise to the same canonical value so callers and caches key on one
    identity per backend (A12).
    """
    normalized = (name or "").strip().lower()
    if not normalized:
        return None
    return _ENGINE_ALIASES.get(normalized)


def default_engine_for_platform(platform_slug: str) -> str:
    """Platform default engine: WhisperKit on macOS, sherpa-onnx elsewhere."""
    return "whisperkit" if platform_slug == "macos" else "sherpa_onnx"


def _detect_platform() -> str:
    """Return a normalised platform slug."""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    # Distinguish WSL2 from plain Linux
    release = platform.release().lower()
    if "microsoft" in release:
        return "wsl2"
    return "linux"


class ASRStrategyFactory:
    """Create the appropriate ASR strategy for the given config and environment.

    Usage::

        factory = ASRStrategyFactory(config, device="cuda:0")
        asr = factory.create_strategy()
        text = asr.transcribe(raw_pcm_bytes)
    """

    def __init__(
        self,
        config: ASRConfig,
        device: str = "cpu",
        platform: str | None = None,
        model_loader=None,
        models_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.platform = platform or _detect_platform()
        self.model_loader = model_loader
        self.models_dir = models_dir or (
            getattr(model_loader, "models_dir", None) if model_loader else None
        )

    def create(self) -> ASRStrategy:
        """Instantiate and return the selected ASR strategy."""
        return self.create_strategy()

    def create_strategy(self) -> ASRStrategy:
        """Instantiate and return the selected ASR strategy.

        Kept as the canonical method name for compatibility with Switchboard's
        original in-repo ASR factory.
        """
        raw_engine = (self.config.engine or "").strip()
        if raw_engine and canonical_engine(raw_engine) is None:
            # Unknown engines are rejected loudly (A12); a typo must not
            # silently select the platform default backend.
            raise ValueError(
                f"unknown ASR engine {self.config.engine!r}; expected one of "
                f"{sorted(set(_ENGINE_ALIASES.values()))} or '' for auto-select"
            )
        engine = canonical_engine(raw_engine)

        if engine is None:
            engine = default_engine_for_platform(self.platform)
            logger.info("ASR engine auto-selected: '%s' for platform '%s'", engine, self.platform)

        if engine == "whisperkit":
            from .whisperkit import WhisperKitASRStrategy

            return WhisperKitASRStrategy(self.config, self.device, self.platform)

        if engine == "sherpa_onnx":
            from .sherpa_onnx import SherpaOnnxASRStrategy

            return SherpaOnnxASRStrategy(
                self.config, self.device, self.platform, models_dir=self.models_dir
            )

        if engine == "parakeet":
            from .parakeet import ParakeetASRStrategy

            return ParakeetASRStrategy(self.config, self.device, self.platform)

        if engine == "whisper":
            from .whisper import WhisperASRStrategy

            return WhisperASRStrategy(self.config, self.model_loader, self.device, self.platform)

        raise AssertionError(  # pragma: no cover - canonical_engine is total
            f"engine {engine!r} has no strategy mapping"
        )
