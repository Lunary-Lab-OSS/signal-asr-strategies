"""Whisper ASR strategy — cross-platform fallback via faster-whisper."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..config import ASRConfig
from .base import ASRStrategy
from .shared import cleanup_temp_file, save_audio_to_wav

logger = logging.getLogger(__name__)

# Compute types supported by CTranslate2 per device family. ``float16`` is
# GPU-only; CPU execution must use an integer or 32-bit float type.
_CPU_COMPUTE_TYPES = ("int8", "int8_float32", "float32")
_GPU_COMPUTE_TYPES = ("float16", "int8_float16", "int8")


def resolve_whisper_device(
    device: str | None,
) -> tuple[str, int | None, str]:
    """Normalise a requested device into faster-whisper constructor arguments.

    Returns ``(device, device_index, compute_type)``:

    - ``"auto"`` (or ``None``) selects CUDA when available, else CPU.
    - ``"cuda"``/``"cuda:0"``/``"cuda:1"`` map to CTranslate2's CUDA device
      with the requested index preserved.
    - ``"cpu"`` selects CPU with the ``int8`` compute type, which CTranslate2
      supports on every CPU; ``float16`` is not CPU-executable.
    - Anything else raises ``ValueError`` instead of being forwarded blindly
      to the backend.

    The compute type can be overridden explicitly via the
    ``SIGNAL_ASR_WHISPER_COMPUTE_TYPE`` environment variable, which must name
    a type supported by the resolved device.
    """
    import os

    requested = (device or "auto").strip().lower()

    def _cuda_available() -> bool:
        try:
            import ctranslate2

            return bool(ctranslate2.get_cuda_device_count() > 0)
        except Exception:  # pragma: no cover - depends on host runtime
            return False

    if requested in ("", "auto"):
        resolved_device = "cuda" if _cuda_available() else "cpu"
        index: int | None = None
    elif requested == "cuda" or requested.startswith("cuda:"):
        resolved_device = "cuda"
        index = None
        if ":" in requested:
            suffix = requested.split(":", 1)[1]
            if not re.fullmatch(r"[0-9]+", suffix):
                raise ValueError(f"invalid CUDA device index in {device!r}")
            index = int(suffix)
    elif requested == "cpu":
        resolved_device = "cpu"
        index = None
    else:
        raise ValueError(
            f"unsupported Whisper device {device!r}; expected 'cpu', 'cuda', "
            "'cuda:<index>', or 'auto'"
        )

    supported = _GPU_COMPUTE_TYPES if resolved_device == "cuda" else _CPU_COMPUTE_TYPES
    default_compute = "float16" if resolved_device == "cuda" else "int8"
    compute_type = os.getenv("SIGNAL_ASR_WHISPER_COMPUTE_TYPE", default_compute)
    if compute_type not in supported:
        raise ValueError(
            f"compute type {compute_type!r} is not supported on device "
            f"{resolved_device!r}; supported: {sorted(supported)}"
        )
    return resolved_device, index, compute_type


class WhisperASRStrategy(ASRStrategy):
    """Whisper ASR via faster-whisper (cross-platform CPU/GPU fallback).

    Works on Windows, Linux, macOS, and WSL2.
    """

    def __init__(self, config: ASRConfig, model_loader, device: str, platform: str) -> None:
        super().__init__(config, device, platform)
        self._model_loader = model_loader

    def load_model(self) -> None:
        if self.model is not None:
            return
        descriptor = self.config.model_priority[0] if self.config.model_priority else None
        local_path = getattr(descriptor, "local_path", None)
        revision = getattr(descriptor, "revision", None)
        model_size = (
            local_path
            or getattr(descriptor, "repo_id", None)
            or getattr(descriptor, "name", "base")
        )
        if not isinstance(model_size, str) or not model_size.strip():
            raise ValueError("Whisper model source must be a nonempty string")
        if local_path and not Path(local_path).is_dir():
            raise ValueError(f"Whisper local_path is not a directory: {local_path!r}")
        if local_path and revision:
            raise ValueError("Whisper revision cannot be verified for a local_path")
        device, device_index, compute_type = resolve_whisper_device(self.device)
        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s, index=%s)...",
            model_size,
            device,
            compute_type,
            device_index,
        )
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper is required: pip install faster-whisper") from None

        kwargs: dict[str, object] = {"device": device, "compute_type": compute_type}
        if device_index is not None:
            kwargs["device_index"] = device_index
        if revision:
            from huggingface_hub import snapshot_download

            if "/" not in model_size:
                raise ValueError("Whisper revision requires an explicit repo_id")
            model_size = snapshot_download(
                repo_id=model_size,
                revision=revision,
                allow_patterns=["*.json", "model.bin", "vocabulary.*"],
            )
        self.model = WhisperModel(model_size, **kwargs)
        logger.info("✅ Whisper ASR ready: %s on %s", model_size, device)

    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        if not audio_chunk:
            return ""
        if self.model is None:
            self.load_model()
        model = self.model
        assert model is not None

        # Drop a trailing partial 16-bit sample so the WAV write stays aligned.
        if len(audio_chunk) % 2:
            audio_chunk = audio_chunk[:-1]
            if not audio_chunk:
                return ""
        path = save_audio_to_wav(audio_chunk)
        try:
            segments, _ = model.transcribe(
                path,
                language=language or self.config.language,
                task=self.config.task,
                beam_size=1,
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as exc:
            raise RuntimeError("Whisper transcription failed") from exc
        finally:
            cleanup_temp_file(path)
