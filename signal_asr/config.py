"""ASR configuration dataclass — dependency-free, importable anywhere."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Minimal model descriptor used by WhisperASRStrategy."""

    name: str
    repo_id: str
    local_path: str | None = None
    quantized: bool = False
    device: str = "auto"
    dtype: str = "float16"
    revision: str | None = None


@dataclass
class ASRConfig:
    """Configuration for an ASR strategy.

    Shared by all strategy implementations so callers only import one thing.
    """

    # Engine selection: "sherpa_onnx" | "whisperkit" | "whisper" | "parakeet"
    # Empty string means auto-select based on platform.
    engine: str = ""

    # Hugging Face model id when engine == "parakeet"
    parakeet_model_name: str | None = None

    # WhisperKit Core ML model variant when engine == "whisperkit"
    whisperkit_model_name: str = "large-v3-v20240930_turbo"

    # Whisper model priority list (used by WhisperASRStrategy)
    model_priority: list = field(default_factory=list)

    language: str | None = None  # None = auto-detect
    task: str = "transcribe"  # "transcribe" | "translate"
    debug: bool = False
    chunk_length_s: float = 30.0
    min_speech_duration_ms: int = 200
    energy_silence_debounce_ms: int = 150
