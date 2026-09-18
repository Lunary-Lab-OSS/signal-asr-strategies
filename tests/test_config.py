"""Tests for ASRConfig and ModelConfig defaults and behavior."""

from signal_asr.config import ASRConfig, ModelConfig


def test_asr_config_defaults():
    config = ASRConfig()
    assert config.engine == ""
    assert config.parakeet_model_name is None
    assert config.whisperkit_model_name == "large-v3-v20240930_turbo"
    assert config.model_priority == []
    assert config.language is None
    assert config.task == "transcribe"
    assert config.debug is False
    assert config.chunk_length_s == 30.0
    assert config.min_speech_duration_ms == 200
    assert config.energy_silence_debounce_ms == 150


def test_asr_config_engine_values():
    for engine in ("sherpa_onnx", "whisperkit", "whisper", "parakeet"):
        assert ASRConfig(engine=engine).engine == engine


def test_model_config_defaults():
    model = ModelConfig(name="m", repo_id="org/m")
    assert model.local_path is None
    assert model.quantized is False
    assert model.device == "auto"
    assert model.dtype == "float16"
    assert model.revision is None
