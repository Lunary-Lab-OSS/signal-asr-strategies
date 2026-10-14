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


def test_every_sherpa_extra_declares_huggingface_hub():
    """Extras promising the Sherpa backend must ship the
    downloader the cold-start ``load_model`` path imports.

    The default Sherpa model is fetched via ``huggingface_hub`` when the
    model directory is empty, so an extra that installs ``sherpa-onnx``
    without also declaring ``huggingface-hub`` does not guarantee a working
    first load.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    extras = data["project"]["optional-dependencies"]
    for extra_name, deps in extras.items():
        installs_sherpa = any(d.startswith("sherpa-onnx") for d in deps)
        if not installs_sherpa:
            continue
        assert any("huggingface-hub" in d for d in deps), (
            f"extra {extra_name!r} installs sherpa-onnx but omits huggingface-hub"
        )
