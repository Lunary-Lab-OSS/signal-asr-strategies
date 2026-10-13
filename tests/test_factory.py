"""Tests for the ASR strategy factory (A07/A12)."""

from __future__ import annotations

import pytest

from signal_asr.config import ASRConfig
from signal_asr.strategies.factory import (
    ASRStrategyFactory,
    canonical_engine,
    default_engine_for_platform,
)


def test_canonical_engine_resolves_all_alias_spellings():
    assert canonical_engine("sherpa_onnx") == "sherpa_onnx"
    assert canonical_engine("sherpa-onnx") == "sherpa_onnx"
    assert canonical_engine("SHERPA_ONNX") == "sherpa_onnx"
    assert canonical_engine(" WhisperKit ") == "whisperkit"
    assert canonical_engine("whisper_kit") == "whisperkit"
    assert canonical_engine("whisper") == "whisper"
    assert canonical_engine("parakeet") == "parakeet"


def test_canonical_engine_empty_and_none_return_none():
    assert canonical_engine("") is None
    assert canonical_engine(None) is None
    assert canonical_engine("   ") is None


def test_canonical_engine_unknown_names_return_none():
    assert canonical_engine("bogus") is None
    assert canonical_engine("whisperx") is None
    assert canonical_engine("deepgram") is None


@pytest.mark.parametrize(
    ("platform_slug", "expected"),
    [
        ("macos", "whisperkit"),
        ("windows", "sherpa_onnx"),
        ("linux", "sherpa_onnx"),
        ("wsl2", "sherpa_onnx"),
    ],
)
def test_default_engine_per_platform(platform_slug, expected):
    assert default_engine_for_platform(platform_slug) == expected


def test_factory_auto_selects_whisperkit_on_macos():
    strategy = ASRStrategyFactory(ASRConfig(), platform="macos").create_strategy()
    assert type(strategy).__name__ == "WhisperKitASRStrategy"


def test_factory_auto_selects_sherpa_on_other_platforms():
    for platform_slug in ("windows", "linux", "wsl2"):
        strategy = ASRStrategyFactory(ASRConfig(), platform=platform_slug).create_strategy()
        assert type(strategy).__name__ == "SherpaOnnxASRStrategy"


def test_factory_explicit_engine_overrides_platform_default():
    strategy = ASRStrategyFactory(ASRConfig(engine="whisper"), platform="macos").create_strategy()
    assert type(strategy).__name__ == "WhisperASRStrategy"


def test_factory_unknown_engine_raises_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="unknown ASR engine"):
        ASRStrategyFactory(ASRConfig(engine="whisperx"), platform="linux").create_strategy()


def test_factory_normalize_engine_alias():
    strategy = ASRStrategyFactory(
        ASRConfig(engine="SHERPA-ONNX"), platform="linux"
    ).create_strategy()
    assert type(strategy).__name__ == "SherpaOnnxASRStrategy"


def test_component_platform_defaults_to_autodetection(monkeypatch):
    """A07: the component must not force a platform on the factory."""
    from signal_asr.component import ASRComponent
    from signal_asr.strategies import factory as factory_module

    detected: list[str] = []
    original = factory_module._detect_platform

    def fake_detect():
        value = original()
        detected.append(value)
        return value

    monkeypatch.setattr(factory_module, "_detect_platform", fake_detect)
    component = ASRComponent(ASRConfig())
    # The resolved platform is exposed and the factory actually detected it.
    assert component.platform == detected[0]
    assert component.platform in ("macos", "windows", "linux", "wsl2")


def test_component_explicit_platform_is_preserved():
    from signal_asr.component import ASRComponent

    component = ASRComponent(ASRConfig(), platform="macos")
    assert component.platform == "macos"
    assert type(component.strategy).__name__ == "WhisperKitASRStrategy"
