"""Additional branch coverage: sherpa CUDA paths, whisper defaults and failures."""

from __future__ import annotations

import sys
import types

import pytest

from signal_asr.config import ASRConfig
from signal_asr.strategies.factory import ASRStrategyFactory
from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy


def _fake_sherpa_module(monkeypatch, recognizer):
    fake = types.ModuleType("sherpa_onnx")
    fake.OfflineRecognizer = types.SimpleNamespace(from_transducer=lambda **kw: recognizer)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)


class FakeRecognizer:
    def __init__(self, warmup_raises=False):
        self.warmup_raises = warmup_raises
        self.streams = []

    def create_stream(self):
        stream = types.SimpleNamespace(
            accepted=[],
            result=types.SimpleNamespace(text="ok"),
        )

        def accept(rate, audio):
            stream.accepted.append((rate, audio))

        stream.accept_waveform = accept
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        pass


def _seed_models(tmp_path):
    model_dir = tmp_path / "csukuangfj--sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    model_dir.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"x")


def test_sherpa_cuda_linux_torch_import_path(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    recognizer = FakeRecognizer()
    _fake_sherpa_module(monkeypatch, recognizer)
    # torch absent: ImportError branch is exercised (logged debug, continues).
    monkeypatch.setitem(sys.modules, "torch", None)

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cuda", platform="linux", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.is_loaded


def test_sherpa_cuda_warmup_failure_leaves_unloaded(monkeypatch, tmp_path):
    _seed_models(tmp_path)

    recognizer = FakeRecognizer()

    def boom(stream):
        raise RuntimeError("cuda warmup broke")

    recognizer.decode_stream = boom
    _fake_sherpa_module(monkeypatch, recognizer)

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cuda", platform="linux", models_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        strategy.load_model()
    assert not strategy.is_loaded


def test_sherpa_missing_dependency_message(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    with pytest.raises(ImportError, match="sherpa-onnx"):
        strategy.load_model()


def test_sherpa_load_model_is_idempotent(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    recognizer = FakeRecognizer()
    _fake_sherpa_module(monkeypatch, recognizer)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    strategy.load_model()
    first = strategy.recognizer
    strategy.load_model()
    assert strategy.recognizer is first


def test_whisper_default_size_when_no_priority(monkeypatch):
    fake_fw = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, size, device=None, compute_type=None):
            self.size = size

        def transcribe(self, path, **kwargs):
            return [], None

    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisper"), device="cpu", platform="linux"
    ).create()
    strategy.load_model()
    assert strategy.model.size == "base"
    assert strategy.transcribe(b"\x01\x00") == ""


def test_whisper_transcribe_failure_propagates(monkeypatch):
    fake_fw = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, size, device=None, compute_type=None):
            pass

        def transcribe(self, path, **kwargs):
            raise RuntimeError("decode failed")

    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisper"), device="cpu", platform="linux"
    ).create()
    strategy.load_model()
    with pytest.raises(RuntimeError, match="Whisper transcription failed"):
        strategy.transcribe(b"\x01\x00\x02\x00")


def test_whisper_odd_chunk_single_byte_returns_empty(monkeypatch):
    fake_fw = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, size, device=None, compute_type=None):
            pass

        def transcribe(self, path, **kwargs):
            return [], None

    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisper"), device="cpu", platform="linux"
    ).create()
    strategy.load_model()
    assert strategy.transcribe(b"\x01") == ""
