"""Mock-based lifecycle tests for strategy backends (no models, no subprocesses)."""

from __future__ import annotations

import subprocess
import sys
import types
from typing import ClassVar

import pytest

from signal_asr.config import ASRConfig
from signal_asr.strategies.factory import ASRStrategyFactory
from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy

# --------------------------------------------------------------------------
# WhisperKit lifecycle
# --------------------------------------------------------------------------


class FakeProc:
    def __init__(self, dies=False):
        self.died = dies
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return 1 if self.died else None

    def terminate(self):
        self.terminated = True
        self.died = True

    def kill(self):
        self.killed = True
        self.died = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.killed and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return 0


def _make_whisperkit(monkeypatch, proc, health_ok=True):
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisperkit"), device="cpu", platform="macos"
    ).create()
    monkeypatch.setattr("shutil.which", lambda name: "/fake/whisperkit-cli")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr(strategy, "_port_available", lambda: True)
    monkeypatch.setattr(strategy, "_owns_listener", lambda: not proc.died)
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="x\nhello")
    )
    monkeypatch.setattr(
        "signal_asr.strategies.whisperkit._bounded_output", lambda *a, **k: "x\nhello"
    )

    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda *a, **k: types.SimpleNamespace(
        status_code=200 if health_ok else 500, close=lambda: None
    )
    fake_requests.post = lambda *a, **k: types.SimpleNamespace(
        status_code=200, iter_content=lambda **kw: [b'{"text":"  hi  "}'], close=lambda: None
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    return strategy


def test_whisperkit_server_mode_happy_path(monkeypatch):
    strategy = _make_whisperkit(monkeypatch, FakeProc(), health_ok=True)
    strategy.load_model()
    assert strategy.is_loaded
    assert strategy.model["mode"] == "server"
    assert strategy.transcribe(b"\x00\x00" * 100) == "hi"


def test_whisperkit_server_down_falls_back_to_cli(monkeypatch):
    strategy = _make_whisperkit(monkeypatch, FakeProc(dies=True), health_ok=False)
    strategy.load_model()
    assert strategy.model["mode"] == "cli"
    assert strategy.transcribe(b"\x00\x00" * 100) == "hello"


def test_whisperkit_shutdown_terminates_then_kills(monkeypatch):
    proc = FakeProc()
    strategy = _make_whisperkit(monkeypatch, proc, health_ok=True)
    strategy.load_model()
    strategy._server = proc
    strategy.shutdown()
    assert proc.terminated, "terminate() must be called before any kill"
    if not getattr(proc, "_wait_timed_out_once", False):
        assert not proc.killed, "kill only after a wait timeout"


def test_whisperkit_port_env_override(monkeypatch):
    monkeypatch.setenv("SIGNAL_WHISPERKIT_PORT", "54321")
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisperkit"), device="cpu", platform="macos"
    ).create()
    assert strategy._port == 54321


# --------------------------------------------------------------------------
# Sherpa-ONNX load paths
# --------------------------------------------------------------------------


def test_sherpa_load_cpu_with_existing_models(monkeypatch, tmp_path):
    model_dir = tmp_path / "csukuangfj--sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    model_dir.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"x")

    fake = types.ModuleType("sherpa_onnx")

    class OfflineRecognizer:
        instances: ClassVar[list] = []

        @staticmethod
        def from_transducer(**kwargs):
            OfflineRecognizer.instances.append(kwargs)
            return object()

    fake.OfflineRecognizer = OfflineRecognizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.is_loaded
    assert OfflineRecognizer.instances[0]["provider"] == "cpu"
    assert OfflineRecognizer.instances[0]["num_threads"] == 4


def test_sherpa_downloads_missing_models(monkeypatch, tmp_path):
    downloaded = {}

    def fake_snapshot_download(**kwargs):
        downloaded.update(kwargs)
        # Materialise the files the strategy requires after download.
        import os

        target = kwargs["local_dir"]
        os.makedirs(target, exist_ok=True)
        for name in (
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ):
            with open(os.path.join(target, name), "wb") as fh:
                fh.write(b"fake")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    strategy._ensure_models_downloaded()
    assert downloaded["repo_id"].endswith("parakeet-tdt-0.6b-v3-int8")
    assert downloaded["revision"] == "2bda32ec70b097a55adaa07d9a7173915b43cc78"
    assert strategy.model_dir == tmp_path / downloaded["repo_id"].replace("/", "--")
    # The download is restricted to the files this strategy loads (A04).
    assert set(downloaded["allow_patterns"]) == {
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    }
    # The removed huggingface_hub 1.x argument must not be passed (A04).
    assert "local_dir_use_symlinks" not in downloaded
    assert "cache_dir" not in downloaded


def test_sherpa_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNAL_ASR_MODELS_DIR", str(tmp_path))
    assert SherpaOnnxASRStrategy._default_models_dir() == tmp_path


def test_sherpa_cuda_requires_runtime_on_windows(monkeypatch):
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cuda:0", platform="windows"
    )
    monkeypatch.setattr(strategy, "_find_cuda_bin_windows", staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match="CUDA runtime DLLs not found"):
        strategy._setup_cuda_windows()


# --------------------------------------------------------------------------
# Whisper (faster-whisper) load and transcribe
# --------------------------------------------------------------------------


def test_whisper_load_and_transcribe(monkeypatch, tmp_path):
    fake_fw = types.ModuleType("faster_whisper")

    class Segment:
        def __init__(self, text):
            self.text = text

    class WhisperModel:
        def __init__(self, size, device=None, compute_type=None):
            self.size = size

        def transcribe(self, path, **kwargs):
            return [Segment("hello"), Segment("world")], None

    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisper", model_priority=[type("M", (), {"name": "tiny"})()]),
        device="cpu",
        platform="linux",
    ).create()
    strategy.load_model()
    assert strategy.model.size == "tiny"
    assert strategy.transcribe(b"\x01\x00\x02\x00") == "hello world"


def test_whisper_missing_dependency_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    strategy = ASRStrategyFactory(
        ASRConfig(engine="whisper"), device="cpu", platform="linux"
    ).create()
    with pytest.raises(ImportError, match="faster-whisper"):
        strategy.load_model()


# --------------------------------------------------------------------------
# Parakeet import guard (torch-dependent module)
# --------------------------------------------------------------------------


def test_parakeet_import_error_message(monkeypatch):
    pytest.importorskip("torch")
    strategy = ASRStrategyFactory(
        ASRConfig(engine="parakeet"), device="cpu", platform="linux"
    ).create()
    fake_nemo = types.ModuleType("nemo")
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", None)
    with pytest.raises(ImportError, match="NeMo"):
        strategy.load_model()
