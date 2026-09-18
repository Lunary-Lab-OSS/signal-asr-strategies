"""Deep coverage for sherpa-onnx and whisperkit paths (A16/D coverage goal).

These tests exercise the platform-specific branches through injectable
seams (fake sherpa module, fake subprocess/requests) so Windows and macOS
code paths run — and assert their behavior — on any host.
"""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any, ClassVar

import pytest

from signal_asr.config import ASRConfig
from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy
from signal_asr.strategies.whisperkit import WhisperKitASRStrategy


def _fake_sherpa_module(monkeypatch, recognizer) -> None:
    fake = types.ModuleType("sherpa_onnx")
    # The recognizer instance doubles as OfflineRecognizer.from_transducer:
    # calling it with the constructor kwargs records them and returns a
    # working recognizer instance.
    fake.OfflineRecognizer = types.SimpleNamespace(from_transducer=recognizer)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)


def _seed_models(tmp_path) -> None:
    model_dir = tmp_path / "csukuangfj--sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"fake")


class RecordingRecognizer:
    def __init__(self, text="hello sherpa"):
        self.text = text
        self.streams: list[Any] = []
        self.decode_calls = 0
        self.constructor_kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs):
        # Captures what OfflineRecognizer.from_transducer received.
        instance = RecordingRecognizer(self.text)
        instance.constructor_kwargs = kwargs
        return instance

    def create_stream(self):
        stream = types.SimpleNamespace(
            accepted=[],
            result=types.SimpleNamespace(text=self.text),
        )

        def accept(rate, audio):
            stream.accepted.append((rate, len(audio)))

        stream.accept_waveform = accept
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        self.decode_calls += 1


@pytest.fixture
def recognizer():
    return RecordingRecognizer()


def _capturing_sherpa_module(monkeypatch, recorder):
    fake = types.ModuleType("sherpa_onnx")
    # OfflineRecognizer.from_transducer(**kwargs) invokes the recorder,
    # which records the constructor kwargs and returns a working instance.
    fake.OfflineRecognizer = types.SimpleNamespace(from_transducer=recorder)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)


def test_sherpa_cuda_uses_single_thread_and_cuda_provider(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    recorder = RecordingRecognizer()
    _capturing_sherpa_module(monkeypatch, recorder)
    monkeypatch.setitem(sys.modules, "torch", None)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cuda", "linux", models_dir=tmp_path
    )
    strategy.load_model()
    kwargs = strategy.recognizer.constructor_kwargs
    # CUDA uses the cuda provider with a single thread (A13/D contract).
    assert kwargs["provider"] == "cuda"
    assert kwargs["num_threads"] == 1


def test_sherpa_transcribe_passes_float32_audio_at_16k(monkeypatch, tmp_path, recognizer):
    _seed_models(tmp_path)
    _fake_sherpa_module(monkeypatch, recognizer)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cpu", "linux", models_dir=tmp_path
    )
    strategy.load_model()

    text = strategy.transcribe(b"\x01\x02" * 8000)
    assert text == "hello sherpa"
    stream = strategy.recognizer.streams[0]
    rate, num_samples = stream.accepted[0]
    assert rate == 16000
    # PCM int16 samples are converted to float32 1:1.
    assert num_samples == 8000


def test_sherpa_transcribe_empty_input_short_circuits(monkeypatch, tmp_path, recognizer):
    _seed_models(tmp_path)
    _fake_sherpa_module(monkeypatch, recognizer)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cpu", "linux", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.transcribe(b"") == ""
    assert recognizer.streams == []


def test_sherpa_transcribe_half_sample_dropped(monkeypatch, tmp_path, recognizer):
    _seed_models(tmp_path)
    _fake_sherpa_module(monkeypatch, recognizer)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cpu", "linux", models_dir=tmp_path
    )
    strategy.load_model()
    text = strategy.transcribe(b"\x01\x02\x03")
    assert text == "hello sherpa"
    _rate, num_samples = strategy.recognizer.streams[0].accepted[0]
    assert num_samples == 1


def test_sherpa_shutdown_clears_state(monkeypatch, tmp_path, recognizer):
    _seed_models(tmp_path)
    _fake_sherpa_module(monkeypatch, recognizer)
    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cpu", "linux", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.is_loaded
    strategy.shutdown()
    assert not strategy.is_loaded
    assert strategy.recognizer is None


def test_sherpa_windows_cuda_setup_paths(monkeypatch, tmp_path, recognizer):
    """Windows CUDA setup: PATH prepend, CUDA_PATH, cuDNN check, pre-load."""
    import ctypes
    import os

    _seed_models(tmp_path)
    _fake_sherpa_module(monkeypatch, recognizer)

    toolkit = tmp_path / "CUDA" / "v12.8" / "bin"
    toolkit.mkdir(parents=True)
    for dll in ("cudart64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        (toolkit / dll).write_bytes(b"")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("CUDA_PATH", str(toolkit.parent))
    monkeypatch.setenv("PATH", "")
    loaded_dlls: list[str] = []

    class _FakeWinDLL:
        def __init__(self, path):
            loaded_dlls.append(path)

    monkeypatch.setattr(ctypes, "WinDLL", _FakeWinDLL, raising=False)

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cuda", "windows", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.is_loaded
    assert loaded_dlls, "cuDNN/cuBLAS DLLs must be pre-loaded on Windows"
    assert os.environ["PATH"].startswith(str(toolkit))


def test_sherpa_windows_cuda_missing_runtime_raises(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("PATH", "")

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cuda", "windows", models_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="CUDA runtime DLLs not found"):
        strategy.load_model()


def test_sherpa_windows_cuda_missing_cudnn_raises(monkeypatch, tmp_path):
    _seed_models(tmp_path)
    toolkit = tmp_path / "CUDA" / "v12.8" / "bin"
    toolkit.mkdir(parents=True)
    (toolkit / "cudart64_12.dll").write_bytes(b"")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("CUDA_PATH", str(toolkit.parent))
    monkeypatch.setenv("PATH", "")

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), "cuda", "windows", models_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="cuDNN not found"):
        strategy.load_model()


def test_sherpa_cuda11_dll_names_recognised(monkeypatch, tmp_path):
    toolkit = tmp_path / "CUDA11" / "bin"
    toolkit.mkdir(parents=True)
    (toolkit / "cudart64_110.dll").write_bytes(b"")
    monkeypatch.setenv("CUDA_PATH", str(toolkit.parent))
    monkeypatch.setenv("PATH", "")

    found = SherpaOnnxASRStrategy._find_cuda_bin_windows()
    assert found is not None
    _bin, runtime_major = found
    assert runtime_major == 11


# --------------------------------------------------------------------- #
# WhisperKit deep paths
# --------------------------------------------------------------------- #


class FakePopen:
    instances: ClassVar[list[FakePopen]] = []

    def __init__(self, cmd, stdout=None, stderr=None):
        self.cmd = cmd
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._wait_timeouts = 0
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self._wait_timeouts > 0:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_fake_popen():
    FakePopen.instances = []
    yield
    FakePopen.instances = []


def _make_whisperkit(monkeypatch, compute_units="cpuAndNeuralEngine"):
    monkeypatch.setattr(
        "shutil.which", lambda name: f"/fake/bin/{name}" if name == "whisperkit-cli" else None
    )
    monkeypatch.setenv("SIGNAL_WHISPERKIT_COMPUTE_UNITS", compute_units)
    strategy = WhisperKitASRStrategy(ASRConfig(engine="whisperkit", language="en"), "cpu", "macos")
    return strategy


def test_whisperkit_cli_command_includes_model_and_language(monkeypatch):
    strategy = _make_whisperkit(monkeypatch)
    strategy._cli_path = "/fake/bin/whisperkit-cli"
    cmd = strategy._build_cli_cmd("/tmp/x.wav", "en")
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == strategy._model_name
    assert "--language" in cmd
    assert cmd[cmd.index("--language") + 1] == "en"
    assert "--without-timestamps" in cmd
    assert "--skip-special-tokens" in cmd


def test_whisperkit_unsupported_compute_units_fall_back(monkeypatch, caplog):
    strategy = _make_whisperkit(monkeypatch, compute_units="quantum-computer")
    assert strategy._compute_units == "cpuAndNeuralEngine"


def test_whisperkit_cli_transcribe_parses_last_nonempty_line(monkeypatch, tmp_path):
    strategy = _make_whisperkit(monkeypatch)
    strategy._cli_path = "/fake/bin/whisperkit-cli"
    strategy._use_server = False

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return "\nloading model...\n  Hello world  \n"

    monkeypatch.setattr("signal_asr.strategies.whisperkit._bounded_output", fake_run)
    text = strategy._transcribe_cli(str(tmp_path / "a.wav"))
    assert text == "Hello world"
    assert captured["cmd"][0] == "/fake/bin/whisperkit-cli"


def test_whisperkit_cli_transcribe_nonzero_exit(monkeypatch, tmp_path):
    strategy = _make_whisperkit(monkeypatch)
    strategy._cli_path = "/fake/bin/whisperkit-cli"

    def fail(cmd, **kwargs):
        raise RuntimeError("WhisperKit subprocess exited with status 2")

    monkeypatch.setattr("signal_asr.strategies.whisperkit._bounded_output", fail)
    with pytest.raises(RuntimeError, match="WhisperKit transcription failed"):
        strategy._transcribe_cli(str(tmp_path / "a.wav"))


def test_whisperkit_server_mode_sends_language(monkeypatch, tmp_path):
    strategy = _make_whisperkit(monkeypatch)
    monkeypatch.setattr(strategy, "_owns_listener", lambda: True)
    strategy._use_server = True
    strategy._cli_path = "/fake/bin/whisperkit-cli"

    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def iter_content(**kwargs):
            return [b'{"text":"  server text  "}']

        @staticmethod
        def close():
            pass

    class FakeRequests:
        @staticmethod
        def post(url, files=None, data=None, timeout=None, allow_redirects=None, **kwargs):
            assert allow_redirects is False
            captured["url"] = url
            captured["data"] = data
            return _Resp()

        @staticmethod
        def get(url, timeout=None):
            return _Resp()

    fake_requests = types.ModuleType("requests")
    fake_requests.post = FakeRequests.post
    fake_requests.get = FakeRequests.get
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert strategy._transcribe_server(str(wav), "en") == "server text"
    assert captured["data"] == {"model": strategy._model_name, "language": "en"}


def test_whisperkit_server_non_200_returns_none(monkeypatch, tmp_path):
    strategy = _make_whisperkit(monkeypatch)
    monkeypatch.setattr(strategy, "_owns_listener", lambda: True)
    strategy._use_server = True

    class _Resp:
        status_code = 500

        @staticmethod
        def json():
            return {}

    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda url, **kw: _Resp()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert strategy._transcribe_server(str(wav)) is None


def test_whisperkit_kill_server_escalates_to_kill(monkeypatch):
    strategy = _make_whisperkit(monkeypatch)
    proc = FakePopen(["whisperkit-cli", "serve"])
    proc._wait_timeouts = 1  # first wait times out -> kill path
    strategy._server = proc
    strategy._kill_server()
    assert proc.terminated
    assert proc.killed


def test_whisperkit_transcribe_falls_back_to_cli_when_server_fails(monkeypatch, tmp_path):
    strategy = _make_whisperkit(monkeypatch)
    strategy._cli_path = "/fake/bin/whisperkit-cli"
    strategy._use_server = True
    strategy.model = {"engine": "whisperkit"}
    monkeypatch.setattr(strategy, "_transcribe_server", lambda path, language: None)
    monkeypatch.setattr(
        "signal_asr.strategies.whisperkit._bounded_output",
        lambda cmd, **kw: "cli fallback text",
    )
    text = strategy.transcribe(b"\x00\x00" * 100)
    assert text == "cli fallback text"


def test_whisperkit_empty_input_returns_empty(monkeypatch):
    strategy = _make_whisperkit(monkeypatch)
    assert strategy.transcribe(b"") == ""


def test_whisperkit_load_rejects_non_macos(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    strategy = WhisperKitASRStrategy(ASRConfig(engine="whisperkit"), "cpu", "linux")
    assert strategy.platform == "linux"
    with pytest.raises(RuntimeError, match="requires macOS"):
        strategy.load_model()


def test_whisperkit_missing_cli_raises_helpful_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    strategy = WhisperKitASRStrategy(ASRConfig(engine="whisperkit"), "cpu", "macos")
    with pytest.raises(RuntimeError, match="brew install whisperkit-cli"):
        strategy.load_model()
