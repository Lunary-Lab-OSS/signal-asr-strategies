"""Unit tests for ASR strategy implementations (A03/A05/A06/A16).

Heavy backends (faster-whisper, NeMo/torch, sherpa-onnx) are exercised
through strict fakes at the module seams; the integration track runs the
real libraries where they are installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

from signal_asr.config import ASRConfig
from signal_asr.strategies import parakeet as parakeet_module
from signal_asr.strategies import sherpa_onnx as sherpa_module
from signal_asr.strategies import whisper as whisper_module
from signal_asr.strategies.whisper import resolve_whisper_device

# --------------------------------------------------------------------- #
# Whisper device / compute-type mapping (A03)
# --------------------------------------------------------------------- #


class _FakeCudaDeviceCount:
    def __init__(self, count: int) -> None:
        self.count = count

    def __call__(self) -> int:
        return self.count


def _install_fake_ctranslate2(monkeypatch, cuda_devices: int) -> types.ModuleType:
    fake = types.ModuleType("ctranslate2")
    fake.get_cuda_device_count = _FakeCudaDeviceCount(cuda_devices)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    return fake


def test_resolve_whisper_device_cpu_uses_int8(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    device, index, compute = resolve_whisper_device("cpu")
    assert (device, index, compute) == ("cpu", None, "int8")


def test_resolve_whisper_device_cuda_preserves_index(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=2)
    device, index, compute = resolve_whisper_device("cuda:1")
    assert (device, index, compute) == ("cuda", 1, "float16")


def test_resolve_whisper_device_cuda_without_index(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=1)
    device, index, compute = resolve_whisper_device("cuda")
    assert (device, index, compute) == ("cuda", None, "float16")


def test_resolve_whisper_device_auto_selects_cuda_when_available(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=1)
    device, _index, compute = resolve_whisper_device("auto")
    assert device == "cuda"
    assert compute == "float16"


def test_resolve_whisper_device_auto_falls_back_to_cpu(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    device, index, compute = resolve_whisper_device(None)
    assert (device, index, compute) == ("cpu", None, "int8")


def test_resolve_whisper_device_rejects_unsupported_names():
    for bad in ("tpu", "mps", "gpu"):
        with pytest.raises(ValueError, match="unsupported Whisper device"):
            resolve_whisper_device(bad)
    with pytest.raises(ValueError, match="invalid CUDA device index"):
        resolve_whisper_device("cuda:x")


def test_resolve_whisper_device_rejects_incompatible_compute_type(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    monkeypatch.setenv("SIGNAL_ASR_WHISPER_COMPUTE_TYPE", "float16")
    with pytest.raises(ValueError, match="not supported on device"):
        resolve_whisper_device("cpu")


def test_resolve_whisper_device_accepts_env_override(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=1)
    monkeypatch.setenv("SIGNAL_ASR_WHISPER_COMPUTE_TYPE", "int8_float16")
    _device, _index, compute = resolve_whisper_device("cuda")
    assert compute == "int8_float16"


class _RecordingWhisperModel:
    instances: ClassVar[list[_RecordingWhisperModel]] = []

    def __init__(self, model_size: str, **kwargs: Any) -> None:
        self.model_size = model_size
        self.kwargs = kwargs
        self.transcribe_calls: list[dict[str, Any]] = []
        _RecordingWhisperModel.instances.append(self)

    def transcribe(self, path, **kwargs):
        self.transcribe_calls.append({"path": path, **kwargs})

        class Segment:
            text = " hello whisper "

        return [Segment()], None


@pytest.fixture(autouse=True)
def _reset_recording_models():
    _RecordingWhisperModel.instances = []
    yield
    _RecordingWhisperModel.instances = []


def test_whisper_strategy_passes_mapped_device_and_compute_type(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _RecordingWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    config = ASRConfig(engine="whisper", model_priority=[])
    strategy = whisper_module.WhisperASRStrategy(config, None, "cpu", "linux")
    strategy.load_model()

    model = _RecordingWhisperModel.instances[-1]
    # A03: CPU must not receive float16; index is absent for plain cpu.
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["compute_type"] == "int8"
    assert "device_index" not in model.kwargs


def test_whisper_strategy_passes_cuda_index(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=2)
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _RecordingWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    config = ASRConfig(engine="whisper")
    strategy = whisper_module.WhisperASRStrategy(config, None, "cuda:1", "linux")
    strategy.load_model()

    model = _RecordingWhisperModel.instances[-1]
    assert model.kwargs["device"] == "cuda"
    assert model.kwargs["device_index"] == 1
    assert model.kwargs["compute_type"] == "float16"


def test_whisper_transcribe_passes_language_and_task(monkeypatch, tmp_path):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _RecordingWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    config = ASRConfig(engine="whisper", language="fr", task="translate")
    strategy = whisper_module.WhisperASRStrategy(config, None, "cpu", "linux")
    text = strategy.transcribe(b"\x00\x00" * 160)
    assert text == "hello whisper"
    call = _RecordingWhisperModel.instances[-1].transcribe_calls[0]
    assert call["language"] == "fr"
    assert call["task"] == "translate"


def test_whisper_transcribe_empty_input_returns_empty():
    config = ASRConfig(engine="whisper")
    strategy = whisper_module.WhisperASRStrategy(config, None, "cpu", "linux")
    # Empty input returns before any backend import is attempted.
    assert strategy.transcribe(b"") == ""


def test_whisper_transcribe_odd_byte_chunk_is_aligned(monkeypatch):
    _install_fake_ctranslate2(monkeypatch, cuda_devices=0)
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _RecordingWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    config = ASRConfig(engine="whisper")
    strategy = whisper_module.WhisperASRStrategy(config, None, "cpu", "linux")
    # One byte (half a sample): nothing transcribable remains.
    assert strategy.transcribe(b"\x00") == ""
    # One and a half samples: the partial sample is dropped, the complete
    # sample is still transcribed.
    assert strategy.transcribe(b"\x00\x00\x00") == "hello whisper"


# --------------------------------------------------------------------- #
# Parakeet via seams (A05/A06/A16)
# --------------------------------------------------------------------- #


class _FakeTensor:
    def __init__(self, data=None):
        self.data = data if data is not None else []

    def __truediv__(self, other):
        return self

    def unsqueeze(self, dim):
        return _FakeTensor([list(self.data)])

    def to(self, device):
        self.device = device
        return self

    @property
    def shape(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return (1, len(self.data[0]))
        return (len(self.data),)

    def item(self):
        if isinstance(self.data, list) and len(self.data) == 1:
            return self.data[0]
        return self.data


class _FakeTorchModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.device_requests: list[str] = []
        self.no_grad_calls = 0

    def device(self, spec):
        self.device_requests.append(spec)

        class D:
            def __init__(self, s):
                self.type = "cuda" if str(s).startswith("cuda") else "cpu"
                self.str = str(s)

            def __str__(self):
                return self.str

        return D(spec)

    def frombuffer(self, buffer, dtype=None):
        return _FakeTensor(list(buffer))

    @property
    def int16(self):
        return "int16"

    @property
    def float32(self):
        return "float32"

    @property
    def long(self):
        return "long"

    def tensor(self, data, device=None, dtype=None):
        return _FakeTensor(data)

    def no_grad(self):
        self.no_grad_calls += 1

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()

    @property
    def cuda(self):
        cuda = self

        class _CudaNS:
            @staticmethod
            def is_available():
                return cuda.cuda_available

        _cuda_ns = _CudaNS()
        cuda.cuda_available = getattr(cuda, "cuda_available", False)
        return _cuda_ns


class _FakeHypothesis:
    def __init__(self, text):
        self.text = text


class _FakeParameter:
    device = "cpu"


class _FakeNemoModel:
    instances: ClassVar[list[_FakeNemoModel]] = []

    def __init__(self, text="hello parakeet", hyps=None, raise_on_decode=False):
        self.text_result = text
        self.hyps_override = hyps
        self.raise_on_decode = raise_on_decode
        self.placed_on = []
        self.eval_called = 0
        self.preprocessor_calls = 0
        self.encoder_calls = 0
        self.decoder_calls = 0
        self._param = _FakeParameter()

    def to(self, device):
        self.placed_on.append(str(device))
        self._param = _FakeParameter()
        self._param.device = str(device)
        return self

    def eval(self):
        self.eval_called += 1
        return self

    def parameters(self):
        return iter([self._param])

    def preprocessor(self, input_signal=None, length=None):
        self.preprocessor_calls += 1
        return input_signal, length

    def encoder(self, audio_signal=None, length=None):
        self.encoder_calls += 1
        return audio_signal, length

    def cpu(self):
        self.placed_on.append("cpu")
        self._param = _FakeParameter()
        self._param.device = "cpu"
        return self


class _FakeDecoding:
    def __init__(self, model):
        self.model = model

    def rnnt_decoder_predictions_tensor(self, encoded, enc_len, return_hypotheses=False):
        assert return_hypotheses is True
        self.model.decoder_calls += 1
        if self.model.raise_on_decode:
            raise RuntimeError("decoder exploded")
        if self.model.hyps_override is not None:
            return self.model.hyps_override
        return [_FakeHypothesis(self.model.text_result)]


def _install_fake_nemo(monkeypatch, model: _FakeNemoModel) -> None:
    model.decoding = _FakeDecoding(model)
    nemo_asr = types.ModuleType("nemo.collections.asr")
    nemo_asr.models = types.SimpleNamespace(
        ASRModel=types.SimpleNamespace(from_pretrained=lambda model_name: model)
    )
    nemo_module = types.ModuleType("nemo")
    collections_module = types.ModuleType("nemo.collections")
    nemo_module.collections = collections_module
    monkeypatch.setitem(sys.modules, "nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_asr)

    def fake_load_nemo():
        return nemo_asr

    monkeypatch.setattr(parakeet_module, "_load_nemo_asr", fake_load_nemo)


@pytest.fixture
def fake_torch(monkeypatch):
    module = _FakeTorchModule()
    monkeypatch.setattr(parakeet_module, "_load_torch", lambda: module)
    return module


def test_parakeet_module_imports_without_torch_or_nemo():
    """A16: the module itself must stay importable on minimal installs."""
    import importlib

    spec = importlib.util.find_spec("signal_asr.strategies.parakeet")
    assert spec is not None
    # Already imported successfully in this process; re-import stays clean.
    importlib.reload(parakeet_module)


def test_parakeet_load_places_model_on_requested_device(monkeypatch, fake_torch):
    model = _FakeNemoModel()
    _install_fake_nemo(monkeypatch, model)

    config = ASRConfig(engine="parakeet")
    strategy = parakeet_module.ParakeetASRStrategy(config, "cpu", "linux")
    strategy.load_model()

    assert model.placed_on == ["cpu"], "requested device must be applied (A05)"
    assert model.eval_called == 1
    assert strategy.is_loaded


def test_parakeet_load_failure_leaves_unloaded(monkeypatch, fake_torch):
    def failing_nemo():
        raise ImportError("nemo missing")

    monkeypatch.setattr(parakeet_module, "_load_nemo_asr", failing_nemo)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "cpu", "linux")
    with pytest.raises(ImportError):
        strategy.load_model()
    assert not strategy.is_loaded


def test_parakeet_rejects_cuda_when_unavailable(monkeypatch, fake_torch):
    fake_torch.cuda_available = False
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "cuda:0", "linux")
    with pytest.raises(ValueError, match=r"torch\.cuda\.is_available"):
        strategy.load_model()


def test_parakeet_rejects_unsupported_device(monkeypatch, fake_torch):
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "mps", "macos")
    with pytest.raises(ValueError, match="unsupported Parakeet device"):
        strategy.load_model()


def test_parakeet_transcribe_normal_text(monkeypatch, fake_torch):
    model = _FakeNemoModel(text="hello parakeet ")
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    assert strategy.transcribe(b"\x00\x00" * 100) == "hello parakeet"
    assert (model.preprocessor_calls, model.encoder_calls, model.decoder_calls) == (1, 1, 1)


def test_parakeet_transcribe_empty_hypotheses_returns_empty(monkeypatch, fake_torch):
    model = _FakeNemoModel(hyps=[])
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    assert strategy.transcribe(b"\x00\x00" * 100) == ""


def test_parakeet_transcribe_empty_text_is_returned_verbatim(monkeypatch, fake_torch):
    """A06: an empty transcript must not become a stringified hypothesis."""
    model = _FakeNemoModel(text="")
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    assert strategy.transcribe(b"\x00\x00" * 100) == ""


def test_parakeet_transcribe_missing_text_attribute_raises(monkeypatch, fake_torch):
    """A06: a backend contract violation must surface, not stringify."""
    model = _FakeNemoModel(hyps=[object()])  # object() has no .text
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    with pytest.raises(RuntimeError, match="usable 'text'"):
        strategy.transcribe(b"\x00\x00" * 100)


def test_parakeet_transcribe_odd_bytes(monkeypatch, fake_torch):
    model = _FakeNemoModel(text="x")
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    # Half a sample: nothing remains after alignment.
    assert strategy.transcribe(b"\x00") == ""
    # One and a half samples: the complete sample is still transcribed.
    assert strategy.transcribe(b"\x00\x00\x00") == "x"


def test_parakeet_shutdown_clears_model(monkeypatch, fake_torch):
    model = _FakeNemoModel()
    _install_fake_nemo(monkeypatch, model)
    strategy = parakeet_module.ParakeetASRStrategy(ASRConfig(engine="parakeet"), "auto", "linux")
    strategy.load_model()
    strategy.shutdown()
    assert not strategy.is_loaded
    assert "cpu" in model.placed_on


# --------------------------------------------------------------------- #
# Sherpa-onnx basics
# --------------------------------------------------------------------- #


def test_sherpa_rejects_indexed_cuda_device():
    config = ASRConfig(engine="sherpa_onnx")
    strategy = sherpa_module.SherpaOnnxASRStrategy(config, "cuda:1", "linux")
    with pytest.raises(ValueError, match="cannot select a specific CUDA index"):
        strategy.load_model()


def test_sherpa_rejects_unknown_device():
    config = ASRConfig(engine="sherpa_onnx")
    strategy = sherpa_module.SherpaOnnxASRStrategy(config, "tpu", "linux")
    with pytest.raises(ValueError, match="unsupported sherpa-onnx device"):
        strategy.load_model()


def test_sherpa_accepts_plain_cuda_and_cpu(monkeypatch):
    class ReachedModelLookup(Exception):
        pass

    def stop_before_download(self):
        raise ReachedModelLookup

    monkeypatch.setattr(
        sherpa_module.SherpaOnnxASRStrategy, "_ensure_models_downloaded", stop_before_download
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    for device in ("cuda", "cpu", "CUDA"):
        strategy = sherpa_module.SherpaOnnxASRStrategy(ASRConfig(), device, "linux")
        with pytest.raises(ReachedModelLookup):
            strategy.load_model()


def test_sherpa_cuda_discovery_uses_dlls_not_directory_names(monkeypatch, tmp_path):
    """A11: a CUDA 12 install in a custom root must be detected via DLLs."""
    custom_root = tmp_path / "Toolkit"  # deliberately un-versioned name
    custom_bin = custom_root / "bin"
    custom_bin.mkdir(parents=True)
    (custom_bin / "cudart64_12.dll").write_bytes(b"")
    monkeypatch.setenv("CUDA_PATH", str(custom_root))
    # Blank PATH so only the CUDA_PATH-derived candidate is considered.
    monkeypatch.setenv("PATH", "")

    class _WinPath(sherpa_module.os.PathLike if hasattr(sherpa_module.os, "PathLike") else object):  # type: ignore[attr-defined]
        pass

    real_isdir = sherpa_module.os.path.isdir

    def fake_isdir(p):
        if str(custom_bin) in str(p):
            return True
        return real_isdir(p)

    real_exists = sherpa_module.os.path.exists

    def fake_exists(p):
        if str(p).endswith("cudart64_12.dll") and str(custom_bin) in str(p):
            return True
        # Reject the default NVIDIA install locations entirely.
        if "NVIDIA GPU Computing Toolkit" in str(p):
            return False
        return real_exists(p)

    monkeypatch.setattr(sherpa_module.os.path, "isdir", fake_isdir)
    monkeypatch.setattr(sherpa_module.os.path, "exists", fake_exists)

    found = sherpa_module.SherpaOnnxASRStrategy._find_cuda_bin_windows()
    assert found is not None
    bin_path, runtime_major = found
    assert runtime_major == 12
    assert str(custom_bin) in bin_path
