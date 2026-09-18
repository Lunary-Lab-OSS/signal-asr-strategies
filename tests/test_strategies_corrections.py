"""Hermetic regressions for strategy request and lifecycle boundaries."""

import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from signal_asr.component import ASRComponent
from signal_asr.config import ASRConfig, ModelConfig
from signal_asr.strategies import whisperkit
from signal_asr.strategies.parakeet import resolve_parakeet_device
from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy
from signal_asr.strategies.whisper import WhisperASRStrategy, resolve_whisper_device
from signal_asr.strategies.whisperkit import WhisperKitASRStrategy


def test_all_sherpa_profiles_declare_and_lock_matching_core():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "signal-asr-strategies")
    profiles = project["project"]["optional-dependencies"]
    for name, dependencies in profiles.items():
        if not any(
            dep.startswith("sherpa-onnx") and not dep.startswith("sherpa-onnx-core")
            for dep in dependencies
        ):
            continue
        assert "sherpa-onnx==1.13.8" in dependencies
        assert "sherpa-onnx-core==1.13.8" in dependencies
        locked = {dep["name"] for dep in package["optional-dependencies"][name]}
        assert {"sherpa-onnx", "sherpa-onnx-core"} <= locked
    for name in ("sherpa-onnx", "sherpa-onnx-core"):
        assert {p["version"] for p in lock["package"] if p["name"] == name} == {"1.13.8"}


@pytest.mark.parametrize("device", ["cudax", "cudafoo:1", "cuda:", "cuda:-1", "cuda:\u0661"])
def test_invalid_devices_rejected_before_loading(device, monkeypatch):
    monkeypatch.setattr("signal_asr.strategies.parakeet._load_torch", lambda: object())
    with pytest.raises(ValueError):
        resolve_whisper_device(device)
    with pytest.raises(ValueError):
        resolve_parakeet_device(device)
    with pytest.raises(ValueError):
        SherpaOnnxASRStrategy(ASRConfig(), device, "linux").load_model()


@pytest.mark.parametrize("server", [True, False])
@pytest.mark.parametrize("default_language", [None, "fr", "en"])
def test_whisperkit_language_is_request_scoped(server, default_language, monkeypatch):
    component = ASRComponent(
        ASRConfig(engine="whisperkit", language=default_language), platform="macos"
    )
    strategy = component.strategy
    strategy.model = {}
    strategy._use_server = server
    strategy._cli_path = "/fake/cli"
    monkeypatch.setattr(strategy, "_owns_listener", lambda: True)
    languages = []

    def post(url, **kwargs):
        languages.append(kwargs["data"].get("language"))
        return SimpleNamespace(
            status_code=200, iter_content=lambda **kw: [b'{"text":""}'], close=lambda: None
        )

    def run(cmd, **kwargs):
        languages.append(cmd[cmd.index("--language") + 1] if "--language" in cmd else None)
        return ""

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))
    monkeypatch.setattr(whisperkit, "_bounded_output", run)
    for language in ("de", "es", None):
        assert component.transcribe(b"\x00\x00", language) == ""
    assert languages == ["de", "es", default_language]


def test_whisperkit_occupied_port_never_spawns(monkeypatch):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    monkeypatch.setattr(strategy, "_port_available", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned"))
    assert strategy._spawn_and_wait("cpuOnly", 1) is False


@pytest.mark.parametrize("owner", ["p456", "p123\np456", ""])
def test_whisperkit_unrelated_listener_never_receives_audio(owner, monkeypatch):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    strategy._server = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(whisperkit, "_bounded_output", lambda *a, **kw: owner)
    monkeypatch.setitem(
        sys.modules, "requests", SimpleNamespace(post=lambda *a, **kw: pytest.fail("uploaded"))
    )
    assert strategy._transcribe_server("not-opened.wav") is None
    strategy._server = None


def test_whisperkit_shutdown_can_reload(monkeypatch):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    monkeypatch.setattr(strategy, "_resolve_cli", lambda: "/fake/cli")
    monkeypatch.setattr(strategy, "_start_server", lambda: True)
    strategy.load_model()
    strategy.shutdown()
    assert not strategy.is_loaded
    assert not strategy._use_server
    strategy.load_model()
    assert strategy.is_loaded


def test_whisperkit_warmup_failure_is_not_loaded(monkeypatch):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    monkeypatch.setattr(strategy, "_resolve_cli", lambda: "/fake/cli")
    monkeypatch.setattr(strategy, "_start_server", lambda: False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="warmup failed"):
        strategy.load_model()
    assert not strategy.is_loaded


@pytest.mark.parametrize("engine", ["whisper", "parakeet", "sherpa_onnx", "whisperkit"])
def test_facade_close_releases_every_strategy(engine):
    component = ASRComponent(ASRConfig(engine=engine), platform="macos")
    component.strategy.model = SimpleNamespace(cpu=lambda: None)
    component.close()
    component.close()
    assert not component.strategy.is_loaded


@pytest.mark.parametrize("source", ["repo", "local", "revision"])
def test_whisper_model_source_is_honored(source, monkeypatch, tmp_path):
    descriptor = ModelConfig(name="ignored", repo_id="owner/converted-whisper")
    downloaded = []
    loaded = []
    if source == "local":
        descriptor.local_path = str(tmp_path)
    if source == "revision":
        descriptor.revision = "specific-revision"

    def download(**kwargs):
        downloaded.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=lambda path, **kw: loaded.append(path) or object()),
    )
    strategy = WhisperASRStrategy(ASRConfig(model_priority=[descriptor]), None, "cpu", "linux")
    strategy.load_model()
    assert loaded == [descriptor.repo_id if source == "repo" else str(tmp_path)]
    if source == "revision":
        assert downloaded[0]["revision"] == descriptor.revision
        assert downloaded[0]["repo_id"] == descriptor.repo_id
    else:
        assert not downloaded


def test_whisper_lazy_segment_failure_propagates(monkeypatch):
    def segments():
        yield SimpleNamespace(text="partial")
        raise ValueError("decode failed")

    strategy = WhisperASRStrategy(ASRConfig(), None, "cpu", "linux")
    strategy.model = SimpleNamespace(transcribe=lambda *a, **kw: (segments(), None))
    with pytest.raises(RuntimeError, match="Whisper transcription failed"):
        strategy.transcribe(b"\x00\x00")


@pytest.mark.parametrize("name", [None, "", 42])
def test_whisper_invalid_model_source_rejected(name):
    strategy = WhisperASRStrategy(
        ASRConfig(model_priority=[SimpleNamespace(name=name)]), None, "cpu", "linux"
    )
    with pytest.raises(ValueError, match="nonempty string"):
        strategy.load_model()


@pytest.mark.parametrize("owner,expected", [("p123\nf4\n", True), ("p123\np456", False)])
def test_whisperkit_listener_requires_exclusive_child_pid(monkeypatch, owner, expected):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    strategy._server = SimpleNamespace(pid=123, poll=lambda: None)

    def output(cmd, **kwargs):
        assert cmd[0] == "/usr/sbin/lsof"
        assert f"-iTCP:{strategy._port}" in cmd
        assert "-sTCP:LISTEN" in cmd
        assert kwargs == {"timeout": 2, "limit": 65536}
        return owner

    monkeypatch.setattr(whisperkit, "_bounded_output", output)
    try:
        assert strategy._owns_listener() is expected
    finally:
        strategy._server = None


def test_whisperkit_missing_lsof_fails_closed(monkeypatch):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    strategy._server = SimpleNamespace(pid=123, poll=lambda: None)

    def missing(*args, **kwargs):
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(whisperkit, "_bounded_output", missing)
    try:
        assert not strategy._owns_listener()
    finally:
        strategy._server = None


@pytest.mark.parametrize(
    "chunks,expected",
    [([b'{"text":"ok"}'], "ok"), ([b"x" * 65], None), ([b"[]"], None), ([b"{"], None)],
)
def test_whisperkit_response_is_bounded_and_closed(monkeypatch, tmp_path, chunks, expected):
    strategy = WhisperKitASRStrategy(ASRConfig(), "cpu", "macos")
    monkeypatch.setattr(strategy, "_owns_listener", lambda: True)
    monkeypatch.setattr(whisperkit, "_MAX_RESPONSE_BYTES", 64)
    closed = []

    def post(url, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["allow_redirects"] is False
        assert kwargs["proxies"] == {"http": "", "https": ""}
        return SimpleNamespace(
            status_code=200,
            iter_content=lambda **kw: iter(chunks),
            close=lambda: closed.append(True),
        )

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    assert strategy._transcribe_server(str(audio)) == expected
    assert closed == [True]


@pytest.mark.parametrize(
    "script,error",
    [
        ("import time; time.sleep(30)", "timed out"),
        ("import sys; sys.stdout.write('x' * 10000)", "exceeds limit"),
        ("raise SystemExit(2)", "status 2"),
    ],
)
def test_whisperkit_subprocess_limits_reap_child(monkeypatch, script, error):
    popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match=error):
        whisperkit._bounded_output([sys.executable, "-c", script], timeout=0.5, limit=64)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdout.closed


def test_whisperkit_bounded_subprocess_success():
    assert whisperkit._bounded_output([sys.executable, "-c", "print('hello')"], 5) == "hello\n"
