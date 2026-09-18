"""Unit tests for the bounded-resource transcription server.

Covers the remediation contract:

- A01 streaming request-body cap (ASGI-level, no Content-Length needed)
- A02 bounded decoding + admission control (503 on overload)
- A08 encoded/raw classification precedence
- A09 request-scoped form cleanup
- A12 canonical engine validation + cache configuration
- A13 single-flight loading, LRU eviction, reference-counted disposal
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

import pytest

from signal_asr import server
from signal_asr.server import (
    MAX_UPLOAD_BYTES,
    OverloadedError,
    RequestBodyLimitMiddleware,
    StrategyCache,
    _engine_from_model,
    _looks_encoded,
    decode_audio_for_strategy,
)

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

requires_ffmpeg = pytest.mark.skipif(
    not server.ffmpeg_available(), reason="ffmpeg binary not on PATH"
)


class FakeStrategy:
    instances: ClassVar[list[FakeStrategy]] = []

    def __init__(self, calls, name="fake"):
        self.calls = calls
        self.name = name
        self.model = None
        self.shutdown_called = 0
        self.in_flight_calls = 0
        FakeStrategy.instances.append(self)

    @property
    def is_loaded(self):
        return self.model is not None

    def load_model(self):
        self.calls.append(("load_model", self.name))
        self.model = object()

    def transcribe(self, audio_bytes, language=None):
        self.in_flight_calls += 1
        try:
            self.calls.append(("transcribe", self.name, audio_bytes, language))
            return f"transcribed:{self.name}"
        finally:
            self.in_flight_calls -= 1

    def shutdown(self):
        self.shutdown_called += 1
        if self.in_flight_calls > 0:  # pragma: no cover - invariant guard
            raise AssertionError("shutdown while in use")


class FakeFactory:
    calls: ClassVar[list] = []

    def __init__(self, config, device="cpu", platform=None, model_loader=None, models_dir=None):
        FakeFactory.calls.append(
            {
                "config": config,
                "device": device,
                "platform": platform,
                "model_loader": model_loader,
                "models_dir": models_dir,
            }
        )

    def create_strategy(self):
        configs = [c for c in FakeFactory.calls if isinstance(c, dict)]
        engine = configs[-1]["config"].engine
        return FakeStrategy(FakeFactory.calls, name=engine or "auto")


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    FakeFactory.calls = []
    FakeStrategy.instances = []
    server.reset_strategy_cache()
    monkeypatch.setattr(server, "ASRStrategyFactory", FakeFactory)
    monkeypatch.delenv("SIGNAL_ASR_ENGINE", raising=False)
    monkeypatch.delenv("SIGNAL_ASR_DEVICE", raising=False)
    yield
    # Tests may leave capacity/env knobs patched; clear them before the
    # cache reset so teardown itself cannot raise.
    for var in (
        "SIGNAL_ASR_CACHE_MAX",
        "SIGNAL_ASR_MAX_CONCURRENT_DECODE",
        "SIGNAL_ASR_MAX_CONCURRENT_INFERENCE",
        "SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS",
        "SIGNAL_ASR_MAX_DECODED_SECONDS",
        "SIGNAL_ASR_FFMPEG_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)
    server.reset_strategy_cache()


@pytest.fixture
def client():
    server.reset_strategy_cache()
    with TestClient(server.create_app()) as c:
        yield c
    server.reset_strategy_cache()


def _factory_configs():
    return [c for c in FakeFactory.calls if isinstance(c, dict)]


# --------------------------------------------------------------------- #
# Engine resolution (A12)
# --------------------------------------------------------------------- #


def test_openai_default_model_uses_auto_engine():
    server.transcribe_audio_bytes(b"pcm", model="whisper-1")
    assert _factory_configs()[0]["config"].engine == server.default_engine_for_platform(
        server._detect_platform()
    )


def test_engine_aliases_normalize_to_one_cache_entry():
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    server.transcribe_audio_bytes(b"pcm", model="SHERPA-ONNX")
    server.transcribe_audio_bytes(b"pcm", model="sherpa-onnx")
    # All aliases are the same canonical engine: exactly one strategy load.
    assert len(_factory_configs()) == 1


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        server.transcribe_audio_bytes(b"pcm", model="not-an-engine")
    # No strategy was created for the invalid name.
    assert _factory_configs() == []


def test_unknown_engine_env_override_is_rejected(monkeypatch):
    monkeypatch.setenv("SIGNAL_ASR_ENGINE", "definitely-not-real")
    with pytest.raises(ValueError, match="unknown engine"):
        server.transcribe_audio_bytes(b"pcm", model="whisper-1")


def test_engine_env_override_wins_over_model_name(monkeypatch):
    monkeypatch.setenv("SIGNAL_ASR_ENGINE", "parakeet")
    server.transcribe_audio_bytes(b"pcm", model="whisper-1")
    assert _factory_configs()[0]["config"].engine == "parakeet"


def test_gpt4o_transcribe_model_names_use_auto_engine():
    for model in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "local_asr/default", ""):
        server.transcribe_audio_bytes(b"pcm", model=model)
    # All auto names collapse onto the same engine entry: a single load.
    assert len(_factory_configs()) == 1


def test_engine_name_is_normalized():
    assert _engine_from_model("Sherpa-ONNX") == "sherpa_onnx"
    assert _engine_from_model("WHISPER_KIT") == "whisperkit"
    assert _engine_from_model("whisper-1") == ""


# --------------------------------------------------------------------- #
# Strategy cache (A13)
# --------------------------------------------------------------------- #


def test_strategy_cache_reuses_loaded_strategy():
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    assert len(_factory_configs()) == 1
    loads = [c for c in FakeFactory.calls if isinstance(c, tuple)]
    assert ("load_model", "sherpa_onnx") in loads


def test_strategy_cache_separates_engines():
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    server.transcribe_audio_bytes(b"pcm", model="parakeet")
    engines = [c["config"].engine for c in _factory_configs()]
    assert engines == ["sherpa_onnx", "parakeet"]


def test_strategy_cache_separates_devices(monkeypatch):
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx", device="cpu")
    monkeypatch.setenv("SIGNAL_ASR_DEVICE", "cuda:0")
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    devices = [c["device"] for c in _factory_configs()]
    assert devices == ["cpu", "cuda:0"]


def test_cache_single_flight_concurrent_same_key():
    """Concurrent requests for one missing key initialise the model once."""
    start = threading.Barrier(4)
    results: list[str] = []
    errors: list[BaseException] = []

    def worker():
        try:
            start.wait()
            results.append(server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx"))
        except BaseException as exc:  # pragma: no cover - diagnostic only
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert len(results) == 4
    assert len(_factory_configs()) == 1


def test_cache_hit_does_not_wait_for_other_key_initialisation():
    """A slow new-key load must not block an already-cached key (A13)."""
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
    assert len(_factory_configs()) == 1

    slow_started = threading.Event()
    slow_release = threading.Event()

    original_create = server.ASRStrategyFactory

    class SlowFactory(original_create):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def create_strategy(self):
            strategy = super().create_strategy()
            if strategy.name == "parakeet":
                slow_started.set()
                assert slow_release.wait(timeout=10)
            return strategy

    server.ASRStrategyFactory = SlowFactory
    try:
        slow = threading.Thread(
            target=lambda: server.transcribe_audio_bytes(b"pcm", model="parakeet")
        )
        slow.start()
        assert slow_started.wait(timeout=5), "slow load never started"
        # Cached engine serves while the new key is still initialising.
        result = server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx")
        assert result == "transcribed:sherpa_onnx"
    finally:
        slow_release.set()
        slow.join(timeout=10)
        server.ASRStrategyFactory = original_create


def test_cache_eviction_closes_least_recently_used_only_when_idle():
    cache = StrategyCache(capacity=2)
    created: dict[str, FakeStrategy] = {}

    def make(name: str):
        def create() -> Any:
            strategy = FakeStrategy([], name=name)
            created[name] = strategy
            return strategy

        return create

    cache.get_or_load(("a",), make("a"), run=lambda s: "ok")
    cache.get_or_load(("b",), make("b"), run=lambda s: "ok")
    # Touch `a` so `b` is the LRU entry.
    cache.get_or_load(("a",), make("a"), run=lambda s: "ok")
    cache.get_or_load(("c",), make("c"), run=lambda s: "ok")

    assert sorted(created) == ["a", "b", "c"]
    assert created["a"].shutdown_called == 0
    assert created["b"].shutdown_called == 1, "LRU victim must be closed on eviction"
    assert created["c"].shutdown_called == 0
    assert len(cache) == 2


def test_cache_defers_disposal_until_in_flight_call_completes():
    cache = StrategyCache(capacity=1)
    inside_call = threading.Event()
    release_call = threading.Event()

    class SlowUser:
        def __init__(self) -> None:
            self.shutdown_called = 0

        def use(self) -> str:
            inside_call.set()
            assert release_call.wait(timeout=5)
            return "done"

        def shutdown(self) -> None:
            self.shutdown_called += 1

    slow = SlowUser()
    result: list[str] = []
    t = threading.Thread(
        target=lambda: result.append(cache.get_or_load(("k",), lambda: slow, run=lambda s: s.use()))
    )
    t.start()
    assert inside_call.wait(timeout=5)

    with pytest.raises(OverloadedError):
        cache.get_or_load(("other",), lambda: FakeStrategy([], "other"), run=lambda s: "ok")
    cache.close_all()
    assert slow.shutdown_called == 0, "must not close while in use"

    release_call.set()
    t.join(timeout=5)
    assert result == ["done"]
    assert slow.shutdown_called == 1, "closed after last in-flight call ends"


def test_cache_close_all_closes_every_idle_entry():
    cache = StrategyCache(capacity=4)
    created: dict[str, FakeStrategy] = {}

    def make(name: str):
        def create() -> Any:
            strategy = FakeStrategy([], name=name)
            created[name] = strategy
            return strategy

        return create

    for i in range(3):
        cache.get_or_load((f"k{i}",), make(f"s{i}"), run=lambda s: "ok")
    assert all(s.shutdown_called == 0 for s in created.values())
    cache.close_all()
    assert all(s.shutdown_called == 1 for s in created.values())
    assert len(cache) == 0


def test_cache_capacity_validation(monkeypatch):
    monkeypatch.delenv("SIGNAL_ASR_CACHE_MAX", raising=False)
    with pytest.raises(ValueError, match="capacity"):
        StrategyCache(capacity=0)
    monkeypatch.setenv("SIGNAL_ASR_CACHE_MAX", "3")
    assert len(StrategyCache()) == 0
    monkeypatch.setenv("SIGNAL_ASR_CACHE_MAX", "0")
    with pytest.raises(RuntimeError, match="SIGNAL_ASR_CACHE_MAX"):
        StrategyCache()


def test_runtime_config_validation(monkeypatch):
    with pytest.raises(RuntimeError, match="SIGNAL_ASR_CACHE_MAX"):
        monkeypatch.setenv("SIGNAL_ASR_CACHE_MAX", "zero")
        server.validate_runtime_config()
    monkeypatch.setenv("SIGNAL_ASR_CACHE_MAX", "2")
    monkeypatch.setenv("SIGNAL_ASR_MAX_CONCURRENT_DECODE", "-1")
    with pytest.raises(RuntimeError, match="SIGNAL_ASR_MAX_CONCURRENT_DECODE"):
        server.validate_runtime_config()


# --------------------------------------------------------------------- #
# Admission control (A02)
# --------------------------------------------------------------------- #


def test_admission_rejects_when_saturated():
    admission = server._Admission.__new__(server._Admission)
    admission.decode = server._BoundedSemaphore(1)
    admission.inference = server._BoundedSemaphore(1)
    admission.queue_timeout = 0.2

    hold = threading.Event()
    release = threading.Event()

    def slow_decode():
        hold.set()
        release.wait(timeout=5)
        return "decoded"

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(admission.run, admission.decode, slow_decode)
        assert hold.wait(timeout=5)
        second = pool.submit(admission.run, admission.decode, lambda: "never")
        with pytest.raises(OverloadedError):
            second.result(timeout=5)
        release.set()
        assert first.result(timeout=5) == "decoded"


def test_server_returns_503_when_inference_saturated(client, monkeypatch):
    # Shrink the real admission to a single permit, then hold it so the
    # request times out waiting for an inference slot.
    monkeypatch.setattr(server._admission, "inference", server._BoundedSemaphore(1))
    monkeypatch.setattr(server._admission, "queue_timeout", 0.2)
    assert server._admission.inference.try_acquire(1)
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.pcm", b"\x00\x00" * 160, "audio/pcm")},
            data={"model": "sherpa_onnx"},
        )
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
    finally:
        server._admission.inference.release()


# --------------------------------------------------------------------- #
# Encoded vs raw classification (A08)
# --------------------------------------------------------------------- #


def test_raw_passthrough_by_content_type():
    assert _looks_encoded(b"\x01\x02", "x.bin", "audio/pcm") is False
    assert _looks_encoded(b"\x01\x02", "x.bin", "audio/L16; rate=16000") is False


def test_raw_passthrough_by_extension():
    assert _looks_encoded(b"\x01\x02", "chunk.pcm", "application/octet-stream") is False
    assert _looks_encoded(b"\x01\x02", "chunk.raw", None) is False


def test_encoded_extension_wins_over_generic_content_type():
    assert _looks_encoded(b"RIFFxxxx", "sample.wav", "application/octet-stream") is True
    assert _looks_encoded(b"\xff\xfbxx", "song.mp3", "application/octet-stream") is True


def test_magic_bytes_force_decoding_without_filename():
    assert _looks_encoded(b"RIFF0000WAVEfmt ", None, "application/octet-stream") is True
    assert _looks_encoded(b"OggS0000", None, None) is True
    assert _looks_encoded(b"\x00\x00\x00\x18ftypmp42", None, None) is True
    # Plain arbitrary bytes (no signature): still decoded by default.
    assert _looks_encoded(b"\x10\x20\x30\x40", None, None) is True


def test_unknown_bytes_default_to_decoding():
    # Generic type with no raw signal and no signature: decode, let ffmpeg
    # produce a clear error rather than transcribing garbage.
    assert _looks_encoded(b"\x01\x02\x03\x04", None, "application/octet-stream") is True


def test_raw_content_type_with_encoded_magic_stays_raw():
    # Explicit raw declaration wins over magic bytes.
    assert _looks_encoded(b"RIFF0000", "audio.wav", "audio/pcm") is False


def test_decode_audio_for_strategy_raw_shortcut(monkeypatch):
    called = []

    def fake_ffmpeg(cmd, data):
        called.append((cmd, data))
        return b"decoded"

    monkeypatch.setattr(server, "_run_bounded_ffmpeg", fake_ffmpeg)
    result = decode_audio_for_strategy(
        b"\x01\x02", filename="x.pcm", content_type="application/octet-stream"
    )
    assert result == b"\x01\x02"
    assert called == []


def test_decode_audio_for_strategy_encoded_invokes_ffmpeg(monkeypatch):
    called = []

    def fake_ffmpeg(cmd, data):
        called.append((cmd, data))
        return b"decoded-pcm"

    monkeypatch.setattr(server, "_run_bounded_ffmpeg", fake_ffmpeg)
    result = decode_audio_for_strategy(b"RIFF-data", filename="clip.wav", content_type="audio/wav")
    assert result == b"decoded-pcm"
    assert called and called[0][1] == b"RIFF-data"
    # Protocol allowlist restricts ffmpeg's input reachability.
    assert any(
        b"protocol_whitelist" in part.encode() or part == "-protocol_whitelist"
        for part in called[0][0]
        if isinstance(part, str)
    )


def test_encoded_upload_with_octet_stream_content_type_is_decoded(client, monkeypatch):
    """A08: generic binary MIME must not bypass decoding for .wav uploads."""
    captured: dict[str, Any] = {}

    def fake_decode(audio_bytes, *, filename=None, content_type=None):
        captured["filename"] = filename
        captured["content_type"] = content_type
        return audio_bytes

    monkeypatch.setattr(server, "decode_audio_for_strategy", fake_decode)
    wav_bytes = b"RIFF" + b"\x00" * 64

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", wav_bytes, "application/octet-stream")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 200
    assert captured["filename"] == "sample.wav"
    assert captured["content_type"] == "application/octet-stream"


# --------------------------------------------------------------------- #
# Bounded ffmpeg execution (A02)
# --------------------------------------------------------------------- #


def test_bounded_ffmpeg_stdout_budget_kills_process(monkeypatch):
    """A decoder producing beyond the PCM budget is killed and reaped."""
    monkeypatch.setattr(server, "MAX_DECODED_AUDIO_BYTES", 1024)
    monkeypatch.setenv("SIGNAL_ASR_FFMPEG_TIMEOUT", "2")
    script = (
        "import sys, time\n"
        "sys.stdout.buffer.write(b'x' * 4096)\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(RuntimeError, match="decoded audio exceeds maximum"):
        server._run_bounded_ffmpeg([server._sys_executable(), "-c", script], b"")


def test_bounded_ffmpeg_timeout_kills_process(monkeypatch):
    monkeypatch.setattr(server, "FFMPEG_TIMEOUT_SECONDS", 1.0)
    script = "import time; time.sleep(30)"
    with pytest.raises(RuntimeError, match="timed out"):
        server._run_bounded_ffmpeg([server._sys_executable(), "-c", script], b"")


def test_bounded_ffmpeg_stderr_is_bounded_and_reported(monkeypatch):
    monkeypatch.setattr(server, "MAX_FFMPEG_STDERR_BYTES", 128)
    script = (
        "import sys\nsys.stderr.buffer.write(b'E' * 8192)\nsys.stderr.buffer.flush()\nsys.exit(3)\n"
    )
    with pytest.raises(RuntimeError, match="could not decode") as error:
        server._run_bounded_ffmpeg([server._sys_executable(), "-c", script], b"")
    assert str(error.value) == "ffmpeg could not decode uploaded audio: " + "E" * 128


def test_bounded_ffmpeg_success_path(monkeypatch):
    script = "import sys\ndata = sys.stdin.buffer.read()\nsys.stdout.buffer.write(data.upper())\n"
    out = server._run_bounded_ffmpeg([server._sys_executable(), "-c", script], b"abc")
    assert out == b"ABC"


def test_bounded_ffmpeg_reaps_on_stdin_broken_pipe():
    script = (
        "import sys\n"
        "sys.stdin.buffer.read(1)\n"  # read a byte then exit non-zero fast
        "sys.exit(1)\n"
    )
    with pytest.raises(RuntimeError):
        server._run_bounded_ffmpeg([server._sys_executable(), "-c", script], b"xy" * 10)


# --------------------------------------------------------------------- #
# Streaming body-size limit (A01) — raw ASGI level
# --------------------------------------------------------------------- #


class _ScriptedASGI:
    """Minimal ASGI app recording what the middleware passes through."""

    def __init__(self):
        self.bodies: list[bytes] = []
        self.disconnects = 0
        self.response_started = False

    async def __call__(self, scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.request":
                self.bodies.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                self.disconnects += 1
                break
        self.response_started = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _run_asgi(app, messages, headers=None):
    """Drive an ASGI app with a scripted receive stream; collect response."""
    import asyncio

    queued = list(messages)
    state = {"response": None, "status": None}

    async def receive():
        if queued:
            return queued.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            state["status"] = message["status"]
            state["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            state["response"] = message.get("body", b"")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers or [],
        "query_string": b"",
    }
    asyncio.run(app(scope, receive, send))
    return state


def _chunk(payload: bytes, more: bool = True) -> dict:
    return {"type": "http.request", "body": payload, "more_body": more}


def test_body_limit_middleware_rejects_chunked_overflow_without_content_length():
    """No Content-Length at all: the cap is enforced while streaming (A01)."""
    limit = 1024
    app = _ScriptedASGI()
    wrapped = RequestBodyLimitMiddleware(app, max_body_bytes=limit)
    state = _run_asgi(wrapped, [_chunk(b"a" * 600), _chunk(b"b" * 600), _chunk(b"", more=False)])
    assert state["status"] == 413
    assert b"exceeds maximum" in state["response"]
    # The app must not have seen the whole body buffered.
    assert sum(len(chunk) for chunk in app.bodies) < 1200


def test_body_limit_middleware_allows_under_limit_chunked_body():
    app = _ScriptedASGI()
    wrapped = RequestBodyLimitMiddleware(app, max_body_bytes=1024)
    state = _run_asgi(wrapped, [_chunk(b"a" * 300), _chunk(b"", more=False)])
    assert state["status"] == 200
    assert b"".join(app.bodies) == b"a" * 300


def test_body_limit_middleware_exact_limit_is_allowed():
    app = _ScriptedASGI()
    wrapped = RequestBodyLimitMiddleware(app, max_body_bytes=100)
    state = _run_asgi(wrapped, [_chunk(b"x" * 100, more=False)])
    assert state["status"] == 200


def test_body_limit_middleware_limit_plus_one_is_rejected():
    app = _ScriptedASGI()
    wrapped = RequestBodyLimitMiddleware(app, max_body_bytes=100)
    state = _run_asgi(wrapped, [_chunk(b"x" * 101, more=False)])
    assert state["status"] == 413


def test_body_limit_middleware_rejects_dishonest_content_length_first():
    """Oversized declared Content-Length is rejected before reading the body."""
    app = _ScriptedASGI()
    wrapped = RequestBodyLimitMiddleware(app, max_body_bytes=100)
    state = _run_asgi(
        wrapped,
        [_chunk(b"x" * 10, more=False)],
        headers=[(b"content-length", b"9999")],
    )
    assert state["status"] == 413
    assert app.bodies == [], "no body may be read after a 413 on the header"


def test_body_limit_middleware_passes_non_http_scopes():
    import asyncio

    captured = {}

    async def latch_app(scope, receive, send):
        captured["type"] = scope["type"]

    wrapped = RequestBodyLimitMiddleware(latch_app, max_body_bytes=10)

    async def noop_receive():
        return {"type": "http.disconnect"}

    async def noop_send(message):
        pass

    asyncio.run(wrapped({"type": "lifespan"}, noop_receive, noop_send))
    # The lifespan scope reached the app untouched by the HTTP limit.
    assert captured.get("type") == "lifespan"


# --------------------------------------------------------------------- #
# HTTP endpoint behaviour
# --------------------------------------------------------------------- #


def test_health_check_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_without_file_field_returns_400(client):
    response = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert response.status_code == 400
    assert "multipart field 'file' is required" in response.json()["detail"]


def test_multiple_file_parts_return_400(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files=[
            ("file", ("a.pcm", b"\x00\x01", "audio/pcm")),
            ("file", ("b.pcm", b"\x00\x01", "audio/pcm")),
        ],
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 400
    assert "only one 'file' part" in response.json()["detail"]


def test_unknown_model_returns_400_without_loading(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x01", "audio/pcm")},
        data={"model": "bogus-engine"},
    )
    assert response.status_code == 400
    assert "unknown model" in response.json()["detail"]
    assert _factory_configs() == []


def test_json_response_format_parses_multipart_request(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 160, "audio/pcm")},
        data={"model": "sherpa_onnx", "response_format": "json"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "transcribed:sherpa_onnx"}


def test_text_response_format_returns_plain_text(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 160, "audio/pcm")},
        data={"model": "sherpa_onnx", "response_format": "text"},
    )
    assert response.status_code == 200
    assert response.text == "transcribed:sherpa_onnx"
    assert response.headers["content-type"].startswith("text/plain")


def test_verbose_json_response_includes_best_effort_timestamp_shape(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 160, "audio/pcm")},
        data={
            "model": "sherpa_onnx",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "transcribed:sherpa_onnx"
    assert payload["language"] is None
    assert payload["segments"] == []
    assert payload["timestamp_granularities"] == ["segment"]


def test_srt_and_vtt_are_explicitly_unsupported(client):
    for fmt in ("srt", "vtt"):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
            data={"model": "sherpa_onnx", "response_format": fmt},
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]


def test_response_format_is_case_insensitive_and_trimmed(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "sherpa_onnx", "response_format": "  JSON "},
    )
    assert response.status_code == 200


def test_malformed_temperature_returns_400(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "sherpa_onnx", "temperature": "hot"},
    )
    assert response.status_code == 400
    assert "temperature" in response.json()["detail"]


def test_actual_body_over_limit_rejected_413(client):
    """Oversized file content yields 413 even with an honest Content-Length."""
    big = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.pcm", big, "audio/pcm")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 413


def test_missing_model_defaults_to_local_asr_default(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
    )
    assert response.status_code == 200
    assert _factory_configs()[0]["config"].engine == server.default_engine_for_platform(
        server._detect_platform()
    )


def test_local_asr_default_model_uses_auto_engine(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "local_asr/default"},
    )
    assert response.status_code == 200
    assert _factory_configs()[0]["config"].engine == server.default_engine_for_platform(
        server._detect_platform()
    )


def test_raw_pcm_passthrough_by_extension(client, monkeypatch):
    decode_calls = []

    def fake_decode(audio_bytes, *, filename=None, content_type=None):
        decode_calls.append((filename, content_type))
        return audio_bytes

    monkeypatch.setattr(server, "decode_audio_for_strategy", fake_decode)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "application/octet-stream")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 200
    # Passed through untouched (no ffmpeg decode attempted for raw input).
    assert decode_calls == [("x.pcm", "application/octet-stream")]


def test_form_resources_are_closed_after_success(client, monkeypatch):
    """A09: parsed upload files must be closed on every request path."""
    closed: list[str] = []

    from starlette.datastructures import UploadFile

    original_close = UploadFile.close

    async def tracking_close(self):
        closed.append(self.filename)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", tracking_close)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("closeme.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 200
    assert "closeme.pcm" in closed


def test_form_resources_are_closed_after_validation_error(client, monkeypatch):
    closed: list[str] = []
    from starlette.datastructures import UploadFile

    original_close = UploadFile.close

    async def tracking_close(self):
        closed.append(self.filename)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", tracking_close)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("bad.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "sherpa_onnx", "response_format": "srt"},
    )
    assert response.status_code == 400
    assert "bad.pcm" in closed


def test_form_resources_are_closed_after_oversize(client, monkeypatch):
    closed: list[str] = []
    from starlette.datastructures import UploadFile

    original_close = UploadFile.close

    async def tracking_close(self):
        closed.append(self.filename)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", tracking_close)
    big = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("huge.pcm", big, "audio/pcm")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 413
    assert "huge.pcm" in closed


# --------------------------------------------------------------------- #
# Real ffmpeg decode path (bounded) — uses the real binary
# --------------------------------------------------------------------- #


@requires_ffmpeg
def test_real_ffmpeg_decodes_wav_to_pcm():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * 44100)  # 1 second of silence at 44.1k
    pcm = decode_audio_for_strategy(
        buffer.getvalue(), filename="tone.wav", content_type="audio/wav"
    )
    # Downsampled to 16 kHz mono s16le.
    assert len(pcm) in range(31000, 34000)


@requires_ffmpeg
def test_real_ffmpeg_rejects_garbage_with_clear_error():
    with pytest.raises(RuntimeError, match="could not decode"):
        decode_audio_for_strategy(
            b"definitely-not-audio" * 32, filename="x.wav", content_type="audio/wav"
        )


def test_missing_ffmpeg_returns_400(client, monkeypatch):
    def fake_broken_cmd(cmd, data):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(server, "_run_bounded_ffmpeg", fake_broken_cmd)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.wav", b"RIFFdata", "audio/wav")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 400
    assert "ffmpeg is required" in response.json()["detail"]


@requires_ffmpeg
def test_corrupt_audio_returns_400(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.wav", b"RIFF-garbage-not-a-real-wave", "audio/wav")},
        data={"model": "sherpa_onnx"},
    )
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


# --------------------------------------------------------------------- #
# Review-fix regressions (adversarial review findings)
# --------------------------------------------------------------------- #


@requires_ffmpeg
def test_real_ffmpeg_long_audio_does_not_deadlock():
    """Regression: >2s clips used to deadlock on pipe capacity.

    stdin was written fully before stdout was drained; once ffmpeg filled
    its output pipe (~64KiB = ~2s of PCM) it stopped reading input and the
    writer blocked forever. A 10-second clip must now decode well under
    the watchdog deadline.
    """
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * 44100 * 10)  # 10 seconds
    pcm = decode_audio_for_strategy(
        buffer.getvalue(), filename="long.wav", content_type="audio/wav"
    )
    assert len(pcm) >= 16_000 * 2 * 9  # ~10s of 16 kHz s16le


def test_decode_admission_returns_503_when_saturated(client, monkeypatch):
    """The HTTP decode path must actually acquire the decode slot."""
    monkeypatch.setattr(server._admission, "decode", server._BoundedSemaphore(1))
    monkeypatch.setattr(server._admission, "queue_timeout", 0.2)
    assert server._admission.decode.try_acquire(1)
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("x.wav", b"RIFFdata", "audio/wav")},
            data={"model": "sherpa_onnx"},
        )
        assert response.status_code == 503
    finally:
        server._admission.decode.release()


def test_invalid_language_returns_400(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
        data={"model": "sherpa_onnx", "language": "../../etc"},
    )
    assert response.status_code == 400
    assert "language" in response.json()["detail"]


def test_valid_language_forms_accepted(client):
    for language in ("en", "pt", "pt-BR", "zh"):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("x.pcm", b"\x00\x00" * 8, "audio/pcm")},
            data={"model": "sherpa_onnx", "language": language},
        )
        assert response.status_code == 200, language


def test_language_is_not_part_of_the_model_cache_key():
    """Distinct languages share one loaded strategy; language is per-call."""
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx", language="en")
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx", language="de")
    configs = _factory_configs()
    assert len(configs) == 1
    loads = [c for c in FakeFactory.calls if isinstance(c, tuple)]
    assert ("load_model", "sherpa_onnx") in loads
    # The per-call language reached transcribe.
    transcribes = [c for c in FakeFactory.calls if isinstance(c, tuple) and c[0] == "transcribe"]
    assert any(t[3] == "de" for t in transcribes)


def test_ffmpeg_protocol_whitelist_is_pipe_only(monkeypatch):
    """Regression: file/http protocols must NOT be whitelisted."""
    captured = {}

    def fake_run(cmd, data):
        captured["cmd"] = cmd
        return b"pcm"

    monkeypatch.setattr(server, "_run_bounded_ffmpeg", fake_run)
    decode_audio_for_strategy(b"RIFFxxxx", filename="a.wav", content_type="audio/wav")
    whitelist = captured["cmd"][captured["cmd"].index("-protocol_whitelist") + 1]
    assert whitelist.split(",") == ["pipe", "fd"]


def test_loading_reservations_are_pruned_after_publication():
    cache = server.StrategyCache(capacity=2)
    for i in range(3):
        cache.get_or_load((f"k{i}",), lambda i=i: FakeStrategy([], f"s{i}"), run=lambda s: "ok")
    with cache._lock:
        assert not cache._loading


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "bad"])
def test_ffmpeg_timeout_requires_finite_positive_value(monkeypatch, value):
    monkeypatch.setenv("SIGNAL_ASR_FFMPEG_TIMEOUT", value)
    with pytest.raises(RuntimeError, match="finite and positive"):
        server.validate_runtime_config()
    with pytest.raises(RuntimeError, match="finite and positive"):
        server._run_bounded_ffmpeg(["must-not-spawn"], b"")


@pytest.mark.parametrize(
    "content_type",
    [
        "audio/L16; rate=16000",
        "audio/pcm; rate=44100",
        "audio/raw; channels=2",
        "audio/x-raw; format=f32le",
        "audio/pcm; endian=big",
    ],
)
def test_raw_unsupported_formats_are_rejected(content_type):
    with pytest.raises(RuntimeError, match=r"unsupported|requires"):
        decode_audio_for_strategy(b"\x01\x02", filename="a.pcm", content_type=content_type)


def test_raw_decoded_budget_and_sample_alignment(monkeypatch):
    monkeypatch.setattr(server, "MAX_DECODED_AUDIO_BYTES", 4)
    assert decode_audio_for_strategy(b"1234", filename="a.pcm") == b"1234"
    with pytest.raises(RuntimeError, match="exceeds maximum"):
        decode_audio_for_strategy(b"123456", filename="a.pcm")
    with pytest.raises(RuntimeError, match="complete 16-bit"):
        decode_audio_for_strategy(b"123", filename="a.pcm")


def test_raw_decoded_budget_http_rejection(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_DECODED_AUDIO_BYTES", 2)
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.pcm", b"1234", "audio/pcm")}
    )
    assert response.status_code == 400
    assert not _factory_configs()


def test_cached_language_does_not_leak_into_auto_detection():
    for language in ("en", "de", None):
        server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx", language=language)
    assert len(_factory_configs()) == 1
    assert _factory_configs()[0]["config"].language is None
    calls = [c for c in FakeFactory.calls if isinstance(c, tuple) and c[0] == "transcribe"]
    assert [c[3] for c in calls] == ["en", "de", None]


def test_auto_and_explicit_default_share_identity(monkeypatch):
    monkeypatch.setattr(server, "_detect_platform", lambda: "linux")
    server.transcribe_audio_bytes(b"pcm", model="whisper-1")
    server.transcribe_audio_bytes(b"pcm", model="sherpa_onnx", platform="linux")
    assert len(_factory_configs()) == 1


def test_app_shutdown_closes_injected_cache_not_global():
    global_cache = server._strategy_cache
    global_strategy = FakeStrategy([], "global")
    global_cache.get_or_load(("global",), lambda: global_strategy, run=lambda s: None)
    cache = StrategyCache()
    with TestClient(server.create_app(cache=cache)) as client:
        response = client.post(
            "/v1/audio/transcriptions", files={"file": ("a.pcm", b"12", "audio/pcm")}
        )
        assert response.status_code == 200
        strategy = FakeStrategy.instances[-1]
        assert len(cache) == 1
    assert strategy.shutdown_called == 1
    assert cache._closed
    assert global_strategy.shutdown_called == 0
    assert server._strategy_cache is global_cache


def test_failed_model_load_is_closed_and_reservation_released(monkeypatch):
    cache = StrategyCache(capacity=1)

    def fail(self):
        raise RuntimeError("load failed")

    monkeypatch.setattr(FakeStrategy, "load_model", fail)
    with pytest.raises(RuntimeError, match="load failed"):
        server.transcribe_audio_bytes(b"pcm", cache=cache)
    assert FakeStrategy.instances[-1].shutdown_called == 1
    assert cache._resident == 0
    assert not cache._loading


def test_loading_and_retiring_models_consume_capacity():
    from concurrent.futures import ThreadPoolExecutor

    cache = StrategyCache(capacity=1)
    retiring = threading.Event()
    release = threading.Event()

    class Retiring(FakeStrategy):
        def shutdown(self):
            retiring.set()
            assert release.wait(5)
            super().shutdown()

    old = Retiring([], "old")
    cache.get_or_load(("old",), lambda: old, run=lambda s: None)
    created = []

    def create():
        created.append(True)
        return FakeStrategy([], "new")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.get_or_load, ("new",), create, run=lambda s: None)
        try:
            assert retiring.wait(5)
            with pytest.raises(OverloadedError):
                cache.get_or_load(("third",), create, run=lambda s: None)
            assert created == []
            assert cache._resident == 1
        finally:
            release.set()
        future.result(5)
    assert old.shutdown_called == 1
    assert created == [True]


def test_same_key_wait_has_deadline_and_failed_load_releases_waiters():
    from concurrent.futures import ThreadPoolExecutor

    cache = StrategyCache(capacity=1)
    cache._queue_timeout = 0
    loading = threading.Event()
    release = threading.Event()

    def fail():
        loading.set()
        assert release.wait(5)
        raise ValueError("failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.get_or_load, ("key",), fail, run=lambda s: None)
        try:
            assert loading.wait(5)
            with pytest.raises(OverloadedError, match="initialisation"):
                cache.get_or_load(("key",), lambda: pytest.fail("duplicate load"), run=str)
        finally:
            release.set()
        with pytest.raises(ValueError, match="failed"):
            future.result(5)
    assert cache.get_or_load(("key",), lambda: "recovered", run=str) == "recovered"


def test_waiter_rechecks_loading_after_publication_and_eviction(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    cache = StrategyCache(capacity=1)
    loading = threading.Event()
    release_load = threading.Event()
    waiting = threading.Event()
    woke = threading.Event()
    release_waiter = threading.Event()
    original_wait = cache._changed.wait
    creations = []

    def wait(timeout):
        waiting.set()
        result = original_wait(timeout)
        cache._changed.release()
        try:
            woke.set()
            assert release_waiter.wait(5)
        finally:
            cache._changed.acquire()
        return result

    monkeypatch.setattr(cache._changed, "wait", wait)

    def create():
        creations.append(True)
        loading.set()
        assert release_load.wait(5)
        return FakeStrategy([], "key")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_load, ("key",), create, run=lambda s: s)
        assert loading.wait(5)
        waiter = pool.submit(cache.get_or_load, ("key",), create, run=lambda s: s)
        try:
            assert waiting.wait(5)
            release_load.set()
            first_strategy = first.result(5)
            assert woke.wait(5)
            cache.get_or_load(("other",), lambda: FakeStrategy([], "other"), run=str)
            replacement = cache.get_or_load(("key",), create, run=lambda s: s)
            release_waiter.set()
            assert waiter.result(5) is replacement
            assert creations == [True, True]
            assert first_strategy.shutdown_called == 1
            assert cache._resident == 1
        finally:
            release_load.set()
            release_waiter.set()


def test_http_admission_precedes_parsing_and_survives_cancellation(monkeypatch):
    import asyncio

    import httpx
    from starlette.requests import Request

    admission = server._Admission()
    admission.request_limit = 1
    app = server.create_app(admission=admission)
    started = threading.Event()
    release = threading.Event()
    parsed = []
    original_form = Request.form

    def form(self, *args, **kwargs):
        parsed.append(True)
        return original_form(self, *args, **kwargs)

    def decode(data, **kwargs):
        started.set()
        assert release.wait(5)
        return data

    monkeypatch.setattr(Request, "form", form)
    monkeypatch.setattr(server, "decode_audio_for_strategy", decode)

    async def scenario():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            first = asyncio.create_task(
                client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("a.pcm", b"12", "audio/pcm")},
                )
            )
            try:
                async with asyncio.timeout(5):
                    while not started.is_set():
                        await asyncio.sleep(0)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                for _ in range(8):
                    response = await client.post(
                        "/v1/audio/transcriptions", content=b"not multipart"
                    )
                    assert response.status_code == 503
                    assert response.headers["retry-after"] == "5"
                assert parsed == [True]
                assert len(app.state.requests) == 1
                assert (await client.get("/health")).status_code == 200
            finally:
                release.set()
            await asyncio.gather(*app.state.requests)
            assert app.state.request_slots.try_acquire(0)
            app.state.request_slots.release()

    asyncio.run(scenario())
