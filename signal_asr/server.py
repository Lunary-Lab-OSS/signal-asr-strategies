"""OpenAI-compatible local transcription server for signal_asr.

Resource-safety contract (enforced here, tested in tests/test_server.py):

- The total HTTP request body is capped *while streaming* from the ASGI
  receive channel, before multipart parsing buffers anything (A01).
- Each uploaded file is capped independently of multipart overhead.
- Decoded PCM output and ffmpeg diagnostics are bounded; ffmpeg is killed
  and reaped when a decoding budget is exceeded (A02).
- Concurrent decoding and inference are admitted through bounded slots with
  queue deadlines; overload produces 503, never unbounded queues (A02).
- Loaded strategies are cached per canonical engine identity with
  per-key single-flight initialisation and reference-counted disposal so a
  strategy is never closed while a request is still using it (A13).

NOTE: keep this module free of ``from __future__ import annotations`` —
FastAPI resolves endpoint parameter annotations against module globals,
and the framework imports live inside ``create_app``.
"""

import argparse
import asyncio
import contextlib
import io
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, cast

from .config import ASRConfig
from .strategies.factory import (
    ASRStrategyFactory,
    _detect_platform,
    canonical_engine,
    default_engine_for_platform,
)

logger = logging.getLogger(__name__)

OPENAI_AUTO_MODEL_NAMES = {
    "",
    "local_asr/default",
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
}
DEFAULT_MODEL_NAME = "local_asr/default"
SUPPORTED_RESPONSE_FORMATS = {"json", "text", "verbose_json"}
UNSUPPORTED_RESPONSE_FORMATS = {"srt", "vtt"}

# --- Upload and decoding budgets --------------------------------------- #

#: Maximum size of a single uploaded audio file (bytes).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
#: Maximum total HTTP request body (bytes). Slightly larger than the file
#: cap to allow multipart framing overhead for a maximum-size file.
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + 2 * 1024 * 1024
#: Maximum decoded PCM bytes (16 kHz mono s16le => 32,000 bytes/second).
MAX_DECODED_AUDIO_BYTES = 16_000 * 2 * int(os.getenv("SIGNAL_ASR_MAX_DECODED_SECONDS", "600"))
#: Maximum retained ffmpeg stderr (bytes).
MAX_FFMPEG_STDERR_BYTES = 64 * 1024
FFMPEG_TIMEOUT_SECONDS = float(os.getenv("SIGNAL_ASR_FFMPEG_TIMEOUT", "60"))

# --- Concurrency admission --------------------------------------------- #


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")
    return value


def _cache_capacity() -> int:
    return _positive_int_env("SIGNAL_ASR_CACHE_MAX", 8)


def _ffmpeg_timeout() -> float:
    try:
        value = float(os.getenv("SIGNAL_ASR_FFMPEG_TIMEOUT", str(FFMPEG_TIMEOUT_SECONDS)))
    except ValueError as exc:
        raise RuntimeError("SIGNAL_ASR_FFMPEG_TIMEOUT must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("SIGNAL_ASR_FFMPEG_TIMEOUT must be finite and positive")
    return value


def validate_runtime_config() -> None:
    """Fail fast on malformed server configuration (called at startup)."""
    _cache_capacity()
    _positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_DECODE", 2)
    _positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_INFERENCE", 2)
    _positive_int_env("SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS", 30)
    _positive_int_env("SIGNAL_ASR_MAX_DECODED_SECONDS", 600)
    _positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_REQUESTS", 4)
    _ffmpeg_timeout()


class _BoundedSemaphore:
    """``threading.Semaphore`` with a deadline-aware ``try_acquire``."""

    def __init__(self, permits: int) -> None:
        self._sem = threading.Semaphore(permits)

    def try_acquire(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        # Poll in small slices so a stuck caller cannot hold us past the
        # deadline without giving up (Semaphore.acquire has no timeout arg
        # compatible with KeyboardInterrupt handling on all platforms).
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._sem.acquire(blocking=False)
            if self._sem.acquire(blocking=True, timeout=min(remaining, 0.1)):
                return True

    def release(self) -> None:
        self._sem.release()


class OverloadedError(RuntimeError):
    """Raised when a decode/inference slot cannot be acquired in time."""


class _Admission:
    """Bounded admission for decoding and inference work."""

    def __init__(self) -> None:
        self.decode = _BoundedSemaphore(_positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_DECODE", 2))
        self.inference = _BoundedSemaphore(
            _positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_INFERENCE", 2)
        )
        self.queue_timeout = float(_positive_int_env("SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS", 30))
        self.request_limit = _positive_int_env("SIGNAL_ASR_MAX_CONCURRENT_REQUESTS", 4)

    def run(self, slot: _BoundedSemaphore, fn, /, *args, **kwargs):
        if not slot.try_acquire(self.queue_timeout):
            raise OverloadedError("server is at capacity; retry after the indicated delay")
        try:
            return fn(*args, **kwargs)
        finally:
            slot.release()


_admission = _Admission()

# --- Strategy cache (A12/A13) ------------------------------------------- #


class _CachedStrategy:
    """Cache entry tracking in-flight use for safe disposal."""

    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy
        self.in_flight = 0
        self.dispose_pending = False


class StrategyCache:
    """Single-flight LRU cache with a strict resident-model budget.

    Loading and retiring models retain a reservation until disposal completes.
    Only idle entries can be evicted; shutdown defers disposal of active entries.
    Initialisation and disposal run outside the map lock.
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity if capacity is not None else _cache_capacity()
        if self._capacity < 1:
            raise ValueError(f"cache capacity must be >= 1, got {self._capacity}")
        self._entries: OrderedDict[tuple[Any, ...], _CachedStrategy] = OrderedDict()
        self._loading: set[tuple[Any, ...]] = set()
        self._resident = 0
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._queue_timeout = float(_positive_int_env("SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS", 30))
        self._closed = False

    def _acquire_entry(
        self,
        key: tuple[Any, ...],
        create: Callable[[], Any],
    ) -> tuple[_CachedStrategy, bool]:
        """Return the entry for ``key``, creating it via ``create`` on miss.

        ``create`` runs at most once per key across concurrent callers.
        """
        deadline = time.monotonic() + self._queue_timeout
        with self._changed:
            while True:
                if self._closed:
                    raise RuntimeError("strategy cache is shut down")
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    entry.in_flight += 1
                    return entry, False
                if key not in self._loading:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OverloadedError("model initialisation is busy; retry later")
                self._changed.wait(remaining)
            victim = None
            if self._resident >= self._capacity:
                for victim_key, candidate in self._entries.items():
                    if candidate.in_flight == 0:
                        victim = candidate
                        del self._entries[victim_key]
                        break
                if victim is None:
                    raise OverloadedError("model cache is at capacity; retry later")
            else:
                self._resident += 1
            self._loading.add(key)
        created = None
        published = False
        try:
            if victim is not None:
                _close_strategy(victim.strategy)
            with self._lock:
                if self._closed:
                    raise RuntimeError("strategy cache is shut down")
            created = create()
            entry = _CachedStrategy(created)
            entry.in_flight = 1
            with self._lock:
                if self._closed:
                    raise RuntimeError("strategy cache is shut down")
                self._entries[key] = entry
                published = True
            return entry, True
        finally:
            if not published:
                _close_strategy(created)
            with self._changed:
                if not published:
                    self._resident -= 1
                self._loading.remove(key)
                self._changed.notify_all()

    def _dispose(self, entry: _CachedStrategy) -> None:
        _close_strategy(entry.strategy)
        entry.strategy = None
        with self._changed:
            self._resident -= 1
            self._changed.notify_all()

    def get_or_load(
        self,
        key: tuple[Any, ...],
        create: Callable[[], Any],
        *,
        run: Callable[[Any], Any],
    ) -> Any:
        """Return ``run(entry.strategy)`` for the cached strategy at ``key``."""
        entry, _ = self._acquire_entry(key, create)
        try:
            return run(entry.strategy)
        finally:
            self._release_entry(entry)

    def _release_entry(self, entry: _CachedStrategy) -> None:
        with self._lock:
            entry.in_flight -= 1
            dispose_now = entry.in_flight == 0 and entry.dispose_pending
        if dispose_now:
            self._dispose(entry)

    def close_all(self) -> None:
        """Close every cached strategy that is not currently in use."""
        with self._changed:
            if self._closed:
                return
            self._closed = True
            entries = []
            for entry in self._entries.values():
                entry.dispose_pending = True
                if entry.in_flight == 0:
                    entries.append(entry)
            self._entries.clear()
            self._changed.notify_all()
        for entry in entries:
            self._dispose(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _close_strategy(strategy: Any) -> None:
    shutdown = getattr(strategy, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            logger.warning("strategy shutdown failed", exc_info=True)


_strategy_cache = StrategyCache()


def reset_strategy_cache() -> None:
    """Close and drop all cached strategies (used by tests and shutdown)."""
    global _strategy_cache
    previous = _strategy_cache
    _strategy_cache = StrategyCache()
    previous.close_all()


def _engine_from_model(model: str) -> str:
    """Map an OpenAI-style ``model`` field to a canonical engine name.

    Raises ``ValueError`` for names that do not name a known backend; the
    HTTP layer turns that into a 400 rather than silently loading the
    default engine for a typo (A12).
    """
    override = os.getenv("SIGNAL_ASR_ENGINE")
    if override is not None:
        canonical = canonical_engine(override)
        if canonical is None:
            raise ValueError(f"SIGNAL_ASR_ENGINE names an unknown engine: {override!r}")
        return canonical

    normalized = (model or "").strip()
    if normalized.lower() in OPENAI_AUTO_MODEL_NAMES:
        return ""
    canonical = canonical_engine(normalized)
    if canonical is None:
        raise ValueError(f"unknown model {model!r}")
    return canonical


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    model: str = "",
    language: str | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    device: str | None = None,
    platform: str | None = None,
    model_loader: Any = None,
    models_dir: Any = None,
    cache: StrategyCache | None = None,
    admission: _Admission | None = None,
) -> str:
    """Transcribe raw 16 kHz mono 16-bit PCM bytes via the strategy factory.

    ``prompt`` and ``temperature`` are accepted for OpenAI API shape parity
    but are not currently interpreted by the strategy interface.

    A loaded strategy is cached and reused across requests, keyed by
    canonical engine/device/platform. Inference is admitted through
    a bounded slot with a queue deadline (503 semantics via
    :class:`OverloadedError` when saturated).
    """
    del prompt, temperature

    resolved_platform = platform or _detect_platform()
    engine = _engine_from_model(model) or default_engine_for_platform(resolved_platform)
    resolved_device = device or os.getenv("SIGNAL_ASR_DEVICE", "cpu")
    # NOTE: language is deliberately NOT part of the cache key — it is a
    # per-call transcribe parameter, not model identity. Keying on it let
    # an unauthenticated form field mint unbounded cache entries and force
    # a model load per value.
    cache_key = (engine, resolved_device, resolved_platform)
    active_cache = cache if cache is not None else _strategy_cache
    active_admission = admission if admission is not None else _admission

    def _create_strategy() -> Any:
        config = ASRConfig(engine=engine)
        factory = ASRStrategyFactory(
            config=config,
            device=resolved_device if resolved_device is not None else "cpu",
            platform=resolved_platform,
            model_loader=model_loader,
            models_dir=models_dir,
        )
        strategy = factory.create_strategy()
        try:
            if not strategy.is_loaded:
                strategy.load_model()
        except BaseException:
            _close_strategy(strategy)
            raise
        return strategy

    def _run(strategy: Any) -> str:
        result: object = active_admission.run(
            active_admission.inference, strategy.transcribe, audio_bytes, language=language
        )
        return str(result)

    text: str = active_cache.get_or_load(cache_key, _create_strategy, run=_run)
    return text


# --- Audio decoding (A02/A08) ------------------------------------------- #

RAW_AUDIO_CONTENT_TYPES = {
    "audio/pcm",
    "audio/raw",
    "audio/s16le",
    "audio/l16",
    "audio/x-raw",
}

#: Valid ``language`` form values: ISO-639-1/2 code with optional region.
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
RAW_AUDIO_EXTENSIONS = (".pcm", ".raw", ".s16le")
# Encoded container extensions that must go through ffmpeg even when the
# client sent a generic content type such as application/octet-stream (A08).
ENCODED_AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".webm",
    ".ogg",
    ".oga",
    ".opus",
    ".flac",
    ".aac",
    ".wma",
    ".aiff",
    ".amr",
)
# Magic-byte detection for encoded containers (used when the filename is
# absent or generic; a generic content type alone is not proof of PCM).
ENCODED_AUDIO_MAGIC = (
    b"RIFF",  # WAV
    b"ID3",  # MP3 with tag
    b"\xff\xfb",  # MP3 frame
    b"\xff\xf3",  # MP3 frame
    b"\xff\xe3",  # MP3 frame
    b"OggS",  # Ogg
    b"fLaC",  # FLAC
)


def _looks_encoded(audio_bytes: bytes, filename: str | None, content_type: str | None) -> bool:
    """Decide whether the upload must be decoded rather than passed through.

    Precedence (A08): an explicit raw content type or raw file extension
    means raw passthrough; otherwise a known encoded container extension or
    a container magic prefix forces decoding — including for generic
    ``application/octet-stream`` uploads, which prove nothing about the
    encoding. Bytes with no raw signal and no container signature are still
    treated as encoded so a mislabelled upload surfaces as a descriptive 400
    instead of garbage transcription.
    """
    normalized_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    normalized_name = (filename or "").lower()
    if normalized_type in RAW_AUDIO_CONTENT_TYPES:
        return False
    if normalized_name.endswith(RAW_AUDIO_EXTENSIONS):
        return False
    if normalized_name.endswith(ENCODED_AUDIO_EXTENSIONS):
        return True
    if any(audio_bytes.startswith(magic) for magic in ENCODED_AUDIO_MAGIC):
        return True
    # ISO base media files carry "ftyp" at byte offset 4.
    if len(audio_bytes) >= 8 and audio_bytes[4:8] == b"ftyp":
        return True
    # No raw signal and no recognised signature: assume an encoded
    # container and let ffmpeg produce a clear diagnostic if it is not.
    return True


def decode_audio_for_strategy(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> bytes:
    """Return raw 16 kHz mono signed-16 PCM bytes for strategy transcription.

    Decoding runs ffmpeg as a subprocess with bounded stdout (decoded PCM
    budget), bounded stderr retention, and an overall deadline; the process
    is killed and reaped when any budget is exceeded (A02).
    """
    if not _looks_encoded(audio_bytes, filename, content_type):
        if len(audio_bytes) > MAX_DECODED_AUDIO_BYTES:
            raise RuntimeError("decoded audio exceeds maximum allowed size")
        if len(audio_bytes) % 2:
            raise RuntimeError("raw PCM must contain complete 16-bit samples")
        parts = (content_type or "").lower().split(";")
        if parts[0].strip() == "audio/l16":
            raise RuntimeError("audio/L16 big-endian PCM is unsupported; use audio/s16le")
        supported = {"rate": "16000", "channels": "1", "format": "s16le"}
        for part in parts[1:]:
            name, separator, value = part.strip().partition("=")
            if not separator or supported.get(name.strip()) != value.strip().strip('"'):
                raise RuntimeError("raw PCM requires 16000 Hz mono s16le parameters")
        return audio_bytes

    # ffmpeg auto-detects the container from the piped bytes. Only pipe
    # input is reachable: file/network protocols are NOT whitelisted, so
    # crafted inputs cannot make ffmpeg read local files or contact
    # services (demuxers that follow embedded playlist/concat references
    # fail instead).
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "pipe,fd",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        return _run_bounded_ffmpeg(cmd, audio_bytes)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode uploaded audio formats") from exc


def ffmpeg_available() -> bool:
    """True when the ffmpeg binary can be found on PATH (test gating)."""
    import shutil

    return shutil.which("ffmpeg") is not None


def _run_bounded_ffmpeg(cmd: list[str], audio_bytes: bytes) -> bytes:
    """Run ffmpeg with bounded stdout/stderr and a hard deadline.

    stdin is fed from a dedicated writer thread and stderr from a drain
    thread, while the main thread incrementally reads stdout with budget
    checks — writing the whole input before reading output would deadlock
    on any clip longer than the pipe capacity once ffmpeg's output buffer
    fills. A watchdog thread enforces the deadline even when the child is
    silent. The child process is always terminated and reaped.
    """
    deadline = time.monotonic() + _ffmpeg_timeout()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stderr_chunks: list[bytes] = []
    stderr_bytes = 0
    stderr_done = threading.Event()
    timed_out = threading.Event()
    finished = threading.Event()
    stdin_error: list[BaseException] = []

    def _feed_stdin() -> None:
        stdin_stream = proc.stdin
        if stdin_stream is None:
            return
        try:
            stdin_stream.write(audio_bytes)
            stdin_stream.close()
        except (BrokenPipeError, OSError) as exc:
            # ffmpeg rejected the input early; the output below still
            # yields the diagnostic.
            stdin_error.append(exc)
            with contextlib.suppress(OSError, ValueError):
                if not stdin_stream.closed:
                    stdin_stream.close()

    def _drain_stderr() -> None:
        nonlocal stderr_bytes
        stderr_stream = proc.stderr
        assert stderr_stream is not None
        raw = getattr(stderr_stream, "raw", stderr_stream)
        try:
            while True:
                chunk = raw.read1(4096) if hasattr(raw, "read1") else stderr_stream.read(4096)
                if not chunk:
                    break
                remaining = MAX_FFMPEG_STDERR_BYTES - stderr_bytes
                if remaining > 0:
                    retained = chunk[:remaining]
                    stderr_chunks.append(retained)
                    stderr_bytes += len(retained)
        except (ValueError, OSError):
            pass
        finally:
            stderr_done.set()

    def _watchdog() -> None:
        while not finished.wait(0.05):
            if time.monotonic() > deadline:
                timed_out.set()
                _terminate(proc)
                return

    stdin_thread = threading.Thread(target=_feed_stdin, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    for thread in (stdin_thread, stderr_thread, watchdog_thread):
        thread.start()

    stdout_chunks: list[bytes] = []
    stdout_bytes = 0

    #: Diagnostic text retained for client-facing errors (bounded).
    max_error_detail = 512

    def _fail(reason: str, detail: str = "") -> RuntimeError:
        err = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        text = err or detail
        if len(text) > max_error_detail:
            text = text[:max_error_detail] + "…"
        suffix = f": {text}" if text else ""
        return RuntimeError(f"{reason}{suffix}")

    try:
        stdout_stream = proc.stdout
        assert stdout_stream is not None
        read1 = cast(io.BufferedReader, stdout_stream).read1
        while True:
            if timed_out.is_set():
                raise RuntimeError("ffmpeg timed out while decoding uploaded audio")
            # read1 returns whatever is available now, so budgets are
            # enforced while the stream is still flowing.
            chunk = read1(65536)
            if not chunk:
                break
            stdout_bytes += len(chunk)
            if stdout_bytes > MAX_DECODED_AUDIO_BYTES:
                _terminate(proc)
                raise RuntimeError(
                    "decoded audio exceeds maximum allowed size of "
                    f"{MAX_DECODED_AUDIO_BYTES} bytes (16 kHz mono PCM)"
                )
            stdout_chunks.append(chunk)

        if timed_out.is_set() or time.monotonic() > deadline:
            raise RuntimeError("ffmpeg timed out while decoding uploaded audio")
        stdin_thread.join(timeout=max(0.1, deadline - time.monotonic()))
        if stdin_thread.is_alive():  # pragma: no cover - writer stuck
            timed_out.set()
            _terminate(proc)
            raise RuntimeError("ffmpeg timed out while decoding uploaded audio")
        returncode = proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        stderr_done.wait(timeout=5)
        if returncode != 0:
            raise _fail("ffmpeg could not decode uploaded audio")
        return b"".join(stdout_chunks)
    except subprocess.TimeoutExpired as exc:
        timed_out.set()
        _terminate(proc)
        raise RuntimeError("ffmpeg timed out while decoding uploaded audio") from exc
    finally:
        finished.set()
        _terminate(proc)
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        stdin_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        watchdog_thread.join(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def _terminate(proc: subprocess.Popen) -> None:
    """Terminate and reap a child process, escalating to SIGKILL."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - extremely rare
            logger.warning("ffmpeg did not exit after SIGKILL")


def _sys_executable() -> str:
    """Return the running interpreter (used by tests to spawn helpers)."""
    import sys

    return sys.executable


def _verbose_json_response(
    *,
    text: str,
    language: str | None,
    timestamp_granularities: Iterable[str],
) -> dict[str, Any]:
    return {
        "text": text,
        "language": language,
        "duration": None,
        "segments": [],
        "timestamp_granularities": list(timestamp_granularities),
    }


# --- ASGI body-size middleware (A01) ------------------------------------ #


class RequestBodyLimitMiddleware:
    """Cap the total HTTP request body while it is being streamed.

    The ASGI ``receive`` channel is wrapped so each ``http.request`` message
    is counted *before* any framework multipart parsing buffers it. When the
    budget is exceeded the middleware answers 413 itself and collapses the
    stream, so unbounded chunked uploads cannot exhaust memory. Requests
    with an honest ``Content-Length`` above the cap are rejected before a
    single body byte is read.
    """

    def __init__(self, app, max_body_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = {"consumed": 0, "exceeded": False, "answered": False}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request" and not state["exceeded"]:
                state["consumed"] += len(message.get("body", b""))
                if state["consumed"] > self.max_body_bytes:
                    state["exceeded"] = True
                    await _send_413()
                    return {"type": "http.disconnect"}
            return message

        async def _send_413() -> None:
            if state["answered"]:
                return
            state["answered"] = True
            payload = json.dumps(
                {
                    "detail": (
                        f"request body exceeds maximum allowed size of {self.max_body_bytes} bytes"
                    )
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload})

        async def guarded_send(message):
            if state["answered"]:
                # Our 413 is already on the wire; drop the app's response.
                return
            await send(message)

        # Reject honest oversized Content-Length before reading any body.
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == b"content-length":
                try:
                    declared = int(header_value)
                except ValueError:
                    continue
                if declared > self.max_body_bytes:
                    state["exceeded"] = True
                    await _send_413()
                    return

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not state["answered"]:
                raise
            # A 413 is already on the wire; the app error that follows is
            # expected (the parser sees a collapsed stream) — log it quietly.
            logger.debug("app error after body-limit 413", exc_info=True)


@asynccontextmanager
async def _app_lifespan(app):
    """Close cached strategies when the server shuts down."""
    try:
        yield
    finally:
        app.state.stopping = True
        if app.state.requests:
            await asyncio.gather(*app.state.requests, return_exceptions=True)
        app.state.executor.shutdown(wait=True)
        app.state.cache.close_all()


def create_app(**kwargs: Any):
    """Create the FastAPI app.

    FastAPI and multipart parsing are optional dependencies. Install with
    ``signal-asr-strategies[server]`` to run the app. Keyword arguments are
    reserved for dependency injection in tests (cache/admission overrides).
    """
    validate_runtime_config()

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent.
        raise RuntimeError(
            "The signal ASR server requires optional dependencies. "
            'Install with: pip install "signal-asr-strategies[server]"'
        ) from exc
    app = FastAPI(
        title="signal-asr local transcription server",
        lifespan=_app_lifespan,
    )
    active_cache = kwargs.get("cache")
    active_cache = active_cache if active_cache is not None else StrategyCache()
    active_admission = kwargs.get("admission") or _admission
    kwargs = {**kwargs, "cache": active_cache, "admission": active_admission}
    app.state.cache = active_cache
    app.state.stopping = False
    app.state.requests = set()
    app.state.request_slots = _BoundedSemaphore(active_admission.request_limit)
    app.state.executor = ThreadPoolExecutor(
        max_workers=active_admission.request_limit, thread_name_prefix="signal-asr"
    )
    app.add_middleware(RequestBodyLimitMiddleware)

    async def run_worker(fn, /, *args, **worker_kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            app.state.executor, partial(fn, *args, **worker_kwargs)
        )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(request: Request):
        if app.state.stopping or not app.state.request_slots.try_acquire(0):
            raise HTTPException(
                status_code=503,
                detail="server is at capacity; retry later",
                headers={"Retry-After": "5"},
            )

        def finished(task):
            app.state.requests.discard(task)
            app.state.request_slots.release()
            if not task.cancelled():
                task.exception()

        task = asyncio.create_task(handle_request(request))
        app.state.requests.add(task)
        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def handle_request(request):
        async with request.form() as form:
            return await handle_form(form)

    async def handle_form(form):
        try:
            return await _handle_transcription_form(form)
        except OverloadedError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "5"}
            ) from exc

    async def _handle_transcription_form(form):
        files = form.getlist("file")
        if len(files) > 1:
            raise HTTPException(status_code=400, detail="only one 'file' part is allowed")
        file = files[0] if files else form.get("file")
        model_value = form.get("model")
        model = str(model_value) if model_value is not None else DEFAULT_MODEL_NAME
        if not file or not hasattr(file, "read"):
            raise HTTPException(status_code=400, detail="multipart field 'file' is required")

        # Fail fast on an unknown engine before reading any file bytes.
        try:
            _engine_from_model(model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        language_value = form.get("language")
        language = str(language_value) if language_value is not None else None
        if language is not None and not _LANGUAGE_RE.fullmatch(language):
            raise HTTPException(
                status_code=400,
                detail="language must be an ISO language code like 'en' or 'pt-BR'",
            )
        prompt_value = form.get("prompt")
        prompt = str(prompt_value) if prompt_value is not None else None
        response_format = str(form.get("response_format") or "json")
        temperature_value = form.get("temperature")
        if temperature_value not in (None, ""):
            try:
                temperature = float(temperature_value)
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="temperature must be a valid number",
                ) from exc
        else:
            temperature = None

        normalized_format = response_format.lower().strip()
        if normalized_format in UNSUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"response_format '{response_format}' is not supported yet",
            )
        if normalized_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"response_format must be one of {sorted(SUPPORTED_RESPONSE_FORMATS)}",
            )

        timestamp_granularities = (
            form.getlist("timestamp_granularities[]")
            or form.getlist("timestamp_granularities")
            or []
        )

        # Bound the file read itself to the cap + 1 byte so oversized files
        # are detected without buffering more than the limit (A01).
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_UPLOAD_BYTES:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(f"uploaded file exceeds maximum allowed size of {MAX_UPLOAD_BYTES} bytes"),
            )
        audio_bytes = b"".join(chunks)

        filename = getattr(file, "filename", None)
        content_type = getattr(file, "content_type", None)
        try:
            # A02: decoding is admitted through a bounded slot just like
            # inference — concurrent uploads cannot spawn unbounded ffmpeg
            # processes.
            audio_bytes = await run_worker(
                active_admission.run,
                active_admission.decode,
                decode_audio_for_strategy,
                audio_bytes,
                filename=filename,
                content_type=content_type,
            )
        except OverloadedError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "5"}
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            text = await run_worker(
                transcribe_audio_bytes,
                audio_bytes,
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                **kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OverloadedError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "5"}
            ) from exc

        if normalized_format == "text":
            return PlainTextResponse(text)
        if normalized_format == "verbose_json":
            return JSONResponse(
                _verbose_json_response(
                    text=text,
                    language=language,
                    timestamp_granularities=timestamp_granularities,
                )
            )
        return JSONResponse({"text": text})

    return app


async def _shutdown_cache() -> None:
    await asyncio.to_thread(reset_strategy_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the signal_asr local ASR server.")
    parser.add_argument("--host", default=os.getenv("SIGNAL_ASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SIGNAL_ASR_PORT", "18500")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("SIGNAL_ASR_LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent.
        raise SystemExit(
            "uvicorn is required for signal-asr-server. "
            'Install with: pip install "signal-asr-strategies[server]"'
        ) from exc

    uvicorn.run(
        "signal_asr.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
