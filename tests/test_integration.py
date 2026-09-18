"""Integration tests: real backends, real server, real container.

Enable with ``SIGNAL_ASR_INTEGRATION=1``. These tests download small real
models on first run and therefore need network access; the docker test
additionally needs a working Docker daemon.

Oracles (A15):
- The container runs its *actual* default command (not curl), on a
  non-default port driven by environment, and the health check matches.
- A real transcription request must return HTTP 200 with a JSON object
  containing a string `text` — connection errors, timeouts, and HTTP
  failures always fail. (When docker is unavailable the test skips;
  CI's integration-docker job provides the daemon.)
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave

import pytest

pytestmark = pytest.mark.integration

INTEGRATION = os.getenv("SIGNAL_ASR_INTEGRATION", "") == "1"
REQUIRES_INTEGRATION = pytest.mark.skipif(
    not INTEGRATION, reason="set SIGNAL_ASR_INTEGRATION=1 to run integration tests"
)

pytest.importorskip("numpy")


def _tone_pcm(seconds: float = 1.0, freq: float = 440.0) -> bytes:
    """Deterministic PCM tone (16 kHz mono s16le)."""
    rate = 16_000
    n = int(rate * seconds)
    return b"".join(
        struct.pack("<h", int(3200 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


def _tone_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(_tone_pcm(0.5))
    return buffer.getvalue()


SPOKEN_SENTENCE = "the quick brown fox jumps over the lazy dog"
SPOKEN_KEYWORDS = ("quick", "brown", "fox", "jumps", "over", "lazy", "dog")


def _speech_wav_bytes() -> bytes:
    """Synthesize real English speech with espeak-ng (22.05 kHz mono WAV)."""
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not available to synthesize speech")
    proc = subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "150", "--stdout", SPOKEN_SENTENCE],
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    wav = proc.stdout
    assert wav[:4] == b"RIFF", "espeak-ng did not emit a WAV"
    return wav


def _normalized_words(text: str) -> set[str]:
    return {w.strip(".,!?;:'\"") for w in text.lower().split()}


def _assert_keywords(transcript: str, minimum: int = 5) -> None:
    """The transcript must contain most spoken content words.

    A fuzzy floor (not exact equality) keeps the oracle robust to TTS
    voice/ASR drift across platforms while still proving that real
    speech was transcribed into real words.
    """
    words = _normalized_words(transcript)
    matched = [k for k in SPOKEN_KEYWORDS if k in words]
    assert len(matched) >= minimum, (
        f"transcript only matched {len(matched)}/{len(SPOKEN_KEYWORDS)} keywords "
        f"({matched!r}): {transcript!r}"
    )


@REQUIRES_INTEGRATION
def test_sherpa_cpu_transcribes_silence_and_tone(tmp_path):
    """Real sherpa-onnx CPU: load the pinned int8 model and transcribe.

    A tone/silence clip has no speech; the oracle asserts the call completes
    and yields a str without crashing the backend, and that a second call
    reuses the loaded recognizer.
    """
    pytest.importorskip("sherpa_onnx")
    from signal_asr.config import ASRConfig
    from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    strategy.load_model()
    assert strategy.is_loaded

    text = strategy.transcribe(_tone_pcm(0.5))
    assert isinstance(text, str)

    text2 = strategy.transcribe(b"\x00\x00" * 8000)
    assert isinstance(text2, str)


@REQUIRES_INTEGRATION
def test_real_speech_transcribes_words(tmp_path):
    """Real voice -> real ffmpeg decode -> real sherpa inference -> words.

    espeak-ng synthesizes spoken English; the transcript must contain
    most of the spoken content words (exact match observed locally, but
    the oracle tolerates TTS/ASR drift).
    """
    pytest.importorskip("sherpa_onnx")
    from signal_asr.config import ASRConfig
    from signal_asr.server import decode_audio_for_strategy
    from signal_asr.strategies.sherpa_onnx import SherpaOnnxASRStrategy

    strategy = SherpaOnnxASRStrategy(
        ASRConfig(engine="sherpa_onnx"), device="cpu", platform="linux", models_dir=tmp_path
    )
    strategy.load_model()

    pcm = decode_audio_for_strategy(
        _speech_wav_bytes(), filename="speech.wav", content_type="audio/wav"
    )
    # ~3.4 s of 16 kHz s16le mono
    assert 16_000 * 2 * 2 <= len(pcm) <= 16_000 * 2 * 6
    _assert_keywords(strategy.transcribe(pcm))


@REQUIRES_INTEGRATION
def test_local_server_end_to_end(tmp_path):
    """Boot the real app stack and drive the OpenAI-compatible API."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from signal_asr import server

    # Route the sherpa downloader at a scratch dir; requests stay bounded.
    old = os.environ.get("SIGNAL_ASR_MODELS_DIR")
    os.environ["SIGNAL_ASR_MODELS_DIR"] = str(tmp_path)
    try:
        with TestClient(server.create_app()) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("tone.wav", _tone_wav_bytes(), "audio/wav")},
                data={"model": "sherpa_onnx", "response_format": "verbose_json"},
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert isinstance(payload["text"], str)
            assert payload["language"] is None
    finally:
        if old is None:
            os.environ.pop("SIGNAL_ASR_MODELS_DIR", None)
        else:
            os.environ["SIGNAL_ASR_MODELS_DIR"] = old


@REQUIRES_INTEGRATION
def test_local_server_transcribes_real_speech(tmp_path):
    """Full HTTP path with real speech: upload -> decode -> infer -> JSON."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from signal_asr import server

    old = os.environ.get("SIGNAL_ASR_MODELS_DIR")
    os.environ["SIGNAL_ASR_MODELS_DIR"] = str(tmp_path)
    try:
        with TestClient(server.create_app()) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("speech.wav", _speech_wav_bytes(), "audio/wav")},
                data={"model": "sherpa_onnx", "response_format": "verbose_json"},
            )
            assert response.status_code == 200, response.text
            _assert_keywords(response.json()["text"])
    finally:
        if old is None:
            os.environ.pop("SIGNAL_ASR_MODELS_DIR", None)
        else:
            os.environ["SIGNAL_ASR_MODELS_DIR"] = old


@REQUIRES_INTEGRATION
def test_docker_image_builds_starts_and_serves():
    """A15: build the image, run its default command, and transcribe.

    The container must serve on the port chosen *via environment*, proving
    the Dockerfile CMD and health check agree (A14). Any connection error,
    timeout, or non-200 response fails the test. When espeak-ng is present
    the upload is real synthesized speech and the transcript must contain
    the spoken words; otherwise the tone oracle still proves the pipeline.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not available")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image = "signal-asr-integration:test"
    port = "18601"

    build = subprocess.run(
        ["docker", "build", "-t", image, repo_root],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert build.returncode == 0, build.stderr[-4000:]

    container = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            "signal-asr-it",
            "-p",
            f"{port}:18601",
            "-e",
            "SIGNAL_ASR_PORT=18601",
            image,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert container.returncode == 0, container.stderr
    container_id = container.stdout.strip()

    try:
        deadline = time.time() + 300
        last_error: Exception | None = None
        healthy = False
        while time.time() < deadline:
            alive = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
            )
            if alive.stdout.strip() != "true":
                logs = subprocess.run(
                    ["docker", "logs", container_id], capture_output=True, text=True
                )
                raise AssertionError(f"container exited early:\n{logs.stdout[-3000:]}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            time.sleep(1.0)
        assert healthy, f"server never became healthy; last error: {last_error}"

        try:
            wav = _speech_wav_bytes()
            transcript_check = True
        except pytest.skip.Exception:  # tone fallback where no TTS exists
            wav = _tone_wav_bytes()
            transcript_check = False
        boundary = "----signalasrintegration"
        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="model"\r\n\r\n'
                f"sherpa_onnx\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="tone.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + wav
            + (f"\r\n--{boundary}--\r\n").encode()
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode())
        assert isinstance(payload.get("text"), str)
        if transcript_check:
            _assert_keywords(payload["text"])
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=60)
