"""WhisperKit ASR strategy — macOS / Apple Silicon only.

Runs a persistent ``whisperkit-cli serve`` process so the Core ML model stays
resident on the Apple Neural Engine (~0.5s/utterance).  Falls back to the
per-call CLI if the server cannot be started (~5s/utterance).

Install: brew install whisperkit-cli
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from typing import cast

from ..config import ASRConfig
from .base import ASRStrategy
from .shared import cleanup_temp_file, save_audio_to_wav

logger = logging.getLogger(__name__)

_CLI_CANDIDATES = ("whisperkit-cli", "argmax-cli")
_DEFAULT_MODEL = "large-v3-v20240930_turbo"
_DEFAULT_PORT = 50060
_MAX_RESPONSE_BYTES = 1024 * 1024
_SERVER_START_TIMEOUT = int(os.getenv("SIGNAL_WHISPERKIT_START_TIMEOUT", "600"))
# The Apple Neural Engine (cpuAndNeuralEngine) is fastest but hangs while
# loading the model on some machines (0% CPU, never binds its port). Probe the
# first attempt for this long, then fall back to cpuAndGPU which loads reliably.
_ANE_PROBE_TIMEOUT = 60
_DEFAULT_COMPUTE_UNITS = "cpuAndNeuralEngine"
_SUPPORTED_COMPUTE_UNITS = frozenset(
    {"all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly", "random"}
)


def _bounded_output(cmd: list[str], timeout: float, limit: int = _MAX_RESPONSE_BYTES) -> str:
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            with contextlib.suppress(ProcessLookupError):
                process.kill()

        timer = threading.Timer(timeout, expire)
        timer.start()
        try:
            assert process.stdout is not None
            output = bytearray()
            while chunk := cast(io.BufferedReader, process.stdout).read1(8192):
                if len(output) + len(chunk) > limit:
                    raise RuntimeError("WhisperKit subprocess output exceeds limit")
                output.extend(chunk)
            returncode = process.wait(timeout=timeout)
            if expired.is_set():
                raise RuntimeError("WhisperKit subprocess timed out")
            if returncode != 0:
                raise RuntimeError(f"WhisperKit subprocess exited with status {returncode}")
            return output.decode("utf-8", errors="replace")
        finally:
            timer.cancel()
            with contextlib.suppress(ProcessLookupError):
                if process.poll() is None:
                    process.kill()
            process.wait(timeout=5)
            timer.join(timeout=5)


class WhisperKitASRStrategy(ASRStrategy):
    """WhisperKit (Argmax Core ML) ASR — Apple Silicon only."""

    def __init__(self, config: ASRConfig, device: str, platform: str) -> None:
        super().__init__(config, device, platform)
        self._cli_path: str | None = None
        self._model_name = getattr(config, "whisperkit_model_name", None) or _DEFAULT_MODEL
        self._port = int(os.environ.get("SIGNAL_WHISPERKIT_PORT", _DEFAULT_PORT))
        self._host = "127.0.0.1"
        self._base_url = f"http://{self._host}:{self._port}"
        self._server: subprocess.Popen | None = None
        self._use_server = False
        # Compute units used for the CoreML encoder/decoder. May be overridden
        # (e.g. SIGNAL_WHISPERKIT_COMPUTE_UNITS=cpuAndGPU) and is updated to the
        # units that actually started successfully so the CLI fallback matches.
        configured_units = os.environ.get(
            "SIGNAL_WHISPERKIT_COMPUTE_UNITS", _DEFAULT_COMPUTE_UNITS
        ).strip()
        if configured_units not in _SUPPORTED_COMPUTE_UNITS:
            logger.warning(
                "Unsupported SIGNAL_WHISPERKIT_COMPUTE_UNITS=%r; using %s",
                configured_units,
                _DEFAULT_COMPUTE_UNITS,
            )
            configured_units = _DEFAULT_COMPUTE_UNITS
        self._compute_units = configured_units

    def load_model(self) -> None:
        if self.model is not None:
            return
        if self.platform != "macos":
            raise RuntimeError("WhisperKit requires macOS with supported Core ML hardware")

        self._cli_path = self._resolve_cli()
        logger.info("Loading WhisperKit '%s' (Core ML / ANE)...", self._model_name)
        self._use_server = self._start_server()
        if not self._use_server:
            logger.warning(
                "WhisperKit server unavailable; falling back to per-call CLI (~5s/utterance)"
            )
            self._warmup_cli()
        self.model = {
            "engine": "whisperkit",
            "model": self._model_name,
            "mode": "server" if self._use_server else "cli",
        }
        logger.info("✅ WhisperKit ASR ready (%s mode)", "server" if self._use_server else "cli")

    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        if not audio_chunk:
            return ""
        if self.model is None:
            self.load_model()
        language = language if language is not None else self.config.language
        path = save_audio_to_wav(audio_chunk)
        try:
            if self._use_server:
                text = self._transcribe_server(path, language)
                if text is not None:
                    return text
                logger.warning("WhisperKit server failed; using CLI fallback")
            return self._transcribe_cli(path, language)
        finally:
            cleanup_temp_file(path)

    # ------------------------------------------------------------------ #

    def _resolve_cli(self) -> str:
        for name in _CLI_CANDIDATES:
            p = shutil.which(name)
            if p:
                return p
        raise RuntimeError(
            f"WhisperKit CLI not found. Install with `brew install whisperkit-cli`. "
            f"Tried: {', '.join(_CLI_CANDIDATES)}"
        )

    def _start_server(self) -> bool:
        # Try the configured compute units first; if that is the ANE and it
        # hangs (a known CoreML issue on some Macs), fall back to cpuAndGPU.
        attempts = [self._compute_units]
        if self._compute_units == "cpuAndNeuralEngine":
            attempts.append("cpuAndGPU")
        for i, units in enumerate(attempts):
            is_last = i == len(attempts) - 1
            # Give a hung ANE load a short probe so we fall back quickly; the
            # final attempt keeps the full timeout for genuinely slow loads.
            timeout = _SERVER_START_TIMEOUT if is_last else _ANE_PROBE_TIMEOUT
            # If every server attempt fails, CLI fallback should use the last
            # attempted units rather than retrying a known-hanging ANE config.
            self._compute_units = units
            if self._spawn_and_wait(units, timeout):
                return True
            self._kill_server()
            if not is_last:
                logger.warning(
                    "WhisperKit '%s' units did not become ready in %ds "
                    "(likely an ANE load hang); retrying with cpuAndGPU",
                    units,
                    timeout,
                )
        return False

    def _spawn_and_wait(self, units: str, timeout: int) -> bool:
        import requests

        if not self._port_available():
            return False
        cli_path = self._cli_path
        assert cli_path is not None
        cmd = [
            cli_path,
            "serve",
            "--model",
            self._model_name,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--audio-encoder-compute-units",
            units,
            "--text-decoder-compute-units",
            units,
        ]
        try:
            self._server = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                return False
            try:
                if not self._owns_listener():
                    time.sleep(0.5)
                    continue
                response = requests.get(
                    f"{self._base_url}/health",
                    timeout=2,
                    allow_redirects=False,
                    stream=True,
                    proxies={"http": "", "https": ""},
                )
                try:
                    healthy = response.status_code == 200
                finally:
                    response.close()
                if healthy and self._owns_listener():
                    logger.info("✅ WhisperKit server ready (%s)", units)
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def _port_available(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((self._host, self._port))
            return True
        except OSError:
            return False

    def _owns_listener(self) -> bool:
        process = self._server
        if process is None or process.poll() is not None:
            return False
        try:
            output = _bounded_output(
                ["/usr/sbin/lsof", "-nP", f"-iTCP:{self._port}", "-sTCP:LISTEN", "-Fp"],
                timeout=2,
                limit=65536,
            )
            owners = {line for line in output.splitlines() if line.startswith("p")}
            return owners == {f"p{process.pid}"} and process.poll() is None
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return False

    def _kill_server(self) -> None:
        self._use_server = False
        process = self._server
        self._server = None
        if not process or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("WhisperKit server did not exit after SIGKILL")

    def _warmup_cli(self) -> None:
        path = save_audio_to_wav(b"\x00\x00" * 8000)
        try:
            result = subprocess.run(
                self._build_cli_cmd(path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_SERVER_START_TIMEOUT,
            )
            if result.returncode != 0:
                raise RuntimeError("WhisperKit CLI warmup failed")
        except Exception as exc:
            raise RuntimeError("WhisperKit CLI warmup failed") from exc
        finally:
            cleanup_temp_file(path)

    def _transcribe_server(self, path: str, language: str | None = None) -> str | None:
        import requests

        if not self._owns_listener():
            return None
        try:
            with open(path, "rb") as f:
                data = {"model": self._model_name}
                if language:
                    data["language"] = language
                r = requests.post(
                    f"{self._base_url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data=data,
                    timeout=120,
                    allow_redirects=False,
                    stream=True,
                    proxies={"http": "", "https": ""},
                )
            try:
                if r.status_code != 200:
                    return None
                body = bytearray()
                deadline = time.monotonic() + 120
                for chunk in r.iter_content(chunk_size=8192):
                    if time.monotonic() > deadline or len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        return None
                    body.extend(chunk)
                payload = json.loads(body)
            finally:
                r.close()
            if not isinstance(payload, dict):
                return None
            if "text" in payload:
                text = payload["text"]
            elif isinstance(payload.get("segments"), list):
                text = "".join(s["text"] for s in payload["segments"])
            else:
                return None
            if not isinstance(text, str):
                return None
            return text.strip()
        except Exception:
            return None

    def _build_cli_cmd(self, path: str, language: str | None = None) -> list[str]:
        cli_path = self._cli_path
        assert cli_path is not None
        cmd: list[str] = [
            cli_path,
            "transcribe",
            "--audio-path",
            path,
            "--model",
            self._model_name,
            "--without-timestamps",
            "--skip-special-tokens",
            "--audio-encoder-compute-units",
            self._compute_units,
            "--text-decoder-compute-units",
            self._compute_units,
        ]
        if language:
            cmd += ["--language", language]
        return cmd

    def _transcribe_cli(self, path: str, language: str | None = None) -> str:
        try:
            output = _bounded_output(self._build_cli_cmd(path, language), timeout=120)
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return lines[-1] if lines else ""
        except Exception as exc:
            raise RuntimeError("WhisperKit transcription failed") from exc

    def shutdown(self) -> None:
        self._kill_server()
        self.model = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()
