# signal-asr-strategies

[![ci](https://github.com/Lunary-Lab-OSS/signal-asr-strategies/actions/workflows/ci.yml/badge.svg)](https://github.com/Lunary-Lab-OSS/signal-asr-strategies/actions/workflows/ci.yml)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Local speech-to-text for Python with one interface for every platform. It picks the best on-device engine for the host (Apple Neural Engine on macOS, NVIDIA Parakeet through ONNX Runtime on Linux and WSL2, and faster-whisper anywhere), and it ships with an OpenAI-compatible transcription server.

Part of the [Signal Switchboard](https://github.com/Lunary-Lab-OSS) voice-agent stack.

## Features

- **One API, several backends.** Swap engines with a config value. Engine names are normalised (`sherpa-onnx`, `SHERPA_ONNX`, and `sherpa_onnx` all mean the same backend), and an unknown name raises a clear error.
- **Platform-aware defaults.** WhisperKit (Core ML) on macOS; sherpa-onnx with NVIDIA Parakeet TDT on Linux, WSL2, and Windows.
- **OpenAI-compatible server.** A `POST /v1/audio/transcriptions` endpoint works with existing OpenAI SDKs and tools. It accepts WAV, MP3, and other ffmpeg-decodable formats, as well as raw PCM.
- **Built for real workloads.** Streaming upload limits, bounded ffmpeg decoding, admission control with `503 Retry-After` under load, and a reference-counted model cache with single-flight loading.
- **Reproducible.** Fully locked dependencies (`uv.lock`), pinned model revisions, and a non-root container image with a health check.

## Backends

| Engine | Name | Runs on | Devices |
|---|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + NVIDIA Parakeet TDT 0.6B v3 | `sherpa_onnx` | Linux, WSL2, Windows | `cpu`, `cuda` |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) | `whisper` | Linux, WSL2, Windows, macOS | `cpu`, `cuda`, `cuda:<n>`, `auto` |
| [WhisperKit](https://github.com/argmaxinc/WhisperKit) (Core ML / ANE) | `whisperkit` | macOS (Apple Silicon) | Neural Engine |

With no engine set, macOS uses `whisperkit` and every other platform uses `sherpa_onnx`. To pick a GPU for sherpa-onnx, set `CUDA_VISIBLE_DEVICES`.

## Installation

Requires Python 3.12+ and `ffmpeg` on `PATH` (the server uses it for encoded audio).

Install straight from GitHub with the extras you need:

```bash
# CPU transcription (sherpa-onnx) + OpenAI-compatible server
pip install "signal-asr-strategies[cpu,server] @ git+https://github.com/Lunary-Lab-OSS/signal-asr-strategies.git"
```

| Extra | Adds |
|---|---|
| `cpu` | sherpa-onnx (CPU) and the Hugging Face model downloader |
| `server` | FastAPI, Uvicorn, and multipart upload support |
| `whisper` | faster-whisper |
| `linux`, `wsl2` | sherpa-onnx with ONNX Runtime GPU |
| `cuda` | sherpa-onnx with ONNX Runtime GPU, plus faster-whisper |
| `all` | Every backend that can be installed with pip |
| `macos` | Nothing extra; install WhisperKit with `brew install whisperkit-cli` |

For a reproducible environment from source, use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Lunary-Lab-OSS/signal-asr-strategies.git
cd signal-asr-strategies
uv sync --locked --extra cpu --extra server
```

## Quick start

```python
from signal_asr import ASRComponent, ASRConfig

asr = ASRComponent(ASRConfig())          # engine auto-selected for this platform
asr.load_model()                         # optional; the first transcribe() loads lazily

text = asr.transcribe(pcm_bytes)         # 16 kHz, mono, signed 16-bit little-endian PCM
print(asr.platform, "->", text)

asr.close()
```

Pin an engine and device:

```python
asr = ASRComponent(ASRConfig(engine="whisper", language="en"), device="cuda")
```

The Parakeet model is fetched at a pinned revision on first use into `./models`, or into `SIGNAL_ASR_MODELS_DIR` if set. Every strategy is safe to call from several threads: concurrent first calls trigger exactly one model load.

## Transcription server

```bash
signal-asr-server --host 127.0.0.1 --port 18500
```

```bash
curl http://127.0.0.1:18500/v1/audio/transcriptions \
  -F model=sherpa_onnx \
  -F file=@sample.wav
# {"text": "..."}
```

Or use the official OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18500/v1", api_key="unused")
with open("sample.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="sherpa_onnx", file=f)
print(result.text)
```

**Request fields**

| Field | Notes |
|---|---|
| `file` | Required. Encoded audio is decoded with ffmpeg. For raw 16 kHz s16le PCM, send `Content-Type: audio/pcm` or a `.pcm` / `.raw` / `.s16le` filename. |
| `model` | `sherpa_onnx`, `whisper`, or `whisperkit` (aliases accepted). |
| `language` | Optional ISO language code such as `en` or `pt-BR`, applied per request. |
| `response_format` | `json` (default), `text`, or `verbose_json`. |
| `prompt`, `temperature` | Accepted for OpenAI SDK compatibility. |

**Status codes**

| Code | Meaning |
|---|---|
| `400` | Invalid request: undecodable audio, malformed form, unknown model, or unsupported format |
| `413` | Upload larger than 25 MiB |
| `503` | All decode or inference slots are busy (`Retry-After: 5`), or ffmpeg is unavailable |

`GET /health` returns `200` when the server is ready.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SIGNAL_ASR_HOST` / `SIGNAL_ASR_PORT` | `127.0.0.1` / `18500` | Bind address |
| `SIGNAL_ASR_DEVICE` | `cpu` | Device for loaded models |
| `SIGNAL_ASR_ENGINE` | auto | Force one engine for every request |
| `SIGNAL_ASR_MODELS_DIR` | `models` | Model download directory |
| `SIGNAL_ASR_CACHE_MAX` | `8` | Loaded-model cache capacity |
| `SIGNAL_ASR_MAX_CONCURRENT_DECODE` | `2` | Concurrent ffmpeg decodes |
| `SIGNAL_ASR_MAX_CONCURRENT_INFERENCE` | `2` | Concurrent transcriptions |
| `SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS` | `30` | Time a request may wait for a slot |
| `SIGNAL_ASR_MAX_DECODED_SECONDS` | `600` | Maximum decoded audio length per request |
| `SIGNAL_ASR_FFMPEG_TIMEOUT` | `60` | Per-request decode deadline (seconds) |
| `SIGNAL_WHISPERKIT_PORT` | `50060` | Port for the managed `whisperkit-cli` process |

Invalid values are rejected at startup with a descriptive error.

The server binds to loopback by default and has no built-in authentication. For access from other machines, put it behind an authenticated TLS reverse proxy.

### Docker

```bash
docker build -t signal-asr .
docker run --rm -p 127.0.0.1:18500:18500 -v asr-models:/models signal-asr
```

The image runs as an unprivileged user, installs only the locked `cpu,server` runtime, and has a health check on `/health`. If you set `SIGNAL_ASR_PORT`, the health check follows it. Mount `/models` to keep downloaded models between runs.

## Architecture

```
signal_asr/
├── component.py        ASRComponent: high-level facade
├── config.py           ASRConfig / ModelConfig dataclasses (dependency-free)
├── server.py           OpenAI-compatible FastAPI app and CLI entry point
└── strategies/
    ├── base.py         ASRStrategy interface + thread-safe lazy loading
    ├── factory.py      Platform detection and engine selection
    ├── sherpa_onnx.py  NVIDIA Parakeet TDT through sherpa-onnx
    ├── whisper.py      faster-whisper
    └── whisperkit.py   WhisperKit server lifecycle (macOS)
```

To add a backend, subclass `ASRStrategy`, implement `load_model()` and `transcribe()` (override `shutdown()` if it holds resources), and register it in `ASRStrategyFactory`.

## Development

```bash
uv sync --locked --extra dev --extra server --extra cpu
uv run --locked --extra dev --extra server --extra cpu pytest tests/ -m "not integration" --cov=signal_asr
uv run --locked --extra dev --extra server --extra cpu mypy signal_asr
uvx ruff@0.15.14 check . && uvx ruff@0.15.14 format --check .
```

End-to-end tests download a real model and run real speech through the library, the server, and the container image:

```bash
SIGNAL_ASR_INTEGRATION=1 uv run --locked --extra dev --extra server --extra cpu pytest tests/test_integration.py
```

CI runs linting, type checking, unit tests with a coverage gate, a package build, a dependency audit of every extra, and the container integration test on each pull request.

## Contributing

Issues and pull requests are welcome. Please run the checks above before you open a PR. To report a security issue, follow the [security policy](https://github.com/Lunary-Lab-OSS/.github/blob/main/.github/SECURITY.md).

## License

[GPL-3.0-or-later](LICENSE). Model weights keep their own upstream licenses.
