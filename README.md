# signal-asr-strategies

Cross-platform speech recognition behind one strategy interface, built for the
[Signal Switchboard](https://github.com/Lunary-Lab-OSS) voice-agent stack.

Backends:

- **WhisperKit** (Core ML / Apple Neural Engine) on macOS
- **sherpa-onnx** with NVIDIA Parakeet TDT on Linux, WSL2, and CUDA
- **faster-whisper** (CTranslate2) as a cross-platform CPU/GPU fallback
- **NVIDIA NeMo Parakeet** source retained, but unsupported (dependency advisory blockers)
- An **OpenAI-compatible local transcription server** (`/v1/audio/transcriptions`)

## Install

```bash
pip install signal-asr-strategies            # core (numpy, requests)
pip install "signal-asr-strategies[server]"  # + fastapi/uvicorn local server
pip install "signal-asr-strategies[cpu,server]" # CPU sherpa + downloader + server
pip install "signal-asr-strategies[linux]"   # GPU-oriented dependencies; verify provider availability
pip install "signal-asr-strategies[wsl2]"    # same as [linux]
pip install "signal-asr-strategies[whisper]" # + faster-whisper
pip install "signal-asr-strategies[macos]"   # core only; brew install whisperkit-cli
```

Every extra that promises a cold-cache sherpa backend also installs the
`huggingface-hub` downloader it imports.

For reproducible source installs use `uv sync --locked --no-dev --extra cpu
--extra server` with uv 0.10.12. Every sherpa profile explicitly pins both
`sherpa-onnx==1.13.8` and `sherpa-onnx-core==1.13.8`: the installed upstream
sherpa wheel metadata requires that exact core version. The lock records
registry provenance and artifact hashes; installing an unrelated ONNX Runtime
GPU wheel does not add CUDA support to a CPU sherpa wheel.

## Usage

```python
from signal_asr import ASRComponent, ASRConfig

asr = ASRComponent(ASRConfig())  # platform auto-detected
asr.load_model()
text = asr.transcribe(raw_pcm_bytes)  # 16 kHz mono s16le PCM
```

The engine can be pinned with `ASRConfig(engine="whisper")` etc.; unknown
engine names raise `ValueError` instead of silently selecting a default.
Aliases like `SHERPA-ONNX` / `sherpa-onnx` / `sherpa_onnx` normalise to one
canonical backend.

### Device selection

- `whisper`: `cpu` (int8), `cuda`, `cuda:<index>` (float16), `auto`.
  Unsupported names raise; the compute type is validated per device.
- `sherpa_onnx`: `cpu` or `cuda` (pick a GPU with `CUDA_VISIBLE_DEVICES`;
  indexed devices are rejected because sherpa-onnx has no index API).
- `parakeet`: `cpu`, `cuda`, `cuda:<index>`, `auto` — the model is moved to
  the requested device after load.

### Local OpenAI-compatible server

```bash
signal-asr-server --host 127.0.0.1 --port 18500
# or via environment:
SIGNAL_ASR_HOST=127.0.0.1 SIGNAL_ASR_PORT=18500 signal-asr-server

curl http://127.0.0.1:18500/v1/audio/transcriptions \
  -F model=sherpa_onnx \
  -F file=@sample.wav
# {"text": "..."}
```

Supported: `response_format=json|text|verbose_json` (`srt`/`vtt` return a
clear 400). Unsupported today: `prompt`, `temperature` (accepted for API
shape parity, not interpreted) and timestamps (`verbose_json` returns
`duration: null`, empty segments).

Resource-safety contract (tested):

- The request body is capped **while streaming** (default 25 MiB file /
  27 MiB body) — oversized chunked uploads without `Content-Length` are
  rejected with 413 before multipart buffering.
- ffmpeg decoding runs with a decoded-PCM budget, bounded stderr, a
  pipe-only protocol allowlist, and a watchdog that kills and reaps
  runaway decoders.
- Concurrent decode and inference slots are bounded; overload returns
  **503** with `Retry-After` instead of queueing without limit.
- The `language` form field is validated (ISO code) and applied per
  request — it never keys the model cache.
- Strategies are cached per engine/device/platform with single-flight
  loading; evicted models are closed only when idle.

Native inference and model loading are not generally interruptible. Admission
and ffmpeg deadlines are not hard execution deadlines for these backends;
process isolation and an external supervisor are needed for hard runtime/RAM
limits. The server has no built-in authentication. Before non-loopback access,
deploy an authenticated TLS reverse proxy, network access controls, and
operator-selected rate/resource limits. Do not expose it directly to the Internet.

> **Behaviour change vs 0.1:** uploads sent as `application/octet-stream`
> with no (or a generic) filename are now treated as *encoded* audio and
> decoded by ffmpeg instead of passed through as raw PCM. Send raw PCM
> with an explicit `audio/pcm` content type or a `.pcm`/`.raw`/`.s16le`
> filename.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SIGNAL_ASR_HOST` / `SIGNAL_ASR_PORT` | `127.0.0.1` / `18500` | server bind |
| `SIGNAL_ASR_DEVICE` | `cpu` | default device for the server |
| `SIGNAL_ASR_ENGINE` | – | force one backend for all requests |
| `SIGNAL_ASR_CACHE_MAX` | `8` | loaded-strategy cache capacity (≥1) |
| `SIGNAL_ASR_MAX_CONCURRENT_DECODE` | `2` | concurrent ffmpeg decodes |
| `SIGNAL_ASR_MAX_CONCURRENT_INFERENCE` | `2` | concurrent transcriptions |
| `SIGNAL_ASR_QUEUE_TIMEOUT_SECONDS` | `30` | admission deadline before 503 |
| `SIGNAL_ASR_MAX_DECODED_SECONDS` | `600` | decoded-PCM budget (seconds) |
| `SIGNAL_ASR_FFMPEG_TIMEOUT` | `60` | per-request ffmpeg deadline |
| `SIGNAL_ASR_MODELS_DIR` | `models` | model download directory |
| `SIGNAL_WHISPERKIT_PORT` | `50060` | whisperkit-cli serve port |

Malformed values fail fast at startup with a clear error.

### Docker

```bash
docker build -t signal-asr .
docker run --rm -p 127.0.0.1:18500:18500 signal-asr
# port override stays consistent with the health check:
docker run --rm -p 127.0.0.1:18601:18601 -e SIGNAL_ASR_PORT=18601 signal-asr
```

The image runs as a non-root user, installs only runtime dependencies
(no tests), and includes the GPL-3.0 license.
Docker and CI select the same locked `cpu,server` runtime. Python base-image
and OS package digests are not pinned; dependency locking alone does not make
the entire image byte-reproducible. Model downloads are separate from uv.lock.

## Development

```bash
uv sync --locked --extra dev --extra server
uv run --locked --extra dev --extra server pytest tests/ -q -m "not integration" --cov=signal_asr --cov-fail-under=85
uv run --locked --extra dev --extra server mypy signal_asr
uvx ruff@0.15.14 check . && uvx ruff@0.15.14 format --check .
```

Integration tests (real models, real container):

```bash
SIGNAL_ASR_INTEGRATION=1 uv run --locked --extra dev --extra server --extra cpu pytest tests/test_integration.py -q
```

Platform support is unit-tested through injectable seams on all hosts
(Windows CUDA DLL discovery, macOS WhisperKit server lifecycle, Parakeet
pipeline). Running those paths on real Windows/macOS hardware is a
**manual** verification step today — the automated integration track
covers Linux CPU (sherpa-onnx + docker); contribute a runner to extend it.

Speech fixtures are synthesized with espeak-ng, not human recordings. CI
requires ffmpeg, espeak-ng, Docker, and sherpa imports and rejects skipped
speech/container cases rather than accepting tone-only success.

## Compatibility evidence

| Backend | Linux | WSL2 | Windows | macOS |
|---|---|---|---|---|
| sherpa-onnx | CPU integration track; CUDA unverified | CPU path; GPU unverified | Unverified | Unverified |
| faster-whisper | Unit seams only | Unverified | Unverified | Unverified |
| WhisperKit | Unsupported | Unsupported | Unsupported | Apple Silicon implementation; hardware unverified |
| Parakeet (NeMo) | Unsupported dependency integration | Unsupported | Unsupported | Unsupported |

Dependency CI audits core and every remaining extra (server, CPU, Whisper,
sherpa, macOS, Linux, WSL2, CUDA, all). A clean Linux profile is not a
clean bill of health for other platforms or model artifacts. Audit failures,
unknown packages, and registry/advisory errors remain blocking; no CVE ignores
are configured.

### Dependency migration (2026-09-17)

The `parakeet` extra is removed; `cuda` and `all` no longer install NeMo.
NeMo 3.0.0 (latest checked on PyPI) requires `hydra-core>1.3,<=1.3.2`
and `lightning>2.2.1,<=2.4.0` through its ASR extra. These constraints cannot
resolve with the advisory fixes without unsupported dependency overrides:

- [PYSEC-2026-3850](https://osv.dev/vulnerability/PYSEC-2026-3850):
  Hydra versions below 1.3.4 affected; fixed in 1.3.4.
- [PYSEC-2026-3972](https://osv.dev/vulnerability/PYSEC-2026-3972):
  Lightning through 2.6.0 affected; no fixed event recorded.
- [PYSEC-2026-3624](https://osv.dev/vulnerability/PYSEC-2026-3624):
  Lightning below 2.6.6 affected; fixed in 2.6.6.

Migrate to `cpu,server` with `engine="sherpa_onnx"` and a compatible ONNX
Parakeet model, or `whisper,server` with `engine="whisper"`. A NeMo checkpoint
is not interchangeable with an ONNX model. The retained `parakeet` source and
unit seams do not imply a supported installation; do not manually install the
vulnerable dependency set. Restoration requires upstream-compatible patched
constraints, a clean audit and real backend validation.

Local verification of this dependency migration: all ten CI runtime profile
exports passed `uv lock --check` and the exact strict pip-audit 2.9.0 CI command
on Linux. Python 3.13 unit suite: 250 passed, 1 skipped, 5 deselected; coverage
90.32%. Mypy passed. Python 3.12, container/model integration and non-Linux
hardware were not rerun for this migration; this is not release approval.

## License

GPL-3.0-or-later. Model weights keep their own upstream licenses.
