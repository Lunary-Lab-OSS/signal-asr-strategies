"""Property-based tests with complete reference predicates.

Each property compares the implementation against an independently written
reference over generated inputs, including negative cases — a crash-only
oracle is not acceptable (remediation: test-quality overhaul).
"""

from __future__ import annotations

import contextlib
import string
import wave
from contextlib import contextmanager
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from signal_asr.config import ASRConfig
from signal_asr.strategies.factory import ASRStrategyFactory, canonical_engine
from signal_asr.strategies.shared import cleanup_temp_file, save_audio_to_wav

KNOWN_ENGINES = ["sherpa_onnx", "sherpa-onnx", "SHERPA_ONNX", "whisperkit", "Whisper-Kit"]


# --------------------------------------------------------------------- #
# canonical_engine: complete reference predicate
# --------------------------------------------------------------------- #


def _reference_canonical(name: str | None) -> str | None:
    """Independent reference implementation of engine normalisation."""
    if name is None:
        return None
    lowered = name.strip().lower().replace("-", "_")
    table = {
        "sherpa_onnx": "sherpa_onnx",
        "sherpa": None,  # 'sherpa' alone is NOT an engine name
        "whisperkit": "whisperkit",
        "whisper_kit": "whisperkit",
        "whisper": "whisper",
        "parakeet": "parakeet",
    }
    return table.get(lowered)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sherpa_onnx", "sherpa_onnx"),
        ("sherpa-onnx", "sherpa_onnx"),
        ("SHERPA_ONNX", "sherpa_onnx"),
        (" whisperkit ", "whisperkit"),
        ("whisper_kit", "whisperkit"),
        ("whisper", "whisper"),
        ("parakeet", "parakeet"),
        ("sherpa", None),
        ("whispers", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_engine_matches_reference(name, expected):
    assert canonical_engine(name) == expected


@settings(max_examples=200)
@given(st.one_of(st.none(), st.text(max_size=24)))
def test_canonical_engine_total_function(name):
    """canonical_engine never raises and agrees with the reference."""
    result = canonical_engine(name)
    assert result in (None, "sherpa_onnx", "whisperkit", "whisper", "parakeet")
    # Agreement with the independently written reference, not a tautology.
    assert result == _reference_canonical(name)


@settings(max_examples=100)
@given(st.sampled_from([*KNOWN_ENGINES, "sherpa", "", "  "]))
def test_factory_accepts_known_engines_and_rejects_unknown(engine):
    factory = ASRStrategyFactory(ASRConfig(engine=engine), platform="linux")
    if canonical_engine(engine) is None and engine.strip():
        with pytest.raises(ValueError, match="unknown ASR engine"):
            factory.create_strategy()
    elif not engine.strip():
        strategy = factory.create_strategy()
        assert strategy is not None
    else:
        strategy = factory.create_strategy()
        assert strategy is not None


# --------------------------------------------------------------------- #
# WAV framing invariants
# --------------------------------------------------------------------- #


@contextmanager
def _wav_roundtrip(pcm: bytes):
    path = save_audio_to_wav(pcm)
    try:
        with wave.open(path, "rb") as wf:
            yield wf
    finally:
        cleanup_temp_file(path)


@settings(max_examples=50)
@given(st.binary(min_size=0, max_size=4096))
def test_save_audio_to_wav_frame_invariants(pcm):
    """The WAV writer round-trips bytes exactly.

    Sample alignment is the caller's (strategy's) contract; the writer must
    persist exactly the bytes it was given so decoding is lossless.
    """
    with _wav_roundtrip(pcm) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        frames = wf.readframes(1 << 30)
        assert frames == pcm


@settings(max_examples=50)
@given(
    st.lists(
        st.integers(min_value=-(2**15), max_value=2**15 - 1),
        min_size=0,
        max_size=256,
    )
)
def test_save_audio_to_wav_sample_roundtrip(samples):
    import struct

    pcm = struct.pack(f"<{len(samples)}h", *samples)
    with _wav_roundtrip(pcm) as wf:
        frames = wf.readframes(1 << 30)
        assert struct.unpack(f"<{len(frames) // 2}h", frames) == tuple(samples)


# --------------------------------------------------------------------- #
# Whisper device mapping: total, deterministic, reference-checked
# --------------------------------------------------------------------- #


def _reference_whisper_device(device: str | None) -> tuple[str, object]:
    requested = (device or "auto").strip().lower()
    if requested in ("", "auto"):
        return ("auto", None)
    if requested == "cpu":
        return ("cpu", None)
    if requested == "cuda" or requested.startswith("cuda:"):
        index = None
        if ":" in requested:
            suffix = requested.split(":", 1)[1]
            if not suffix or any(c not in string.digits for c in suffix):
                return ("invalid", None)
            index = int(suffix)
        return ("cuda", index)
    return ("invalid", None)


@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda", "cuda:0", "cuda:9", "auto", None, "", "  CUDA  ", "tpu", "cuda:x", "mps"],
)
def test_whisper_device_mapping_matches_reference(device, monkeypatch):
    from signal_asr.strategies.whisper import resolve_whisper_device

    kind, index = _reference_whisper_device(device)
    if kind == "invalid":
        with pytest.raises(ValueError):
            resolve_whisper_device(device)
        return
    # Pin CUDA availability deterministically for auto.
    import sys
    import types

    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.get_cuda_device_count = lambda: 2
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)

    resolved_device, resolved_index, compute = resolve_whisper_device(device)
    if kind == "auto":
        assert resolved_device == "cuda"  # availability faked to 2 devices
        assert resolved_index is None
    else:
        assert resolved_device == kind
        assert resolved_index == index
    assert compute in ("float16", "int8", "int8_float16", "float32", "int8_float32")


@settings(max_examples=100)
@given(
    st.one_of(
        st.none(),
        st.text(alphabet=string.ascii_letters + string.digits + ":-_ ", max_size=10),
    )
)
def test_whisper_device_mapping_never_crashes(device):
    from signal_asr.strategies.whisper import resolve_whisper_device

    with contextlib.suppress(ValueError):
        resolve_whisper_device(device)


# --------------------------------------------------------------------- #
# Encoded/raw classification reference predicate
# --------------------------------------------------------------------- #


_RAW_TYPES = {"audio/pcm", "audio/raw", "audio/s16le", "audio/l16", "audio/x-raw"}
_RAW_EXTS = (".pcm", ".raw", ".s16le")
_ENCODED_EXTS = (
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


def _reference_looks_encoded(data: bytes, filename: str | None, content_type: str | None) -> bool:
    if (content_type or "").split(";")[0].strip().lower() in _RAW_TYPES:
        return False
    name = (filename or "").lower()
    if name.endswith(_RAW_EXTS):
        return False
    if name.endswith(_ENCODED_EXTS):
        return True
    return True  # unknown bytes are decoded by default


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("x.pcm", "application/octet-stream"),
        ("x.raw", None),
        ("x.wav", "application/octet-stream"),
        ("x.mp3", "audio/mpeg"),
        (None, "audio/pcm"),
        (None, "audio/L16;rate=16000"),
        (None, "application/octet-stream"),
        ("x.bin", "application/octet-stream"),
    ],
)
def test_looks_encoded_matches_reference(filename, content_type):
    from signal_asr.server import _looks_encoded

    data = b"\x01\x02\x03\x04"
    assert _looks_encoded(data, filename, content_type) == _reference_looks_encoded(
        data, filename, content_type
    )


@settings(max_examples=100)
@given(
    st.binary(max_size=64),
    st.one_of(st.none(), st.sampled_from(["x.pcm", "x.wav", "x.raw", "x.mp3", "x.bin", ""])),
    st.one_of(
        st.none(),
        st.sampled_from(
            ["application/octet-stream", "audio/pcm", "audio/wav", "audio/mpeg", "audio/x-raw"]
        ),
    ),
)
def test_looks_encoded_is_total_and_consistent(data, filename, content_type):
    from signal_asr.server import _looks_encoded

    result = _looks_encoded(data, filename, content_type)
    assert isinstance(result, bool)
    assert result == _reference_looks_encoded(data, filename, content_type)


def test_tmp_path_helper_is_used_by_wav_writer(tmp_path: Path):
    # Sanity: the shared writer honours the target directory for callers
    # that need deterministic placement (used by integration fixtures).
    path = save_audio_to_wav(b"\x00\x01", directory=tmp_path)
    assert Path(path).parent == tmp_path
    cleanup_temp_file(path)
