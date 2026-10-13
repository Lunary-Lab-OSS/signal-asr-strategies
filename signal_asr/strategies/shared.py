"""Shared utilities used by multiple ASR strategy implementations."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1  # mono


def save_audio_to_wav(audio_chunk: bytes, *, directory: str | Path | None = None) -> str:
    """Write raw PCM bytes to a temporary WAV file and return its path.

    The temporary file is removed again if writing the WAV header or frames
    raises, so callers never receive a path to a partially written or empty
    file they would then have to clean up.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=directory) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_chunk)
    except BaseException:
        # Header or frame writing failed: remove the temporary file and
        # re-raise so callers never see a half-written WAV.
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise
    return path


def cleanup_temp_file(path: str) -> None:
    """Best-effort removal of a temporary file."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except (OSError, PermissionError):
        pass
