"""Tests for shared ASR utilities (A10)."""

from __future__ import annotations

import os
import wave
from unittest import mock

import pytest

from signal_asr.strategies.shared import cleanup_temp_file, save_audio_to_wav


def _read_wav(path: str) -> tuple[int, int, int, bytes]:
    with wave.open(path, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(1 << 30)


def test_save_audio_to_wav_writes_valid_header_and_frames(tmp_path):
    pcm = b"\x01\x02" * 100
    path = save_audio_to_wav(pcm, directory=tmp_path)
    try:
        channels, width, rate, frames = _read_wav(path)
        assert channels == 1
        assert width == 2
        assert rate == 16_000
        assert frames == pcm
    finally:
        cleanup_temp_file(path)


def test_save_audio_to_wav_cleans_up_when_header_write_fails(tmp_path, monkeypatch):
    """A10: a failure mid-write must not leak the temporary file."""
    created: list[str] = []

    real_named_temporary_file = save_audio_to_wav.__globals__["tempfile"].NamedTemporaryFile

    def tracking_factory(*args, **kwargs):
        handle = real_named_temporary_file(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(
        save_audio_to_wav.__globals__["tempfile"], "NamedTemporaryFile", tracking_factory
    )

    real_open = wave.open

    def failing_wave_open(*args, **kwargs):
        wf = real_open(*args, **kwargs)
        original_setframerate = wf.setframerate

        def fail_once(*f_args, **f_kwargs):
            original_setframerate(*f_args, **f_kwargs)
            raise OSError("simulated disk failure")

        wf.setframerate = fail_once
        return wf

    monkeypatch.setattr(wave, "open", failing_wave_open)

    with pytest.raises(OSError, match="simulated disk failure"):
        save_audio_to_wav(b"\x01\x02" * 10, directory=tmp_path)

    assert created, "no temporary file was created"
    assert not os.path.exists(created[0]), "temporary file leaked after write failure"


def test_save_audio_to_wav_cleans_up_when_frame_write_fails(tmp_path, monkeypatch):
    created: list[str] = []

    real_named_temporary_file = save_audio_to_wav.__globals__["tempfile"].NamedTemporaryFile

    def tracking_factory(*args, **kwargs):
        handle = real_named_temporary_file(*args, **kwargs)
        created.append(handle.name)
        return handle

    monkeypatch.setattr(
        save_audio_to_wav.__globals__["tempfile"], "NamedTemporaryFile", tracking_factory
    )

    class ExplodingWriter:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def setnchannels(self, n): ...

        def setsampwidth(self, w): ...

        def setframerate(self, r): ...

        def writeframes(self, data):
            raise OSError("disk full")

    monkeypatch.setattr(wave, "open", lambda *a, **k: ExplodingWriter())

    with pytest.raises(OSError, match="disk full"):
        save_audio_to_wav(b"\x01\x02" * 10, directory=tmp_path)

    assert created
    assert not os.path.exists(created[0])


def test_cleanup_temp_file_removes_existing_file(tmp_path):
    target = tmp_path / "deleteme.wav"
    target.write_bytes(b"data")
    cleanup_temp_file(str(target))
    assert not target.exists()


def test_cleanup_temp_file_tolerates_missing_file():
    cleanup_temp_file("/nonexistent/path/should/not/raise")


def test_cleanup_temp_file_tolerates_unlink_failure(tmp_path, monkeypatch):
    target = tmp_path / "locked.wav"
    target.write_bytes(b"data")
    real_unlink = os.unlink

    def failing_unlink(path, *args, **kwargs):
        if str(path) == str(target):
            raise PermissionError("file locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", failing_unlink)
    cleanup_temp_file(str(target))  # must not raise
    assert target.exists()  # untouched because unlink failed


def test_save_audio_to_wav_directory_parameter_is_honoured(tmp_path):
    path = save_audio_to_wav(b"\x00\x01", directory=tmp_path)
    assert os.path.dirname(path) == str(tmp_path)
    cleanup_temp_file(path)


def test_wav_independent_helpers_use_mock_lock(tmp_path):
    # Guard against regressions in cleanup semantics under concurrent calls.
    with mock.patch.object(os, "unlink", side_effect=PermissionError) as failing:
        cleanup_temp_file(str(tmp_path / "whatever.wav"))
    failing.assert_not_called()
