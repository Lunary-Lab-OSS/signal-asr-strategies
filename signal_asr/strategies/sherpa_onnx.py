"""Sherpa-ONNX ASR strategy — cross-platform (Windows/Linux/WSL2).

Default backend for non-Apple platforms.  Uses
``csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`` from HuggingFace.

Platform notes:
- Windows/CUDA: handles CUDA DLL discovery and cuDNN pre-loading automatically.
- Linux / WSL2: imports torch first so the shared CUDA/cuDNN runtime is already
  resident before sherpa_onnx is imported (avoids duplicate CUDA init).
- CPU: supported; set ``device="cpu"`` in config.

sherpa-onnx ``OfflineRecognizer`` does not document thread-safe concurrent
``decode_stream`` use, so each strategy instance serialises transcription
through :attr:`_decode_lock` (A13/D: per-backend concurrency policy).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

from ..config import ASRConfig
from .base import ASRStrategy

logger = logging.getLogger(__name__)

_MODEL_ID = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
_MODEL_REVISION = "2bda32ec70b097a55adaa07d9a7173915b43cc78"
_REQUIRED_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")

# CUDA runtime DLL names grouped by major runtime version. Discovery keys off
# the actual DLLs present in a directory, never off directory-name guessing.
_CUDA_RUNTIME_DLLS: dict[int, tuple[str, ...]] = {
    12: ("cudart64_12.dll",),
    11: ("cudart64_110.dll", "cudart64_11.dll"),
}
_CUDNN_DLLS = ("cudnn64_9.dll", "cudnn64_8.dll")


class SherpaOnnxASRStrategy(ASRStrategy):
    """Sherpa-ONNX (Parakeet TDT 0.6B v3 INT8) ASR strategy.

    Works on Windows (CUDA), Linux, and WSL2.  Falls back to CPU only when
    ``device`` does not start with ``"cuda"``.
    """

    def __init__(
        self,
        config: ASRConfig,
        device: str,
        platform: str,
        models_dir: Path | None = None,
    ) -> None:
        super().__init__(config, device, platform)
        self.models_dir = models_dir
        self.model_dir: Path | None = None
        self.recognizer = None
        self._decode_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        if self.recognizer is not None:
            return

        requested = str(self.device).strip().lower()
        if requested != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", requested):
            raise ValueError(
                f"unsupported sherpa-onnx device {self.device!r}; expected "
                "'cpu', 'cuda', or 'cuda:<index>'"
            )
        # sherpa-onnx exposes a binary cuda/cpu provider without a device
        # index; an explicit index cannot be honoured and is rejected rather
        # than silently ignored.
        if requested.startswith("cuda") and ":" in requested:
            raise ValueError(
                f"sherpa-onnx cannot select a specific CUDA index (got {self.device!r}); "
                "use 'cuda' and set CUDA_VISIBLE_DEVICES instead"
            )
        use_cuda = requested.startswith("cuda")

        if use_cuda:
            if sys.platform == "win32":
                self._setup_cuda_windows()
            else:
                # Linux / WSL2: ensure PyTorch has already loaded CUDA/cuDNN
                try:
                    import torch  # noqa: F401
                except ImportError:
                    logger.debug("torch not available; sherpa-onnx will initialise CUDA itself")

        self._ensure_models_downloaded()

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx is not installed.\n"
                "  GPU (Windows): pip install sherpa-onnx -f "
                "https://k2-fsa.github.io/sherpa/onnx/cuda.html\n"
                "  CPU/Linux:     pip install sherpa-onnx"
            ) from None

        provider = "cuda" if use_cuda else "cpu"
        num_threads = 1 if use_cuda else 4

        model_dir = self.model_dir
        assert model_dir is not None  # set by _ensure_models_downloaded above

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            provider=provider,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
            hotwords_file="",
            hotwords_score=0.0,
            debug=False,
        )
        self.recognizer = recognizer
        self.model = recognizer

        if use_cuda:
            try:
                self._warmup()
            except Exception:
                self.shutdown()
                raise

        logger.info("✅ SherpaOnnx ASR ready (provider=%s, threads=%d)", provider, num_threads)

    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        # language is accepted for interface parity; the pinned Parakeet
        # model is monolingual and ignores it.
        del language
        if not audio_chunk:
            return ""
        if self.recognizer is None:
            self.load_model()

        # Streaming chunk boundaries can deliver an odd number of bytes;
        # drop the trailing partial 16-bit sample so frombuffer stays aligned.
        if len(audio_chunk) % 2:
            audio_chunk = audio_chunk[:-1]
            if not audio_chunk:
                return ""

        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        # sherpa-onnx does not document concurrent decode_stream safety;
        # serialise access per recognizer instance.
        recognizer = self.recognizer
        assert recognizer is not None
        with self._decode_lock:
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            recognizer.decode_stream(stream)
        text = stream.result.text
        if not isinstance(text, str):
            raise RuntimeError("Sherpa decoder returned non-string text")
        return text.strip()

    def shutdown(self) -> None:
        """Drop the recognizer so native memory can be reclaimed."""
        self.recognizer = None
        self.model = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _warmup(self) -> None:
        try:
            silence = np.zeros(16000, dtype=np.float32)
            recognizer = self.recognizer
            assert recognizer is not None
            with self._decode_lock:
                s = recognizer.create_stream()
                s.accept_waveform(16000, silence)
                recognizer.decode_stream(s)
            logger.info("✅ CUDA warmup complete")
        except Exception as exc:
            raise RuntimeError("Sherpa CUDA warmup failed") from exc

    def _ensure_models_downloaded(self) -> None:
        safe_name = _MODEL_ID.replace("/", "--")
        base = self.models_dir or self._default_models_dir()
        model_dir = base / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if all((model_dir / f).exists() for f in _REQUIRED_FILES):
            self.model_dir = model_dir
            return

        # Import lazily so offline installs with pre-downloaded models work.
        from huggingface_hub import snapshot_download

        logger.info("Downloading %s to %s ...", _MODEL_ID, model_dir)
        # Note: ``local_dir_use_symlinks`` was removed in huggingface_hub 1.x;
        # current versions always materialise real files in ``local_dir``.
        # The download is restricted to the files this strategy loads.
        snapshot_download(
            repo_id=_MODEL_ID,
            revision=_MODEL_REVISION,
            local_dir=str(model_dir),
            allow_patterns=list(_REQUIRED_FILES),
        )
        missing = [f for f in _REQUIRED_FILES if not (model_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"snapshot_download for {_MODEL_ID} did not produce required files: {missing}"
            )
        self.model_dir = model_dir

    @staticmethod
    def _default_models_dir() -> Path:
        override = os.getenv("SIGNAL_ASR_MODELS_DIR")
        if override:
            return Path(override)
        if sys.platform == "win32":
            return Path(r"C:\workspace\artifacts\models")
        return Path("models")

    # ------------------------------------------------------------------ #
    # Windows CUDA helpers
    # ------------------------------------------------------------------ #

    def _setup_cuda_windows(self) -> None:
        """Locate CUDA, verify cuDNN, add to PATH, pre-load DLLs."""
        found = self._find_cuda_bin_windows()
        if found is None:
            raise RuntimeError(
                "CUDA runtime DLLs not found on Windows. Install CUDA 12.x "
                "with cuDNN 9.x and ensure the bin directory is in PATH."
            )
        cuda_bin, runtime_major = found

        # Prioritise CUDA bin in PATH
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if cuda_bin not in path_parts:
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

        cuda_root = str(Path(cuda_bin).parent)
        os.environ.setdefault("CUDA_PATH", cuda_root)

        self._verify_cudnn_windows(cuda_bin)
        self._preload_cuda_dlls_windows(cuda_bin, runtime_major)

    @staticmethod
    def _cuda_candidate_dirs() -> list[str]:
        """Directories that may contain a CUDA runtime, most specific first."""
        candidates: list[str] = []
        env_keys = (
            "CUDA_PATH_V12_8",
            "CUDA_PATH_V12_6",
            "CUDA_PATH_V12_4",
            "CUDA_PATH_V11_8",
            "CUDA_PATH",
        )
        for key in env_keys:
            val = os.environ.get(key)
            if val:
                candidates.append(os.path.join(val, "bin"))
        # PATH entries can also host a runtime (e.g. portable installs).
        candidates.extend(p for p in os.environ.get("PATH", "").split(os.pathsep) if p)
        # Versioned default install locations.
        candidates.extend(
            rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\bin"
            for version in ("v12.8", "v12.6", "v12.4", "v12.2", "v12.1", "v12.0", "v11.8")
        )
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for cand in candidates:
            norm = os.path.normcase(os.path.normpath(cand))
            if cand and norm not in seen:
                seen.add(norm)
                unique.append(cand)
        return unique

    @classmethod
    def _find_cuda_bin_windows(cls) -> tuple[str, int] | None:
        """Locate a CUDA runtime directory by the DLLs it actually contains.

        Returns ``(bin_path, runtime_major)`` or ``None``. Detection is based
        on real runtime DLL filenames rather than assumptions encoded in the
        directory name, so custom install roots work as well as the NVIDIA
        defaults (A11).
        """
        for directory in cls._cuda_candidate_dirs():
            if not directory or not os.path.isdir(directory):
                continue
            for runtime_major, dll_names in sorted(_CUDA_RUNTIME_DLLS.items(), reverse=True):
                if any(os.path.exists(os.path.join(directory, dll)) for dll in dll_names):
                    return directory, runtime_major
        return None

    @staticmethod
    def _verify_cudnn_windows(cuda_bin: str) -> None:
        for dll in _CUDNN_DLLS:
            if os.path.exists(os.path.join(cuda_bin, dll)):
                return
        raise RuntimeError(
            f"cuDNN not found in {cuda_bin}.\n"
            "Install cuDNN 9.x for CUDA 12.x (or 8.x for CUDA 11.x) "
            "and copy DLLs to the CUDA bin directory."
        )

    @staticmethod
    def _preload_cuda_dlls_windows(cuda_bin: str, runtime_major: int) -> None:
        """Pre-load key CUDA/cuDNN DLLs so ONNX Runtime can find them."""
        import ctypes
        import glob

        ctypes = ctypes  # keep mypy aware of the platform-only API
        load_dll = getattr(ctypes, "WinDLL", None)
        core = (
            ["cudart64_12.dll", "cublas64_12.dll"]
            if runtime_major == 12
            else ["cudart64_110.dll", "cublas64_11.dll"]
        )
        dlls = [
            os.path.join(cuda_bin, d) for d in core if os.path.exists(os.path.join(cuda_bin, d))
        ]
        dlls += glob.glob(os.path.join(cuda_bin, "cudnn*.dll"))

        for p in dlls:
            if load_dll is None:
                logger.debug("WinDLL unavailable on this platform; skipping %s", p)
                continue
            try:
                load_dll(p)
            except Exception:
                logger.debug("Could not pre-load %s", os.path.basename(p), exc_info=True)
