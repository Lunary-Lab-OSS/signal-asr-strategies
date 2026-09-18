"""Parakeet ASR strategy — NVIDIA NeMo toolkit.

Uses ``nvidia/parakeet-tdt-0.6b-v3`` (or a config-specified variant).
Feeds audio directly through the model preprocessor/encoder/decoder to
avoid NeMo's file-based manifest path and its Windows file-lock issues.

``torch`` and ``nemo_toolkit`` are optional heavy dependencies; they are
imported lazily through :func:`_load_torch` / :func:`_load_nemo_asr` seams
so the module itself stays importable (and unit-testable) without them.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from .base import ASRStrategy

logger = logging.getLogger(__name__)


def _load_torch():
    """Import torch lazily (injectable seam for tests)."""
    import torch

    return torch


def _load_nemo_asr():
    """Import ``nemo.collections.asr`` lazily (injectable seam for tests)."""
    import nemo.collections.asr as nemo_asr

    return nemo_asr


def resolve_parakeet_device(device: str | None):
    """Return the ``torch.device`` the model should be placed on.

    ``"auto"``/``None`` keeps the model wherever NeMo loaded it. ``"cuda"``,
    ``"cuda:<n>"`` and ``"cpu"`` are honoured explicitly; anything else is
    rejected rather than silently ignored.
    """
    torch = _load_torch()
    requested = (device or "auto").strip().lower()
    if requested in ("", "auto"):
        return None
    if requested == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", requested):
        try:
            parsed = torch.device(requested)
        except RuntimeError as exc:
            raise ValueError(f"invalid device {device!r}: {exc}") from exc
        if parsed.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device {device!r} requested but torch.cuda.is_available() is False"
            )
        return parsed
    raise ValueError(
        f"unsupported Parakeet device {device!r}; expected 'cpu', 'cuda', 'cuda:<index>', or 'auto'"
    )


class ParakeetASRStrategy(ASRStrategy):
    """NVIDIA Parakeet TDT ASR via NeMo (in-memory pipeline)."""

    def load_model(self) -> None:
        if self.model is not None:
            return
        model_name = self.config.parakeet_model_name or "nvidia/parakeet-tdt-0.6b-v3"
        target_device = resolve_parakeet_device(self.device)
        logger.info("Loading Parakeet '%s' (requested device %r)...", model_name, self.device)
        nemo_asr = _load_nemo_asr()

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        # Apply the requested placement explicitly and switch to eval mode.
        # ``self.model`` is only published once initialisation fully
        # succeeded, so a failed load leaves ``is_loaded`` False (A05).
        if target_device is not None:
            model.to(target_device)
        model.eval()

        # Prevent Windows file-lock issues in NeMo's data loader
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            for attr in ("test_ds", "decoding"):
                section = getattr(cfg, attr, None)
                if section is not None and hasattr(section, "num_workers"):
                    with contextlib.suppress(Exception):
                        section.num_workers = 0

        self.model = model
        logger.info("✅ Parakeet ASR ready: %s on %s", model_name, self._model_device())

    def _model_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)  # type: ignore[union-attr]
        except StopIteration:
            return "cpu"

    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        # language is accepted for interface parity; the model's own
        # multilingual handling applies.
        del language
        if not audio_chunk:
            return ""
        if self.model is None:
            self.load_model()

        torch = _load_torch()
        model = self.model
        assert model is not None

        # Drop a trailing partial 16-bit sample from streaming chunk boundaries.
        if len(audio_chunk) % 2:
            audio_chunk = audio_chunk[:-1]
            if not audio_chunk:
                return ""

        signal = torch.frombuffer(audio_chunk, dtype=torch.int16).to(torch.float32) / 32768.0
        input_signal = signal.unsqueeze(0)

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        input_signal = input_signal.to(model_device)
        length = torch.tensor([input_signal.shape[1]], device=model_device, dtype=torch.long)

        if length.item() == 0:
            return ""

        with torch.no_grad():
            processed, proc_len = model.preprocessor(input_signal=input_signal, length=length)
            encoded, enc_len = model.encoder(audio_signal=processed, length=proc_len)
            hyps: list[Any] = model.decoding.rnnt_decoder_predictions_tensor(
                encoded, enc_len, return_hypotheses=True
            )

        if not hyps:
            return ""

        # NeMo hypotheses always carry a ``.text`` attribute. An absent or
        # non-string attribute means the backend contract changed — that is
        # an error to surface, not something to stringify into the
        # transcript. An empty transcript is legitimate and returned (A06).
        hypothesis = hyps[0]
        text = getattr(hypothesis, "text", None)
        if not isinstance(text, str):
            raise RuntimeError(
                "NeMo decoder returned a hypothesis without a usable 'text' "
                f"attribute (type={type(hypothesis).__name__}); "
                "the NeMo backend contract may have changed."
            )
        return text.strip()

    def shutdown(self) -> None:
        """Release model memory eagerly when a model is loaded."""
        if self.model is not None:
            with contextlib.suppress(Exception):
                self.model.cpu()
            self.model = None
