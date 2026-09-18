"""High-level ASR facade."""

from __future__ import annotations

from .config import ASRConfig
from .strategies.base import ASRStrategy
from .strategies.factory import ASRStrategyFactory


class ASRComponent:
    """Small facade over the ASR strategy factory."""

    def __init__(
        self,
        config: ASRConfig,
        model_loader=None,
        device: str = "cpu",
        platform: str | None = None,
    ) -> None:
        self.config = config
        self.device = device
        factory = ASRStrategyFactory(
            config=config,
            model_loader=model_loader,
            device=device,
            platform=platform,
        )
        # Expose the *resolved* platform (detected when not supplied) so
        # callers can see which backend family was selected on this machine.
        self.platform = factory.platform
        self._strategy: ASRStrategy = factory.create_strategy()

    def load_model(self) -> None:
        self._strategy.load_model()

    def transcribe(self, audio_chunk: bytes, language: str | None = None) -> str:
        return self._strategy.transcribe(audio_chunk, language=language)

    def close(self) -> None:
        self._strategy.shutdown()

    def shutdown(self) -> None:
        self.close()

    @property
    def strategy(self) -> ASRStrategy:
        return self._strategy
