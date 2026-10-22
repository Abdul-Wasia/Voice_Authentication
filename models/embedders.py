"""Lazy wrappers around real pretrained speaker-embedding models.

ECAPA-TDNN (SpeechBrain, ``spkrec-ecapa-voxceleb``) and Resemblyzer both require
one-time download of pretrained weights. Loading is lazy and failures are
reported via ``available()`` instead of silently falling back to MFCC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

_BASE_DIR = Path(__file__).resolve().parents[1]
_PRETRAINED_DIR = _BASE_DIR / "data" / "pretrained_models"


class PretrainedEmbedder:
    name = "base"
    dim = 0

    def __init__(self, savedir: Path | None = None):
        self.savedir = Path(savedir) if savedir else _PRETRAINED_DIR / self.name
        self._model: Any = None
        self._loaded = False
        self._error: Optional[str] = None

    def _load(self) -> Any:
        raise NotImplementedError

    def available(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            self.savedir.mkdir(parents=True, exist_ok=True)
            self._model = self._load()
        except Exception as exc:  # pragma: no cover - depends on environment/network
            self._model = None
            self._error = f"{type(exc).__name__}: {exc}"
        return self._model is not None

    def is_cached(self) -> bool:
        """True when weights are already present locally (no download needed)."""
        raise NotImplementedError

    def error(self) -> Optional[str]:
        return self._error

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if not self.available():
            raise RuntimeError(f"Model '{self.name}' is not available: {self._error or 'weights unavailable'}")
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size == 0:
            raise ValueError("Empty audio cannot be embedded.")
        vector = np.asarray(self._embed(audio, sample_rate), dtype=np.float32).ravel()
        return vector

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError


class EcapaTdnnEmbedder(PretrainedEmbedder):
    name = "ecapa_tdnn"
    dim = 192

    def _load(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:  # pragma: no cover - legacy import path
            from speechbrain.pretrained import EncoderClassifier
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.savedir),
            run_opts={"device": "cpu"},
        )

    def is_cached(self) -> bool:
        if not self.savedir.exists():
            return False
        return any(self.savedir.glob("*.ckpt"))

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            embeddings = self._model.encode_batch(wav)
        return embeddings.squeeze().cpu().numpy().astype(np.float32)


class ResemblyzerEmbedder(PretrainedEmbedder):
    name = "resemblyzer"
    dim = 256

    def _load(self):
        from resemblyzer import VoiceEncoder

        return VoiceEncoder()

    def is_cached(self) -> bool:
        try:
            import os

            import resemblyzer

            return os.path.exists(os.path.join(os.path.dirname(resemblyzer.__file__), "pretrained.pt"))
        except Exception:
            return False

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa

        wav = np.asarray(audio, dtype=np.float32).ravel()
        if sample_rate != 16000:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1e-8:
            wav = wav / peak * 0.95
        return self._model.embed_utterance(wav).astype(np.float32)


_REGISTRY: dict[str, PretrainedEmbedder] = {}


def get_embedder(name: str) -> PretrainedEmbedder | None:
    if name == "ecapa_tdnn":
        if name not in _REGISTRY:
            _REGISTRY[name] = EcapaTdnnEmbedder()
        return _REGISTRY[name]
    if name == "resemblyzer":
        if name not in _REGISTRY:
            _REGISTRY[name] = ResemblyzerEmbedder()
        return _REGISTRY[name]
    return None


def model_status() -> dict[str, str]:
    """Return a truthful availability report for every optional model."""
    status: dict[str, str] = {}
    for name in ("ecapa_tdnn", "resemblyzer"):
        embedder = get_embedder(name)
        if embedder is None:
            status[name] = "unavailable"
        elif embedder.available():
            status[name] = "available (pretrained weights loaded)"
        else:
            status[name] = f"unavailable ({embedder.error()})"
    return status
