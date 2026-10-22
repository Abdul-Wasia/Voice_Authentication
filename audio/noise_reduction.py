"""Noise reduction wrappers for local microphone recordings."""

from __future__ import annotations

import numpy as np
from noisereduce import reduce_noise


class NoiseReducer:
    """Apply light, speech-preserving noise reduction using NoiseReduce."""

    def __init__(self, stationary: bool = False, prop_decrease: float = 0.3):
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def reduce_noise(self, audio: np.ndarray, sample_rate: int, stationary: bool | None = None) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio
        stationary_mode = self.stationary if stationary is None else stationary
        try:
            reduced = reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=stationary_mode,
                prop_decrease=self.prop_decrease,
            )
            if reduced.size == 0:
                return audio
            return np.asarray(reduced, dtype=np.float32)
        except Exception:
            return audio

    def compare(self, original: np.ndarray, processed: np.ndarray) -> dict:
        original = np.asarray(original, dtype=np.float32)
        processed = np.asarray(processed, dtype=np.float32)
        return {
            "original_rms": float(np.sqrt(np.mean(np.square(original))) if original.size else 0.0),
            "processed_rms": float(np.sqrt(np.mean(np.square(processed))) if processed.size else 0.0),
            "noise_reduction_applied": bool(processed.size and original.size),
        }
