"""Compatibility shim for Resemblyzer on Windows.

The upstream `webrtcvad` package is a native extension and cannot be reliably
built in all Windows developer environments without the full C++ SDK. This local
module provides the minimal `Vad` API expected by Resemblyzer so the project can
remain functional and CPU-friendly without requiring a difficult native build.
"""

from __future__ import annotations

import numpy as np


class Vad:
    """Simple fallback voice activity detector used by Resemblyzer."""

    def __init__(self, mode: int = 3):
        self.mode = int(mode)

    def is_speech(self, buf, sample_rate: int = 16000) -> bool:
        if buf is None:
            return False

        if isinstance(buf, (bytes, bytearray)):
            if len(buf) == 0:
                return False
            samples = np.frombuffer(buf, dtype="<i2").astype(np.float32)
        else:
            samples = np.asarray(buf, dtype=np.float32).ravel()

        if samples.size == 0:
            return False

        if sample_rate <= 0:
            return False

        samples = np.clip(samples / 32768.0, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < 0.005:
            return False

        zero_crossings = np.mean(np.abs(np.diff(np.signbit(samples))))
        energy = float(np.mean(np.square(samples)))
        return bool((energy > 0.0005) or (zero_crossings > 0.02))
