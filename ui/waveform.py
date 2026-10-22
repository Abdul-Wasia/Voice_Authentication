"""Waveform rendering helpers.

Produces the vertical-bar waveform strips where the bars rise with voice
loudness and fall with quiet gaps.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BAR_COLOR = "#334155"


def _envelope(audio: np.ndarray, bars: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return np.zeros(bars, dtype=np.float32)
    if audio.size >= bars:
        chunk = audio.size // bars
        trimmed = audio[: chunk * bars].reshape(bars, chunk)
        amp = np.max(np.abs(trimmed), axis=1)
    else:
        amp = np.zeros(bars, dtype=np.float32)
        amp[: audio.size] = np.abs(audio)
    scale = max(float(amp.max()), 0.08)
    return np.clip(amp / scale, 0.0, 1.0)


def waveform_figure(audio: np.ndarray, bars: int = 120, height: float = 1.25, color: str = BAR_COLOR):
    amp = _envelope(audio, bars)
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=110)
    x = np.arange(bars)
    ax.vlines(x, -amp, amp, color=color, linewidth=1.25)
    ax.axhline(0.0, color="#cbd5e1", linewidth=0.6)
    ax.set_xlim(-1, bars)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return fig


def level_figure(level: float, height: float = 0.55):
    level = float(np.clip(level, 0.0, 100.0)) / 100.0
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=110)
    ax.barh([0], [1.0], color="#e2e8f0", height=0.6)
    ax.barh([0], [level], color="#2563eb", height=0.6)
    ax.set_xlim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.6, bottom=0.4)
    return fig
