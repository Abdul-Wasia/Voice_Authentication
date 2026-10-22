"""MFCC feature extraction and compact voice embeddings."""

from __future__ import annotations

import numpy as np
import librosa


class MFCCExtractor:
    """Extract compact MFCC vectors from cleaned audio."""

    def __init__(self, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Audio is empty and cannot be converted into MFCC features.")
        n_fft = min(self.n_fft, max(256, 1 << int(np.floor(np.log2(len(audio))))))
        if n_fft < 256:
            n_fft = 256
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=n_fft,
            hop_length=min(self.hop_length, max(64, n_fft // 4)),
        )
        feature_vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
        return feature_vector

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.extract(audio, sample_rate)

    def save_embedding(self, file_path: str, embedding: np.ndarray) -> str:
        np.save(file_path, np.asarray(embedding, dtype=np.float32))
        return file_path
