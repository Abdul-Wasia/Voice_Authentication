"""Speaker verification using MFCC similarity/SVM and real pretrained models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.svm import SVC

from models.embedders import get_embedder, model_status
from models.feature_extraction import MFCCExtractor

PRETRAINED_MODELS = {"ecapa_tdnn", "resemblyzer"}
IMPLEMENTED_MODELS = ["mfcc_svm", "mfcc_cosine", "ecapa_tdnn", "resemblyzer"]

# Cosine thresholds per model. Speaker embeddings (Resemblyzer/ECAPA) are
# discriminative, so a genuine match sits well above impostors. Raw MFCC
# mean/std vectors are NOT discriminative (different speakers often score
# >0.9), so the MFCC threshold is set very high to avoid false accepts.
MODEL_THRESHOLDS = {
    "resemblyzer": 0.80,
    "ecapa_tdnn": 0.30,
    "mfcc_cosine": 0.95,
    "mfcc_svm": 0.75,
}

# Security ranking: earlier = stronger speaker discrimination.
MODEL_PREFERENCE = ("resemblyzer", "ecapa_tdnn", "mfcc_svm", "mfcc_cosine")


class SpeakerVerifier:
    """Support MFCC + cosine, MFCC + SVM, and real ECAPA-TDNN / Resemblyzer."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._mfcc = MFCCExtractor(n_mfcc=20)

    @staticmethod
    def model_threshold(model_name: str) -> float:
        return float(MODEL_THRESHOLDS.get(model_name, 0.75))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---------------------------------------------------------------- embedding
    @staticmethod
    def model_available(model_name: str) -> bool:
        """Availability WITHOUT triggering a download (uses locally cached weights)."""
        if model_name in {"mfcc_svm", "mfcc_cosine"}:
            return True
        embedder = get_embedder(model_name)
        return bool(embedder and embedder.is_cached())

    @staticmethod
    def cached_model_status() -> dict[str, str]:
        status = {"mfcc_svm": "available", "mfcc_cosine": "available"}
        for name in ("ecapa_tdnn", "resemblyzer"):
            embedder = get_embedder(name)
            status[name] = "available" if (embedder and embedder.is_cached()) else "requires one-time weight download"
        return status

    @staticmethod
    def model_status() -> dict[str, str]:
        """Probe availability, downloading weights if necessary (explicit action)."""
        status = {"mfcc_svm": "available", "mfcc_cosine": "available"}
        status.update(model_status())
        return status

    def embed(self, audio: np.ndarray, sample_rate: int, model_name: str) -> np.ndarray:
        if model_name in PRETRAINED_MODELS:
            embedder = get_embedder(model_name)
            if embedder is None or not embedder.available():
                raise RuntimeError(f"Model '{model_name}' is not available in this environment.")
            return embedder.embed(audio, sample_rate)
        return self._mfcc.extract(audio, sample_rate)

    # ---------------------------------------------------------------- MFCC paths
    def verify_mfcc_cosine(self, sample_embedding: np.ndarray, reference_embedding: np.ndarray, threshold: float | None = None) -> Dict[str, Any]:
        cutoff = float(self.threshold if threshold is None else threshold)
        score = self.cosine_similarity(sample_embedding, reference_embedding)
        accepted = bool(score >= cutoff)
        return {"model": "mfcc_cosine", "score": float(score), "threshold": cutoff, "accepted": accepted}

    def build_svm_model(self, embeddings: list[np.ndarray]) -> SVC | None:
        if len(embeddings) < 2:
            return None
        data = np.vstack([np.asarray(item, dtype=np.float32).ravel() for item in embeddings])
        mean_vector = np.mean(data, axis=0)
        std_vector = np.std(data, axis=0)
        negative_vectors = []
        for _ in range(max(2, len(data))):
            noise = np.random.default_rng(42).normal(loc=mean_vector, scale=np.maximum(std_vector, 1e-3))
            negative_vectors.append(noise)
        negative_data = np.vstack(negative_vectors)
        combined = np.vstack([data, negative_data])
        labels = np.concatenate([np.ones(len(data), dtype=np.int32), np.zeros(len(negative_data), dtype=np.int32)])
        model = SVC(kernel="linear", probability=True)
        model.fit(combined, labels)
        return model

    def save_svm_model(self, model: SVC | None, file_path: str | Path) -> None:
        if model is None:
            return
        with open(file_path, "wb") as file_handle:
            pickle.dump(model, file_handle)

    def load_svm_model(self, file_path: str | Path) -> SVC | None:
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as file_handle:
                return pickle.load(file_handle)
        except Exception:
            return None

    def verify_mfcc_svm(self, sample_embedding: np.ndarray, svm_model: SVC | None, threshold: float | None = None) -> Dict[str, Any]:
        cutoff = float(self.threshold if threshold is None else threshold)
        if svm_model is None:
            return {"model": "mfcc_svm", "score": 0.0, "threshold": cutoff, "accepted": False}
        sample = np.asarray(sample_embedding, dtype=np.float32).reshape(1, -1)
        decision = float(svm_model.decision_function(sample)[0])
        score = float(np.clip((decision + 1.0) / 2.0, 0.0, 1.0))
        accepted = bool(score >= cutoff)
        return {"model": "mfcc_svm", "score": score, "threshold": cutoff, "accepted": accepted}

    # ------------------------------------------------------------- pretrained
    def _verify_pretrained(self, model_name: str, sample_embedding: np.ndarray, reference_embedding: np.ndarray, threshold: float | None = None) -> Dict[str, Any]:
        cutoff = float(self.threshold if threshold is None else threshold)
        if reference_embedding is None or np.asarray(reference_embedding).size != np.asarray(sample_embedding).size:
            return {
                "model": model_name,
                "score": 0.0,
                "threshold": cutoff,
                "accepted": False,
                "reason": "Stored reference for this model is missing or incompatible.",
            }
        score = self.cosine_similarity(sample_embedding, reference_embedding)
        accepted = bool(score >= cutoff)
        return {"model": model_name, "score": float(score), "threshold": cutoff, "accepted": accepted}

    def verify(self, sample_embedding: np.ndarray, reference_embedding: np.ndarray, model_name: str, svm_model: SVC | None = None, threshold: float | None = None) -> Dict[str, Any]:
        if model_name == "mfcc_svm":
            return self.verify_mfcc_svm(sample_embedding, svm_model, threshold)
        if model_name == "mfcc_cosine":
            return self.verify_mfcc_cosine(sample_embedding, reference_embedding, threshold)
        if model_name in PRETRAINED_MODELS:
            return self._verify_pretrained(model_name, sample_embedding, reference_embedding, threshold)
        return self.verify_mfcc_cosine(sample_embedding, reference_embedding, threshold)
