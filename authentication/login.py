"""Login workflow: password verification followed by speaker verification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf

from audio.noise_reduction import NoiseReducer
from audio.preprocessing import AudioPreprocessor
from audio.recorder import AudioRecorder
from authentication.passwords import verify_password
from database.database import DatabaseManager
from models.embedders import get_embedder
from models.speaker_verification import PRETRAINED_MODELS, MODEL_PREFERENCE, SpeakerVerifier


class LoginManager:
    """Validate a login attempt using password + recorded voice."""

    def __init__(self, db_manager: DatabaseManager | None = None):
        self.db = db_manager or DatabaseManager()
        self.preprocessor = AudioPreprocessor(sample_rate=16000)
        self.verifier = SpeakerVerifier(threshold=0.75)

    def _read_profile_metadata(self, profile_dir: Path) -> Dict[str, Any]:
        metadata_path = profile_dir / "user_profile.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No user profile metadata found in {profile_dir}.")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _profile_dir(self, user: Dict[str, Any]) -> Path:
        return Path(user["profile_path"]) if user.get("profile_path") else Path(user["folder_name"])

    def _load_reference_vector(self, profile_dir: Path, model_name: str) -> np.ndarray:
        metadata = self._read_profile_metadata(profile_dir)

        if model_name in {"mfcc_cosine", "mfcc_svm"}:
            if "reference_centroid" in metadata:
                return np.asarray(metadata["reference_centroid"], dtype=np.float32)
            feature_dir = profile_dir / "profiles" / "mfcc"
            files = sorted(feature_dir.glob("*.npy"))
            if not files:
                raise FileNotFoundError(f"No stored embedding for model '{model_name}'.")
            return np.mean(np.vstack([np.load(p).astype(np.float32) for p in files]), axis=0)

        expected = self._expected_dim(model_name)
        embeddings = metadata.get("reference_embeddings")
        if isinstance(embeddings, dict) and model_name in embeddings:
            stored = np.asarray(embeddings[model_name], dtype=np.float32)
            if expected is None or stored.size == expected:
                return stored

        feature_dir = profile_dir / "profiles" / model_name
        vectors = []
        for path in sorted(feature_dir.glob("*.npy")):
            vector = np.load(path).astype(np.float32)
            if expected is None or vector.size == expected:
                vectors.append(vector)
        if vectors:
            return np.mean(np.vstack(vectors), axis=0)

        rebuilt = self._rebuild_reference(profile_dir, model_name)
        if rebuilt is not None:
            return rebuilt
        raise FileNotFoundError(f"The stored '{model_name}' voice reference is missing or incompatible.")

    def _load_svm_model(self, profile_dir: Path):
        return self.verifier.load_svm_model(profile_dir / "profiles" / "svm" / "svm_model.pkl")

    @staticmethod
    def _noise_reducer_for_profile(profile_dir: Path) -> NoiseReducer:
        metadata_path = profile_dir / "user_profile.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No user profile metadata found in {profile_dir}.")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        # Profiles created before the setting was recorded used the old default.
        prop_decrease = float(metadata.get("noise_reduction_prop_decrease", 0.8))
        if not 0.0 <= prop_decrease <= 1.0:
            raise ValueError("Voice profile has an invalid noise-reduction setting.")
        return NoiseReducer(stationary=False, prop_decrease=prop_decrease)

    @staticmethod
    def _expected_dim(model_name: str) -> int | None:
        if model_name in PRETRAINED_MODELS:
            embedder = get_embedder(model_name)
            return getattr(embedder, "dim", None)
        return None

    @staticmethod
    def _enrollment_audio_files(profile_dir: Path) -> list[Path]:
        for subfolder in ("processed", "raw"):
            files = sorted((profile_dir / subfolder).glob("*.wav"))
            if files:
                return files
        return []

    def _rebuild_reference(self, profile_dir: Path, model_name: str) -> np.ndarray | None:
        """Recompute a pretrained embedding from the original enrollment audio.

        Older profiles sometimes saved the wrong feature type (e.g. MFCC) under
        the model folder. Rebuilding from the stored recordings restores a valid,
        discriminative reference without forcing the user to re-enroll.
        """
        embedder = get_embedder(model_name)
        if embedder is None or not embedder.available():
            return None
        files = self._enrollment_audio_files(profile_dir)
        if not files:
            return None
        vectors: list[np.ndarray] = []
        for path in files:
            try:
                audio, sample_rate = sf.read(path, dtype="float32")
            except Exception:
                continue
            if sample_rate != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            cleaned = self.preprocessor.preprocess_audio(audio, 16000, 16000)
            denoised = self._noise_reducer_for_profile(profile_dir).reduce_noise(cleaned, 16000)
            try:
                vectors.append(embedder.embed(denoised, 16000))
            except Exception:
                continue
        if not vectors:
            return None
        output_dir = profile_dir / "profiles" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, vector in enumerate(vectors, start=1):
            np.save(output_dir / f"voice_{index:02d}.npy", vector.astype(np.float32))
        return np.mean(np.vstack(vectors), axis=0).astype(np.float32)

    def _reference_compatible(self, profile_dir: Path, model_name: str) -> bool:
        """True when a correctly shaped reference exists or can be rebuilt."""
        expected = self._expected_dim(model_name)
        try:
            metadata = self._read_profile_metadata(profile_dir)
        except Exception:
            return False

        if expected is None:
            if "reference_centroid" in metadata:
                return True
            return any((profile_dir / "profiles" / "mfcc").glob("*.npy"))

        embeddings = metadata.get("reference_embeddings")
        if isinstance(embeddings, dict) and model_name in embeddings:
            if np.asarray(embeddings[model_name]).size == expected:
                return True
        for path in sorted((profile_dir / "profiles" / model_name).glob("*.npy")):
            try:
                if np.load(path).size == expected:
                    return True
            except Exception:
                continue
        return self._enrollment_audio_files(profile_dir) != []

    def _select_model(self, profile_dir: Path, requested: str | None) -> str:
        """Pick the strongest model that is available AND has a usable reference.

        Strong speaker models are preferred over the requested/default model so
        that profiles enrolled with a weak MFCC setting but which already have a
        Resemblyzer/ECAPA embedding are transparently upgraded.
        """
        candidates: list[str] = []
        for name in (*MODEL_PREFERENCE, requested):
            if name and name not in candidates:
                candidates.append(name)
        for name in candidates:
            if name in PRETRAINED_MODELS:
                if self.verifier.model_available(name) and self._reference_compatible(profile_dir, name):
                    return name
            elif self._reference_compatible(profile_dir, name):
                return name
        return requested or "mfcc_cosine"

    def verify_password_only(self, user_id: str, password: str) -> Dict[str, Any]:
        user = self.db.get_user_by_id(user_id)
        if not user:
            return {"ok": False, "reason": "Invalid user ID or password."}
        if not int(user.get("is_active") or 0):
            return {"ok": False, "reason": "This account is disabled. Contact an administrator."}
        if not user.get("password_hash"):
            return {"ok": False, "reason": "This account has no password set. Contact an administrator."}
        if not verify_password(password, user.get("password_hash")):
            return {"ok": False, "reason": "Invalid user ID or password."}
        return {"ok": True, "reason": "Password accepted.", "user": user}

    def authenticate(
        self,
        user_id: str,
        password: str | None = None,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        duration: float = 3.0,
        model_name: str = "mfcc_cosine",
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        user = self.db.get_user_by_id(user_id)
        if not user:
            return self._fail(user_id, model_name, "Invalid user ID or password.")
        if not int(user.get("is_active") or 0):
            return self._fail(user_id, model_name, "This account is disabled. Contact an administrator.")
        if user.get("password_hash") is not None:
            if password is None or not verify_password(password, user.get("password_hash")):
                return self._fail(user_id, model_name, "Invalid user ID or password.")
        return self._voice_authenticate(user, user_id, audio, sample_rate, duration, model_name, threshold)

    def authenticate_voice_only(
        self,
        user_id: str,
        audio: np.ndarray | None = None,
        sample_rate: int = 16000,
        duration: float = 3.0,
        model_name: str = "mfcc_cosine",
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Run ONLY the voice step. Caller must have verified the password first."""
        user = self.db.get_user_by_id(user_id)
        if not user:
            return self._fail(user_id, model_name, "Invalid account.")
        if not int(user.get("is_active") or 0):
            return self._fail(user_id, model_name, "This account is disabled. Contact an administrator.")
        return self._voice_authenticate(user, user_id, audio, sample_rate, duration, model_name, threshold)

    @staticmethod
    def _fail(user_id: str, model_name: str, reason: str, score: float = 0.0) -> Dict[str, Any]:
        return {
            "authenticated": False,
            "user_id": user_id,
            "reason": reason,
            "score": score,
            "threshold": SpeakerVerifier.model_threshold(model_name),
            "model": model_name,
            "processing_time_ms": 0.0,
        }

    def _voice_authenticate(
        self,
        user: Dict[str, Any],
        user_id: str,
        audio: np.ndarray | None,
        sample_rate: int,
        duration: float,
        model_name: str,
        threshold: float | None,
    ) -> Dict[str, Any]:
        profile_dir = self._profile_dir(user)
        if not profile_dir.exists():
            return self._fail(user_id, model_name, "The voice profile is unavailable for this account.")

        chosen_model = self._select_model(profile_dir, model_name)
        if chosen_model in PRETRAINED_MODELS and not self.verifier.model_available(chosen_model):
            return self._fail(user_id, chosen_model, f"The '{chosen_model}' model is not available in this environment.")

        start = time.perf_counter()
        if audio is None:
            recorder = AudioRecorder(sample_rate=sample_rate)
            audio, sample_rate = recorder.record_sample(duration=duration)

        cleaned = self.preprocessor.preprocess_audio(audio, sample_rate, sample_rate)
        denoised = self._noise_reducer_for_profile(profile_dir).reduce_noise(cleaned, sample_rate)

        try:
            sample_vector = self.verifier.embed(denoised, sample_rate, chosen_model)
            reference_vector = self._load_reference_vector(profile_dir, chosen_model)
        except Exception as exc:
            return self._fail(user_id, chosen_model, f"Voice profile could not be processed: {exc}")

        model = self._load_svm_model(profile_dir) if chosen_model == "mfcc_svm" else None
        effective_threshold = threshold if threshold is not None else SpeakerVerifier.model_threshold(chosen_model)
        decision = self.verifier.verify(
            sample_vector, reference_vector, model_name=chosen_model, svm_model=model, threshold=effective_threshold
        )
        accepted = bool(decision["accepted"])

        processing_time_ms = (time.perf_counter() - start) * 1000.0
        result = "SUCCESS" if accepted else "FAILURE"
        notes = "Password and voice matched the registered speaker." if accepted else "Voice did not match the registered speaker."
        self.db.save_auth_attempt(
            user_id=user_id,
            result=result,
            score=float(decision.get("score", 0.0)),
            model_name=chosen_model,
            processing_time_ms=processing_time_ms,
            notes=notes,
        )

        return {
            "authenticated": accepted,
            "user_id": user_id,
            "reason": notes,
            "score": float(decision.get("score", 0.0)),
            "threshold": float(decision.get("threshold", effective_threshold)),
            "model": decision.get("model", chosen_model),
            "processing_time_ms": round(processing_time_ms, 3),
        }
