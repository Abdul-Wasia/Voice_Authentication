"""Registration workflow for per-user voice-profile folders."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from audio.noise_reduction import NoiseReducer
from audio.preprocessing import AudioPreprocessor
from audio.recorder import AudioRecorder
from authentication.passwords import hash_password
from database.database import DatabaseManager
from models.embedders import get_embedder
from models.feature_extraction import MFCCExtractor
from models.speaker_verification import IMPLEMENTED_MODELS, SpeakerVerifier


def sanitize_path_component(value: str) -> str:
    cleaned = (value or "user").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in cleaned)
    cleaned = cleaned.replace("__", "_")
    return cleaned or "user"


class RegistrationManager:
    """Create and persist a per-user voice profile directory."""

    def __init__(self, root_dir: str | Path | None = None, db_manager: DatabaseManager | None = None):
        base_dir = Path(__file__).resolve().parents[1]
        self.root_dir = Path(root_dir) if root_dir else base_dir / "data" / "voice_profiles"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_manager or DatabaseManager()

    def __user_folder_name(self, user_id: str, name: str) -> str:
        sanitized_name = sanitize_path_component(name)
        return f"{sanitize_path_component(user_id)}_{sanitized_name}"

    def _ensure_unique_folder(self, folder_path: Path) -> Path:
        if not folder_path.exists():
            return folder_path
        suffix = 1
        while True:
            candidate = folder_path.parent / f"{folder_path.name}_{suffix}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def create_user_folder(self, user_id: str, name: str) -> Path:
        folder_name = self.__user_folder_name(user_id, name)
        folder_path = self.root_dir / folder_name
        folder_path = self._ensure_unique_folder(folder_path)
        for subfolder in ("raw", "processed", "profiles", "profiles/mfcc", "profiles/ecapa", "profiles/resemblyzer", "profiles/svm"):
            (folder_path / subfolder).mkdir(parents=True, exist_ok=True)
        return folder_path

    def _resolve_default_model(self, requested: str | None, verifier: SpeakerVerifier) -> str:
        """Return the strongest speaker model available in this environment."""
        if requested and requested not in {"auto", "default"}:
            return requested
        for name in ("resemblyzer", "ecapa_tdnn"):
            if verifier.model_available(name):
                return name
        return "mfcc_cosine"

    def register_user(
        self,
        user_id: str,
        name: str,
        sample_count: int = 4,
        duration: float = 3.0,
        default_model: str | None = None,
        secondary_pin: str | None = None,
        recording_factory: Any | None = None,
        password: str | None = None,
        role: str = "user",
        voice_min_duration: float | None = None,
    ) -> Dict[str, Any]:
        if not user_id or not name:
            raise ValueError("User ID and name are required.")
        if sample_count < 3 or sample_count > 6:
            raise ValueError("Record between 3 and 6 voice samples for enrollment.")
        if password is not None and len(str(password)) < 8:
            raise ValueError("Password must be at least 8 characters long.")

        existing = self.db.get_user_by_id(user_id)
        if existing:
            raise ValueError(f"User ID '{user_id}' already exists.")

        verifier = SpeakerVerifier(threshold=0.75)
        default_model = self._resolve_default_model(default_model, verifier)

        folder_path = self.create_user_folder(user_id, name)
        metadata, centroid, reference_embeddings = self._build_voice_profile(
            folder_path=folder_path,
            user_id=user_id,
            name=name,
            sample_count=sample_count,
            duration=duration,
            default_model=default_model,
            recording_factory=recording_factory,
        )

        secondary_pin_hash = None
        if secondary_pin:
            secondary_pin_hash = hashlib.sha256(secondary_pin.strip().encode("utf-8")).hexdigest()

        password_hash = hash_password(password) if password else None
        self.db.create_user(
            user_id=user_id,
            name=name,
            folder_name=folder_path.name,
            profile_path=str(folder_path),
            default_model=default_model,
            password_hash=password_hash,
            secondary_pin_hash=secondary_pin_hash,
            profile_version="1.0",
            role=role,
            voice_min_duration=float(voice_min_duration or duration),
        )

        self._save_profile_records(user_id, folder_path, centroid, reference_embeddings)
        return {"user_id": user_id, "name": name, "folder_path": str(folder_path), "metadata": metadata}

    def update_voice_profile(
        self,
        user_id: str,
        sample_count: int = 4,
        duration: float = 3.0,
        recording_factory: Any | None = None,
    ) -> Dict[str, Any]:
        """Re-enroll an EXISTING user's voice without touching their account data."""
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found.")
        if sample_count < 3 or sample_count > 6:
            raise ValueError("Record between 3 and 6 voice samples for enrollment.")

        folder_path = Path(user["profile_path"]) if user.get("profile_path") else self.root_dir / user["folder_name"]
        folder_path.mkdir(parents=True, exist_ok=True)
        for subfolder in (
            "raw",
            "processed",
            "profiles",
            "profiles/mfcc",
            "profiles/ecapa",
            "profiles/resemblyzer",
            "profiles/svm",
        ):
            (folder_path / subfolder).mkdir(parents=True, exist_ok=True)

        resolved_model = self._resolve_default_model(user.get("default_model"), SpeakerVerifier(threshold=0.75))
        metadata, centroid, reference_embeddings = self._build_voice_profile(
            folder_path=folder_path,
            user_id=user_id,
            name=user["name"],
            sample_count=sample_count,
            duration=duration,
            default_model=resolved_model,
            recording_factory=recording_factory,
        )
        self._save_profile_records(user_id, folder_path, centroid, reference_embeddings)
        self.db.update_user(user_id, voice_min_duration=float(duration), default_model=resolved_model)
        return {"user_id": user_id, "folder_path": str(folder_path), "metadata": metadata}

    def _save_profile_records(self, user_id: str, folder_path: Path, centroid: np.ndarray, reference_embeddings: Dict[str, Any]) -> None:
        self.db.save_profile_record(user_id, "mfcc_svm", str(folder_path / "profiles" / "svm"), {"reference_centroid": centroid.astype(float).tolist()})
        self.db.save_profile_record(user_id, "mfcc_cosine", str(folder_path / "profiles" / "mfcc"), {"reference_centroid": centroid.astype(float).tolist()})
        for model_name in ("ecapa_tdnn", "resemblyzer"):
            if model_name in reference_embeddings:
                self.db.save_profile_record(user_id, model_name, str(folder_path / "profiles" / model_name), {"reference_embedding": reference_embeddings[model_name]})

    def _build_voice_profile(
        self,
        folder_path: Path,
        user_id: str,
        name: str,
        sample_count: int,
        duration: float,
        default_model: str,
        recording_factory: Any | None,
    ) -> tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
        preprocessor = AudioPreprocessor(sample_rate=16000)
        recorder = AudioRecorder(sample_rate=16000)
        noise_reducer = NoiseReducer(stationary=False)
        mfcc_extractor = MFCCExtractor(n_mfcc=20)
        verifier = SpeakerVerifier(threshold=0.75)

        models_status = verifier.cached_model_status()
        embedders = {}
        for model_name in ("ecapa_tdnn", "resemblyzer"):
            if verifier.model_available(model_name):
                embedders[model_name] = get_embedder(model_name)

        feature_vectors: List[np.ndarray] = []
        pretrained_vectors: Dict[str, List[np.ndarray]] = {"ecapa_tdnn": [], "resemblyzer": []}
        for index in range(1, sample_count + 1):
            if recording_factory is not None:
                audio, sample_rate = recording_factory(index, duration)
            else:
                audio, sample_rate = recorder.record_sample(duration=duration)
            sf.write(folder_path / "raw" / f"voice_{index:02d}.wav", np.asarray(audio, dtype=np.float32), sample_rate)

            cleaned = preprocessor.preprocess_audio(audio, sample_rate, sample_rate)
            denoised = noise_reducer.reduce_noise(cleaned, sample_rate)
            sf.write(folder_path / "processed" / f"voice_{index:02d}.wav", np.asarray(denoised, dtype=np.float32), sample_rate)

            feature_vector = mfcc_extractor.extract(denoised, sample_rate)
            feature_vectors.append(feature_vector)
            np.save(folder_path / "profiles" / "mfcc" / f"voice_{index:02d}.npy", feature_vector)

            for model_name, embedder in embedders.items():
                try:
                    vector = embedder.embed(denoised, sample_rate)
                    pretrained_vectors[model_name].append(vector)
                    np.save(folder_path / "profiles" / model_name / f"voice_{index:02d}.npy", vector)
                except Exception:
                    models_status[model_name] = "unavailable (embedding failed during enrollment)"

        svm_model = verifier.build_svm_model(feature_vectors)
        if svm_model is not None:
            verifier.save_svm_model(svm_model, folder_path / "profiles" / "svm" / "svm_model.pkl")

        centroid = np.mean(np.vstack(feature_vectors), axis=0)
        reference_embeddings: Dict[str, Any] = {"mfcc_cosine": centroid.astype(float).tolist()}
        for model_name, vectors in pretrained_vectors.items():
            if vectors:
                reference_embeddings[model_name] = np.mean(np.vstack(vectors), axis=0).astype(float).tolist()

        metadata = {
            "user_id": user_id,
            "name": name,
            "sample_count": sample_count,
            "sample_rate": 16000,
            "models": IMPLEMENTED_MODELS,
            "model_status": models_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile_version": "1.0",
            "folder_name": folder_path.name,
            "profile_path": str(folder_path),
            "noise_reduction_prop_decrease": noise_reducer.prop_decrease,
            "reference_centroid": centroid.astype(float).tolist(),
            "reference_embeddings": reference_embeddings,
            "default_model": default_model,
        }
        with open(folder_path / "user_profile.json", "w", encoding="utf-8") as profile_file:
            json.dump(metadata, profile_file, indent=2)
        return metadata, centroid, reference_embeddings

    def delete_user(self, user_id: str, archive: bool = False) -> Dict[str, Any]:
        user = self.db.get_user_by_id(user_id)
        if not user:
            return {"deleted": False, "reason": "User not found."}
        profile_path = Path(user["profile_path"]) if user.get("profile_path") else self.root_dir / user["folder_name"]

        if archive:
            archive_root = self.root_dir / "_archive"
            archive_root.mkdir(parents=True, exist_ok=True)
            archive_path = archive_root / profile_path.name
            if archive_path.exists():
                suffix = 1
                while archive_path.exists():
                    archive_path = archive_root / f"{profile_path.name}_{suffix}"
                    suffix += 1
            if profile_path.exists():
                import shutil
                shutil.move(str(profile_path), str(archive_path))
        else:
            if profile_path.exists():
                import shutil
                shutil.rmtree(profile_path)

        self.db.delete_user(user_id)
        return {"deleted": True, "user_id": user_id, "path": str(profile_path)}
