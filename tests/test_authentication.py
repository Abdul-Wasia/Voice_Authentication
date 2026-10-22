import json
from pathlib import Path

import numpy as np

from authentication.registration import RegistrationManager
from authentication.login import LoginManager
from database.database import DatabaseManager


def _synthetic_recording(index, duration):
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Add harmonics to make it more speech-like for VAD/SNR estimation
    audio = (
        0.5 * np.sin(2 * np.pi * (180 + index * 10) * t)
        + 0.2 * np.sin(2 * np.pi * (360 + index * 20) * t)
        + 0.1 * np.sin(2 * np.pi * (540 + index * 30) * t)
    ).astype(np.float32)
    return audio, sample_rate


def test_registration_creates_user_folder(tmp_path):
    manager = RegistrationManager(
        root_dir=tmp_path / "voice_profiles",
        db_manager=DatabaseManager(db_path=tmp_path / "voice_auth.db"),
    )
    result = manager.register_user(
        user_id="user_001",
        name="Test User",
        sample_count=3,
        duration=1.5,
        default_model="mfcc_cosine",
        recording_factory=_synthetic_recording,
    )
    assert result["user_id"] == "user_001"
    profile_dir = Path(result["folder_path"])
    assert (profile_dir / "raw").exists()
    assert (profile_dir / "processed").exists()
    assert (profile_dir / "profiles").exists()
    assert (profile_dir / "user_profile.json").exists()
    metadata = json.loads((profile_dir / "user_profile.json").read_text(encoding="utf-8"))
    assert metadata["noise_reduction_prop_decrease"] == 0.3


def test_legacy_profile_keeps_its_noise_reduction_setting(tmp_path):
    profile_dir = tmp_path / "legacy_profile"
    profile_dir.mkdir()
    (profile_dir / "user_profile.json").write_text("{}", encoding="utf-8")

    reducer = LoginManager._noise_reducer_for_profile(profile_dir)
    assert reducer.prop_decrease == 0.8


def test_login_rejects_unknown_user():
    db = DatabaseManager(db_path=Path("data") / "voice_auth_test.db")
    login = LoginManager(db)
    result = login.authenticate(user_id="user_missing", audio=np.ones(1600, dtype=np.float32), threshold=0.75)
    assert result["authenticated"] is False


def _register_user(tmp_path, password="password123"):
    manager = RegistrationManager(
        root_dir=tmp_path / "voice_profiles",
        db_manager=DatabaseManager(db_path=tmp_path / "voice_auth.db"),
    )
    manager.register_user(
        user_id="user_001",
        name="Test User",
        sample_count=3,
        duration=1.5,
        default_model="mfcc_cosine",
        password=password,
        recording_factory=_synthetic_recording,
    )
    return manager


def test_registration_stores_hashed_password(tmp_path):
    manager = _register_user(tmp_path)
    user = manager.db.get_user_by_id("user_001")
    assert user["password_hash"] and user["password_hash"].startswith("$2")
    assert user["password_hash"] != "password123"
    assert user["role"] == "user"


def test_password_only_verification(tmp_path):
    manager = _register_user(tmp_path)
    login = LoginManager(manager.db)
    assert login.verify_password_only("user_001", "password123")["ok"] is True
    assert login.verify_password_only("user_001", "wrong-password")["ok"] is False


def test_login_requires_correct_password(tmp_path):
    manager = _register_user(tmp_path)
    login = LoginManager(manager.db)
    audio, _ = _synthetic_recording(1, 1.5)
    result = login.authenticate(user_id="user_001", password="wrong-password", audio=audio)
    assert result["authenticated"] is False


def test_full_login_with_password_and_voice(tmp_path):
    manager = _register_user(tmp_path)
    login = LoginManager(manager.db)
    audio, _ = _synthetic_recording(1, 1.5)
    result = login.authenticate(user_id="user_001", password="password123", audio=audio, threshold=0.75)
    assert result["authenticated"] is True


def test_login_rejects_impostor_voice(tmp_path):
    manager = _register_user(tmp_path)
    login = LoginManager(manager.db)
    sample_rate = 16000
    t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
    impostor = (
        0.5 * np.sin(2 * np.pi * 900 * t)
        + 0.2 * np.sin(2 * np.pi * 1800 * t)
        + 0.1 * np.sin(2 * np.pi * 2700 * t)
    ).astype(np.float32)
    result = login.authenticate(user_id="user_001", password="password123", audio=impostor)
    assert result["authenticated"] is False


def test_registration_auto_selects_strong_model(tmp_path):
    from models.speaker_verification import SpeakerVerifier

    manager = RegistrationManager(
        root_dir=tmp_path / "voice_profiles",
        db_manager=DatabaseManager(db_path=tmp_path / "voice_auth.db"),
    )
    manager.register_user(
        user_id="user_auto",
        name="Auto User",
        sample_count=3,
        duration=1.5,
        password="password123",
        recording_factory=_synthetic_recording,
    )
    user = manager.db.get_user_by_id("user_auto")
    if SpeakerVerifier.model_available("resemblyzer"):
        assert user["default_model"] == "resemblyzer"
