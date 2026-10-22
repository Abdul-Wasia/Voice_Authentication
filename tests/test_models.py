import numpy as np

from models.speaker_verification import SpeakerVerifier


def test_mfcc_models_always_available():
    assert SpeakerVerifier.model_available("mfcc_cosine") is True
    assert SpeakerVerifier.model_available("mfcc_svm") is True


def test_per_model_thresholds_are_sane():
    assert SpeakerVerifier.model_threshold("resemblyzer") == 0.80
    assert SpeakerVerifier.model_threshold("ecapa_tdnn") == 0.30
    # MFCC is a weak fallback and must be very strict.
    assert SpeakerVerifier.model_threshold("mfcc_cosine") >= 0.9


def test_embedding_is_self_similar():
    verifier = SpeakerVerifier(threshold=0.75)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    vector = verifier.embed(audio, 16000, "mfcc_cosine")
    assert vector.size == 40
    assert abs(verifier.cosine_similarity(vector, vector) - 1.0) < 1e-6


def test_pretrained_reference_mismatch_is_rejected():
    verifier = SpeakerVerifier(threshold=0.75)
    sample = np.ones(192, dtype=np.float32)
    wrong = np.ones(40, dtype=np.float32)
    decision = verifier.verify(sample, wrong, "ecapa_tdnn")
    assert decision["accepted"] is False


def test_pretrained_rejects_different_voice():
    verifier = SpeakerVerifier(threshold=0.75)
    if not verifier.model_available("resemblyzer"):
        return
    sample_rate = 16000
    t = np.linspace(0, 3.0, 3 * sample_rate, endpoint=False)
    reference = verifier.embed((0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), sample_rate, "resemblyzer")
    other = verifier.embed((0.4 * np.sin(2 * np.pi * 900 * t)).astype(np.float32), sample_rate, "resemblyzer")
    decision = verifier.verify(other, reference, "resemblyzer", threshold=0.80)
    assert decision["accepted"] is False
