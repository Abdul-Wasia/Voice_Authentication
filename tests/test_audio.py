import numpy as np

from audio.noise_reduction import NoiseReducer
from audio.preprocessing import AudioPreprocessor


def test_preprocessing_creates_valid_audio():
    # Use a longer synthetic signal that passes quality checks
    sample_rate = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    processor = AudioPreprocessor(sample_rate=sample_rate)
    cleaned = processor.preprocess_audio(audio, original_sample_rate=sample_rate, sample_rate=sample_rate)
    assert cleaned.size > 0
    assert np.isfinite(cleaned).all()


def test_preprocessing_for_test_bypasses_quality_check():
    """Unit test helper should bypass quality gates for short synthetic signals."""
    audio = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    processor = AudioPreprocessor(sample_rate=16000)
    cleaned = processor.preprocess_audio_for_test(audio, original_sample_rate=16000, sample_rate=16000)
    assert cleaned.size > 0
    assert np.isfinite(cleaned).all()


def test_noise_reduction_returns_array():
    audio = np.sin(np.linspace(0, 2 * np.pi * 10, 2000), dtype=np.float32)
    reducer = NoiseReducer()
    assert reducer.prop_decrease == 0.3
    reduced = reducer.reduce_noise(audio, sample_rate=16000)
    assert reduced.shape == audio.shape
    assert np.isfinite(reduced).all()


def test_noise_reduction_improves_snr_without_aggressive_suppression():
    sample_rate = 16000
    time = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 3 * time)
    voice = envelope * (
        0.16 * np.sin(2 * np.pi * 180 * time)
        + 0.08 * np.sin(2 * np.pi * 360 * time)
        + 0.04 * np.sin(2 * np.pi * 540 * time)
    )
    noise = 0.06 * np.random.default_rng(42).normal(size=time.size)
    mixed = (voice + noise).astype(np.float32)
    reduced = NoiseReducer().reduce_noise(mixed, sample_rate)

    input_snr = np.mean(voice**2) / np.mean((mixed - voice) ** 2)
    output_snr = np.mean(voice**2) / np.mean((reduced - voice) ** 2)
    assert output_snr > input_snr
    assert reduced.shape == mixed.shape
    assert np.isfinite(reduced).all()
