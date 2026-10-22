import numpy as np

from models.feature_extraction import MFCCExtractor


def test_mfcc_extraction_produces_vector():
    audio = np.sin(np.linspace(0, 2 * np.pi * 220, 1600)).astype(np.float32)
    extractor = MFCCExtractor(n_mfcc=12)
    vector = extractor.extract(audio, 16000)
    assert vector.size == 24
    assert np.isfinite(vector).all()
