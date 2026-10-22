"""Audio preprocessing utilities used before feature extraction."""

from __future__ import annotations

import numpy as np
import librosa

try:
    import webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except Exception:
    _WEBRTCVAD_AVAILABLE = False


class AudioPreprocessor:
    """Normalize, trim, and standardize speech recordings with VAD and quality checks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        min_speech_duration: float = 1.0,
        min_snr_db: float = 5.0,
        frame_duration_ms: int = 30,
    ):
        self.sample_rate = sample_rate
        self.vad_aggressiveness = vad_aggressiveness
        self.min_speech_duration = min_speech_duration
        self.min_snr_db = min_snr_db
        self.frame_duration_ms = frame_duration_ms
        self._vad = None
        if _WEBRTCVAD_AVAILABLE:
            try:
                self._vad = webrtcvad.Vad(vad_aggressiveness)
            except Exception:
                self._vad = None

    def ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        return audio.astype(np.float32, copy=False)

    def _frame_generator(self, audio: np.ndarray, sample_rate: int):
        """Yield audio frames of frame_duration_ms for VAD."""
        n = int(sample_rate * self.frame_duration_ms / 1000)
        for i in range(0, len(audio), n):
            frame = audio[i:i + n]
            if len(frame) == n:
                yield frame

    def vad_segments(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Return list of (start, end) sample indices for speech segments using WebRTC VAD."""
        if self._vad is None:
            return self._energy_vad_segments(audio, sample_rate)

        segments = []
        in_speech = False
        start = 0
        for i, frame in enumerate(self._frame_generator(audio, sample_rate)):
            # Convert float32 [-1,1] to int16 for WebRTC VAD
            frame_int16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
            is_speech = self._vad.is_speech(frame_int16.tobytes(), sample_rate)
            if is_speech and not in_speech:
                in_speech = True
                start = i * int(sample_rate * self.frame_duration_ms / 1000)
            elif not is_speech and in_speech:
                in_speech = False
                end = i * int(sample_rate * self.frame_duration_ms / 1000)
                segments.append((start, end))
        if in_speech:
            segments.append((start, len(audio)))
        return segments

    def _energy_vad_segments(self, audio: np.ndarray, sample_rate: int, top_db: float = 25.0) -> list[tuple[int, int]]:
        """Fallback energy-based VAD using librosa."""
        segments = librosa.effects.split(audio, top_db=top_db)
        return [(int(s), int(e)) for s, e in segments]

    def trim_silence(self, audio: np.ndarray, top_db: float = 25.0) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio
        segments = self.vad_segments(audio, self.sample_rate)
        if len(segments) == 0:
            return audio
        if len(segments) == 1:
            start, end = segments[0]
            return audio[start:end]
        return np.concatenate([audio[start:end] for start, end in segments])

    def estimate_snr(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate SNR in dB using spectral flatness as noise proxy."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < sample_rate * 0.1:
            return 0.0
        # Use spectral flatness: low flatness = tonal (speech), high flatness = noise-like
        stft = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
        power = stft ** 2
        geometric_mean = np.exp(np.mean(np.log(power + 1e-10), axis=0))
        arithmetic_mean = np.mean(power, axis=0)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        # Speech has low flatness (~0.1-0.3), noise has high (~0.5-0.9)
        median_flatness = float(np.median(flatness))
        # Map flatness to approximate SNR: flatness 0.1 -> ~20dB, 0.5 -> ~0dB
        snr_est = max(0.0, 20.0 * (1.0 - median_flatness / 0.5))
        return snr_est

    def speech_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """Total duration of detected speech segments in seconds."""
        segments = self.vad_segments(audio, sample_rate)
        total_samples = sum(e - s for s, e in segments)
        return total_samples / sample_rate

    def quality_check(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Run quality checks and return metrics dict with pass/fail."""
        speech_dur = self.speech_duration(audio, sample_rate)
        snr_db = self.estimate_snr(audio, sample_rate)
        total_dur = len(audio) / sample_rate
        speech_ratio = speech_dur / total_dur if total_dur > 0 else 0.0

        return {
            "speech_duration_sec": round(speech_dur, 3),
            "total_duration_sec": round(total_dur, 3),
            "speech_ratio": round(speech_ratio, 3),
            "snr_db": round(snr_db, 2),
            "passes_speech_duration": speech_dur >= self.min_speech_duration,
            "passes_snr": snr_db >= self.min_snr_db,
            "passes_all": (speech_dur >= self.min_speech_duration) and (snr_db >= self.min_snr_db),
        }

    def fix_length(self, audio: np.ndarray, duration: float, sample_rate: int | None = None) -> np.ndarray:
        """Pad with zeros or crop so every recording has an identical length."""
        rate = sample_rate or self.sample_rate
        target = int(round(duration * rate))
        audio = np.asarray(audio, dtype=np.float32)
        if target <= 0:
            return audio
        if audio.size >= target:
            return audio[:target]
        padded = np.zeros(target, dtype=np.float32)
        padded[: audio.size] = audio
        return padded

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return audio
        amplitude = np.max(np.abs(audio))
        if amplitude < 1e-8:
            return audio
        return (audio / amplitude) * 0.95

    def resample_audio(self, audio: np.ndarray, original_sample_rate: int, target_sample_rate: int | None = None) -> np.ndarray:
        target = target_sample_rate or self.sample_rate
        if original_sample_rate == target or audio.size == 0:
            return np.asarray(audio, dtype=np.float32)
        return librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=original_sample_rate, target_sr=target)

    def preprocess_audio(
        self,
        audio: np.ndarray,
        original_sample_rate: int,
        sample_rate: int | None = None,
        top_db: float = 25.0,
        apply_quality_check: bool = True,
    ) -> np.ndarray:
        target_rate = sample_rate or self.sample_rate
        audio = self.ensure_mono(audio)
        audio = self.resample_audio(audio, original_sample_rate, target_rate)
        audio = self.trim_silence(audio, top_db=top_db)
        if audio.size == 0:
            raise ValueError("Audio does not contain enough usable speech after trimming silence.")
        if apply_quality_check:
            q = self.quality_check(audio, target_rate)
            if not q["passes_all"]:
                raise ValueError(
                    f"Audio quality check failed: speech_duration={q['speech_duration_sec']}s "
                    f"(min {self.min_speech_duration}s), snr={q['snr_db']}dB (min {self.min_snr_db}dB)"
                )
        audio = self.normalize_audio(audio)
        return audio.astype(np.float32, copy=False)

    # Convenience method for tests / controlled environments
    def preprocess_audio_for_test(
        self,
        audio: np.ndarray,
        original_sample_rate: int,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """Preprocess without quality gates — for unit tests with synthetic signals only."""
        return self.preprocess_audio(
            audio, original_sample_rate, sample_rate=sample_rate, apply_quality_check=False
        )
