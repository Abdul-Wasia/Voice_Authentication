"""Microphone recording utilities for the voice authentication project."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Capture microphone recordings and provide real-time waveform/level updates."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: int | None = None,
        max_seconds: float = 60.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.max_seconds = float(max_seconds)
        self._max_buffer = int(self.max_seconds * self.sample_rate)
        self._stream = None
        self._lock = threading.Lock()
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._live_level = 0.0
        self._voice_active = False
        self._device_name = None
        self._started_at = None
        self._stream_stopped = False

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        try:
            devices = sd.query_devices()
        except Exception:
            return []
        if isinstance(devices, dict):
            devices = [devices]
        output: list[dict[str, Any]] = []
        for index, device in enumerate(devices):
            if isinstance(device, dict) and device.get("max_input_channels", 0) > 0:
                output.append({
                    "index": index,
                    "name": str(device.get("name") or f"Microphone {index + 1}"),
                    "max_input_channels": int(device.get("max_input_channels", 0)),
                })
        return output

    @staticmethod
    def resolve_device(name: str | None, preferred_names: tuple[str, ...] = ("laptop microphone", "handsfree", "headset", "microphone")) -> int | None:
        devices = AudioRecorder.list_devices()
        if not devices:
            return None
        if name:
            normalized = name.lower()
            for device in devices:
                if normalized in device["name"].lower():
                    return device["index"]
        for token in preferred_names:
            for device in devices:
                if token in device["name"].lower():
                    return device["index"]
        return devices[0]["index"]

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        chunk = np.asarray(indata, dtype=np.float32)
        mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
        mono = np.asarray(mono, dtype=np.float32)
        with self._lock:
            self._audio_buffer = np.append(self._audio_buffer, mono)
            if self._audio_buffer.size > self._max_buffer:
                self._audio_buffer = self._audio_buffer[-self._max_buffer:]
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            self._live_level = min(100.0, max(0.0, rms * 500.0))
            self._voice_active = self._live_level > 8.0

    def start_stream(self, device_index: int | None = None):
        if self._stream is not None:
            self.stop_stream()
        target_index = device_index if device_index is not None else self.device_index
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._stream_stopped = False
        self._stream = sd.InputStream(
            device=target_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback,
            dtype="float32",
        )
        self._stream.start()
        self._started_at = time.time()
        self._device_name = AudioRecorder.list_devices()[target_index]["name"] if target_index is not None and target_index < len(AudioRecorder.list_devices()) else "Selected microphone"

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stream_stopped = True

    def get_buffer(self) -> np.ndarray:
        """Return the FULL captured buffer since the stream started."""
        with self._lock:
            return self._audio_buffer.copy()

    def get_elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.time() - self._started_at

    def is_recording(self) -> bool:
        return self._stream is not None

    def clear_buffer(self) -> None:
        with self._lock:
            self._audio_buffer = np.empty(0, dtype=np.float32)

    def get_live_waveform(self, length: int = 512) -> np.ndarray:
        with self._lock:
            samples = self._audio_buffer.copy()
        if samples.size == 0:
            return np.zeros(length, dtype=np.float32)
        if samples.size > length:
            samples = samples[-length:]
        if samples.size < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[-samples.size:] = samples
            return padded
        return samples

    def get_live_level(self) -> float:
        with self._lock:
            return float(self._live_level)

    def is_voice_active(self) -> bool:
        with self._lock:
            return bool(self._voice_active)

    def record_sample(self, duration: float = 3.0, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
        sampling_rate = sample_rate or self.sample_rate
        try:
            devices = sd.query_devices()
        except Exception:
            devices = None
        if devices is None:
            raise RuntimeError("Audio device detection failed. Please verify microphone access.")

        recording = sd.rec(
            int(duration * sampling_rate),
            samplerate=sampling_rate,
            channels=self.channels,
            dtype="float32",
            blocking=True,
        )
        sd.wait()
        audio = np.asarray(recording, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.size == 0 or not np.any(np.abs(audio) > 1e-6):
            raise ValueError("Recording is empty or unusable. Please speak closer to the microphone.")
        return audio, sampling_rate

    def save_wav(self, file_path: str, audio: np.ndarray, sample_rate: int) -> str:
        sf.write(file_path, np.asarray(audio, dtype=np.float32), sample_rate)
        return file_path
