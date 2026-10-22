"""Reusable microphone recorder widget with a live voice waveform."""

from __future__ import annotations

import numpy as np
import streamlit as st

from audio.recorder import AudioRecorder
from ui.waveform import level_figure, waveform_figure

TARGET_SR = 16000


def get_recorder() -> AudioRecorder:
    recorder = st.session_state.get("va_recorder")
    if recorder is None:
        recorder = AudioRecorder(sample_rate=TARGET_SR, max_seconds=30.0)
        st.session_state.va_recorder = recorder
    return recorder


def set_device(name: str) -> int | None:
    index = AudioRecorder.resolve_device(name)
    st.session_state.va_device_index = index
    get_recorder().device_index = index
    return index


def device_selector(key: str = "device_selector") -> str | None:
    devices = AudioRecorder.list_devices()
    if not devices:
        st.warning("No microphone detected. Connect a microphone to record voice.")
        return None
    names = [device["name"] for device in devices]
    current = st.session_state.get("va_device_name")
    index = names.index(current) if current in names else 0
    choice = st.selectbox("Microphone device", names, index=index, key=key)
    st.session_state.va_device_name = choice
    set_device(choice)
    return choice


def clear_capture(key: str) -> None:
    st.session_state[f"va_capture_{key}"] = None
    st.session_state[f"va_recording_{key}"] = False


def _capture_key(key: str) -> str:
    return f"va_capture_{key}"


@st.fragment(run_every=0.1)
def _live_strip(key: str, target_seconds: float) -> None:
    recorder = st.session_state.get("va_recorder")
    rec_key = f"va_recording_{key}"
    cap_key = _capture_key(key)
    if recorder is None or not recorder.is_recording():
        return
    buffer = recorder.get_buffer()
    window = buffer[-int(4 * TARGET_SR):] if buffer.size else buffer
    st.pyplot(waveform_figure(window), use_container_width=True)
    elapsed = recorder.get_elapsed()
    st.progress(min(elapsed / target_seconds, 1.0), text=f"{elapsed:0.1f}s / {target_seconds:0.0f}s")
    st.pyplot(level_figure(recorder.get_live_level()), use_container_width=True)
    if elapsed >= target_seconds:
        recorder.stop_stream()
        st.session_state[cap_key] = recorder.get_buffer()
        st.session_state[rec_key] = False
        st.rerun(scope="app")


def recorder_widget(key: str, target_seconds: float, sample_rate: int = TARGET_SR) -> np.ndarray | None:
    """Render a recorder. Returns captured audio (np.ndarray) once available."""
    recorder = get_recorder()
    recorder.sample_rate = sample_rate
    recorder.max_seconds = max(30.0, target_seconds + 2.0)
    recorder._max_buffer = int(recorder.max_seconds * sample_rate)
    cap_key = _capture_key(key)
    rec_key = f"va_recording_{key}"

    captured = st.session_state.get(cap_key)
    if captured is not None:
        st.pyplot(waveform_figure(captured), use_container_width=True)
        if st.button("Re-record", key=f"{key}_redo", use_container_width=True):
            clear_capture(key)
            st.rerun(scope="app")
        return captured

    if not st.session_state.get(rec_key, False):
        st.caption(f"Records up to {target_seconds:.0f} seconds, then stops automatically.")
        if st.button("●  Record voice", key=f"{key}_start", type="primary", use_container_width=True):
            index = st.session_state.get("va_device_index")
            recorder.device_index = index
            try:
                recorder.start_stream(device_index=index)
            except Exception as exc:  # pragma: no cover - device dependent
                st.error(f"Could not access the microphone: {exc}")
                return None
            st.session_state[rec_key] = True
            st.rerun(scope="app")
        return None

    if st.button("■  Stop now", key=f"{key}_stop", use_container_width=True):
        recorder.stop_stream()
        st.session_state[cap_key] = recorder.get_buffer()
        st.session_state[rec_key] = False
        st.rerun(scope="app")
    _live_strip(key, target_seconds)
    return None
