"""User portal: re-enroll the user's own voice profile."""

from __future__ import annotations

import streamlit as st

from authentication.registration import RegistrationManager
from database.database import DatabaseManager
from ui import recorder_ui
from ui.guards import guard
from ui.theme import hero
from ui.waveform import waveform_figure

db = DatabaseManager()
record = guard("user", db)
user_id = record["subject_id"]
user = db.get_user_by_id(user_id)
if not user:
    st.error("Account not found.")
    st.stop()

hero("Re-enroll your voice", "Record fresh samples to update your voice profile. Your account data is unchanged.")

RE_KEY = "reenroll_voice"
samples = st.session_state.setdefault("reenroll_samples", [])
sample_count = st.session_state.setdefault("reenroll_count", 4)
duration = st.session_state.setdefault("reenroll_duration", 6.0)

cols = st.columns(3)
sample_count = cols[0].selectbox("Samples (4–6)", [4, 5, 6], index=[4, 5, 6].index(sample_count))
duration = cols[1].selectbox("Seconds each (5–8)", [5.0, 6.0, 7.0, 8.0], index=[5.0, 6.0, 7.0, 8.0].index(duration))
cols[2].metric("Recorded", f"{len(samples)} / {sample_count}")
st.session_state.reenroll_count = sample_count
st.session_state.reenroll_duration = duration

recorder_ui.device_selector(key="reenroll_device")

if len(samples) >= sample_count:
    st.subheader("Preview")
    for sample in samples:
        st.pyplot(waveform_figure(sample), use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Restart", use_container_width=True):
        st.session_state.reenroll_samples = []
        recorder_ui.clear_capture(RE_KEY)
        st.rerun(scope="app")
    if c2.button("Save new voice profile", type="primary", use_container_width=True):
        try:
            with st.spinner("Updating your voice profile..."):
                RegistrationManager(db_manager=db).update_voice_profile(
                    user_id=user_id,
                    sample_count=len(samples),
                    duration=float(duration),
                    recording_factory=lambda idx, _d: (samples[idx - 1], 16000),
                )
            st.session_state.reenroll_samples = []
            recorder_ui.clear_capture(RE_KEY)
            st.success("Voice profile updated successfully.")
        except Exception as exc:
            st.error(f"Could not update the voice profile: {exc}")
    st.stop()

index = len(samples) + 1
st.markdown(f"**Recording sample {index} of {sample_count}**")
captured = recorder_ui.recorder_widget(RE_KEY, float(duration))
if captured is not None and getattr(captured, "size", 0) > 0:
    if st.button("Save this sample", type="primary", use_container_width=True):
        samples.append(captured)
        st.session_state.reenroll_samples = samples
        recorder_ui.clear_capture(RE_KEY)
        st.rerun(scope="app")
