"""Admin portal: model availability and evaluation (honest labeling)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from evaluation.testing import EvaluationSuite
from models.speaker_verification import SpeakerVerifier
from ui.guards import guard
from ui.theme import hero, status_pill

db = DatabaseManager()
guard("admin", db)

hero("Models & Performance", "What is really implemented, and how it is evaluated.")

st.subheader("Verification models")
st.caption("Verification uses the strongest available model. Resemblyzer (256-d) ships with bundled "
           "weights and is used by default; ECAPA-TDNN downloads weights on first use. MFCC + cosine "
           "and MFCC + SVM are always available but are weak, non-discriminative fallbacks.")

if st.button("Download / verify pretrained weights", type="primary"):
    with st.spinner("Loading pretrained models (may download weights)..."):
        st.session_state["model_status"] = SpeakerVerifier.model_status()

st.markdown(f"**MFCC + cosine** — {status_pill('available')}", unsafe_allow_html=True)
st.markdown(f"**MFCC + SVM** — {status_pill('available')}", unsafe_allow_html=True)

status = st.session_state.get("model_status") or SpeakerVerifier.cached_model_status()
for name, state in status.items():
    if name in {"mfcc_svm", "mfcc_cosine"}:
        continue
    st.markdown(f"**{name}** — {status_pill(state)}", unsafe_allow_html=True)

st.divider()
st.subheader("Evaluation metrics")
st.warning(
    "These metrics come from a SYNTHETIC validation set for demonstration only. They are not measured "
    "on real speaker data and must not be reported as real-world accuracy."
)
results = EvaluationSuite().run()
metrics = results["metrics"]
cols = st.columns(6)
for col, name in zip(cols, ["accuracy", "precision", "recall", "far", "frr", "eer"]):
    col.metric(name.upper(), round(float(metrics[name]), 4))
st.caption(results["notes"])

st.divider()
st.subheader("Security scope")
st.markdown(
    """
    - Password + voice are both required for user login.
    - Passwords are stored as salted bcrypt hashes; plain-text passwords are never stored.
    - Brute-force attempts are rate-limited per identifier.
    - **Not implemented:** liveness detection and replay/synthetic-voice (anti-spoofing) protection.
    """
)
