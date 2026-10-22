from __future__ import annotations

import os
import subprocess
import sys

if __name__ == "__main__" and os.environ.get("VOICE_AUTH_STREAMLIT_APP") != "1":
    env = os.environ.copy()
    env["VOICE_AUTH_STREAMLIT_APP"] = "1"
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
            env=env,
        )
    )

import streamlit as st

from authentication.session import SessionManager
from database.database import DatabaseManager
from ui.landing import render_landing
from ui.theme import inject_global_styles

st.set_page_config(page_title="Voice Authentication System", page_icon="🎙️", layout="wide")

db = DatabaseManager()
session = SessionManager(db)
inject_global_styles()

record = session.get(st.session_state)

if not record:
    render_landing(db, session)
    st.stop()

role = record["role"]
subject = record["subject_id"]

with st.sidebar:
    st.markdown("## 🎙️ Voice Auth")
    st.markdown(f"**{subject}**")
    st.caption("Administrator" if role == "admin" else "User session")
    if st.button("Log out", use_container_width=True):
        session.logout(st.session_state)
        st.rerun()

if role == "admin":
    pages = [
        st.Page("portals/admin/dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("portals/admin/users.py", title="Manage Users", icon="👥"),
        st.Page("portals/admin/history.py", title="Authentication History", icon="🕘"),
        st.Page("portals/admin/attempts.py", title="Login Attempts", icon="🔐"),
        st.Page("portals/admin/security.py", title="Models & Performance", icon="🧪"),
    ]
else:
    pages = [
        st.Page("portals/user/dashboard.py", title="My Profile", icon="👤", default=True),
        st.Page("portals/user/history.py", title="My Login History", icon="🕘"),
        st.Page("portals/user/voice.py", title="Re-enroll Voice", icon="🎙️"),
    ]

navigation = st.navigation(pages)
navigation.run()
