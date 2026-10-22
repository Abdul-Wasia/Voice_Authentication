"""Authorization guards for portal pages.

Each page calls ``guard(...)`` so authorization is enforced in the page logic,
not only by hiding navigation items.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from authentication.session import AuthError, SessionManager
from database.database import DatabaseManager


def guard(role: str, db_manager: DatabaseManager | None = None) -> Dict[str, Any]:
    db = db_manager or DatabaseManager()
    session = SessionManager(db)
    try:
        return session.require(st.session_state, role)
    except AuthError as exc:
        st.error(str(exc))
        st.info("Please return to the home page and log in with the correct account.")
        st.stop()


def sidebar_identity(record: Dict[str, Any]) -> None:
    role = record.get("role", "user")
    subject = record.get("subject_id", "")
    label = "Administrator" if role == "admin" else "User"
    st.sidebar.markdown(f"### {subject}")
    st.sidebar.caption(f"{label} session")
