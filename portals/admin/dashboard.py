"""Admin portal: system overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
guard("admin", db)

hero("Admin Dashboard", "System-wide overview of accounts and authentication activity.")

users = db.get_all_users()
logs = db.get_auth_history()
attempts = db.recent_login_attempts(limit=500)

total = len(users)
active = sum(1 for u in users if int(u.get("is_active") or 0))
success = sum(1 for row in logs if str(row.get("result")).upper() == "SUCCESS")
failure = len(logs) - success
rate_limited = sum(1 for a in attempts if a.get("reason") == "rate_limited")

cols = st.columns(5)
cols[0].metric("Registered users", total)
cols[1].metric("Active users", active)
cols[2].metric("Successful logins", success)
cols[3].metric("Failed logins", failure)
cols[4].metric("Rate-limited events", rate_limited)

st.subheader("Recent authentication events")
if logs:
    frame = pd.DataFrame(logs)[["timestamp", "user_id", "result", "score", "model_name"]]
    frame = frame.rename(
        columns={"timestamp": "Time", "user_id": "User ID", "result": "Result", "score": "Score", "model_name": "Model"}
    )
    st.dataframe(frame.head(25), use_container_width=True, hide_index=True)
else:
    st.info("No authentication events recorded yet.")
