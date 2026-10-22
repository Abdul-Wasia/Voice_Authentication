"""Admin portal: login attempt / brute-force monitoring."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
guard("admin", db)

hero("Login Attempts", "Password and admin login attempts, including rate-limit triggers.")

attempts = db.recent_login_attempts(limit=500)
if attempts:
    frame = pd.DataFrame(attempts)[["timestamp", "identifier", "role", "success", "reason"]]
    frame["success"] = frame["success"].map(lambda v: "Success" if int(v) else "Failure")
    frame = frame.rename(
        columns={"timestamp": "Time", "identifier": "Identifier", "role": "Role", "success": "Outcome", "reason": "Reason"}
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
else:
    st.info("No login attempts recorded yet.")
