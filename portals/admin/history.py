"""Admin portal: full authentication history across all users."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
guard("admin", db)

hero("Authentication History", "Every voice authentication attempt across the system.")

filter_id = st.text_input("Filter by User ID (optional)", placeholder="user_001")
logs = db.get_auth_history(filter_id.strip() or None)

if logs:
    frame = pd.DataFrame(logs)[
        ["timestamp", "user_id", "result", "score", "model_name", "processing_time_ms", "notes"]
    ]
    frame = frame.rename(
        columns={
            "timestamp": "Time",
            "user_id": "User ID",
            "result": "Result",
            "score": "Score",
            "model_name": "Model",
            "processing_time_ms": "Processing (ms)",
            "notes": "Notes",
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
else:
    st.info("No authentication attempts recorded yet.")
