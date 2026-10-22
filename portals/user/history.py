"""User portal: the logged-in user's own authentication history only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
record = guard("user", db)
user_id = record["subject_id"]

hero("My Login History", "Every authenticated attempt made on your account.")

history = db.get_auth_history(user_id)
if history:
    frame = pd.DataFrame(history)[["timestamp", "result", "score", "model_name", "processing_time_ms", "notes"]]
    frame = frame.rename(
        columns={
            "timestamp": "Time",
            "result": "Result",
            "score": "Score",
            "model_name": "Model",
            "processing_time_ms": "Processing (ms)",
            "notes": "Notes",
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
else:
    st.info("No login attempts recorded yet.")
