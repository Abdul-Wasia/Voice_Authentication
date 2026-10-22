"""Admin portal: view and manage all registered users."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from authentication.passwords import hash_password
from authentication.registration import RegistrationManager
from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
guard("admin", db)

hero("Manage Users", "View and administer every registered account.")

users = db.get_all_users()
if not users:
    st.info("No users have registered yet.")
    st.stop()

rows = [
    {
        "User ID": u["user_id"],
        "Name": u["name"],
        "Status": "Active" if int(u.get("is_active") or 0) else "Disabled",
        "Model": u.get("default_model"),
        "Registered": u.get("registration_date"),
        "Last login": u.get("last_login") or "—",
    }
    for u in users
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
selected = st.selectbox("Select a user to manage", [u["user_id"] for u in users])
user = db.get_user_by_id(selected)
if not user:
    st.warning("User not found.")
    st.stop()

st.caption(f"Managing **{user['name']}** (`{user['user_id']}`)")

col1, col2, col3 = st.columns(3)
with col1:
    active = int(user.get("is_active") or 0) == 1
    if st.button("Disable account" if active else "Enable account", use_container_width=True):
        db.update_user(selected, is_active=0 if active else 1)
        st.success("Account updated.")
        st.rerun(scope="app")
with col2:
    if st.button("Revoke active sessions", use_container_width=True):
        db.revoke_sessions_for(selected)
        st.success("All sessions for this user were revoked.")
with col3:
    if st.button("Archive account", use_container_width=True):
        try:
            RegistrationManager(db_manager=db).delete_user(selected, archive=True)
            st.success(f"Archived {selected}.")
            st.rerun(scope="app")
        except Exception as exc:
            st.error(f"Archive failed: {exc}")

st.subheader("Reset password")
with st.form("admin_reset_password"):
    new_password = st.text_input("New password", type="password", help="At least 8 characters.")
    submitted = st.form_submit_button("Set password", type="primary")
if submitted:
    if len(new_password) < 8:
        st.error("Password must be at least 8 characters.")
    else:
        db.set_user_password(selected, hash_password(new_password))
        db.revoke_sessions_for(selected)
        st.success("Password updated. Share it with the user through a secure channel.")

st.subheader("Permanent delete")
st.caption("This removes the account, its voice profile folder, and its logs. This cannot be undone.")
confirm = st.text_input("Type the User ID to permanently delete", placeholder=selected)
if st.button("Permanently delete", type="secondary"):
    if confirm == selected:
        try:
            RegistrationManager(db_manager=db).delete_user(selected, archive=False)
            st.success(f"Deleted {selected}.")
            st.rerun(scope="app")
        except Exception as exc:
            st.error(f"Delete failed: {exc}")
    else:
        st.warning("The typed User ID does not match.")
