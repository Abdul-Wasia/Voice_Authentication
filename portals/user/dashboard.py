"""User portal: personal profile (only the logged-in user's own details)."""

from __future__ import annotations

import streamlit as st

from authentication.passwords import hash_password, verify_password
from database.database import DatabaseManager
from ui.guards import guard
from ui.theme import hero

db = DatabaseManager()
record = guard("user", db)
user_id = record["subject_id"]
user = db.get_user_by_id(user_id)

if not user:
    st.error("Your account could not be found. Please log in again.")
    st.stop()

hero(f"Welcome, {user['name']}", "This is your private profile. Only you can see these details.")

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Profile")
    st.markdown('<div class="va-panel">', unsafe_allow_html=True)
    st.write(f"**Name:** {user['name']}")
    st.write(f"**User ID:** {user['user_id']}")
    status = "Active" if int(user.get('is_active') or 0) else "Disabled"
    st.write(f"**Status:** {status}")
    st.write(f"**Voice model:** {user.get('default_model', 'mfcc_cosine')}")
    st.write(f"**Registered:** {user.get('registration_date', '—')}")
    st.write(f"**Last login:** {user.get('last_login') or '—'}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader("Change password")
    with st.form("change_password"):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password", help="At least 8 characters.")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password", type="primary", use_container_width=True)
    if submitted:
        if not verify_password(current, user.get("password_hash")):
            st.error("Current password is incorrect.")
        elif len(new) < 8:
            st.error("New password must be at least 8 characters.")
        elif new != confirm:
            st.error("New passwords do not match.")
        else:
            db.set_user_password(user_id, hash_password(new))
            st.success("Password updated.")
