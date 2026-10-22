"""Server-side session handling and role-based access control.

Sessions are persisted in the ``sessions`` table so they can expire and be
revoked. The opaque token lives in the caller's session store (Streamlit's
``st.session_state`` for the web app, or a plain dict in tests).
"""

from __future__ import annotations

from typing import Any, MutableMapping

from database.database import DatabaseManager

SESSION_KEY = "auth_session"


class AuthError(PermissionError):
    """Raised when an action is attempted without the required role/session."""


class SessionManager:
    VALID_ROLES = {"user", "admin"}

    def __init__(self, db_manager: DatabaseManager | None = None, ttl_minutes: int = 60):
        self.db = db_manager or DatabaseManager()
        self.ttl_minutes = ttl_minutes

    def login(self, store: MutableMapping[str, Any], subject_id: str, role: str, user_agent: str | None = None) -> dict:
        if role not in self.VALID_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        record = self.db.create_session(subject_id, role, ttl_minutes=self.ttl_minutes, user_agent=user_agent)
        store[SESSION_KEY] = record["token"]
        return record

    def get(self, store: MutableMapping[str, Any]) -> dict | None:
        token = store.get(SESSION_KEY)
        if not token:
            return None
        record = self.db.get_session(token)
        if not record:
            store.pop(SESSION_KEY, None)
            return None
        return record

    def logout(self, store: MutableMapping[str, Any]) -> None:
        token = store.pop(SESSION_KEY, None)
        if token:
            self.db.revoke_session(token)

    def require(self, store: MutableMapping[str, Any], *roles: str) -> dict:
        record = self.get(store)
        if not record:
            raise AuthError("No active session. Please log in again.")
        if roles and record.get("role") not in roles:
            raise AuthError("You are not authorized to view this page.")
        return record

    def is_authenticated(self, store: MutableMapping[str, Any]) -> bool:
        return self.get(store) is not None
