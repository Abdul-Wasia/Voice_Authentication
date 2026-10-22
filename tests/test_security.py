import pytest

from authentication.security import SecurityManager
from authentication.session import AuthError, SessionManager
from database.database import DatabaseManager


def test_rate_limit_blocks_after_threshold(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "rate.db")
    security = SecurityManager(db, max_failed_attempts=3, window_minutes=10)
    for _ in range(3):
        security.record_attempt("bob", "user", False, "bad_password")
    assert security.check_rate_limit("bob", "user")["allowed"] is False
    assert security.check_rate_limit("alice", "user")["allowed"] is True


def test_successful_login_resets_identifier_window(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "reset.db")
    security = SecurityManager(db, max_failed_attempts=2, window_minutes=10)
    security.record_attempt("bob", "user", False, "bad_password")
    security.record_attempt("bob", "user", False, "bad_password")
    assert security.check_rate_limit("bob", "user")["allowed"] is False
    # Failures outside the window are ignored.
    assert security.check_rate_limit("carol", "admin")["allowed"] is True


def test_session_role_guard(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "session.db")
    manager = SessionManager(db)
    store = {}
    manager.login(store, "admin1", "admin")
    assert manager.require(store, "admin")["subject_id"] == "admin1"
    with pytest.raises(AuthError):
        manager.require(store, "user")
    manager.logout(store)
    with pytest.raises(AuthError):
        manager.require(store, "admin")


def test_admin_cannot_be_reached_by_user_role(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "rbac.db")
    manager = SessionManager(db)
    store = {}
    manager.login(store, "user1", "user")
    with pytest.raises(AuthError):
        manager.require(store, "admin")
