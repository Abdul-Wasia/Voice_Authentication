import sqlite3

from authentication.passwords import hash_password
from database.database import DatabaseManager

OLD_USERS_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    folder_name TEXT,
    profile_path TEXT,
    registration_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    profile_version TEXT DEFAULT '1.0',
    default_model TEXT DEFAULT 'mfcc_cosine',
    secondary_pin_hash TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    lockout_until TEXT
)
"""


def test_migration_is_additive_and_preserves_data(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(OLD_USERS_SCHEMA)
    conn.execute("INSERT INTO users (user_id, name) VALUES ('legacy', 'Legacy User')")
    conn.commit()
    conn.close()

    db = DatabaseManager(db_path=db_path)
    user = db.get_user_by_id("legacy")
    assert user["name"] == "Legacy User"
    assert user["role"] == "user"
    assert "password_hash" in user
    assert user["password_hash"] is None

    # Re-initializing must be idempotent (no errors, data intact).
    DatabaseManager(db_path=db_path)
    assert db.get_user_by_id("legacy")["name"] == "Legacy User"


def test_user_password_and_admin_lifecycle(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "auth.db")
    db.create_user("u1", "User One", "u1_user", str(tmp_path / "u1"), password_hash=hash_password("password123"))
    assert db.get_user_by_id("u1")["password_hash"].startswith("$2")

    db.create_admin("admin1", hash_password("adminpass123"))
    assert db.get_admin_by_id("admin1")["admin_id"] == "admin1"
    assert db.count_admins() == 1


def test_sessions_expire_and_revoke(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "sessions.db")
    record = db.create_session("u1", "user", ttl_minutes=60)
    assert db.get_session(record["token"]) is not None
    db.revoke_session(record["token"])
    assert db.get_session(record["token"]) is None


def test_login_attempt_counting(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "attempts.db")
    db.record_login_attempt("bob", "user", False, "bad_password")
    db.record_login_attempt("bob", "user", False, "bad_password")
    db.record_login_attempt("bob", "user", True, "success")
    assert db.count_recent_failures("bob", "user", window_minutes=10) == 2
