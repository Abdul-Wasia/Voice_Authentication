import json
import os
import sqlite3
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatabaseManager:
    """SQLite-backed data layer for accounts, voice profiles, sessions and logs."""

    USER_BASE_COLUMNS = [
        "id",
        "user_id",
        "name",
        "folder_name",
        "profile_path",
        "registration_date",
        "profile_version",
        "default_model",
        "secondary_pin_hash",
        "is_active",
        "failed_attempts",
        "lockout_until",
    ]

    def __init__(self, db_path: str | None = None):
        base_dir = Path(__file__).resolve().parents[1]
        override = os.environ.get("VOICE_AUTH_DB")
        default_path = Path(override) if override else base_dir / "data" / "voice_auth.db"
        self.db_path = Path(db_path) if db_path else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # ------------------------------------------------------------------ schema
    def _initialize(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
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
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    profile_path TEXT,
                    embedding_data TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    model_name TEXT,
                    score REAL,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    lockout_until TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    user_agent TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,
                    role TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT
                )
                """
            )
            self._migrate(conn)
            conn.commit()

    def _migrate(self, conn) -> None:
        """Additive, idempotent migrations. Never drops or rewrites existing data."""
        for column, definition in (
            ("password_hash", "TEXT"),
            ("role", "TEXT NOT NULL DEFAULT 'user'"),
            ("last_login", "TEXT"),
            ("must_change_password", "INTEGER NOT NULL DEFAULT 0"),
            ("voice_min_duration", "REAL DEFAULT 3.0"),
        ):
            self._ensure_column(conn, "users", column, definition)

    @staticmethod
    def _ensure_column(conn, table: str, column: str, definition: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def connect(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _rows_to_dicts(cursor, rows) -> List[Dict[str, Any]]:
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------- users
    def create_user(
        self,
        user_id: str,
        name: str,
        folder_name: str,
        profile_path: str,
        default_model: str = "mfcc_cosine",
        password_hash: str | None = None,
        secondary_pin_hash: str | None = None,
        profile_version: str = "1.0",
        role: str = "user",
        voice_min_duration: float = 3.0,
    ) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    user_id, name, folder_name, profile_path, default_model,
                    password_hash, secondary_pin_hash, profile_version, role, voice_min_duration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    name,
                    folder_name,
                    profile_path,
                    default_model,
                    password_hash,
                    secondary_pin_hash,
                    profile_version,
                    role,
                    voice_min_duration,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            columns = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
        return dict(zip(columns, row)) if row else {}

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(zip([d[0] for d in cursor.description], row))

    def get_all_users(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM users ORDER BY registration_date DESC")
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def next_user_id(self, prefix: str = "user_", width: int = 4) -> str:
        """Generate the next unique User ID automatically (e.g. user_0003)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT user_id FROM users WHERE user_id LIKE ?", (f"{prefix}%",)
            ).fetchall()
        max_n = 0
        for (uid,) in rows:
            suffix = str(uid)[len(prefix):]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
        candidate = f"{prefix}{max_n + 1:0{width}d}"
        while self.get_user_by_id(candidate):
            max_n += 1
            candidate = f"{prefix}{max_n + 1:0{width}d}"
        return candidate

    def update_user(self, user_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = tuple(fields.values()) + (user_id,)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE users SET {assignments} WHERE user_id = ?", values)
            conn.commit()

    def set_user_password(self, user_id: str, password_hash: str) -> None:
        self.update_user(user_id, password_hash=password_hash, must_change_password=0)

    def touch_user_login(self, user_id: str) -> None:
        self.update_user(user_id, last_login=_utcnow().isoformat(), failed_attempts=0, lockout_until=None)

    # ------------------------------------------------------------------ admins
    def create_admin(self, admin_id: str, password_hash: str, role: str = "superadmin") -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO admin_users (admin_id, password_hash, role) VALUES (?, ?, ?)",
                (admin_id, password_hash, role),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM admin_users WHERE id = ?", (cursor.lastrowid,)).fetchone()
            columns = [d[0] for d in conn.execute("SELECT * FROM admin_users LIMIT 0").description]
        return dict(zip(columns, row)) if row else {}

    def get_admin_by_id(self, admin_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM admin_users WHERE admin_id = ?", (admin_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(zip([d[0] for d in cursor.description], row))

    def get_all_admins(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM admin_users ORDER BY created_at DESC")
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def update_admin(self, admin_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = tuple(fields.values()) + (admin_id,)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE admin_users SET {assignments} WHERE admin_id = ?", values)
            conn.commit()

    def count_admins(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM admin_users").fetchone()[0])

    # ---------------------------------------------------------------- profiles
    def save_profile_record(self, user_id: str, model_name: str, profile_path: str, embedding_data: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO voice_profiles (user_id, model_name, profile_path, embedding_data)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, model_name, profile_path, json.dumps(embedding_data, default=str)),
            )
            conn.commit()

    def get_profile_records(self, user_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM voice_profiles WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
            )
            return self._rows_to_dicts(cursor, cursor.fetchall())

    # ----------------------------------------------------------------- auth log
    def save_auth_attempt(
        self,
        user_id: str,
        result: str,
        score: float,
        model_name: str,
        processing_time_ms: float,
        notes: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO auth_logs (user_id, model_name, score, result, processing_time_ms, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, model_name, score, result, processing_time_ms, notes),
            )
            conn.commit()

    def get_auth_history(self, user_id: str | None = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM auth_logs"
        params: tuple[Any, ...] = ()
        if user_id:
            query += " WHERE user_id = ?"
            params = (user_id,)
        query += " ORDER BY timestamp DESC"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return self._rows_to_dicts(cursor, cursor.fetchall())

    def delete_user(self, user_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM auth_logs WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM voice_profiles WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()

    # ----------------------------------------------------------------- sessions
    def create_session(self, subject_id: str, role: str, ttl_minutes: int = 60, user_agent: str | None = None) -> Dict[str, Any]:
        token = secrets.token_urlsafe(32)
        issued = _utcnow()
        expires = issued + timedelta(minutes=ttl_minutes)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (token, subject_id, role, issued_at, expires_at, user_agent) VALUES (?, ?, ?, ?, ?, ?)",
                (token, subject_id, role, issued.isoformat(), expires.isoformat(), user_agent),
            )
            conn.commit()
        return {
            "token": token,
            "subject_id": subject_id,
            "role": role,
            "issued_at": issued.isoformat(),
            "expires_at": expires.isoformat(),
        }

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,))
            row = cursor.fetchone()
            if row is None:
                return None
            record = dict(zip([d[0] for d in cursor.description], row))
        try:
            if _utcnow() > datetime.fromisoformat(record["expires_at"]):
                self.revoke_session(token)
                return None
        except (ValueError, TypeError):
            return None
        if int(record.get("revoked") or 0) == 1:
            return None
        return record

    def revoke_session(self, token: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sessions SET revoked = 1 WHERE token = ?", (token,))
            conn.commit()

    def revoke_sessions_for(self, subject_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE sessions SET revoked = 1 WHERE subject_id = ?", (subject_id,))
            conn.commit()

    # ------------------------------------------------------------ login attempts
    def record_login_attempt(self, identifier: str, role: str, success: bool, reason: str | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO login_attempts (identifier, role, success, timestamp, reason) VALUES (?, ?, ?, ?, ?)",
                (identifier, role, 1 if success else 0, _utcnow().isoformat(), reason),
            )
            conn.commit()

    def count_recent_failures(self, identifier: str, role: str, window_minutes: int = 10) -> int:
        since = (_utcnow() - timedelta(minutes=window_minutes)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM login_attempts WHERE identifier = ? AND role = ? AND success = 0 AND timestamp >= ?",
                    (identifier, role, since),
                ).fetchone()[0]
            )

    def recent_login_attempts(self, limit: int = 200) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM login_attempts ORDER BY timestamp DESC LIMIT ?", (limit,))
            return self._rows_to_dicts(cursor, cursor.fetchall())

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def _user_columns():
        return list(DatabaseManager.USER_BASE_COLUMNS)

    @staticmethod
    def _auth_columns():
        return [
            "id",
            "user_id",
            "model_name",
            "score",
            "result",
            "timestamp",
            "processing_time_ms",
            "notes",
        ]
