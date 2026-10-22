"""Secure password hashing helpers.

Uses bcrypt with a per-password random salt. Plain-text passwords are never
stored or logged anywhere in the system.
"""

from __future__ import annotations

import bcrypt

BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Return a bcrypt hash for ``password`` with a random salt."""
    if password is None or not str(password):
        raise ValueError("Password must not be empty.")
    encoded = str(password).encode("utf-8")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """Constant-time verification of ``password`` against ``password_hash``."""
    if not password or not password_hash:
        return False
    try:
        return bool(bcrypt.checkpw(str(password).encode("utf-8"), str(password_hash).encode("utf-8")))
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str | None, target_rounds: int = BCRYPT_ROUNDS) -> bool:
    """Return True when a stored hash should be upgraded to the current cost."""
    if not password_hash:
        return True
    parts = str(password_hash).split("$")
    if len(parts) < 4 or not parts[1].startswith("2"):
        return True
    try:
        return int(parts[2]) < target_rounds
    except (ValueError, IndexError):
        return True
