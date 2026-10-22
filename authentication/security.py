"""Security helpers: brute-force throttling and an EXPERIMENTAL replay heuristic.

IMPORTANT: This project does NOT implement genuine anti-spoofing / liveness
detection. ``experimental_replay_risk`` is a weak signal-only heuristic and must
not be presented to users as replay protection.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np

from database.database import DatabaseManager


class SecurityManager:
    """Rate limiting backed by the ``login_attempts`` table."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        max_failed_attempts: int = 5,
        window_minutes: int = 10,
    ):
        self.db = db_manager or DatabaseManager()
        self.max_failed_attempts = max_failed_attempts
        self.window_minutes = window_minutes

    def check_rate_limit(self, identifier: str, role: str = "user") -> Dict[str, Any]:
        failures = self.db.count_recent_failures(identifier, role, window_minutes=self.window_minutes)
        if failures >= self.max_failed_attempts:
            return {
                "allowed": False,
                "reason": (
                    f"Too many failed attempts. Please wait about {self.window_minutes} minutes "
                    "before trying again."
                ),
                "failures": failures,
            }
        return {"allowed": True, "reason": "Within allowed retry window.", "failures": failures}

    def record_attempt(self, identifier: str, role: str, success: bool, reason: str | None = None) -> None:
        self.db.record_login_attempt(identifier, role, success, reason)

    # Legacy per-user counters kept for backward compatibility with earlier code.
    def record_failure(self, user_id: str) -> Dict[str, Any]:
        user = self.db.get_user_by_id(user_id)
        if not user:
            return {"updated": False}
        failed_attempts = int(user.get("failed_attempts") or 0) + 1
        lockout_until = None
        if failed_attempts >= self.max_failed_attempts:
            lockout_until = (datetime.now() + timedelta(minutes=self.window_minutes)).isoformat()
        self.db.update_user(user_id, failed_attempts=failed_attempts, lockout_until=lockout_until)
        return {"updated": True, "failed_attempts": failed_attempts, "lockout_until": lockout_until}

    def reset_failures(self, user_id: str) -> None:
        self.db.update_user(user_id, failed_attempts=0, lockout_until=None)

    def experimental_replay_risk(self, audio: np.ndarray, enrolled_audio: np.ndarray | None = None) -> Dict[str, Any]:
        """Weak, experimental heuristic. NOT anti-spoofing; do not advertise as such."""
        sample = np.asarray(audio, dtype=np.float32).ravel()
        if sample.size == 0:
            return {"risk": "unknown", "reason": "Empty audio.", "experimental": True}

        if enrolled_audio is not None:
            enrolled = np.asarray(enrolled_audio, dtype=np.float32).ravel()
            n = min(len(sample), len(enrolled))
            if n > 1:
                corr = float(np.corrcoef(sample[:n], enrolled[:n])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                if corr > 0.99:
                    return {"risk": "medium", "reason": "Signal unusually similar to an enrolled sample.", "experimental": True}

        zero_crossing = float(np.mean(np.abs(np.diff(np.sign(sample + 1e-8)))))
        if zero_crossing < 1e-3:
            return {"risk": "medium", "reason": "Waveform is unusually constant.", "experimental": True}

        return {"risk": "low", "reason": "No obvious artefact detected (heuristic only).", "experimental": True}

    # Backwards-compatible alias.
    basic_replay_risk = experimental_replay_risk
