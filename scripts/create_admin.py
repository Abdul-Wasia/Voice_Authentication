"""Create the first administrator account.

Usage (interactive):
    python scripts/create_admin.py

Usage (environment variables):
    set VOICE_AUTH_ADMIN_ID=admin
    set VOICE_AUTH_ADMIN_PASSWORD=your-strong-password
    python scripts/create_admin.py

There is intentionally NO default admin account and no hardcoded credentials.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authentication.passwords import hash_password  # noqa: E402
from database.database import DatabaseManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an administrator account.")
    parser.add_argument("--admin-id", default=os.environ.get("VOICE_AUTH_ADMIN_ID"))
    parser.add_argument("--password", default=os.environ.get("VOICE_AUTH_ADMIN_PASSWORD"))
    parser.add_argument("--role", default="superadmin")
    args = parser.parse_args()

    admin_id = (args.admin_id or input("Admin ID: ")).strip()
    if not admin_id:
        print("Admin ID is required.", file=sys.stderr)
        return 1

    password = args.password
    if not password:
        password = getpass.getpass("Password (min 8 chars): ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.", file=sys.stderr)
            return 1
    if len(password) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        return 1

    db = DatabaseManager()
    if db.get_admin_by_id(admin_id):
        print(f"Admin '{admin_id}' already exists.", file=sys.stderr)
        return 1

    db.create_admin(admin_id, hash_password(password), role=args.role)
    print(f"Admin '{admin_id}' created. You can now log in through the Admin portal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
