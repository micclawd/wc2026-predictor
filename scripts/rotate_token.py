#!/usr/bin/env python3
"""
WC 2026 — token rotation helper.

Triggers a fresh classic PAT (repo scope) rotation. Since this script
cannot interactively create a token on github.com, it just:
  1. Reminds Michael to create a new token at github.com/settings/tokens
  2. Reads the new token from $NEW_TOKEN env var (passed in via cron prompt)
  3. Updates ~/.git-credentials
  4. Tests the new token with `git ls-remote`
  5. Posts a summary

Usage (interactive, called by cron):
    NEW_TOKEN=ghp_NEW_TOKEN_HERE python3 rotate_token.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CRED_FILE = Path("~/.git-credentials").expanduser()
REPO_URL = "https://github.com/micclawd/wc2026-predictor.git"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def main():
    new_token = os.environ.get("NEW_TOKEN")
    if not new_token:
        log("❌ NEW_TOKEN env var not set. Cron prompt should pass it.")
        log("   To rotate manually: export NEW_TOKEN=ghp_NEW...")
        log("   Then: python3 scripts/rotate_token.py")
        sys.exit(1)

    log("Rotating GitHub token ...")

    # 1. Backup old credentials
    if CRED_FILE.exists():
        backup = CRED_FILE.with_suffix(".credentials.bak")
        backup.write_text(CRED_FILE.read_text())
        log(f"  backed up old credentials to {backup}")

    # 2. Write new credentials
    new_url = f"https://micclawd:{new_token}@github.com"
    CRED_FILE.write_text(new_url + "\n")
    CRED_FILE.chmod(0o600)
    log(f"  wrote new credentials to {CRED_FILE}")

    # 3. Test with git ls-remote
    log(f"  testing new token against {REPO_URL} ...")
    r = subprocess.run(["git", "ls-remote", REPO_URL], capture_output=True, text=True)
    if r.returncode == 0:
        log("  ✅ new token works")
    else:
        log(f"  ❌ new token failed: {r.stderr.strip()}")
        # Restore backup
        if CRED_FILE.with_suffix(".credentials.bak").exists():
            CRED_FILE.write_text(CRED_FILE.with_suffix(".credentials.bak").read_text())
            log("  restored old credentials")
        sys.exit(1)

    log("Token rotated. Old token revoked (assumed; Michael revokes on github.com).")
    log("⚠ REMINDER: revoke the OLD token on https://github.com/settings/tokens")
    log(f"   Old token to revoke: stored in {CRED_FILE.with_suffix('.credentials.bak')}")


if __name__ == "__main__":
    main()
