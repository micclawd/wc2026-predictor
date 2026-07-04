#!/usr/bin/env bash
#
# Push the WC 2026 predictor to GitHub.
#
# Usage:
#   bash push_to_github.sh <github-url>
#
# Example:
#   bash push_to_github.sh https://github.com/yourusername/wc2026-predictor.git
#
# Or if you have a personal access token:
#   bash push_to_github.sh https://<token>@github.com/yourusername/wc2026-predictor.git
#
set -e

if [ -z "$1" ]; then
  echo "Usage: bash push_to_github.sh <github-url>"
  echo ""
  echo "Examples:"
  echo "  bash push_to_github.sh https://github.com/yourusername/wc2026-predictor.git"
  echo "  bash push_to_github.sh https://<token>@github.com/yourusername/wc2026-predictor.git"
  exit 1
fi

REPO_URL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Pushing WC 2026 Predictor to GitHub ==="
echo "Repo URL: $REPO_URL"
echo ""

cd "$SCRIPT_DIR"

# Check if git repo is initialized
if [ ! -d ".git" ]; then
  echo "[1/3] Initializing git repo ..."
  git init -b main
  git add .
  git commit -m "[v1.4.0] Initial release: scoreline + betting markets prediction engine"
else
  echo "[1/3] Git repo already initialized"
  git add .
  if ! git diff --cached --quiet; then
    git commit -m "[v1.4.0] Update: scoreline + betting markets prediction engine"
  fi
fi

echo ""
echo "[2/3] Adding remote 'origin' ..."
# Remove existing origin if present
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

echo ""
echo "[3/3] Pushing to GitHub ..."
git push -u origin main

echo ""
echo "=== Done! ==="
echo "Your repo is live at: ${REPO_URL%.git}"
echo ""
echo "To clone on another machine:"
echo "  git clone $REPO_URL"
echo "  cd wc2026-predictor"
echo "  bash setup.sh"
echo "  python3 scripts/wc_betting_predictor.py"
