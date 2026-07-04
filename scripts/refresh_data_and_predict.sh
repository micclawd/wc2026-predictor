#!/usr/bin/env bash
# WC 2026 — refresh upstream data + re-run betting-markets predictor.
#
# Step 1: cd data/26worldcup.github.io && git pull
#         (upstream updates matches.json within hours of final whistle)
# Step 2: python3 scripts/wc_betting_predictor.py --force-refresh
#         (re-derives everything; rolling ELO picks up new finished
#          matches automatically via matches.json)
#
# Idempotent — safe to run hourly, no side effects on success.
# Use this AFTER each R16 match settles to feed post-match form data
# into the next match's prediction.

set -euo pipefail

export HOME=/Users/michaellee
PREDICTOR_DIR=~/Projects/wc2026-predictor
DATA_DIR="$PREDICTOR_DIR/data/26worldcup.github.io"
LOG="$PREDICTOR_DIR/data/refresh.log"
mkdir -p "$PREDICTOR_DIR/data"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== refresh_data_and_predict START ==="

# --- Step 1: pull upstream data ---
if [ -d "$DATA_DIR/.git" ]; then
  log "pulling upstream data repo ..."
  if git -C "$DATA_DIR" pull --ff-only 2>&1 | tee -a "$LOG"; then
    NEW=$(git -C "$DATA_DIR" log --oneline -1 2>/dev/null || echo "?")
    log "  upstream: $NEW"
  else
    log "  WARN: git pull failed (probably local edits); continuing with existing data"
  fi
else
  log "  WARN: $DATA_DIR is not a git repo; cannot pull upstream"
fi

# --- Step 2: re-run predictor with fresh data ---
log "running wc_betting_predictor.py --force-refresh ..."
cd "$PREDICTOR_DIR"
python3 scripts/wc_betting_predictor.py --force-refresh 2>&1 | tee -a "$LOG" | tail -30

# --- Verify output ---
OUT="$PREDICTOR_DIR/download/wc2026_betting_markets.json"
if [ -f "$OUT" ]; then
  GENERATED=$(python3 -c "import json; print(json.load(open('$OUT'))['generated_at'])")
  N_MATCHES=$(python3 -c "import json; print(len(json.load(open('$OUT'))['matches']))")
  log "✅ markets updated: $GENERATED ($N_MATCHES matches)"
else
  log "❌ markets file not written — check logs above"
  exit 1
fi

log "=== refresh_data_and_predict DONE ==="
