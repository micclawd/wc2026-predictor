#!/usr/bin/env bash
# WC 2026 — run a retrain (iter-5/6/7/final).
# Usage: retrain_runner.sh <iter-version>
#   iter-version: 5 (after R16), 6 (after QF), 7 (after SF), final (after Jul 19)

set -euo pipefail
export HOME=/Users/michaellee
PREDICTOR_DIR=~/Projects/wc2026-predictor
LOG="$PREDICTOR_DIR/data/retrain.log"
mkdir -p "$PREDICTOR_DIR/data"

ITER="${1:-5}"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== retrain iter-$ITER START ==="

# Pull latest data
DATA_DIR="$PREDICTOR_DIR/data/26worldcup.github.io"
if [ -d "$DATA_DIR/.git" ]; then
  log "pulling upstream data ..."
  git -C "$DATA_DIR" pull --ff-only 2>&1 | tail -3 | tee -a "$LOG" || log "  WARN: pull failed"
fi

cd "$PREDICTOR_DIR"

if [ "$ITER" = "final" ]; then
  log "running final_wind_down.py"
  python3 scripts/final_wind_down.py 2>&1 | tee -a "$LOG"
else
  log "running wc_predictor_iter$ITER.py"
  python3 "scripts/wc_predictor_iter$ITER.py" 2>&1 | tee -a "$LOG" | tail -50

  # Check tier promotion (0 → 1, 1 → 2)
  log "checking tier promotion ..."
  python3 scripts/tier_promotion.py "$ITER" 2>&1 | tee -a "$LOG"
fi

log "=== retrain iter-$ITER DONE ==="
