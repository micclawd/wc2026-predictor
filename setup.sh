#!/usr/bin/env bash
#
# Setup script — clones the data repo and installs Python dependencies.
#
# Usage:
#   bash setup.sh
#
set -e

echo "=== WC 2026 Prediction Engine — Setup ==="
echo ""

# 1. Clone the data repo if not already present
DATA_DIR="data/26worldcup.github.io"
if [ ! -d "$DATA_DIR" ]; then
  echo "[1/3] Cloning data repository (26worldcup.github.io) ..."
  mkdir -p data
  git clone --depth 1 https://github.com/26worldcup/26worldcup.github.io.git "$DATA_DIR"
  echo "  ✓ Cloned to $DATA_DIR"
else
  echo "[1/3] Data repository already exists at $DATA_DIR"
  echo "  (to refresh: rm -rf $DATA_DIR && bash setup.sh)"
fi

# 2. Verify Python version
echo ""
echo "[2/3] Checking Python ..."
if command -v python3 &>/dev/null; then
  PYVER=$(python3 --version 2>&1)
  echo "  ✓ $PYVER"
else
  echo "  ✗ Python 3 not found. Please install Python 3.8+."
  exit 1
fi

# 3. Install Python dependencies
echo ""
echo "[3/3] Installing Python dependencies ..."
pip3 install --quiet numpy 2>&1 | tail -2
echo "  ✓ numpy installed"

# 4. Verify setup
echo ""
echo "=== Verification ==="
python3 scripts/config.py

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the betting markets predictor:"
echo "  python3 scripts/wc_betting_predictor.py"
echo ""
echo "To test with sample bookmaker odds:"
echo "  python3 scripts/wc_betting_predictor.py --bookmaker-odds scripts/sample_bookmaker_odds.json"
