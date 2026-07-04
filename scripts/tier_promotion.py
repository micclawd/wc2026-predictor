#!/usr/bin/env python3
"""
WC 2026 — tier promotion check (called by iter-5/6/7/final runners).

Reads the latest wc2026_iter5_results.json (or iter6/iter7) and
checks if the new backtest passes the promotion criteria.
If so, updates sizing_tiers.CURRENT_TIER.

Tier 0 → 1: 24 matches, exact_scoreline_acc >= 0.50, brier < 0.20
Tier 1 → 2: 28 matches, exact_scoreline_acc >= 0.45
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PREDICTOR_DIR = Path("~/Projects/wc2026-predictor").expanduser()
sys.path.insert(0, str(PREDICTOR_DIR / "scripts"))

from sizing_tiers import TIER_0, TIER_1, TIER_2, CURRENT_TIER  # noqa


def check_tier_promotion(iter_version: str) -> dict:
    """
    Check if backtest results for iter_version pass tier promotion criteria.
    Returns dict {promoted: bool, from_tier, to_tier, reason}.
    """
    results_path = PREDICTOR_DIR / "download" / f"wc2026_iter{iter_version}_results.json"
    if not results_path.exists():
        return {"promoted": False, "reason": f"no results at {results_path}"}

    results = json.loads(results_path.read_text())
    n_matches = results.get("n_matches", 0)
    best = results.get("best", {})

    exact = best.get("exact_scoreline_acc", 0)
    brier = best.get("brier", 1.0)

    if iter_version in ("5", "5_final") and n_matches >= 24 and exact >= 0.50 and brier < 0.20:
        return {"promoted": True, "from_tier": "0", "to_tier": "1",
                "reason": f"iter-5: {n_matches} matches, {exact*100:.1f}% exact, brier {brier:.3f}"}
    if iter_version in ("6",) and n_matches >= 28 and exact >= 0.45:
        return {"promoted": True, "from_tier": "1", "to_tier": "2",
                "reason": f"iter-6: {n_matches} matches, {exact*100:.1f}% exact"}
    return {"promoted": False, "reason": f"iter-{iter_version}: {n_matches} matches, {exact*100:.1f}% exact, brier {brier:.3f} — did not pass criteria"}


def apply_tier_promotion(promo: dict, iter_version: str) -> bool:
    """Update sizing_tiers.py with the new CURRENT_TIER."""
    if not promo["promoted"]:
        return False

    sizes_path = PREDICTOR_DIR / "scripts" / "sizing_tiers.py"
    text = sizes_path.read_text()

    new_tier_name = f"TIER_{promo['to_tier']}"
    new_tier_var = f"TIER_{promo['to_tier']}"

    # Replace CURRENT_TIER assignment
    old_line = f"CURRENT_TIER = TIER_{promo['from_tier']}  # updated by the iter-5/6 retrain runners"
    new_line = f"CURRENT_TIER = {new_tier_var}  # PROMOTED from tier {promo['from_tier']} by iter-{iter_version}: {promo['reason']}"

    if old_line not in text:
        return False

    text = text.replace(old_line, new_line)
    sizes_path.write_text(text)
    return True


def main():
    if len(sys.argv) < 2:
        print("usage: tier_promotion.py <iter_version>")
        sys.exit(1)
    iter_version = sys.argv[1]
    promo = check_tier_promotion(iter_version)
    print(json.dumps(promo, indent=2))
    if apply_tier_promotion(promo, iter_version):
        print(f"  ✓ TIER PROMOTED: {promo['from_tier']} → {promo['to_tier']}")
    else:
        print("  no promotion applied")


if __name__ == "__main__":
    main()
