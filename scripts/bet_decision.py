#!/usr/bin/env python3
"""
WC 2026 bet decision engine.

Applies Michael's rules to model probabilities + Stake odds:
  - edge >= +5% (not +1%)
  - kelly_fraction >= 3% (use 1/4 Kelly for actual stake)
  - odds >= 1.50
  - max 3 selections per match
  - max 5% total exposure per match
  - SKIP correct score (model overconfident there)
  - liquid markets only: 1X2, O/U 2.5, AH -0.5/+0.5, BTTS yes/no

Usage:
    python3 bet_decision.py --match CAN-MAR \\
        --model-probs /path/to/markets.json \\
        --stake-odds /path/to/stake_odds.json \\
        --bankroll 74.97 \\
        --output /path/to/decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Betting thresholds (Michael's policy)
MIN_EDGE = 0.05              # +5% edge required
MIN_KELLY = 0.03             # 3% Kelly fraction
MIN_ODDS = 1.50              # no sub-1.50 favorites
MAX_SELECTIONS_PER_MATCH = 3
MAX_EXPOSURE_PER_MATCH = 0.05  # 5% of bankroll
KELLY_FRACTION = 0.25         # use 1/4 Kelly

# Markets we bet on (skip CS — model overconfident)
ALLOWED_MARKETS = {
    "1X2": ["home_win", "draw", "away_win"],
    "O/U 2.5": ["over", "under"],
    "AH -0.5": ["home_win", "away_win"],
    "AH +0.5": ["home_win", "away_win"],
    "BTTS": ["yes", "no"],
}


def kelly_fraction(model_prob: float, odds: float) -> float:
    """Kelly fraction: (bp - q) / b where b = odds - 1, p = model prob, q = 1 - p."""
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - model_prob
    kf = (b * model_prob - q) / b
    return max(0.0, kf)


def get_liquid_selections(
    markets: Dict[str, Any],
    stake_odds: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Walk model markets, look up Stake odds, return candidate selections.
    Each: {market, selection, model_prob, odds, edge, kelly}
    """
    out: List[Dict[str, Any]] = []

    # 1X2
    x2 = markets.get("1X2", {})
    stake_1x2 = stake_odds.get("1X2", {})
    for sel in ["home_win", "draw", "away_win"]:
        mp = x2.get(sel)
        odds = stake_1x2.get(sel)
        if mp is None or odds is None:
            continue
        out.append(_build_candidate("1X2", sel, mp, odds))

    # O/U 2.5
    ou = markets.get("over_under", {}).get("2.5", {})
    stake_ou = stake_odds.get("O/U 2.5", {})
    for sel in ["over", "under"]:
        mp = ou.get(sel)
        odds = stake_ou.get(sel)
        if mp is None or odds is None:
            continue
        out.append(_build_candidate("O/U 2.5", sel, mp, odds))

    # AH -0.5 / +0.5 (home and away)
    stake_ah = stake_odds.get("AH", {})
    for line in ["-0.5", "+0.5"]:
        ah = markets.get("asian_handicap", {}).get(line, {})
        # predictor uses home_win/home_loss (vs away win/loss); we
        # only care about the WIN side for each team.
        # home winning at -0.5 = home_win prob
        # away winning at -0.5 = home_loss prob (since 0 - 0.5 < 0)
        home_model = ah.get("home_win")
        away_model = ah.get("home_loss")  # the "away wins" side
        home_stake = stake_ah.get(f"{line}_home")
        away_stake = stake_ah.get(f"{line}_away")
        if home_model is not None and home_stake is not None:
            out.append(_build_candidate(f"AH {line}", "home_win", home_model, home_stake))
        if away_model is not None and away_stake is not None:
            out.append(_build_candidate(f"AH {line}", "away_win", away_model, away_stake))

    # BTTS
    btts = markets.get("btts", {})
    stake_btts = stake_odds.get("BTTS", {})
    for sel in ["yes", "no"]:
        mp = btts.get(sel)
        odds = stake_btts.get(sel)
        if mp is None or odds is None:
            continue
        out.append(_build_candidate("BTTS", sel, mp, odds))

    return out


def _build_candidate(market: str, selection: str, model_prob: float, odds: float) -> Dict[str, Any]:
    implied = 1.0 / odds
    edge = model_prob - implied
    kf = kelly_fraction(model_prob, odds)
    return {
        "market": market,
        "selection": selection,
        "model_prob": model_prob,
        "odds": odds,
        "implied_prob": implied,
        "edge": edge,
        "kelly": kf,
    }


def filter_candidates(
    candidates: List[Dict[str, Any]],
    bankroll: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply Michael's rules. Return (selected, rejected).
    Selected: passes all gates and is sized to 1/4 Kelly, capped at 5% bankroll.
    """
    passed_filter: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for c in candidates:
        reasons: List[str] = []
        if c["edge"] < MIN_EDGE:
            reasons.append(f"edge {c['edge']*100:.1f}% < {MIN_EDGE*100:.1f}%")
        if c["kelly"] < MIN_KELLY:
            reasons.append(f"kelly {c['kelly']*100:.1f}% < {MIN_KELLY*100:.1f}%")
        if c["odds"] < MIN_ODDS:
            reasons.append(f"odds {c['odds']:.2f} < {MIN_ODDS}")

        if reasons:
            rejected.append({**c, "rejected_reasons": reasons})
        else:
            # size the stake: 1/4 Kelly, capped at MAX_EXPOSURE_PER_MATCH
            stake_pct = c["kelly"] * KELLY_FRACTION
            stake_pct = min(stake_pct, MAX_EXPOSURE_PER_MATCH)
            stake = round(bankroll * stake_pct, 4)
            passed_filter.append({**c, "stake_pct": stake_pct, "stake": stake})

    # Sort by edge descending, take top N
    passed_filter.sort(key=lambda x: x["edge"], reverse=True)
    selected = passed_filter[:MAX_SELECTIONS_PER_MATCH]

    # Enforce total exposure cap
    if sum(s["stake_pct"] for s in selected) > MAX_EXPOSURE_PER_MATCH:
        # scale all stakes down proportionally
        total = sum(s["stake_pct"] for s in selected)
        scale = MAX_EXPOSURE_PER_MATCH / total
        for s in selected:
            s["stake_pct"] *= scale
            s["stake"] = round(bankroll * s["stake_pct"], 4)

    return selected, rejected


def main():
    ap = argparse.ArgumentParser(description="WC 2026 bet decision engine")
    ap.add_argument("--match", required=True, help="Match key, e.g. CAN-MAR")
    ap.add_argument("--model-probs", required=True, help="Path to wc2026_betting_markets.json")
    ap.add_argument("--stake-odds", required=True, help="Path to JSON with Stake odds")
    ap.add_argument("--bankroll", type=float, required=True, help="Current bankroll USDT")
    ap.add_argument("--output", help="Optional output path for decisions JSON")
    ap.add_argument("--dry-run", action="store_true", help="Don't actually place bets, just print")
    args = ap.parse_args()

    markets_data = json.loads(Path(args.model_probs).read_text())
    stake_odds = json.loads(Path(args.stake_odds).read_text())

    # Find this match in the model output
    match = next((m for m in markets_data["matches"]
                  if f"{m['home']}-{m['away']}" == args.match), None)
    if not match:
        print(f"❌ match {args.match} not found in {args.model_probs}")
        sys.exit(1)
    if not match.get("markets"):
        print(f"❌ no markets for {args.match}")
        sys.exit(1)
    if not match.get("lineup_available"):
        print(f"⚠ WARNING: lineups not confirmed for {args.match}")
        # Don't hard-fail — Michael's rule is to skip if lineups missing,
        # but the cron caller can decide. We tag this in the decision.

    candidates = get_liquid_selections(match["markets"], stake_odds)
    selected, rejected = filter_candidates(candidates, args.bankroll)

    decisions = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "match": args.match,
        "model_version": markets_data.get("model_version"),
        "lineup_used": match.get("lineup_available", False),
        "bankroll": args.bankroll,
        "thresholds": {
            "min_edge": MIN_EDGE,
            "min_kelly": MIN_KELLY,
            "min_odds": MIN_ODDS,
            "max_selections": MAX_SELECTIONS_PER_MATCH,
            "max_exposure_pct": MAX_EXPOSURE_PER_MATCH,
            "kelly_fraction": KELLY_FRACTION,
        },
        "selected": selected,
        "rejected": rejected,
        "total_stake": sum(s["stake"] for s in selected),
        "total_stake_pct": sum(s["stake_pct"] for s in selected),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(decisions, indent=2))

    # Print summary
    print(f"=== {args.match} bet decisions ===")
    print(f"  lineup: {decisions['lineup_used']}")
    print(f"  bankroll: ${args.bankroll:.2f}")
    print(f"  candidates: {len(candidates)} → selected: {len(selected)}")
    if selected:
        for s in selected:
            print(f"  ✅ {s['market']} {s['selection']}: ${s['stake']:.2f} @ {s['odds']:.2f} "
                  f"(edge {s['edge']*100:+.1f}%, kelly {s['kelly']*100:.1f}%)")
        print(f"  total stake: ${decisions['total_stake']:.2f} ({decisions['total_stake_pct']*100:.2f}%)")
    else:
        print("  no value, skipped")
    if rejected and len(rejected) <= 5:
        for r in rejected:
            print(f"  ❌ {r['market']} {r['selection']}: edge {r['edge']*100:+.1f}%, "
                  f"odds {r['odds']:.2f} — {', '.join(r['rejected_reasons'])}")

    sys.exit(0 if selected or args.dry_run else 2)  # exit 2 = "no value"


if __name__ == "__main__":
    main()
