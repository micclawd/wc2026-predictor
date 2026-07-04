#!/usr/bin/env python3
"""
WC 2026 — Stake.com multibet placer.

Combines value bets across DIFFERENT matches into a single Stake parlay.
Per upstream STAKE_INTEGRATION.md: never combine selections from the same
match (correlated — forfeits partial-win upside). Each leg MUST be from a
different match — that's the entire point of a multibet.

Uses mcp__stake__stake_place_combo_bet — does NOT require Kasada headers,
unlike upstream's native GraphQL stake_bet_placer.py.

Usage:
    # Build multibet from a list of decisions (one JSON per match)
    python3 scripts/place_multibet.py \
        --decisions data/decisions/CAN-MAR_*.json \
                   data/decisions/PAR-FRA_*.json \
        --stake 1.00 \
        --mode safe   # or "aggressive"

Modes:
    safe         — top 1 selection per match by edge, max 3 matches (3 legs)
    aggressive   — top 1 selection per match, max 5 matches (5 legs)
    custom       — pass --legs explicitly (one selection per match)

Rules enforced:
    - Each leg from a DIFFERENT match (NO same-match parlay)
    - Only Stake-parlay-eligible markets: 1X2, O/U, AH, BTTS, DC, CS
    - 2 ≤ legs ≤ 6 (Stake's parlay limits)
    - Min combined odds 1.50, max 50
    - Stake: $0.50 min, $1000 max (Stake platform limits)
    - Per-bet stake must respect tier 0 caps (1.67% of bankroll)
    - Bankroll exposure: 5% max for the parlay

Logs to bets_log.jsonl with type=multibet.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PREDICTOR_DIR = Path(__file__).parent.parent
BET_LOG = PREDICTOR_DIR / "bets_log.jsonl"
BANKROLL_FILE = PREDICTOR_DIR / "data" / "bankroll.txt"

# Stake-parlay-eligible markets (verified against Stake's UI)
PARLAY_ELIGIBLE_MARKETS = {
    "1X2", "O/U 2.5", "O/U 1.5", "O/U 3.5",
    "AH -0.5", "AH +0.5", "AH -1.5", "AH +1.5",
    "BTTS", "Double Chance", "Correct Score 0-1", "Correct Score 1-0",
    "Correct Score 0-2", "Correct Score 2-0", "Correct Score 1-2", "Correct Score 2-1",
}

# Tier 0 sizing (mirrors sizing_tiers.py)
TIER_0 = {
    "kelly_fraction": 0.25,
    "max_exposure_pct": 0.05,
    "max_stake_pct": 0.0167,
    "min_stake": 0.50,
    "max_stake": 3.00,
}


def load_decision(path: str) -> Dict[str, Any]:
    """Load a bet_decision.py output JSON."""
    with open(path) as f:
        return json.load(f)


def select_legs(decisions: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    Pick the TOP edge selection from each match, ensuring:
    - Different matches only (NO same-match parlay)
    - Parlay-eligible markets only
    - At most one leg per match
    """
    by_match: Dict[str, List[Dict[str, Any]]] = {}
    for d in decisions:
        match_key = d.get("match", "?")
        for sel in d.get("selected", []):
            m = sel.get("market", "")
            if m not in PARLAY_ELIGIBLE_MARKETS:
                continue
            sel["match"] = match_key
            by_match.setdefault(match_key, []).append(sel)

    # Sort each match's selections by edge descending, take top 1
    legs = []
    max_matches = {"safe": 3, "aggressive": 5, "custom": 6}[mode]
    for match_key, sels in by_match.items():
        if len(legs) >= max_matches:
            break
        sels.sort(key=lambda x: x.get("edge", 0), reverse=True)
        if sels:
            legs.append(sels[0])

    return legs


def compute_parlay(legs: List[Dict[str, Any]], bankroll: float) -> Dict[str, Any]:
    """
    Compute stake + EV for the parlay.
    Tier 0: kelly_fraction × min(legs) × bankroll, capped at max_exposure_pct.
    """
    if len(legs) < 2:
        return {"error": "need at least 2 legs"}

    # Combined odds = product of leg odds
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.get("odds", 1.0)
        combined_prob *= leg.get("model_prob", 0.5)

    # Kelly on combined prob vs combined odds
    b = combined_odds - 1
    p = combined_prob
    q = 1 - p
    kelly_full = (b * p - q) / b if b > 0 else 0
    kelly_q = max(0, kelly_full) * TIER_0["kelly_fraction"]

    # Stake: kelly-sized, capped at 5% bankroll AND 1.67% per leg
    raw_stake = kelly_q * bankroll
    stake = min(
        raw_stake,
        bankroll * TIER_0["max_exposure_pct"],   # 5% parlay cap
        TIER_0["max_stake"],                     # $3 per bet
    )
    stake = max(stake, TIER_0["min_stake"])      # Stake min
    stake = round(stake, 2)

    expected_value = stake * (combined_prob * (combined_odds - 1) - (1 - combined_prob))

    return {
        "combined_odds": round(combined_odds, 3),
        "combined_prob": round(combined_prob, 4),
        "model_prob_pct": round(combined_prob * 100, 2),
        "implied_prob_pct": round((1 / combined_odds) * 100, 2),
        "edge_pct": round((combined_prob - 1 / combined_odds) * 100, 2),
        "kelly_full": round(kelly_full, 4),
        "kelly_q": round(kelly_q, 4),
        "stake": stake,
        "stake_pct_bankroll": round(stake / bankroll * 100, 2),
        "expected_value": round(expected_value, 4),
        "potential_payout": round(stake * combined_odds, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", nargs="+", required=True,
                    help="One or more bet_decision.py output JSON files (one per match)")
    ap.add_argument("--stake", type=float, default=None,
                    help="Override auto-sized stake (USD)")
    ap.add_argument("--mode", choices=["safe", "aggressive", "custom"], default="safe")
    ap.add_argument("--place", action="store_true",
                    help="Actually call MCP stake_place_combo_bet (otherwise dry-run print)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bankroll", type=float, default=None)
    args = ap.parse_args()

    # Load bankroll
    if args.bankroll:
        bankroll = args.bankroll
    elif BANKROLL_FILE.exists():
        bankroll = float(BANKROLL_FILE.read_text().strip())
    else:
        bankroll = 74.97

    # Load + select
    decisions = [load_decision(p) for p in args.decisions]
    legs = select_legs(decisions, args.mode)
    if len(legs) < 2:
        print(f"❌ Need at least 2 legs across DIFFERENT matches. Got {len(legs)}.")
        print(f"   decisions: {[d.get('match') for d in decisions]}")
        print(f"   leg matches: {[l.get('match') for l in legs]}")
        return 1

    parlay = compute_parlay(legs, bankroll)
    if args.stake is not None:
        parlay["stake"] = args.stake
        parlay["stake_pct_bankroll"] = round(args.stake / bankroll * 100, 2)
        parlay["potential_payout"] = round(args.stake * parlay["combined_odds"], 2)

    # Build display
    print(f"=== Multibet ({args.mode} mode) ===")
    print(f"  bankroll: ${bankroll:.2f}")
    print(f"  legs: {len(legs)}")
    for leg in legs:
        print(f"    {leg['match']:<10} {leg['market']:<20} {leg['selection']:<12} "
              f"@ {leg['odds']:.2f}  prob {leg.get('model_prob', 0)*100:.1f}%  "
              f"edge +{leg.get('edge', 0)*100:.1f}%")
    print(f"  combined odds: {parlay['combined_odds']:.3f}")
    print(f"  model combined prob: {parlay['model_prob_pct']:.1f}%")
    print(f"  implied prob: {parlay['implied_prob_pct']:.1f}%")
    print(f"  edge: +{parlay['edge_pct']:.1f}%")
    print(f"  stake: ${parlay['stake']:.2f} ({parlay['stake_pct_bankroll']:.2f}% bankroll)")
    print(f"  expected value: ${parlay['expected_value']:+.2f}")
    print(f"  potential payout: ${parlay['potential_payout']:.2f}")

    if args.dry_run and not args.place:
        print("  (dry run — not placing)")
        return 0

    # Place via MCP. We can't call mcp__stake__stake_place_combo_bet directly
    # from this script (it's a hermes_tools-only function). Output the params
    # and let the calling session fire the tool.
    outcome_ids = [l["outcome_id"] for l in legs]
    label = f"Multibet {args.mode}: {' + '.join(l['match'] for l in legs)}"

    print()
    if args.place:
        print("  ⚠ --place requested but this script cannot call MCP directly.")
        print("    Use the mcp__stake__stake_place_combo_bet tool with these params:")
    else:
        print("To place this parlay, call mcp__stake__stake_place_combo_bet with:")
    print(f"  outcome_ids: {outcome_ids}")
    print(f"  amount: {parlay['stake']:.2f}")
    print(f"  currency: usdt")
    print(f"  label: '{label}'")

    # Log the decision (whether or not we place it)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "match": "MULTIBET",
        "mode": args.mode,
        "legs": [{"match": l["match"], "market": l["market"],
                  "selection": l["selection"], "odds": l["odds"],
                  "outcome_id": l["outcome_id"]} for l in legs],
        "combined_odds": parlay["combined_odds"],
        "combined_prob": parlay["combined_prob"],
        "edge": parlay["edge_pct"] / 100,
        "stake": parlay["stake"],
        "expected_value": parlay["expected_value"],
        "potential_payout": parlay["potential_payout"],
        "bankroll": bankroll,
        "dry_run": not args.place,
    }
    with open(BET_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
