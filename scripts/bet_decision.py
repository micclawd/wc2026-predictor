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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
try:
    from sizing_tiers import CURRENT_TIER as TIER
except ImportError:
    TIER = {
        "max_exposure_pct": 0.05,
        "kelly_fraction": 0.25,
        "max_stake_pct": 0.0167,
        "min_edge": 0.05,
        "min_kelly": 0.03,
        "min_odds": 1.50,
        "max_selections": 3,
        "global_exposure_cap": 0.25,
    }

# Betting thresholds (Michael's policy)
MIN_EDGE = TIER["min_edge"]                # +5% edge required
MIN_KELLY = TIER["min_kelly"]              # 3% Kelly fraction
MIN_ODDS = TIER["min_odds"]                # no sub-1.50 favorites
MAX_SELECTIONS_PER_MATCH = TIER["max_selections"]
MAX_EXPOSURE_PER_MATCH = TIER["max_exposure_pct"]
KELLY_FRACTION = TIER["kelly_fraction"]    # tier-controlled
MAX_STAKE_PCT_PER_BET = TIER["max_stake_pct"]
GLOBAL_EXPOSURE_CAP = TIER["global_exposure_cap"]

# Markets we bet on
# CS 0-1 added 2026-07-04 per Michael directive (tier 0 validation test)
ALLOWED_MARKETS = {
    "1X2": ["home_win", "draw", "away_win"],
    "O/U 2.5": ["over", "under"],
    "AH -0.5": ["home_win", "away_win"],
    "AH +0.5": ["home_win", "away_win"],
    "BTTS": ["yes", "no"],
    "Correct Score 0-1": ["0-1"],  # CS allowed at tier 0 only
}
CS_MARKET = "Correct Score 0-1"
CS_MAX_PER_MATCH = 1


# ---------------------------------------------------------------------------
# v1.5.x Stake odds parser (added 2026-07-06)
#
# fetch_stake_odds_mcp.py writes a FLAT dict of "Market|Selection" -> odds,
# not the nested {"1X2": {"home_win": 1.77, ...}} shape this script used
# to read. The old get_liquid_selections looked up stake_odds["1X2"]["home_win"]
# which always returns None against the v1.5.x file, so EVERY selection
# silently dropped to 0 candidates. This parser converts flat -> nested
# so the rest of the pipeline can stay unchanged.
# ---------------------------------------------------------------------------

# Stake "1X2 (90' + Stoppage Time)" is the main moneyline market
_RE_1X2 = re.compile(r"^1[xX]2[^|]*\|\s*(Brazil|Norway|Draw|Home|Away|.+?)\s*$", re.IGNORECASE)
_RE_AH_LINE = re.compile(r"Asian Handicap[^|]*\((\d+)\)[^|]*\|\s*([A-Za-z][\w\s\.\-]*?)\s*\(([+\-][\d\.]+)\)\s*$")
_RE_OU_LINE = re.compile(r"Asian Total[^|]*\((\d+)\)[^|]*\|\s*(Over|Under)\s+([\d\.]+)\s*$")
_RE_OU_BARE = re.compile(r"^Over/Under[^|]*\|\s*(Over|Under)\s+([\d\.]+)\s*$", re.IGNORECASE)
_RE_BTTS = re.compile(r"^Both Teams? to Score[^|]*\|\s*(Yes|No)\s*$", re.IGNORECASE)
_RE_CS_SCORE = re.compile(r"^Correct Score[^|]*\|\s*(\d+)[:\-](\d+)\s*$", re.IGNORECASE)


def _parse_stake_odds_v15(flat_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert v1.5.x flat markets_flat -> nested dict shape.

    Input keys look like:
      "1x2 - 1x2 (90' + Stoppage Time)|Brazil"  -> 1.77
      "Asian Handicap - Asian Handicap (90' + Stoppage Time) (4)|Brazil (-0.5)" -> 1.77
      "Asian Total - Asian Total (90' + Stoppage Time) (5)|Over 2.5" -> 1.68
      "Both Teams to Score - Both Teams to Score (90' + Stoppage Time)|Yes" -> 1.59
      "Correct Score - Correct Score (90' + Stoppage Time)|0:1" -> 15.0

    Output shape matches what get_liquid_selections() expects:
      {"1X2": {"home_win": 1.77, "draw": 3.75, "away_win": 4.5},
       "O/U 2.5": {"over": 1.68, "under": 2.15},
       "AH": {"-0.5_home": 1.77, "-0.5_away": 2.05, "+0.5_home": 1.22, "+0.5_away": 1.66},
       "BTTS": {"yes": 1.59, "no": 2.27},
       "Correct Score": {"0-1": 15.0, "0:1": 15.0}}
    """
    # Handle both shapes: the v1.5.x file has top-level "markets_flat" key,
    # or you can pass the inner dict directly.
    if "markets_flat" in flat_payload and isinstance(flat_payload["markets_flat"], dict):
        flat = flat_payload["markets_flat"]
    else:
        flat = flat_payload

    nested: Dict[str, Any] = {"1X2": {}, "O/U 2.5": {}, "AH": {}, "BTTS": {}, "Correct Score": {}}
    home_abbr = None
    away_abbr = None

    for key, odds in flat.items():
        if not isinstance(odds, (int, float)) or odds <= 1.0:
            continue
        # 1X2
        if key.lower().startswith("1x2 - 1x2"):
            m = re.match(r"^1[xX]2[^|]*\|\s*(.+?)\s*$", key, re.IGNORECASE)
            if not m:
                continue
            sel = m.group(1).strip()
            if sel.lower() in ("draw", "x"):
                nested["1X2"]["draw"] = float(odds)
            elif "home" in sel.lower() or sel.lower() == "1":
                nested["1X2"]["home_win"] = float(odds)
            elif "away" in sel.lower() or sel.lower() == "2":
                nested["1X2"]["away_win"] = float(odds)
            else:
                # first time we see a non-Draw team name, lock it in as home_abbr
                if home_abbr is None:
                    home_abbr = sel
                    nested["1X2"]["home_win"] = float(odds)
                elif away_abbr is None and sel != home_abbr:
                    away_abbr = sel
                    nested["1X2"]["away_win"] = float(odds)
            continue

        # Asian Handicap with (N) suffix
        m = re.match(r"Asian Handicap[^|]*\((\d+)\)[^|]*\|\s*([A-Za-z][\w\s\.\-]*?)\s*\(([+\-][\d\.]+)\)\s*$", key)
        if m:
            _, team, line = m.group(1), m.group(2).strip(), m.group(3)
            side = "home" if team == (home_abbr or "Brazil") else "away"
            nested["AH"][f"{line}_{side}"] = float(odds)
            continue

        # Asian Total with (N) suffix
        m = re.match(r"Asian Total[^|]*\((\d+)\)[^|]*\|\s*(Over|Under)\s+([\d\.]+)\s*$", key)
        if m:
            line = float(m.group(3))
            side = m.group(2).lower()
            if line == 2.5:
                nested["O/U 2.5"][side] = float(odds)
            continue

        # Bare Over/Under (no (N) suffix)
        m = re.match(r"^Over/Under[^|]*\|\s*(Over|Under)\s+([\d\.]+)\s*$", key, re.IGNORECASE)
        if m:
            line = float(m.group(2))
            side = m.group(1).lower()
            if line == 2.5:
                nested["O/U 2.5"].setdefault(side, float(odds))
            continue

        # BTTS
        m = re.match(r"^Both Teams? to Score[^|]*\|\s*(Yes|No)\s*$", key, re.IGNORECASE)
        if m:
            nested["BTTS"][m.group(1).lower()] = float(odds)
            continue

        # Correct Score
        m = re.match(r"^Correct Score[^|]*\|\s*(\d+)[:\-](\d+)\s*$", key, re.IGNORECASE)
        if m:
            score = f"{m.group(1)}-{m.group(2)}"
            nested["Correct Score"][score] = float(odds)
            nested["Correct Score"][f"{m.group(1)}:{m.group(2)}"] = float(odds)
            continue

    return nested


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

    # Correct score 0-1 (CS — only allowed at tier 0 validation)
    cs_top5 = markets.get("correct_score_top5", [])
    if cs_top5:
        # find the 0-1 entry in the model output
        cs_01 = next((c for c in cs_top5 if c.get("score") == "0-1"), None)
        if cs_01:
            mp = cs_01.get("prob", 0)
            # Stake odds for 0-1 (use exact key)
            stake_cs = stake_odds.get("Correct Score", {})
            odds = stake_cs.get("0-1", stake_cs.get("0:1", 0))
            if odds and odds >= MIN_ODDS:
                out.append(_build_candidate(CS_MARKET, "0-1", mp, odds))

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
            # size the stake: KELLY_FRACTION, capped at MAX_STAKE_PCT_PER_BET
            # and the per-match exposure cap
            stake_pct = min(c["kelly"] * KELLY_FRACTION, MAX_STAKE_PCT_PER_BET)
            stake = round(bankroll * stake_pct, 4)
            passed_filter.append({**c, "stake_pct": stake_pct, "stake": stake})

    # Sort by edge descending, take top N
    passed_filter.sort(key=lambda x: x["edge"], reverse=True)
    selected = passed_filter[:MAX_SELECTIONS_PER_MATCH]

    # CS-specific cap: max 1 CS leg per match (only 1 candidate so usually a no-op,
    # but defensive if the data ever has more)
    cs_in_selected = [s for s in selected if s["market"] == CS_MARKET]
    if len(cs_in_selected) > CS_MAX_PER_MATCH:
        cs_in_selected.sort(key=lambda x: x["edge"], reverse=True)
        keep_ids = {id(s) for s in cs_in_selected[:CS_MAX_PER_MATCH]}
        selected = [s for s in selected if s["market"] != CS_MARKET or id(s) in keep_ids]

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
    raw_stake = json.loads(Path(args.stake_odds).read_text())
    # v1.5.x fetch_stake_odds_mcp.py writes a flat "markets_flat" dict.
    # Convert to the nested {"1X2": {home_win:.., ..}} shape that
    # get_liquid_selections() expects. Old nested payloads are passed through.
    if isinstance(raw_stake, dict) and (
        "markets_flat" in raw_stake or any("|" in str(k) for k in raw_stake.keys())
    ):
        stake_odds = _parse_stake_odds_v15(raw_stake)
        print(f"  parsed {sum(len(v) for v in stake_odds.values()) if isinstance(stake_odds, dict) else 0} Stake outcomes from v1.5.x flat format")
    else:
        stake_odds = raw_stake

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
