#!/usr/bin/env python3
"""
WC 2026 — place bets from a bet_decision.py decisions JSON.

For each selected bet, looks up the Stake outcome_id from the Stake
fixture odds (passed in as a JSON dict), places the bet, logs to
~/Projects/wc2026-predictor/bets_log.jsonl.

Usage:
    python3 place_bet.py \\
        --decisions /path/to/decisions.json \\
        --stake-odds /path/to/stake_odds_raw.json \\
        --match CAN-MAR \\
        --fixture-id <UUID> \\
        --bankroll 74.97
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PATH = Path("~/Projects/wc2026-predictor/bets_log.jsonl").expanduser()


def find_outcome_id(stake_odds: Dict[str, Any], market: str, selection: str) -> Optional[str]:
    """
    Walk the raw Stake odds JSON (from mcp__stake__stake_get_fixture_odds)
    and find the outcome_id matching (market, selection).
    """
    # We accept a flattened dict or the raw markets tree
    if "outcome_map" in stake_odds:
        return stake_odds["outcome_map"].get(f"{market}|{selection}")

    # Raw: walk markets
    for mkt_name, outcomes in stake_odds.get("markets", {}).items():
        # mkt_name might be "Asian Total - Asian Total (90' + Stoppage Time) (3)"
        # We match on our market+selection
        for o in outcomes:
            name = o.get("name", "")
            if matches(mkt_name, name, market, selection):
                return o.get("id")
    return None


def matches(mkt_name: str, sel_name: str, market: str, selection: str) -> bool:
    """
    Match Stake market/selection strings to our normalized market/selection.
    """
    m = market.lower()
    s = selection.lower()
    name = sel_name.lower()
    mkt = mkt_name.lower()

    if m == "1x2":
        if "1x2" not in mkt:
            return False
        if s == "home_win":
            return "draw" not in name  # home team, not draw
        if s == "away_win":
            return False  # away team — Stake lists home/away/draw, away is the third one
        if s == "draw":
            return "draw" in name

    if m == "o/u 2.5":
        if "asian total" not in mkt:
            return False
        if "2.5" not in mkt and "2.5" not in sel_name:
            return False
        if s == "over":
            return "over" in name
        if s == "under":
            return "under" in name

    if m.startswith("ah "):
        line = m.replace("ah ", "").strip()
        if "asian handicap" not in mkt:
            return False
        if line not in sel_name and line not in mkt:
            return False
        if s == "home_win":
            # home team has the line on the home side
            return line in name
        if s == "away_win":
            return line in name

    if m == "btts":
        if "both teams" not in mkt:
            return False
        if s == "yes":
            return name == "yes"
        if s == "no":
            return name == "no"

    if m == "correct score 0-1":
        if "correct score" not in mkt:
            return False
        if s == "0-1":
            return name in ("0:1", "0-1")

    return False


def log_bet(rec: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def place_one_bet(
    decision: Dict[str, Any],
    stake_odds: Dict[str, Any],
    match: str,
    fixture_id: str,
    bankroll: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Place a single bet via Stake API (MCP). Returns log record.
    """
    market = decision["market"]
    selection = decision["selection"]
    odds = decision["odds"]
    stake = decision["stake"]

    outcome_id = find_outcome_id(stake_odds, market, selection)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "match": match,
        "market": market,
        "selection": selection,
        "odds": odds,
        "stake": stake,
        "bankroll": bankroll,
        "model_prob": decision["model_prob"],
        "implied_prob": decision["implied_prob"],
        "edge": decision["edge"],
        "kelly": decision["kelly"],
        "model_version": decision.get("model_version"),
        "lineup_used": decision.get("lineup_used", False),
        "outcome_id": outcome_id,
        "fixture_id": fixture_id,
        "dry_run": dry_run,
    }

    if not outcome_id:
        rec["status"] = "skipped"
        rec["reason"] = f"no outcome_id for {market} {selection}"
        log_bet(rec)
        return rec

    if dry_run:
        rec["status"] = "dry_run"
        log_bet(rec)
        return rec

    # Real placement — call Stake MCP
    try:
        from hermes_tools import mcp  # type: ignore
        result = mcp.stake.stake_place_bet(
            outcome_id=outcome_id,
            amount=stake,
            currency="usdt",
            label=f"{match} {market} {selection}",
        )
        rec["status"] = "placed"
        rec["result"] = result
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = str(e)

    log_bet(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--stake-odds", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--fixture-id", required=True)
    ap.add_argument("--bankroll", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    decisions = json.loads(Path(args.decisions).read_text())
    stake_odds = json.loads(Path(args.stake_odds).read_text())

    if not decisions.get("selected"):
        print(f"NO BETS: {decisions['match']} has no value")
        return

    print(f"=== Placing {len(decisions['selected'])} bet(s) for {decisions['match']} ===")
    for sel in decisions["selected"]:
        rec = place_one_bet(sel, stake_odds, args.match, args.fixture_id,
                            args.bankroll, dry_run=args.dry_run)
        print(f"  {rec['status']}: {sel['market']} {sel['selection']} "
              f"${sel['stake']:.2f} @ {sel['odds']:.2f} "
              f"(edge {sel['edge']*100:+.1f}%, kelly {sel['kelly']*100:.1f}%)")
        if rec.get("outcome_id"):
            print(f"    outcome_id: {rec['outcome_id']}")
        if rec.get("status") == "placed":
            print(f"    bet_id: {rec.get('result', {}).get('bet_id', '?')}")
        time.sleep(2)  # rate limit buffer


if __name__ == "__main__":
    main()
