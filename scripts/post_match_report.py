#!/usr/bin/env python3
"""
WC 2026 — post-match report generator.

Reads bets_log.jsonl entries for a given match, compares against the
actual result (from ESPN), and emits Michael's report format:

  "{match} result: {actual_score}
   Bets: {n_placed} placed, {n_won} won, {n_lost} lost
   P&L: {pnl}
   Model calibration: predicted {home} win {p_home}%, actual {outcome}
   Pre-edge: {pre_edges}
   Post-edge (if {home} won): {post_edges}
   Action for {next_match}: {action}"

Usage:
    python3 post_match_report.py --match CAN-MAR --event <ESPN_ID> \\
        --next-match BRA-NOR

Reads the actual score from mcp__espn__espn_get_match_summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BETS_LOG = Path("~/Projects/wc2026-predictor/bets_log.jsonl").expanduser()
MARKETS_JSON = Path("~/Projects/wc2026-predictor/download/wc2026_betting_markets.json")


def load_match_bets(match: str) -> List[Dict[str, Any]]:
    if not BETS_LOG.exists():
        return []
    out = []
    for line in BETS_LOG.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("match") == match and rec.get("status") in ("placed", "dry_run", "settled"):
            out.append(rec)
    return out


def load_model_probs(match: str) -> Optional[Dict[str, Any]]:
    if not MARKETS_JSON.exists():
        return None
    data = json.loads(MARKETS_JSON.read_text())
    for m in data.get("matches", []):
        if f"{m['home']}-{m['away']}" == match:
            return m
    return None


def fetch_actual_score(event_id: str) -> Optional[Dict[str, Any]]:
    """Call ESPN MCP for the final score."""
    try:
        from hermes_tools import mcp  # type: ignore
    except ImportError:
        print("⚠ hermes_tools.mcp not available", file=sys.stderr)
        return None
    try:
        r = mcp.espn.espn_get_match_summary(event_id=event_id)
        # Walk the result to find final score
        # ESPN returns: header, boxscore, scoring, events, lineups
        return r
    except Exception as e:
        print(f"⚠ ESPN fetch failed: {e}", file=sys.stderr)
        return None


def settle_bets(
    bets: List[Dict[str, Any]],
    actual_score: Dict[str, Any],
) -> tuple:
    """
    Walk each bet, determine win/loss vs actual score.
    Returns (settled, total_stake, total_payout, pnl).
    """
    home_score = actual_score.get("home_score")
    away_score = actual_score.get("away_score")
    if home_score is None or away_score is None:
        return bets, 0, 0, 0

    settled = []
    total_stake = 0
    total_payout = 0
    n_won = 0
    n_lost = 0

    for b in bets:
        stake = b.get("stake", 0)
        odds = b.get("odds", 0)
        market = b.get("market")
        selection = b.get("selection")
        won = False
        push = False

        if market == "1X2":
            if selection == "home_win" and home_score > away_score:
                won = True
            elif selection == "away_win" and away_score > home_score:
                won = True
            elif selection == "draw" and home_score == away_score:
                won = True
        elif market == "O/U 2.5":
            total = home_score + away_score
            if selection == "over" and total > 2.5:
                won = True
            elif selection == "under" and total < 2.5:
                won = True
        elif market == "BTTS":
            if selection == "yes" and home_score > 0 and away_score > 0:
                won = True
            elif selection == "no" and (home_score == 0 or away_score == 0):
                won = True
        elif market.startswith("AH "):
            line = float(market.replace("AH ", ""))
            if selection == "home_win" and (home_score - away_score) > line:
                won = True
            elif selection == "away_win" and (away_score - home_score) > -line:
                won = True
            elif (home_score - away_score) == line:
                push = True  # void

        if push:
            payout = stake
        elif won:
            payout = stake * odds
            n_won += 1
        else:
            payout = 0
            n_lost += 1

        total_stake += stake
        total_payout += payout

        settled.append({**b, "status": "settled", "won": won, "push": push,
                        "payout": payout, "pnl": payout - stake})

    pnl = total_payout - total_stake
    return settled, total_stake, total_payout, pnl, n_won, n_lost


def recompute_post_edge(model_match: Dict[str, Any], actual_score: Dict[str, Any]) -> List[str]:
    """
    Recompute model edges if the actual score had been the model prediction.
    Useful for "if the model was right, how much would the post-edge be?"
    Returns list of formatted edges for the report.
    """
    # For now just return the actual outcome and the model's pre-game
    # fair odds for that outcome. Detailed post-edge requires actually
    # running the model with the actual lineups — out of scope here.
    home = model_match.get("home", "?")
    away = model_match.get("away", "?")
    hs = actual_score.get("home_score", "?")
    as_ = actual_score.get("away_score", "?")
    if hs == as_:
        outcome = "draw"
    elif hs > as_:
        outcome = f"{home} win"
    else:
        outcome = f"{away} win"
    return [f"actual {outcome}"]


def decide_action(n_won: int, n_lost: int, n_total: int) -> str:
    if n_won == n_total and n_total > 0:
        return "SIZE UP to tier 1 immediately for next match (model sharper than expected)"
    if n_won >= n_total - 1 and n_total > 0:
        return "STAY at tier 0 through R16, bump at iter-5"
    return "INVESTIGATE before next match (model likely miscalibrated, do not size up)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--event", required=True, help="ESPN event_id")
    ap.add_argument("--next-match", required=True, help="e.g. BRA-NOR")
    args = ap.parse_args()

    bets = load_match_bets(args.match)
    if not bets:
        print(f"NO BETS LOGGED for {args.match}")
        return

    # Try to fetch actual score
    summary = fetch_actual_score(args.event)
    if not summary:
        print("⚠ could not fetch actual score from ESPN")
        return

    # Parse actual score from summary
    # ESPN summary has header.competitions[0].competitors[] with score
    actual = {"home_score": None, "away_score": None}
    try:
        comps = summary.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
        for c in comps:
            if c.get("homeAway") == "home":
                actual["home_score"] = int(c.get("score", 0))
            elif c.get("homeAway") == "away":
                actual["away_score"] = int(c.get("score", 0))
    except (KeyError, ValueError, IndexError) as e:
        print(f"⚠ could not parse score: {e}")
        return

    if actual["home_score"] is None:
        print("⚠ match not yet settled or score unavailable")
        return

    settled, total_stake, total_payout, pnl, n_won, n_lost = settle_bets(bets, actual)
    model_match = load_model_probs(args.match)
    home = model_match.get("home", args.match.split("-")[0]) if model_match else args.match.split("-")[0]
    p_home = (model_match.get("markets", {}).get("1X2", {}).get("home_win", 0) * 100) if model_match else 0
    pre_edges = [f"{b['market']} {b['selection']} +{b['edge']*100:.1f}%" for b in bets]
    post_edges = recompute_post_edge(model_match, actual)
    action = decide_action(n_won, n_lost, len(bets))

    actual_score_str = f"{actual['home_score']}-{actual['away_score']}"

    report = f"""{args.match} result: {actual_score_str}
Bets: {len(bets)} placed, {n_won} won, {n_lost} lost
P&L: {pnl:+.2f}
Model calibration: predicted {home} win {p_home:.0f}%, actual {actual_score_str}
Pre-edge: {', '.join(pre_edges)}
Post-edge: {', '.join(post_edges)}
Action for {args.next_match}: {action}"""

    # Write to log file and stdout
    log_path = Path("~/Projects/wc2026-predictor/data/post_match_reports.log").expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n")
        f.write(report + "\n")

    # Persist settled bets
    for s in settled:
        with BETS_LOG.open("a") as f:
            f.write(json.dumps(s) + "\n")

    print(report)


if __name__ == "__main__":
    main()
