#!/usr/bin/env python3
"""
WC 2026 — full bet placement workflow (Michael 2026-07-05 policy).

Called by the assistant in main session after the run-phase cron notifies.
Reads the bet_decision.py decisions JSON, then places:

  1. Up to MAX_SINGLES (4) single bets from decisions (1.67% bankroll each)
  2. ALWAYS add 1 CS 0-1 bet (1% bankroll) — model is overconfident but
     this is the tier-0 validation test
  3. ALWAYS add 1 SGP (3% bankroll) — assistant-curated combo, typically
     the top ML pick + top secondary market (e.g. Under 2.5) on the
     same match at multiplied odds

For halftime (T+60min), re-runs bet_decision with the in-play score, then
places 1-2 CS bets on the current scoreline if value still exists.

Usage (called by assistant):
    python3 place_bet_v2.py place-t30 \
        --decisions /path/to/decisions.json \
        --stake-odds /path/to/stake_odds.json \
        --fixture-id <UUID> \
        --bankroll 74.97

    python3 place_bet_v2.py place-ht \
        --match CAN-MAR \
        --fixture-id <UUID> \
        --bankroll 70.00 \
        --score 0-1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure predictor scripts dir is on the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("WARN: mcp not available - will return dry-run plan", file=sys.stderr)


# Michael 2026-07-05 sizing policy
MAX_SINGLES = 4                  # top-4 single bets from decisions
SINGLE_STAKE_PCT = 0.0167        # 1.67% bankroll per single (1/4 Kelly @ tier 0)
CS_STAKE_PCT = 0.01              # 1% bankroll for the CS bet
SGP_STAKE_PCT = 0.03              # 3% bankroll for the curated SGP
HT_CS_STAKE_PCT = 0.015           # 1.5% per halftime CS bet
HT_CS_MAX = 2                     # 1-2 CS bets at halftime


def _outcome_id(stake_odds: Dict, market: str, selection: str) -> Optional[str]:
    """Find outcome_id from stake odds dict."""
    # Try flat map first (key format: "Market - Submarket|name")
    if "outcome_map" in stake_odds:
        for key, oid in stake_odds["outcome_map"].items():
            mkt, name = key.split("|", 1) if "|" in key else (key, "")
            if market_match(mkt, market) and selection_match(mkt, name, selection):
                return oid

    # Walk raw markets
    for mkt_name, outcomes in stake_odds.get("markets", {}).items():
        mkt_l = mkt_name.lower()
        for o in outcomes:
            name = o.get("name", "")
            if market == "1X2" and is_1x2_market(mkt_l) and sel_matches_1x2(mkt_l, name, selection):
                return o.get("id")
            if market.startswith("AH ") and "handicap" in mkt_l:
                line = market.split(" ")[1]
                if sel_matches_ah(line, name, selection):
                    return o.get("id")
            if market == "O/U 2.5" and "total" in mkt_l and "over" in mkt_l and "under" in mkt_l:
                if sel_matches_ou(name, selection):
                    return o.get("id")
            if market == "BTTS" and "both teams" in mkt_l:
                if sel_matches_btts(name, selection):
                    return o.get("id")
            if market == "Correct Score 0-1" and "correct score" in mkt_l:
                if name in ("0:1", "0-1"):
                    return o.get("id")
    return None


def market_match(mkt_name: str, market: str) -> bool:
    """Check if a Stake market name matches our internal market key."""
    m = mkt_name.lower()
    if market == "1X2":
        return is_1x2_market(m)
    if market.startswith("AH "):
        return "handicap" in m
    if market == "O/U 2.5":
        return "total" in m and "over" in m and "under" in m
    if market == "BTTS":
        return "both teams" in m
    if market.startswith("Correct Score"):
        return "correct score" in m
    return False


def is_1x2_market(mkt_l: str) -> bool:
    return ("1x2" in mkt_l) or ("match winner" in mkt_l) or ("match result" in mkt_l)


def selection_match(mkt_name: str, name: str, selection: str) -> bool:
    """Check if a Stake selection name matches our internal selection key."""
    m = mkt_name.lower()
    n = name.lower()
    if is_1x2_market(m):
        return sel_matches_1x2(m, name, selection)
    if "handicap" in m:
        return True  # rely on stake-side filter
    if "total" in m:
        return sel_matches_ou(name, selection)
    if "both teams" in m:
        return sel_matches_btts(name, selection)
    if "correct score" in m:
        return name in ("0:1", "0-1") and selection == "0-1"
    return False


def sel_matches_1x2(mkt_l, name, sel):
    return name.lower() in (
        "home", "draw", "away", "1", "x", "2",
        "morocco", "canada", "france", "paraguay", "brazil", "norway",
        "mexico", "england", "portugal", "spain", "usa", "belgium",
        "argentina", "egypt", "switzerland", "colombia",
    ) and sel_map_1x2(name) == sel


def sel_map_1x2(name):
    n = name.lower()
    if n in ("home", "1") or "home" in n:
        return "home_win"
    if n in ("draw", "x"):
        return "draw"
    if n in ("away", "2") or "away" in n:
        return "away_win"
    return None


def sel_matches_ah(line, name, sel):
    # AH -0.5 home means "home (-0.5)" — the home wins outright
    n = name.lower()
    if line not in name:
        return False
    if sel == "home_win" and ("home" in n or n.startswith("1 ")):
        return True
    if sel == "away_win" and ("away" in n or n.startswith("2 ")):
        return True
    return False


def sel_matches_ou(name, sel):
    n = name.lower()
    if sel == "over" and "over" in n:
        return True
    if sel == "under" and "under" in n:
        return True
    return False


def sel_matches_btts(name, sel):
    n = name.lower()
    if sel == "yes" and n == "yes":
        return True
    if sel == "no" and n == "no":
        return True
    return False


async def _mcp_call(tool: str, args: Dict) -> Dict:
    """Call a Stake MCP tool."""
    if not MCP_AVAILABLE:
        return {"success": False, "error": "MCP not available"}

    params = StdioServerParameters(
        command="/Users/michaellee/.hermes/hermes-agent/venv/bin/python3.11",
        args=["/Users/michaellee/.hermes/mcp-servers/stake/server.py"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            r = await s.call_tool(tool, args)
            return json.loads(r.content[0].text)


def plan_t30(decisions: Dict, stake_odds: Dict, bankroll: float) -> Dict:
    """Build the T-30 placement plan: 4 singles + 1 CS + 1 SGP (curated)."""
    plan = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "t30",
        "bankroll": bankroll,
        "singles": [],
        "cs": None,
        "sgp": None,
    }

    # 1. Top MAX_SINGLES from decisions
    selected = decisions.get("selected", [])
    for s in selected[:MAX_SINGLES]:
        market = s["market"]
        sel = s["selection"]
        oid = _outcome_id(stake_odds, market, sel)
        if not oid:
            continue
        stake = round(bankroll * SINGLE_STAKE_PCT, 4)
        plan["singles"].append({
            "market": market,
            "selection": sel,
            "odds": s["odds"],
            "model_prob": s["model_prob"],
            "edge": s["edge"],
            "stake": stake,
            "outcome_id": oid,
        })

    # 2. CS 0-1 (tier 0 validation bet — model overconfident but small stake)
    cs_oid = _outcome_id(stake_odds, "Correct Score 0-1", "0-1")
    if cs_oid:
        cs_odds = _find_odds(stake_odds, "Correct Score 0-1", "0-1") or 0
        plan["cs"] = {
            "market": "Correct Score 0-1",
            "selection": "0-1",
            "odds": cs_odds,
            "stake": round(bankroll * CS_STAKE_PCT, 4),
            "outcome_id": cs_oid,
        }

    # 3. SGP — assistant-curated. The default is: top ML pick + best
    # supporting market (typically Under 2.5 or BTTS No) at multiplied odds.
    # Caller overrides this by editing the plan before submitting.
    ml_leg = next(
        (s for s in plan["singles"] if s["market"] == "1X2"),
        None,
    )
    supporting_legs = [s for s in plan["singles"] if s["market"] in ("O/U 2.5", "BTTS")]
    if ml_leg and supporting_legs:
        legs = [ml_leg] + supporting_legs[:1]
        combined_odds = 1.0
        for leg in legs:
            combined_odds *= leg["odds"]
        plan["sgp"] = {
            "legs": [{"market": l["market"], "selection": l["selection"],
                      "odds": l["odds"], "outcome_id": l["outcome_id"]}
                     for l in legs],
            "combined_odds": round(combined_odds, 4),
            "stake": round(bankroll * SGP_STAKE_PCT, 4),
            "label": f"Curated SGP: {ml_leg['market']} {ml_leg['selection']} + {legs[1]['market']} {legs[1]['selection']}",
        }

    return plan


def plan_ht(decisions: Dict, stake_odds: Dict, bankroll: float, score: str) -> Dict:
    """Build the halftime (T+60min) plan: 1-2 CS bets if value exists."""
    plan = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "halftime",
        "score_at_ht": score,
        "bankroll": bankroll,
        "cs_bets": [],
    }

    # CS at halftime: model re-evaluated with the in-play score. Look for
    # CS candidates that match the current 0-X or X-0 state with strong edge.
    selected = decisions.get("selected", [])
    cs_candidates = [s for s in selected if s["market"].startswith("Correct Score")]

    # If the model didn't generate CS candidates for the current score, fall
    # back to a simple heuristic: 0-0 → bet on next goal side or goalless.
    if not cs_candidates:
        # Halftime CS bets: if 0-0, bet on goalless (0-0 final @ ~3.50) or
        # the away team to win 0-1. If 1-0, bet on 1-0 final or 1-1 draw.
        # The assistant will override this in main session with model output.
        plan["cs_bets"] = []  # caller fills in
        return plan

    for c in cs_candidates[:HT_CS_MAX]:
        market = c["market"]
        sel = c["selection"]
        oid = _outcome_id(stake_odds, market, sel)
        if not oid:
            continue
        stake = round(bankroll * HT_CS_STAKE_PCT, 4)
        plan["cs_bets"].append({
            "market": market,
            "selection": sel,
            "odds": c["odds"],
            "edge": c["edge"],
            "stake": stake,
            "outcome_id": oid,
        })

    return plan


def _find_odds(stake_odds: Dict, market: str, selection: str) -> Optional[float]:
    """Find odds for a (market, selection) pair."""
    # Try flat
    if "markets_flat" in stake_odds:
        for key, odds in stake_odds["markets_flat"].items():
            mkt, name = key.split("|", 1) if "|" in key else (key, "")
            if market_match(mkt, market) and selection_match(mkt, name, selection):
                return odds
    # Walk raw
    for mkt_name, outcomes in stake_odds.get("markets", {}).items():
        mkt_l = mkt_name.lower()
        for o in outcomes:
            name = o.get("name", "")
            if market == "Correct Score 0-1" and "correct score" in mkt_l and name in ("0:1", "0-1"):
                return o.get("odds")
            if market == "1X2" and is_1x2_market(mkt_l) and sel_matches_1x2(mkt_l, name, selection):
                return o.get("odds")
    return None


def main():
    ap = argparse.ArgumentParser(description="WC 2026 v2 bet placer (T-30 + HT workflow)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan-t30", help="Build T-30 placement plan (4 singles + 1 CS + 1 SGP)")
    p1.add_argument("--decisions", required=True)
    p1.add_argument("--stake-odds", required=True)
    p1.add_argument("--bankroll", type=float, required=True)
    p1.add_argument("--output")

    p2 = sub.add_parser("plan-ht", help="Build halftime plan (1-2 CS bets)")
    p2.add_argument("--decisions", required=True)
    p2.add_argument("--stake-odds", required=True)
    p2.add_argument("--bankroll", type=float, required=True)
    p2.add_argument("--score", required=True, help="HT score, e.g. 0-1")
    p2.add_argument("--output")

    p3 = sub.add_parser("submit", help="Submit a placement plan via MCP")
    p3.add_argument("--plan", required=True, help="Path to plan JSON")

    args = ap.parse_args()

    if args.cmd == "plan-t30":
        decisions = json.loads(Path(args.decisions).read_text())
        stake_odds = json.loads(Path(args.stake_odds).read_text())
        plan = plan_t30(decisions, stake_odds, args.bankroll)
        out = args.output or "/tmp/place_plan_t30.json"
        Path(out).write_text(json.dumps(plan, indent=2))
        print(f"Plan written to {out}")
        print(json.dumps(plan, indent=2))

    elif args.cmd == "plan-ht":
        decisions = json.loads(Path(args.decisions).read_text())
        stake_odds = json.loads(Path(args.stake_odds).read_text())
        plan = plan_ht(decisions, stake_odds, args.bankroll, args.score)
        out = args.output or "/tmp/place_plan_ht.json"
        Path(out).write_text(json.dumps(plan, indent=2))
        print(f"Plan written to {out}")
        print(json.dumps(plan, indent=2))

    elif args.cmd == "submit":
        plan = json.loads(Path(args.plan).read_text())
        # Submit each bet via MCP
        results = {"singles": [], "cs": None, "sgp": None, "cs_bets": []}
        for s in plan.get("singles", []):
            r = asyncio.run(_mcp_call("stake_place_bet", {
                "outcome_id": s["outcome_id"],
                "amount": s["stake"],
                "currency": "usdt",
                "label": f"WC2026 T-30 {s['market']} {s['selection']}",
            }))
            results["singles"].append({"plan": s, "result": r})
        if plan.get("cs"):
            r = asyncio.run(_mcp_call("stake_place_bet", {
                "outcome_id": plan["cs"]["outcome_id"],
                "amount": plan["cs"]["stake"],
                "currency": "usdt",
                "label": f"WC2026 T-30 CS {plan['cs']['selection']}",
            }))
            results["cs"] = {"plan": plan["cs"], "result": r}
        if plan.get("sgp"):
            r = asyncio.run(_mcp_call("stake_place_custom_bet", {
                "outcome_ids": [l["outcome_id"] for l in plan["sgp"]["legs"]],
                "amount": plan["sgp"]["stake"],
                "currency": "usdt",
                "label": plan["sgp"]["label"],
            }))
            results["sgp"] = {"plan": plan["sgp"], "result": r}
        for c in plan.get("cs_bets", []):
            r = asyncio.run(_mcp_call("stake_place_bet", {
                "outcome_id": c["outcome_id"],
                "amount": c["stake"],
                "currency": "usdt",
                "label": f"WC2026 HT CS {c['selection']}",
            }))
            results["cs_bets"].append({"plan": c, "result": r})
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
