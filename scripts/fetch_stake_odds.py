#!/usr/bin/env python3
"""
WC 2026 — fetch fresh Stake odds for a fixture and write a JSON the
bet_decision.py + place_bet.py can consume.

Uses the hermes MCP `mcp__stake__stake_get_fixture_odds` tool.
Writes a flattened dict mapping our normalized market+selection → odds,
plus the raw markets tree (for outcome_id lookup in place_bet.py).

Usage:
    python3 fetch_stake_odds.py --event <ESPN_EVENT_ID> --output <path>
    python3 fetch_stake_odds.py --fixture-id <STAKE_FIXTURE_ID> --output <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def flatten_markets(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk the raw Stake markets tree and build:
      - 1X2: {home_win, draw, away_win}
      - O/U 2.5: {over, under}
      - AH -0.5 / +0.5: {"-0.5_home", "-0.5_away", "+0.5_home", "+0.5_away"}
      - BTTS: {yes, no}
    """
    flat: Dict[str, Any] = {"1X2": {}, "O/U 2.5": {}, "AH": {}, "BTTS": {}, "Correct Score": {}}
    outcome_map: Dict[str, str] = {}
    fixture_id = raw.get("fixture_id", "")

    markets = raw.get("markets", {})

    # 1X2
    for mkt_name, outcomes in markets.items():
        if "1x2" in mkt_name.lower():
            for o in outcomes:
                n = o["name"].lower()
                oid = o["id"]
                if "draw" in n:
                    flat["1X2"]["draw"] = o["odds"]
                    outcome_map["1X2|draw"] = oid
                else:
                    # home team is the one without "draw"
                    # We don't know which is home from the name, so we use
                    # the order: 1X2 list is [home, draw, away] in our raw fetch
                    if "home_win" not in flat["1X2"]:
                        flat["1X2"]["home_win"] = o["odds"]
                        outcome_map["1X2|home_win"] = oid
                    else:
                        flat["1X2"]["away_win"] = o["odds"]
                        outcome_map["1X2|away_win"] = oid
            break

    # O/U 2.5
    for mkt_name, outcomes in markets.items():
        m = mkt_name.lower()
        if "asian total" in m and "2.5" in m:
            for o in outcomes:
                n = o["name"].lower()
                if "over" in n:
                    flat["O/U 2.5"]["over"] = o["odds"]
                    outcome_map["O/U 2.5|over"] = o["id"]
                elif "under" in n:
                    flat["O/U 2.5"]["under"] = o["odds"]
                    outcome_map["O/U 2.5|under"] = o["id"]
            break

    # AH
    for mkt_name, outcomes in markets.items():
        m = mkt_name.lower()
        if "asian handicap" not in m:
            continue
        for line in ["-0.5", "+0.5"]:
            if line in m or any(line in o["name"] for o in outcomes):
                for o in outcomes:
                    if line in o["name"]:
                        # home or away? The first one is the home team
                        if f"{line}_home" not in flat["AH"]:
                            flat["AH"][f"{line}_home"] = o["odds"]
                            outcome_map[f"AH {line}|home_win"] = o["id"]
                        else:
                            flat["AH"][f"{line}_away"] = o["odds"]
                            outcome_map[f"AH {line}|away_win"] = o["id"]
                break

    # BTTS
    for mkt_name, outcomes in markets.items():
        if "both teams" in mkt_name.lower():
            for o in outcomes:
                n = o["name"].lower()
                if n == "yes":
                    flat["BTTS"]["yes"] = o["odds"]
                    outcome_map["BTTS|yes"] = o["id"]
                elif n == "no":
                    flat["BTTS"]["no"] = o["odds"]
                    outcome_map["BTTS|no"] = o["id"]
            break

    # Correct Score (only 0-1, for tier 0 CS leg)
    for mkt_name, outcomes in markets.items():
        if "correct score" in mkt_name.lower():
            for o in outcomes:
                n = o["name"]
                # Stake uses "0:1" or "0-1"
                if n in ("0:1", "0-1"):
                    flat["Correct Score"]["0-1"] = o["odds"]
                    outcome_map["Correct Score 0-1|0-1"] = o["id"]
            break

    return {
        "fixture_id": fixture_id,
        "match": raw.get("match", ""),
        "status": raw.get("status", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "1X2": flat["1X2"],
        "O/U 2.5": flat["O/U 2.5"],
        "AH": flat["AH"],
        "BTTS": flat["BTTS"],
        "Correct Score": flat["Correct Score"],
        "outcome_map": outcome_map,
        "raw": raw,  # keep raw for place_bet.py fallback matching
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", help="ESPN event ID (will look up Stake fixture_id)")
    ap.add_argument("--fixture-id", help="Stake fixture_id directly")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        from hermes_tools import mcp  # type: ignore
    except ImportError:
        print("❌ hermes_tools.mcp not available; cannot call Stake MCP", file=sys.stderr)
        sys.exit(1)

    fixture_id = args.fixture_id
    if not fixture_id and args.event:
        # We need to map ESPN event → Stake fixture_id. Use search.
        # This is best-effort; in practice the T-45 cron already knows the fixture_id.
        # For now, search by date.
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        r = mcp.stake.stake_search_fixture(date_from=today, date_to=today)
        # Match by event... we don't actually have a clean way without an
        # ESPN-to-Stake mapping table. Skip for now.
        print(f"⚠ no fixture_id provided and event→fixture mapping not implemented; "
              f"using args.fixture_id only", file=sys.stderr)
        sys.exit(1)

    if not fixture_id:
        print("❌ --fixture-id is required", file=sys.stderr)
        sys.exit(1)

    raw = mcp.stake.stake_get_fixture_odds(fixture_id=fixture_id, main_only=False)
    flat = flatten_markets(raw.get("result", raw))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(flat, indent=2))
    print(f"  wrote {args.output}")
    print(f"  1X2: {flat['1X2']}")
    print(f"  O/U 2.5: {flat['O/U 2.5']}")
    print(f"  AH: {flat['AH']}")
    print(f"  BTTS: {flat['BTTS']}")
    print(f"  CS 0-1: {flat['Correct Score']}")


if __name__ == "__main__":
    main()
