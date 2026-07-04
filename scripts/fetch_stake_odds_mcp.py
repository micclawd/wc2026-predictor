#!/usr/bin/env python3
"""
WC 2026 — fetch fresh Stake odds for a fixture via the MCP stdio client.
Works in cron env (no hermes_tools dependency).

Maps ESPN event_id → Stake fixture_id via search.

Usage:
    python3 fetch_stake_odds_mcp.py --event <ESPN_EVENT_ID> --output <path>
    python3 fetch_stake_odds_mcp.py --fixture-id <STAKE_FIXTURE_ID> --output <path>
    python3 fetch_stake_odds_mcp.py --match "Canada - Morocco" --output <path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MCP_SERVER = "/Users/michaellee/.hermes/mcp-servers/stake/server.py"
TOKEN_FILE = "/Users/michaellee/.openclaw/workspace/stake_token.txt"


def _flatten_markets(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Walk raw markets tree and build flat + outcome_map.

    Returns:
      {
        "fixture_id": ...,
        "ts": ...,
        "markets_flat": {"Market Name|Selection": odds, ...},
        "outcome_map": {"Market Name|Selection": outcome_uuid, ...},
        "raw": {...}
      }
    """
    fixture_id = raw.get("fixture_id", "") or raw.get("id", "")
    out: Dict[str, Any] = {
        "fixture_id": fixture_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "markets_flat": {},
        "outcome_map": {},
        "raw": raw,
    }
    markets = raw.get("markets", {})
    for mkt_name, outcomes in markets.items():
        for o in outcomes:
            sel = o.get("name", "")
            odds = o.get("odds", 0)
            oid = o.get("id", "")
            if not sel:
                continue
            key = f"{mkt_name}|{sel}"
            out["markets_flat"][key] = odds
            out["outcome_map"][key] = oid
    return out


async def _mcp_call(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP tool directly via stdio client (no agent loop)."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Check token
    if not os.path.exists(TOKEN_FILE):
        return {"error": f"token file missing: {TOKEN_FILE}"}
    os.environ["STAKE_ACCESS_TOKEN"] = open(TOKEN_FILE).read().strip()

    params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER], env=os.environ.copy())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            # Extract text content
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"text": content.text}
            return {"raw": str(result.content)}


def fetch_fixture_odds(fixture_id: str) -> Dict[str, Any]:
    return asyncio.run(_mcp_call("stake_get_fixture_odds", {"fixture_id": fixture_id, "main_only": False}))


def search_fixtures(date_from: str, date_to: str) -> List[Dict[str, Any]]:
    r = asyncio.run(_mcp_call("stake_search_fixture", {
        "date_from": date_from, "date_to": date_to,
        "tournament_id": "24556add-af26-4844-b338-07fc47991731",
    }))
    return r.get("result", r.get("fixtures", r if isinstance(r, list) else []))


def find_fixture_for_match(match_query: str) -> Optional[str]:
    """Search Stake for a match by name (e.g. 'Canada - Morocco' or 'CAN - MAR')."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Search ±1 day to handle UTC vs SGT mismatches
    from datetime import timedelta
    today_dt = datetime.now(timezone.utc)
    date_from = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    date_to = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    fixtures = search_fixtures(date_from, date_to)

    # Try multiple splits
    raw = match_query.replace(" vs ", "-").replace(" vs. ", "-").replace(" - ", "-")
    parts = re.split(r"[-–—]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        return None
    a, b = parts[0].lower(), parts[1].lower()

    ABBR = {
        "arg": "argentina", "aus": "australia", "bra": "brazil", "can": "canada",
        "col": "colombia", "crc": "costa rica", "civ": "ivory coast", "cze": "czech",
        "den": "denmark", "egy": "egypt", "eng": "england", "fra": "france",
        "ger": "germany", "ir": "iran", "ir-iran": "iran", "ira": "iran",
        "jpn": "japan", "kor": "korea", "mls": "united states", "mar": "morocco",
        "mex": "mexico", "ned": "netherlands", "nga": "nigeria",
        "nor": "norway", "par": "paraguay", "pol": "poland", "por": "portugal",
        "rsa": "south africa", "ksa": "saudi arabia", "sau": "saudi arabia",
        "sen": "senegal", "srb": "serbia", "spa": "spain", "esp": "spain",
        "sui": "switzerland", "swe": "sweden", "tun": "tunisia", "tur": "turkey",
        "ukr": "ukraine", "uru": "uruguay", "usa": "united states", "us": "united states",
        "wal": "wales", "crc": "costa rica", "pan": "panama", "ecu": "ecuador",
        "uru": "uruguay", "ven": "venezuela", "gha": "ghana", "cmr": "cameroon",
        "alg": "algeria", "irq": "iraq",
    }
    a_full = ABBR.get(a, a)
    b_full = ABBR.get(b, b)

    for f in fixtures:
        name = (f.get("name", "")).lower().replace(" - ", "-").replace("–", "-").replace("—", "-")
        # Try full names first
        if a_full in name and b_full in name:
            return f.get("id")
        # Try abbreviation match if names have parens (e.g. "USA (United States)")
        if a in name and b in name:
            return f.get("id")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event")
    ap.add_argument("--fixture-id")
    ap.add_argument("--match")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fixture_id = args.fixture_id
    if not fixture_id and args.match:
        fixture_id = find_fixture_for_match(args.match)
        if not fixture_id:
            print(f"❌ could not find Stake fixture for '{args.match}'", file=sys.stderr)
            sys.exit(1)
    if not fixture_id and args.event:
        # Try to search today's fixtures
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fixtures = search_fixtures(today, today)
        if not fixtures:
            print("❌ no Stake fixtures found today", file=sys.stderr)
            sys.exit(1)
        # Pick first one (best-effort)
        fixture_id = fixtures[0].get("id")
        print(f"  using first fixture: {fixtures[0].get('name')} → {fixture_id}", file=sys.stderr)
    if not fixture_id:
        print("❌ --fixture-id, --event, or --match required", file=sys.stderr)
        sys.exit(1)

    print(f"  fixture_id: {fixture_id}", file=sys.stderr)
    raw = fetch_fixture_odds(fixture_id)
    if not raw or "error" in raw and "result" not in raw:
        print(f"❌ fetch failed: {raw}", file=sys.stderr)
        sys.exit(1)
    # The MCP returns {"result": {...}} or just {...}
    markets = raw.get("result", raw)
    if "markets" not in markets:
        print(f"❌ no markets in response: {list(markets.keys())}", file=sys.stderr)
        sys.exit(1)

    flat = _flatten_markets(markets)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(flat, indent=2, default=str))
    n = len(flat["markets_flat"])
    print(f"  wrote {args.output}  ({n} outcomes)")


if __name__ == "__main__":
    main()
