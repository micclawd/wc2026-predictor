#!/usr/bin/env python3
"""
Stake.com Bet Placer — bridges model output to Stake bets
==========================================================

Reads the betting markets JSON from wc_betting_predictor.py, finds value
bets, looks up the corresponding outcomes on Stake.com, validates them,
and places the bets.

Supports two modes:
  1. SINGLES  — place each value bet as an independent single bet
                (safer, no correlation risk, default)
  2. MULTIBET — combine value bets across DIFFERENT matches into one multibet
                (higher odds, but only across matches — never within)

NEVER combines selections from the same match into a multibet (correlated).

USAGE
-----
    # Place singles (default)
    python3 stake_bet_placer.py --session-file stake_session.json

    # Place multibet across matches (for QF/SF rounds with multiple matches)
    python3 stake_bet_placer.py --mode multibet

    # Dry run (validate but don't place)
    python3 stake_bet_placer.py --dry-run

    # Specify fixture slugs (needed because we can't auto-discover them)
    python3 stake_bet_placer.py --fixtures fixtures.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from stake_api import StakeAPIClient, ManualSessionProvider, Outcome
from betting_markets import find_value_bets, to_implied_odds

try:
    from config import DOWNLOAD_DIR
except ImportError:
    DOWNLOAD_DIR = Path("/home/z/my-project/download")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BET_LOG = DOWNLOAD_DIR.parent / "bets_log.jsonl"
FIXTURES_FILE = DOWNLOAD_DIR.parent / "fixtures.json"

# Bet sizing rules (matches the rules given to Hermes)
MIN_EDGE = 0.05           # 5% minimum edge to consider
MIN_KELLY = 0.03          # 3% minimum Kelly fraction
MIN_ODDS = 1.50           # no sub-1.50 favorites
MAX_SELECTIONS_PER_MATCH = 3
MAX_EXPOSURE_PER_MATCH_PCT = 0.05   # 5% of bankroll per match
MAX_TOTAL_EXPOSURE_PCT = 0.25       # 25% across all open bets
KELLY_FRACTION = 0.25               # quarter-Kelly for actual stake
DEFAULT_CURRENCY = "usdt"
MIN_BET_AMOUNT = 0.01               # Stake.com minimum


# --------------------------------------------------------------------------
# Selection key → Stake market name mapping
# --------------------------------------------------------------------------
# Our model produces selection keys like "ou_2.5_over", "ah_-0.5_home_win"
# Stake uses market names like "Total Goals", "Asian Handicap", "1X2"
# This mapping tells the placer how to look up each selection on Stake.

SELECTION_TO_STAKE = {
    # 1X2
    "home_win":   ("1X2", "Home", None),
    "draw":       ("1X2", "Draw", None),
    "away_win":   ("1X2", "Away", None),
    # Double chance
    "dc_1X":      ("Double Chance", "Home/Draw", None),
    "dc_12":      ("Double Chance", "Home/Away", None),
    "dc_X2":      ("Double Chance", "Draw/Away", None),
    # Over/Under (specifier is the line)
    "ou_0.5_over":  ("Total Goals", "Over", "0.5"),
    "ou_0.5_under": ("Total Goals", "Under", "0.5"),
    "ou_1.5_over":  ("Total Goals", "Over", "1.5"),
    "ou_1.5_under": ("Total Goals", "Under", "1.5"),
    "ou_2.5_over":  ("Total Goals", "Over", "2.5"),
    "ou_2.5_under": ("Total Goals", "Under", "2.5"),
    "ou_3.5_over":  ("Total Goals", "Over", "3.5"),
    "ou_3.5_under": ("Total Goals", "Under", "3.5"),
    "ou_4.5_over":  ("Total Goals", "Over", "4.5"),
    "ou_4.5_under": ("Total Goals", "Under", "4.5"),
    # Asian Handicap (specifier is the line)
    "ah_-2.5_home_win":  ("Asian Handicap", "Home", "-2.5"),
    "ah_-2.5_away_win":  ("Asian Handicap", "Away", "-2.5"),
    "ah_-2.0_home_win":  ("Asian Handicap", "Home", "-2.0"),
    "ah_-2.0_away_win":  ("Asian Handicap", "Away", "-2.0"),
    "ah_-1.5_home_win":  ("Asian Handicap", "Home", "-1.5"),
    "ah_-1.5_away_win":  ("Asian Handicap", "Away", "-1.5"),
    "ah_-1.0_home_win":  ("Asian Handicap", "Home", "-1.0"),
    "ah_-1.0_away_win":  ("Asian Handicap", "Away", "-1.0"),
    "ah_-0.5_home_win":  ("Asian Handicap", "Home", "-0.5"),
    "ah_-0.5_away_win":  ("Asian Handicap", "Away", "-0.5"),
    "ah_0.0_home_win":   ("Asian Handicap", "Home", "0.0"),
    "ah_0.0_away_win":   ("Asian Handicap", "Away", "0.0"),
    "ah_0.5_home_win":   ("Asian Handicap", "Home", "0.5"),
    "ah_0.5_away_win":   ("Asian Handicap", "Away", "0.5"),
    "ah_1.0_home_win":   ("Asian Handicap", "Home", "1.0"),
    "ah_1.0_away_win":   ("Asian Handicap", "Away", "1.0"),
    "ah_1.5_home_win":   ("Asian Handicap", "Home", "1.5"),
    "ah_1.5_away_win":   ("Asian Handicap", "Away", "1.5"),
    "ah_2.0_home_win":   ("Asian Handicap", "Home", "2.0"),
    "ah_2.0_away_win":   ("Asian Handicap", "Away", "2.0"),
    "ah_2.5_home_win":   ("Asian Handicap", "Home", "2.5"),
    "ah_2.5_away_win":   ("Asian Handicap", "Away", "2.5"),
    # BTTS
    "btts_yes":   ("Both Teams to Score", "Yes", None),
    "btts_no":    ("Both Teams to Score", "No", None),
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class ValueBet:
    """A value bet detected by the model and resolved to a Stake outcome."""
    match_key: str           # e.g. "CAN-MAR"
    selection: str           # e.g. "ah_-0.5_away_win"
    model_prob: float
    stake_odds: float        # odds found on Stake
    implied_prob: float
    edge: float
    kelly_fraction: float
    stake_amount: float      # calculated stake
    outcome_id: str          # Stake outcome UUID
    market_name: str
    market_specifier: Optional[str]
    selection_name: str      # human-readable selection


@dataclass
class PlacedBet:
    """A bet that was actually placed on Stake."""
    bet_id: str
    amount: float
    currency: str
    potential_multiplier: float
    selections: List[Dict[str, Any]]
    placed_at: str


# --------------------------------------------------------------------------
# Bankroll management
# --------------------------------------------------------------------------
def load_bankroll() -> float:
    """Load current bankroll from bets_log.jsonl (or return default)."""
    if not BET_LOG.exists():
        return 75.0   # default starting bankroll
    # Bankroll = starting - settled losses + settled wins
    # For now, just return the latest "bankroll_after" field if present
    last_line = None
    with BET_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line:
        try:
            entry = json.loads(last_line)
            if "bankroll_after" in entry:
                return float(entry["bankroll_after"])
        except json.JSONDecodeError:
            pass
    return 75.0


def calculate_stake(kelly_fraction: float, bankroll: float,
                    max_stake: float) -> float:
    """Calculate the actual stake using quarter-Kelly, capped."""
    stake = bankroll * kelly_fraction * KELLY_FRACTION
    stake = min(stake, max_stake)
    stake = max(stake, MIN_BET_AMOUNT)   # Stake.com minimum
    # Round to 2 decimal places (USDT precision)
    return round(stake, 2)


# --------------------------------------------------------------------------
# Fixture slug resolution
# --------------------------------------------------------------------------
def load_fixtures() -> Dict[str, str]:
    """Load the match → fixture_slug mapping.

    Since we can't auto-discover Stake fixture slugs from team codes,
    we maintain a manual mapping in fixtures.json. Update this file
    before each match round.
    """
    if not FIXTURES_FILE.exists():
        return {}
    return json.loads(FIXTURES_FILE.read_text())


# --------------------------------------------------------------------------
# Find value bets in the model output
# --------------------------------------------------------------------------
def find_value_bets_in_output(output: Dict[str, Any],
                              stake_client: StakeAPIClient,
                              bankroll: float,
                              fixtures: Dict[str, str],
                              dry_run: bool = False) -> List[ValueBet]:
    """Find value bets in the betting markets output and resolve to Stake outcomes.

    Returns a list of ValueBet objects with outcome IDs filled in.
    """
    value_bets: List[ValueBet] = []
    max_per_match = bankroll * MAX_EXPOSURE_PER_MATCH_PCT

    for match in output.get("matches", []):
        if not match.get("markets"):
            continue
        home = match.get("home", "")
        away = match.get("away", "")
        if home == "TBD" or away == "TBD":
            continue
        match_key = f"{home}-{away}"
        fixture_slug = fixtures.get(match_key)
        if not fixture_slug:
            print(f"  [SKIP] {match_key}: no fixture slug in fixtures.json")
            continue

        print(f"\n  [{match_key}] Fetching Stake markets for fixture {fixture_slug} ...")
        try:
            fixture_data = stake_client.get_fixture_markets(fixture_slug)
        except Exception as e:
            print(f"  [ERROR] Failed to fetch fixture: {e}")
            continue

        # Flatten model markets to selection → probability
        model_probs = _flatten_markets(match["markets"])
        # Filter to selections we know how to look up on Stake
        candidate_selections = [
            (sel, p) for sel, p in model_probs.items()
            if sel in SELECTION_TO_STAKE and p > 0
        ]
        if not candidate_selections:
            continue

        # Resolve each to a Stake outcome and check odds
        match_bets: List[ValueBet] = []
        for sel, model_p in candidate_selections:
            market_name, selection_name, specifier = SELECTION_TO_STAKE[sel]
            outcome = stake_client.find_outcome(fixture_data, market_name,
                                                 selection_name, specifier)
            if outcome is None or not outcome.active:
                continue
            stake_odds = outcome.odds
            if stake_odds < MIN_ODDS:
                continue
            implied_p = 1.0 / stake_odds
            edge = model_p - implied_p
            if edge < MIN_EDGE:
                continue
            # Kelly fraction
            kelly = (model_p * stake_odds - 1) / (stake_odds - 1)
            kelly = max(0, min(kelly, 1.0))
            if kelly < MIN_KELLY:
                continue
            stake = calculate_stake(kelly, bankroll, max_per_match)
            vb = ValueBet(
                match_key=match_key,
                selection=sel,
                model_prob=model_p,
                stake_odds=stake_odds,
                implied_prob=implied_p,
                edge=edge,
                kelly_fraction=kelly,
                stake_amount=stake,
                outcome_id=outcome.id,
                market_name=market_name,
                market_specifier=specifier,
                selection_name=selection_name,
            )
            match_bets.append(vb)
            print(f"    ✓ {sel}: model={model_p:.1%} stake={stake_odds} edge={edge:+.1%} kelly={kelly:.1%} stake=${stake}")

        # Limit per match
        match_bets.sort(key=lambda x: x.edge, reverse=True)
        match_bets = match_bets[:MAX_SELECTIONS_PER_MATCH]
        value_bets.extend(match_bets)

    return value_bets


def _flatten_markets(markets: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the nested markets dict into {selection: probability}."""
    flat = {}
    for k in ["home_win", "draw", "away_win"]:
        if k in markets.get("1X2", {}):
            flat[k] = markets["1X2"][k]
    for k in ["1X", "12", "X2"]:
        if k in markets.get("double_chance", {}):
            flat[f"dc_{k}"] = markets["double_chance"][k]
    for line, probs in markets.get("over_under", {}).items():
        flat[f"ou_{line}_over"] = probs.get("over", 0)
        flat[f"ou_{line}_under"] = probs.get("under", 0)
    for line, probs in markets.get("asian_handicap", {}).items():
        flat[f"ah_{line}_home_win"] = probs.get("home_win", 0)
        flat[f"ah_{line}_away_win"] = probs.get("away_win", 0)
    if "btts" in markets:
        flat["btts_yes"] = markets["btts"].get("yes", 0)
        flat["btts_no"] = markets["btts"].get("no", 0)
    return flat


# --------------------------------------------------------------------------
# Place bets
# --------------------------------------------------------------------------
def place_singles(client: StakeAPIClient, value_bets: List[ValueBet],
                  currency: str, dry_run: bool = False) -> List[PlacedBet]:
    """Place each value bet as an independent single bet."""
    placed = []
    for vb in value_bets:
        if dry_run:
            print(f"  [DRY-RUN] Would place ${vb.stake_amount} {currency} on "
                  f"{vb.match_key} {vb.selection_name} @ {vb.stake_odds}")
            placed.append(PlacedBet(
                bet_id="dry-run",
                amount=vb.stake_amount,
                currency=currency,
                potential_multiplier=vb.stake_odds,
                selections=[{
                    "match": vb.match_key,
                    "selection": vb.selection,
                    "outcome_id": vb.outcome_id,
                    "odds": vb.stake_odds,
                }],
                placed_at=datetime.now(timezone.utc).isoformat(),
            ))
            continue
        try:
            bet = client.place_single_bet(vb.stake_amount, currency, vb.outcome_id)
            placed.append(PlacedBet(
                bet_id=bet.id,
                amount=bet.amount,
                currency=bet.currency,
                potential_multiplier=bet.potential_multiplier,
                selections=[{
                    "match": vb.match_key,
                    "selection": vb.selection,
                    "outcome_id": vb.outcome_id,
                    "odds": vb.stake_odds,
                }],
                placed_at=datetime.now(timezone.utc).isoformat(),
            ))
            print(f"  ✓ Placed ${bet.amount} on {vb.match_key} {vb.selection_name} "
                  f"@ {vb.stake_odds} (bet ID: {bet.id})")
        except Exception as e:
            print(f"  ✗ Failed to place bet on {vb.match_key} {vb.selection}: {e}")
    return placed


def place_multibet(client: StakeAPIClient, value_bets: List[ValueBet],
                   currency: str, dry_run: bool = False) -> Optional[PlacedBet]:
    """Combine value bets ACROSS matches into one multibet.

    IMPORTANT: Never combines selections from the same match (correlated).
    Picks the top selection from each match.
    """
    # Group by match, pick top edge from each
    by_match: Dict[str, ValueBet] = {}
    for vb in value_bets:
        if vb.match_key not in by_match or vb.edge > by_match[vb.match_key].edge:
            by_match[vb.match_key] = vb
    if len(by_match) < 2:
        print("  [SKIP] Multibet requires 2+ matches with value bets")
        return None

    selections = list(by_match.values())
    outcome_ids = [s.outcome_id for s in selections]
    total_stake = min(s.stake_amount for s in selections)   # use smallest stake

    # Validate first
    print(f"  Validating multibet ({len(outcome_ids)} selections) ...")
    validation = client.validate_multibet(outcome_ids)
    if not validation["compatible"]:
        print(f"  ✗ Multibet has conflicts: {validation['conflicts']}")
        return None
    combined_odds = validation["odds"]
    print(f"  Combined odds: {combined_odds}")

    if dry_run:
        print(f"  [DRY-RUN] Would place ${total_stake} {currency} multibet at {combined_odds}×")
        return PlacedBet(
            bet_id="dry-run",
            amount=total_stake,
            currency=currency,
            potential_multiplier=combined_odds,
            selections=[{
                "match": s.match_key,
                "selection": s.selection,
                "outcome_id": s.outcome_id,
                "odds": s.stake_odds,
            } for s in selections],
            placed_at=datetime.now(timezone.utc).isoformat(),
        )

    try:
        bet = client.place_multibet(total_stake, currency, outcome_ids)
        print(f"  ✓ Multibet placed: ${bet.amount} at {bet.potential_multiplier}× (ID: {bet.id})")
        return PlacedBet(
            bet_id=bet.id,
            amount=bet.amount,
            currency=bet.currency,
            potential_multiplier=bet.potential_multiplier,
            selections=[{
                "match": s.match_key,
                "selection": s.selection,
                "outcome_id": s.outcome_id,
                "odds": s.stake_odds,
            } for s in selections],
            placed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        print(f"  ✗ Failed to place multibet: {e}")
        return None


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log_placed_bets(placed: List[PlacedBet], value_bets: List[ValueBet],
                    bankroll_before: float, model_version: str) -> None:
    """Append placed bets to bets_log.jsonl."""
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    bankroll_after = bankroll_before
    vb_by_outcome = {vb.outcome_id: vb for vb in value_bets}
    with BET_LOG.open("a") as f:
        for bet in placed:
            for sel in bet["selections"] if isinstance(bet, dict) else bet.selections:
                vb = vb_by_outcome.get(sel["outcome_id"])
                if vb is None:
                    continue
                bankroll_after -= bet.amount if isinstance(bet, PlacedBet) else 0
                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "bet_id": bet.bet_id if isinstance(bet, PlacedBet) else bet.get("bet_id"),
                    "match": vb.match_key,
                    "market": vb.market_name,
                    "selection": vb.selection,
                    "selection_name": vb.selection_name,
                    "odds": vb.stake_odds,
                    "stake": bet.amount if isinstance(bet, PlacedBet) else bet.get("amount"),
                    "currency": bet.currency if isinstance(bet, PlacedBet) else bet.get("currency"),
                    "model_prob": vb.model_prob,
                    "implied_prob": vb.implied_prob,
                    "edge": vb.edge,
                    "kelly": vb.kelly_fraction,
                    "model_version": model_version,
                    "bankroll_after": round(bankroll_after, 2),
                }
                f.write(json.dumps(entry) + "\n")
    print(f"\n  Logged {len(placed)} bets to {BET_LOG}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stake.com Bet Placer")
    ap.add_argument("--session-file", default="stake_session.json",
                    help="Path to Stake session JSON file")
    ap.add_argument("--mode", choices=["singles", "multibet"], default="singles",
                    help="Bet mode: 'singles' (one bet per selection) or 'multibet' (combine across matches)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and show what would be placed, but don't actually place")
    ap.add_argument("--fixtures", default=None,
                    help="Path to fixtures.json (match → fixture_slug mapping)")
    ap.add_argument("--input", default=None,
                    help="Path to betting markets JSON (default: download/wc2026_betting_markets.json)")
    ap.add_argument("--bankroll", type=float, default=None,
                    help="Override bankroll (default: read from bets_log.jsonl)")
    ap.add_argument("--currency", default=DEFAULT_CURRENCY,
                    help=f"Bet currency (default: {DEFAULT_CURRENCY})")
    args = ap.parse_args()

    print("="*70)
    print("Stake.com Bet Placer")
    print("="*70)

    # Load session
    try:
        provider = ManualSessionProvider.from_file(args.session_file)
        print(f"\n[1/5] Loaded Stake session from {args.session_file}")
        print(f"  Session valid: {provider.is_valid()}")
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nCreate stake_session.json from stake_session.example.json")
        sys.exit(1)

    # Load fixtures mapping
    fixtures_path = Path(args.fixtures) if args.fixtures else FIXTURES_FILE
    fixtures = json.loads(fixtures_path.read_text()) if fixtures_path.exists() else {}
    print(f"\n[2/5] Loaded {len(fixtures)} fixture mappings from {fixtures_path}")

    # Load betting markets
    input_path = Path(args.input) if args.input else (DOWNLOAD_DIR / "wc2026_betting_markets.json")
    if not input_path.exists():
        print(f"\n✗ Betting markets file not found: {input_path}")
        print("Run wc_betting_predictor.py first.")
        sys.exit(1)
    output = json.loads(input_path.read_text())
    model_version = output.get("model_version", "unknown")
    print(f"\n[3/5] Loaded betting markets (model_version: {model_version})")

    # Bankroll
    bankroll = args.bankroll if args.bankroll else load_bankroll()
    print(f"\n[4/5] Bankroll: ${bankroll:.2f}")

    # Find value bets
    client = StakeAPIClient(provider, verbose=True)
    print(f"\n[5/5] Finding value bets ...")
    value_bets = find_value_bets_in_output(output, client, bankroll, fixtures, args.dry_run)

    if not value_bets:
        print("\n✗ No value bets found (none cleared the thresholds)")
        print(f"  Thresholds: edge≥{MIN_EDGE:.0%}, kelly≥{MIN_KELLY:.0%}, odds≥{MIN_ODDS}")
        sys.exit(0)

    print(f"\n=== Found {len(value_bets)} value bets ===")
    total_exposure = sum(vb.stake_amount for vb in value_bets)
    print(f"Total exposure: ${total_exposure:.2f} ({total_exposure/bankroll:.1%} of bankroll)")

    if total_exposure > bankroll * MAX_TOTAL_EXPOSURE_PCT:
        print(f"\n✗ Total exposure exceeds {MAX_TOTAL_EXPOSURE_PCT:.0%} cap. Trimming ...")
        # Sort by edge and keep top selections until under cap
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        kept = []
        running = 0
        for vb in value_bets:
            if running + vb.stake_amount > bankroll * MAX_TOTAL_EXPOSURE_PCT:
                continue
            kept.append(vb)
            running += vb.stake_amount
        value_bets = kept
        print(f"  Trimmed to {len(value_bets)} bets, ${running:.2f} total")

    # Place bets
    print(f"\n=== Placing bets (mode: {args.mode}) ===")
    if args.mode == "singles":
        placed = place_singles(client, value_bets, args.currency, args.dry_run)
    else:
        bet = place_multibet(client, value_bets, args.currency, args.dry_run)
        placed = [bet] if bet else []

    if not placed:
        print("\n✗ No bets placed")
        sys.exit(0)

    # Log
    if not args.dry_run:
        log_placed_bets(placed, value_bets, bankroll, model_version)
    else:
        print("\n[DRY-RUN] No bets logged")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Bets placed: {len(placed)}")
    print(f"Total staked: ${sum(b.amount for b in placed):.2f} {args.currency}")
    if args.mode == "multibet" and placed:
        print(f"Combined odds: {placed[0].potential_multiplier:.2f}×")
        print(f"Potential return: ${placed[0].amount * placed[0].potential_multiplier:.2f}")
    print(f"\nBets logged to: {BET_LOG}")


if __name__ == "__main__":
    main()
