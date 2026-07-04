#!/usr/bin/env python3
"""
WC 2026 Betting Markets Predictor
==================================

End-to-end runner that:
1. Loads the prediction engine (iter-6 with ESPN lineup support)
2. Computes the full scoreline distribution for each upcoming match
3. Derives all betting market probabilities (1X2, O/U, AH, BTTS, etc.)
4. Optionally compares to bookmaker odds to find value bets
5. Outputs a clean JSON file for Hermes (or any betting bot) to consume

Usage:
    python3 wc_betting_predictor.py                          # default run
    python3 wc_betting_predictor.py --force-refresh          # re-fetch ESPN lineups
    python3 wc_betting_predictor.py --no-lineups             # skip ESPN, use squads.json
    python3 wc_betting_predictor.py --bookmaker-odds odds.json  # find value bets
    python3 wc_betting_predictor.py --event 760501           # test on a specific event

Output:
    /home/z/my-project/download/wc2026_betting_markets.json

Output format:
    {
      "generated_at": "2026-07-04T12:00:00Z",
      "model_version": "v1.4.0",
      "lineups_used": true,
      "matches": [
        {
          "n": 89,
          "stage": "r16",
          "date": "2026-07-04",
          "home": "PAR",
          "away": "FRA",
          "lineup_available": false,
          "markets": {
            "1X2": {"home_win": 0.02, "draw": 0.05, "away_win": 0.93, ...},
            "double_chance": {"1X": ..., "12": ..., "X2": ...},
            "over_under": {"2.5": {"over": ..., "under": ...}, ...},
            "asian_handicap": {"-0.5": {"home_win": ..., "push": ..., "home_loss": ...}, ...},
            "btts": {"yes": ..., "no": ...},
            "correct_score_top5": [{"score": "0-2", "prob": 0.58}, ...],
            "expected_goals": {"home": 1.4, "away": 2.5, "total": 3.9}
          }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from wc_predictor import (
    HOST_BONUS_DEFAULT, Match, PredictionEngine, ScoreDist,
    load_intl_history, load_matches, load_teams, predict_upcoming,
)
from wc_predictor_iter3 import _TEAM_RATINGS_REGISTRY
from wc_predictor_iter4 import SquadFeatures, load_squad_features
from wc_predictor_iter4b import DecisionRuleEngine
from wc_predictor_iter4e import TargetedDecisionRuleModel
from wc_predictor_iter6 import adjust_elo_with_lineups
from espn_lineups import fetch_all_upcoming_lineups, fetch_lineups_with_cache
from betting_markets import compute_all_markets, add_fair_odds, find_value_bets

try:
    from config import DOWNLOAD_DIR
except ImportError:
    DOWNLOAD_DIR = Path("/home/z/my-project/download")
MODEL_VERSION = "v1.4.0"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="WC 2026 Betting Markets Predictor")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Ignore ESPN cache, re-fetch all lineups")
    ap.add_argument("--no-lineups", action="store_true",
                    help="Skip ESPN fetch; use full-squad star_power")
    ap.add_argument("--event", type=str, default=None,
                    help="Test on a specific ESPN event ID")
    ap.add_argument("--bookmaker-odds", type=str, default=None,
                    help="Path to JSON file with bookmaker odds for value bet detection")
    ap.add_argument("--output", type=str, default=None,
                    help="Output file path (default: download/wc2026_betting_markets.json)")
    args = ap.parse_args()

    print("="*70)
    print(f"WC 2026 Betting Markets Predictor ({MODEL_VERSION})")
    print("="*70)

    # 1. Load base data
    print("\n[1/5] Loading base data ...")
    matches = load_matches()
    teams = load_teams()
    history = load_intl_history()
    squads = load_squad_features()
    print(f"  Matches: {len(matches)} ({sum(1 for m in matches if m.status == 'finished')} finished)")
    print(f"  Teams:   {len(teams)}")

    # 2. Fetch lineups (or skip)
    lineups = {}
    if args.event:
        print(f"\n[2/5] Fetching lineup for test event {args.event} ...")
        lineup = fetch_lineups_with_cache(args.event, force_refresh=args.force_refresh)
        if lineup:
            lineups[f"{lineup.home.fifa_code}-{lineup.away.fifa_code}"] = lineup
    elif not args.no_lineups:
        print("\n[2/5] Fetching ESPN lineups for upcoming matches ...")
        lineups = fetch_all_upcoming_lineups()
    else:
        print("\n[2/5] Skipping ESPN lineups (--no-lineups)")

    print(f"  Lineups fetched: {len(lineups)}")

    # 3. Adjust ELO with lineups
    print(f"\n[3/5] Adjusting ELO with lineup-derived star_power ...")
    teams_adj, lineup_debug = adjust_elo_with_lineups(teams, squads, lineups, star_weight=40.0)
    if lineup_debug:
        for code, d in lineup_debug.items():
            if d.get("fallback"):
                continue
            print(f"  {code}: SP {d['old_star_power']:.3f} → {d['new_star_power']:.3f} "
                  f"(Δ={d['delta']:+.3f}, matched {d['matched_count']}/11)")

    # 4. Build the engine
    print(f"\n[4/5] Building prediction engine (iter-4e decision rule) ...")
    model = TargetedDecisionRuleModel(
        big_fav=300, strong_fav=150, slight_fav=50,
        low_star_threshold=0.5, enable_low_star_fix=True,
        enable_high_underdog_concession=False, high_underdog_threshold=0.4,
    )
    engine = DecisionRuleEngine(
        teams=teams_adj, intl_history=history,
        score_model=model, host_bonus=80.0,
        use_form_elo=True, form_weight=0.5,
        squads=squads,
    )

    # 5. Compute betting markets for each upcoming match
    print(f"\n[5/5] Computing betting markets for upcoming matches ...")
    # We need the raw distribution (not just the mode), so re-run predict_upcoming
    # but capture the full ScoreDist
    betting_matches = []
    for m in matches:
        if m.status == "finished":
            continue
        if not m.home_code or not m.away_code:
            betting_matches.append({
                "n": m.n, "stage": m.stage, "date": m.date.date().isoformat(),
                "home": "TBD", "away": "TBD",
                "note": "Bracket-dependent; teams not yet determined.",
                "markets": None,
            })
            continue
        try:
            dist = engine.predict(m.home_code, m.away_code, m.venue_country)
        except Exception as e:
            print(f"  ERROR on match {m.n}: {e}")
            continue
        # Compute all betting markets from the distribution
        markets = compute_all_markets(dist.probs, m.home_code, m.away_code, dist.max_goals)
        markets_dict = markets.to_dict()
        markets_dict_with_odds = add_fair_odds(markets_dict)

        match_key = f"{m.home_code}-{m.away_code}"
        lineup = lineups.get(match_key)
        lineup_info = None
        if lineup:
            lineup_info = {
                "home_formation": lineup.home.formation,
                "away_formation": lineup.away.formation,
                "home_starters_count": len(lineup.home.starters),
                "away_starters_count": len(lineup.away.starters),
            }
        home_debug = lineup_debug.get(m.home_code, {})
        away_debug = lineup_debug.get(m.away_code, {})

        betting_matches.append({
            "n": m.n,
            "stage": m.stage,
            "date": m.date.date().isoformat(),
            "kickoff_utc": m.date.isoformat(),
            "home": m.home_code,
            "away": m.away_code,
            "venue_country": m.venue_country,
            "lineup_available": lineup is not None,
            "lineup_info": lineup_info,
            "home_star_power": {
                "squad": home_debug.get("old_star_power"),
                "lineup": home_debug.get("new_star_power"),
                "delta": home_debug.get("delta"),
            } if home_debug else None,
            "away_star_power": {
                "squad": away_debug.get("old_star_power"),
                "lineup": away_debug.get("new_star_power"),
                "delta": away_debug.get("delta"),
            } if away_debug else None,
            "markets": markets_dict_with_odds["markets"],
        })

    # 6. Optional: value bet detection
    value_bets_by_match = {}
    if args.bookmaker_odds:
        print(f"\n[6/6] Loading bookmaker odds from {args.bookmaker_odds} ...")
        try:
            bookmaker_odds = json.loads(Path(args.bookmaker_odds).read_text())
            for match in betting_matches:
                if not match.get("markets"):
                    continue
                match_key = f"{match['home']}-{match['away']}"
                match_odds = bookmaker_odds.get(match_key, {})
                if not match_odds:
                    continue
                # Flatten model probs into a single dict for comparison
                model_probs = _flatten_markets(match["markets"])
                vbs = find_value_bets(model_probs, match_odds)
                if vbs:
                    value_bets_by_match[match_key] = vbs
            print(f"  Found {sum(len(v) for v in value_bets_by_match.values())} value bets "
                  f"across {len(value_bets_by_match)} matches")
        except Exception as e:
            print(f"  [WARN] could not load bookmaker odds: {e}")

    # 7. Build output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "engine_config": {
            "base_model": "iter-4e targeted decision rule",
            "star_weight": 40.0,
            "host_bonus": 80.0,
            "form_weight": 0.5,
            "lineups_used": len(lineups) > 0,
            "lineup_match_count": len(lineups),
        },
        "matches": betting_matches,
    }
    if value_bets_by_match:
        output["value_bets"] = value_bets_by_match

    # 8. Write output
    out_path = Path(args.output) if args.output else (DOWNLOAD_DIR / "wc2026_betting_markets.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Wrote betting markets -> {out_path}")

    # 9. Print summary table
    print(f"\n=== Summary (matches with predictions) ===")
    print(f"{'#':>3} {'Stage':<6} {'Date':<12} {'Home':<5} {'Away':<5} {'1X2 (H/D/A)':<22} {'O/U 2.5':<14} {'AH -0.5 (H)':<14} {'Lineup':<7}")
    print("-" * 100)
    for m in betting_matches:
        if not m.get("markets"):
            continue
        mk = m["markets"]
        h, d, a = mk["1X2"]["home_win"], mk["1X2"]["draw"], mk["1X2"]["away_win"]
        ou = mk["over_under"].get("2.5", {}).get("over", 0)
        ah = mk["asian_handicap"].get("-0.5", {}).get("home_win", 0)
        lineup_str = "YES" if m.get("lineup_available") else "no"
        print(f"{m['n']:>3} {m['stage']:<6} {m['date']:<12} {m['home']:<5} {m['away']:<5} "
              f"{h:.0%}/{d:.0%}/{a:.0%}        "
              f"{ou:.0%}          "
              f"{ah:.0%}          "
              f"{lineup_str:<7}")

    print(f"\nFull output: {out_path}")
    print(f"Model version: {MODEL_VERSION}")


def _flatten_markets(markets: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the nested markets dict into {selection: probability} for value bet comparison.

    Selection keys follow a Hermes-friendly naming convention:
      - 1X2: home_win, draw, away_win
      - double_chance: dc_1X, dc_12, dc_X2
      - over_under: ou_<line>_over, ou_<line>_under
      - asian_handicap: ah_<line>_home_win, ah_<line>_away_win
      - btts: btts_yes, btts_no
      - correct_score: cs_<h>-<a>
    """
    flat = {}
    # 1X2
    for k in ["home_win", "draw", "away_win"]:
        if k in markets.get("1X2", {}):
            flat[k] = markets["1X2"][k]
    # Double chance
    for k in ["1X", "12", "X2"]:
        if k in markets.get("double_chance", {}):
            flat[f"dc_{k}"] = markets["double_chance"][k]
    # O/U
    for line, probs in markets.get("over_under", {}).items():
        flat[f"ou_{line}_over"] = probs.get("over", 0)
        flat[f"ou_{line}_under"] = probs.get("under", 0)
    # AH (use home_win and away_win, ignoring push for value bet purposes)
    for line, probs in markets.get("asian_handicap", {}).items():
        flat[f"ah_{line}_home_win"] = probs.get("home_win", 0)
        flat[f"ah_{line}_away_win"] = probs.get("away_win", 0)
    # BTTS
    if "btts" in markets:
        flat["btts_yes"] = markets["btts"].get("yes", 0)
        flat["btts_no"] = markets["btts"].get("no", 0)
    # Correct score top-5
    for cs in markets.get("correct_score_top5", []):
        flat[f"cs_{cs['score']}"] = cs["prob"]
    return flat


if __name__ == "__main__":
    main()
