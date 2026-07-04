#!/usr/bin/env python3
"""
WC 2026 Predictor — Iteration 6: Lineup-Adjusted Predictions
=============================================================

Integrates ESPN lineup data into the R16 prediction model.

Pipeline:
1. Fetch starting XIs from ESPN (~1h before kickoff)
2. Map ESPN starters to squads.json player IDs
3. Recompute star_power using ONLY the 11 starters
4. Adjust ELO with the new star_power
5. Predict scoreline using iter-4e's decision rule

If lineups aren't available yet, falls back to full-squad star_power (iter-4e).

Usage:
    python3 wc_predictor_iter6.py                    # fetch lineups + predict
    python3 wc_predictor_iter6.py --force-refresh    # ignore cache, re-fetch
    python3 wc_predictor_iter6.py --no-lineups       # skip ESPN, use squads.json
    python3 wc_predictor_iter6.py --event 760501     # test on a specific event
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from wc_predictor import (
    HOST_BONUS_DEFAULT, HOST_OF, Match, PoissonModel, PredictionEngine,
    ScoreDist, TeamRating, _brier, _log_loss, load_intl_history, load_matches,
    load_teams, predict_upcoming,
)
from wc_predictor_iter3 import _TEAM_RATINGS_REGISTRY
from wc_predictor_iter4 import (
    SquadFeatures, load_squad_features, adjust_elo_with_stars,
)
from wc_predictor_iter4b import DecisionRuleEngine
from wc_predictor_iter4e import TargetedDecisionRuleModel
from espn_lineups import (
    ESPN_TO_FIFA, MatchLineups, StarterPlayer, TeamLineup,
    compute_lineup_star_power, fetch_all_upcoming_lineups,
    fetch_lineups_with_cache, fetch_match_lineups, get_scoreboard_events,
)

try:
    from config import DOWNLOAD_DIR, SQUADS_JSON
except ImportError:
    DOWNLOAD_DIR = Path("/home/z/my-project/download")
    SQUADS_JSON = Path("/home/z/my-project/26worldcup.github.io/public/data/squads.json")


# --------------------------------------------------------------------------
# Adjust team ratings using lineup-derived star_power
# --------------------------------------------------------------------------
def adjust_elo_with_lineups(
    teams: Dict[str, TeamRating],
    squads: Dict[str, SquadFeatures],
    lineups: Dict[str, MatchLineups],
    star_weight: float = 40.0,
) -> Tuple[Dict[str, TeamRating], Dict[str, Dict[str, Any]]]:
    """Adjust ELO using lineup-derived star_power.

    Args:
        teams: prebuilt team ratings
        squads: squad features (with 26-man star_power)
        lineups: dict mapping "HOME-AWAY" to MatchLineups
        star_weight: ELO points per star_power unit

    Returns:
        (adjusted_teams, debug_info)
    """
    # Build a per-team lineup-derived star_power map
    lineup_star_power: Dict[str, float] = {}
    debug: Dict[str, Dict[str, Any]] = {}
    squads_raw = json.loads(SQUADS_JSON.read_text())

    for match_key, lineup in lineups.items():
        for team_lineup in [lineup.home, lineup.away]:
            fifa_code = team_lineup.fifa_code
            sq_features = squads.get(fifa_code)
            sq_players = squads_raw.get(fifa_code, {}).get("players", [])
            if not sq_features or not sq_players:
                continue
            new_sp, sp_debug = compute_lineup_star_power(
                team_lineup.starters, sq_features, sq_players
            )
            lineup_star_power[fifa_code] = new_sp
            debug[fifa_code] = sp_debug

    # Now adjust ELO using the lineup-derived star_power
    adjusted: Dict[str, TeamRating] = {}
    for code, t in teams.items():
        sq = squads.get(code)
        if code in lineup_star_power:
            # Use lineup-derived star_power
            new_sp = lineup_star_power[code]
            bonus = (new_sp - 0.5) * 2 * star_weight
        elif sq:
            # Fall back to full-squad star_power
            new_sp = sq.star_power
            bonus = (new_sp - 0.5) * 2 * star_weight
        else:
            bonus = 0
        adjusted[code] = TeamRating(
            code=code,
            elo_current=t.elo_current + bonus,
            elo_form=(t.elo_form + bonus) if t.elo_form is not None else None,
            fifa_ranking=t.fifa_ranking,
            group=t.group,
        )
    return adjusted, debug


# --------------------------------------------------------------------------
# Engine that uses lineup-adjusted ELO
# --------------------------------------------------------------------------
class LineupAdjustedEngine(PredictionEngine):
    """Engine that injects team ratings into the registry before predicting."""
    def predict(self, home: str, away: str, venue_country: Optional[str] = None) -> ScoreDist:
        global _TEAM_RATINGS_REGISTRY
        _TEAM_RATINGS_REGISTRY = self.teams
        return super().predict(home, away, venue_country)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-refresh", action="store_true",
                    help="Ignore cache, re-fetch all lineups from ESPN")
    ap.add_argument("--no-lineups", action="store_true",
                    help="Skip ESPN fetch; use full-squad star_power (iter-4e baseline)")
    ap.add_argument("--event", type=str, default=None,
                    help="Test on a specific ESPN event ID (e.g. 760501 for COL-GHA)")
    args = ap.parse_args()

    print("="*70)
    print("WC 2026 — Iteration 6: Lineup-Adjusted Predictions")
    print("="*70)

    # Load base data
    print("\n[1/5] Loading base data ...")
    matches = load_matches()
    teams = load_teams()
    history = load_intl_history()
    squads = load_squad_features()
    print(f"  Matches: {len(matches)} ({sum(1 for m in matches if m.status == 'finished')} finished)")
    print(f"  Teams:   {len(teams)}")
    print(f"  Squads:  {len(squads)}")

    # Fetch lineups
    lineups: Dict[str, MatchLineups] = {}
    lineup_debug: Dict[str, Dict[str, Any]] = {}
    if args.event:
        print(f"\n[2/5] Fetching lineups for test event {args.event} ...")
        lineup = fetch_lineups_with_cache(args.event, force_refresh=args.force_refresh)
        if lineup:
            match_key = f"{lineup.home.fifa_code}-{lineup.away.fifa_code}"
            lineups[match_key] = lineup
    elif not args.no_lineups:
        print("\n[2/5] Fetching lineups from ESPN for upcoming R16 matches ...")
        lineups = fetch_all_upcoming_lineups()
    else:
        print("\n[2/5] Skipping ESPN fetch (--no-lineups); using full-squad star_power")

    if lineups:
        print(f"\n  Got lineups for {len(lineups)} matches:")
        for key, lu in lineups.items():
            print(f"    {key}: {lu.home.fifa_code} ({lu.home.formation}) vs "
                  f"{lu.away.fifa_code} ({lu.away.formation})")
    else:
        print("\n  No lineups available. Will use full-squad star_power as fallback.")
        print("  (ESPN typically publishes lineups ~1 hour before kickoff)")

    # Adjust ELO with lineups
    print(f"\n[3/5] Adjusting ELO with lineup-derived star_power ...")
    teams_adj, lineup_debug = adjust_elo_with_lineups(teams, squads, lineups, star_weight=40.0)
    if lineup_debug:
        print(f"\n  Star_power adjustments (lineup vs squad):")
        print(f"  {'Team':<6} {'Old SP':>7} {'New SP':>7} {'Δ SP':>7} {'Matched':>9} {'Top5':>5} {'WCGoals':>8}")
        for code, d in lineup_debug.items():
            if d.get("fallback"):
                print(f"  {code:<6} {d['old_star_power']:>7.3f} {'FALLBACK':>7} {'—':>7} {'—':>9} {'—':>5} {'—':>8}")
            else:
                print(f"  {code:<6} {d['old_star_power']:>7.3f} {d['new_star_power']:>7.3f} "
                      f"{d['delta']:>+7.3f} {d['matched_count']:>4}/11 {d.get('starters_top5_count','?'):>5} "
                      f"{d.get('starters_wc_goals','?'):>8}")

    # Build the model (iter-4e config: the winning R32 backtest config)
    print(f"\n[4/5] Building prediction model (iter-4e config) ...")
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

    # Predict upcoming matches
    print(f"\n[5/5] Predicting upcoming R16 matches ...")
    preds = predict_upcoming(engine, matches)

    # Augment predictions with lineup info
    augmented_preds = []
    for p in preds:
        if "predicted_scoreline_mode" not in p:
            augmented_preds.append(p)
            continue
        match_key = f"{p['home']}-{p['away']}"
        lineup = lineups.get(match_key)
        lineup_info = None
        if lineup:
            lineup_info = {
                "home_formation": lineup.home.formation,
                "away_formation": lineup.away.formation,
                "home_starters": [
                    {"jersey": s.jersey, "pos": s.position, "name": s.name}
                    for s in lineup.home.starters
                ],
                "away_starters": [
                    {"jersey": s.jersey, "pos": s.position, "name": s.name}
                    for s in lineup.away.starters
                ],
            }
        home_debug = lineup_debug.get(p["home"], {})
        away_debug = lineup_debug.get(p["away"], {})
        p["lineup_info"] = lineup_info
        p["home_star_power"] = {
            "squad": home_debug.get("old_star_power"),
            "lineup": home_debug.get("new_star_power"),
            "delta": home_debug.get("delta"),
            "matched_count": home_debug.get("matched_count"),
        } if home_debug else None
        p["away_star_power"] = {
            "squad": away_debug.get("old_star_power"),
            "lineup": away_debug.get("new_star_power"),
            "delta": away_debug.get("delta"),
            "matched_count": away_debug.get("matched_count"),
        } if away_debug else None
        augmented_preds.append(p)

    # Print predictions
    print(f"\n{'#':>3} {'Date':<12} {'Home':<5} {'Away':<5} {'Pred':<8} {'H/D/A':<18} {'Lineup?':<8}")
    print("-" * 75)
    for p in augmented_preds:
        if "predicted_scoreline_mode" not in p:
            continue
        t1 = p["top5_scorelines"][0]
        has_lineup = "YES" if p.get("lineup_info") else "no"
        sp_h = p.get("home_star_power") or {}
        sp_a = p.get("away_star_power") or {}
        sp_str = ""
        if sp_h.get("lineup") is not None:
            sp_str = f" (H SP: {sp_h['squad']:.2f}→{sp_h['lineup']:.2f}, A SP: {sp_a['squad']:.2f}→{sp_a['lineup']:.2f})"
        print(f"{p['n']:>3} {p['date']:<12} {p['home']:<5} {p['away']:<5} "
              f"{p['predicted_scoreline_mode']:<8} "
              f"{p['p_home_win']:.0%}/{p['p_draw']:.0%}/{p['p_away_win']:.0%}      "
              f"{has_lineup:<8}{sp_str}")

    # Print lineup details for matches where we have them
    matches_with_lineups = [p for p in augmented_preds if p.get("lineup_info")]
    if matches_with_lineups:
        print(f"\n=== Starting XIs for matches with lineups ===")
        for p in matches_with_lineups:
            li = p["lineup_info"]
            print(f"\n#{p['n']} {p['home']} vs {p['away']} "
                  f"(formation: {li['home_formation']} vs {li['away_formation']})")
            print(f"  {p['home']} starting XI:")
            for s in li["home_starters"]:
                print(f"    #{s['jersey']:>2} {s['pos']:<5} {s['name']}")
            print(f"  {p['away']} starting XI:")
            for s in li["away_starters"]:
                print(f"    #{s['jersey']:>2} {s['pos']:<5} {s['name']}")

    # Save results
    out = {
        "iteration": 6,
        "description": "Lineup-adjusted predictions using ESPN starting XIs",
        "lineups_fetched": len(lineups),
        "lineup_debug": lineup_debug,
        "predictions_upcoming_v6": augmented_preds,
    }
    out_path = DOWNLOAD_DIR / "wc2026_engine_iter6_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote iter-6 results -> {out_path}")

    # Update canonical predictions
    (DOWNLOAD_DIR / "wc2026_predictions.json").write_text(json.dumps(augmented_preds, indent=2))
    print(f"Updated canonical predictions -> {DOWNLOAD_DIR / 'wc2026_predictions.json'}")

    # Write report
    write_iter6_report(out, augmented_preds, lineup_debug)


def write_iter6_report(out: Dict[str, Any], preds: List[Dict[str, Any]],
                       lineup_debug: Dict[str, Dict[str, Any]]):
    from datetime import datetime, timezone
    lines: List[str] = []
    lines.append("# World Cup 2026 Scoreline Prediction — Iteration 6 Report\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append("## Goal\n")
    lines.append("Integrate **ESPN starting XI lineups** into the prediction model to capture "
                 "the +8-12% accuracy improvement identified as Lever 2 in the R16 playbook.\n")
    lines.append("## What's new in iter-6\n")
    lines.append("### ESPN lineup integration")
    lines.append("- Pulls starting XIs from `site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<ID>`")
    lines.append("- ESPN publishes lineups ~1 hour before kickoff")
    lines.append("- Caches lineups to `/home/z/my-project/cache/lineup_<event_id>.json` (1-hour TTL)")
    lines.append("- Falls back to full-squad star_power when lineups aren't available\n")
    lines.append("### Star_power recalculation")
    lines.append("For each team with a lineup:")
    lines.append("1. Map ESPN starter names to `squads.json` player IDs (fuzzy match on name + jersey + position)")
    lines.append("2. Recompute `star_power` using ONLY the 11 starters (not the 26-man squad)")
    lines.append("3. Adjust ELO: `bonus = (new_star_power - 0.5) * 2 * star_weight`")
    lines.append("4. Use the adjusted ELO in the iter-4e decision-rule model\n")
    lines.append("## Lineup fetch results\n")
    lines.append(f"- Lineups fetched: **{out['lineups_fetched']}**")
    if out['lineups_fetched'] == 0:
        lines.append("- **No lineups available yet** — ESPN publishes them ~1 hour before kickoff.")
        lines.append("- Predictions below use the full-squad star_power (same as iter-4e).")
        lines.append("- Re-run this script closer to kickoff to get lineup-adjusted predictions:")
        lines.append("  ```bash")
        lines.append("  python3 /home/z/my-project/scripts/wc_predictor_iter6.py --force-refresh")
        lines.append("  ```\n")
    else:
        lines.append("\n### Star_power adjustments")
        lines.append("| Team | Old SP (squad) | New SP (lineup) | Δ | Matched | Top5 | WCGoals |")
        lines.append("|------|---------------:|----------------:|--:|--------:|-----:|--------:|")
        for code, d in lineup_debug.items():
            if d.get("fallback"):
                lines.append(f"| {code} | {d['old_star_power']:.3f} | FALLBACK | — | — | — | — |")
            else:
                lines.append(f"| {code} | {d['old_star_power']:.3f} | {d['new_star_power']:.3f} | "
                             f"{d['delta']:+.3f} | {d['matched_count']}/11 | "
                             f"{d.get('starters_top5_count', '?')} | {d.get('starters_wc_goals', '?')} |")
    lines.append("\n## Predictions for 8 R16 matches\n")
    lines.append("| # | Date | Home | Away | Pred | P(H) | P(D) | P(A) | Lineup? |")
    lines.append("|---|------|------|------|------|-----:|-----:|-----:|---------|")
    for p in preds:
        if "predicted_scoreline_mode" not in p:
            continue
        has_lineup = "✓ YES" if p.get("lineup_info") else "✗ no"
        lines.append(f"| {p['n']} | {p['date']} | {p['home']} | {p['away']} | "
                     f"{p['predicted_scoreline_mode']} | {p['p_home_win']} | {p['p_draw']} | "
                     f"{p['p_away_win']} | {has_lineup} |")
    lines.append("")
    # Show lineups where available
    matches_with_lineups = [p for p in preds if p.get("lineup_info")]
    if matches_with_lineups:
        lines.append("## Starting XIs (where lineups are available)\n")
        for p in matches_with_lineups:
            li = p["lineup_info"]
            lines.append(f"### #{p['n']} {p['home']} vs {p['away']} "
                         f"(formation: {li['home_formation']} vs {li['away_formation']})\n")
            lines.append(f"**{p['home']} starting XI:**")
            lines.append("| # | Pos | Player |")
            lines.append("|---|-----|--------|")
            for s in li["home_starters"]:
                lines.append(f"| {s['jersey']} | {s['pos']} | {s['name']} |")
            lines.append(f"\n**{p['away']} starting XI:**")
            lines.append("| # | Pos | Player |")
            lines.append("|---|-----|--------|")
            for s in li["away_starters"]:
                lines.append(f"| {s['jersey']} | {s['pos']} | {s['name']} |")
            lines.append("")
    lines.append("## Top-5 scorelines per match\n")
    for p in preds:
        if "top5_scorelines" not in p:
            continue
        lines.append(f"**#{p['n']} {p['home']} vs {p['away']}** ({p['date']}):")
        for s in p["top5_scorelines"]:
            lines.append(f"  - {s['score']} : {s['prob']:.1%}")
        lines.append("")
    lines.append("## How to re-run\n")
    lines.append("```bash")
    lines.append("# Standard: fetch lineups (uses cache if fresh) and predict")
    lines.append("python3 /home/z/my-project/scripts/wc_predictor_iter6.py")
    lines.append("")
    lines.append("# Force refresh: ignore cache, re-fetch from ESPN")
    lines.append("python3 /home/z/my-project/scripts/wc_predictor_iter6.py --force-refresh")
    lines.append("")
    lines.append("# Skip lineups: use full-squad star_power (iter-4e baseline)")
    lines.append("python3 /home/z/my-project/scripts/wc_predictor_iter6.py --no-lineups")
    lines.append("")
    lines.append("# Test on a specific event (e.g. completed COL-GHA match)")
    lines.append("python3 /home/z/my-project/scripts/wc_predictor_iter6.py --event 760501")
    lines.append("```\n")
    lines.append("## ESPN API details\n")
    lines.append("- **Scoreboard**: `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard`")
    lines.append("- **Match summary (with lineups)**: `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<ID>`")
    lines.append("- **Lineup availability**: ESPN publishes lineups ~1 hour before kickoff")
    lines.append("- **Roster format**: `rosters[].roster[]` with `starter: true/false` flag")
    lines.append("- **Player fields**: `athlete.displayName`, `athlete.id`, `jersey`, `position.abbreviation`, `formationPlace`\n")
    lines.append("## Files written\n")
    lines.append("- `/home/z/my-project/scripts/espn_lineups.py` — ESPN lineup fetcher module")
    lines.append("- `/home/z/my-project/scripts/wc_predictor_iter6.py` — iter-6 engine")
    lines.append("- `/home/z/my-project/download/wc2026_engine_iter6_results.json` — backtest + predictions")
    lines.append("- `/home/z/my-project/download/wc2026_predictions.json` — canonical predictions (updated)")
    lines.append("- `/home/z/my-project/download/wc2026_engine_iter6_report.md` — this report")
    lines.append("- `/home/z/my-project/cache/lineup_<event_id>.json` — cached lineups (1-hour TTL)\n")
    (DOWNLOAD_DIR / "wc2026_engine_iter6_report.md").write_text("\n".join(lines))
    print(f"Wrote report -> {DOWNLOAD_DIR / 'wc2026_engine_iter6_report.md'}")


if __name__ == "__main__":
    main()
