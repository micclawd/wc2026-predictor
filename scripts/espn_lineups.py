#!/usr/bin/env python3
"""
ESPN Lineup Fetcher for WC 2026
================================

Pulls starting XI lineups from ESPN's public API:
  https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<ID>

Workflow:
1. Get all upcoming WC 2026 events from the scoreboard endpoint.
2. For each event, fetch the summary endpoint.
3. Parse the `rosters` array: each entry has a `team` (with abbreviation) and
   a `roster` list of 26 players. Filter to starters (roster[i].starter == true)
   to get the 11-man starting XI.
4. Cache lineups to disk (lineups expire ~1 hour before kickoff when ESPN
   publishes them; before that, rosters[] is empty).
5. Map ESPN player names to squads.json player IDs using fuzzy matching.
6. Recompute `star_power` using ONLY the 11 starters (not the 26-man squad).

This module is designed to fail gracefully:
- If ESPN is unreachable → fall back to squads.json star_power
- If lineups aren't announced yet → fall back to squads.json star_power
- If player names don't match → use jersey number + position fallback
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-use existing modules
sys.path.insert(0, str(Path(__file__).parent))
from wc_predictor_iter4 import SquadFeatures, load_squad_features, TOP5_LEAGUES

try:
    from config import CACHE_DIR, SQUADS_JSON
except ImportError:
    CACHE_DIR = Path("/home/z/my-project/cache")
    SQUADS_JSON = Path("/home/z/my-project/26worldcup.github.io/public/data/squads.json")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event={event_id}"

# Map ESPN team abbreviations to our FIFA codes
# ESPN uses ISO 3-letter codes mostly, but some differ
ESPN_TO_FIFA = {
    "PAR": "PAR", "FRA": "FRA", "CAN": "CAN", "MAR": "MAR",
    "BRA": "BRA", "NOR": "NOR", "MEX": "MEX", "ENG": "ENG",
    "POR": "POR", "ESP": "ESP", "USA": "USA", "BEL": "BEL",
    "ARG": "ARG", "EGY": "EGY", "SUI": "SUI", "COL": "COL",
    # R32 teams (for backtest / lookup)
    "RSA": "RSA", "GER": "GER", "NED": "NED", "JPN": "JPN",
    "SWE": "SWE", "CIV": "CIV", "ECU": "ECU", "COD": "COD",
    "BIH": "BIH", "SEN": "SEN", "CRO": "CRO", "AUT": "AUT",
    "ALG": "ALG", "CPV": "CPV", "GHA": "GHA", "AUS": "AUS",
    # Common alternates
    "SUI": "SUI",  # Switzerland
    "NED": "NED",  # Netherlands
}


@dataclass
class StarterPlayer:
    """A single starting player from ESPN lineups."""
    name: str
    jersey: str
    position: str          # ESPN position abbreviation (G, CD-L, RB, CM, F, etc.)
    athlete_id: str        # ESPN athlete ID (for caching)
    formation_place: str   # 1-11 position in the formation


@dataclass
class TeamLineup:
    """Starting XI for one team."""
    team_abbrev: str       # ESPN abbreviation (will map to FIFA code)
    fifa_code: str         # Our FIFA code
    formation: str         # e.g. "4-3-3"
    starters: List[StarterPlayer] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_abbrev": self.team_abbrev,
            "fifa_code": self.fifa_code,
            "formation": self.formation,
            "starters": [
                {"name": s.name, "jersey": s.jersey, "position": s.position,
                 "athlete_id": s.athlete_id, "formation_place": s.formation_place}
                for s in self.starters
            ],
        }


@dataclass
class MatchLineups:
    """Lineups for a single match (both teams)."""
    event_id: str
    date: str
    home: TeamLineup
    away: TeamLineup
    fetched_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "date": self.date,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "fetched_at": self.fetched_at,
        }


# --------------------------------------------------------------------------
# HTTP fetch with retry and caching
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 15.0, retries: int = 2) -> Optional[Dict]:
    """GET JSON from URL with simple retry."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; wc2026-predictor/1.0)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    print(f"  [WARN] ESPN fetch failed for {url}: {last_err}")
    return None


def get_scoreboard_events(target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch WC 2026 events from ESPN scoreboard.

    Args:
        target_date: YYYYMMDD format; if None, fetches today + upcoming.
    Returns:
        List of event dicts with at least: id, date, home_abbrev, away_abbrev, status.
    """
    url = ESPN_SCOREBOARD
    if target_date:
        url += f"?dates={target_date}"
    data = _http_get(url)
    if not data:
        return []
    events = []
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        c = comps[0]
        competitors = c.get("competitors", [])
        if len(competitors) < 2:
            continue
        # ESPN puts home first
        home = competitors[0].get("team", {})
        away = competitors[1].get("team", {})
        events.append({
            "id": ev.get("id"),
            "date": ev.get("date"),
            "home_abbrev": home.get("abbreviation"),
            "away_abbrev": away.get("abbreviation"),
            "home_name": home.get("displayName"),
            "away_name": away.get("displayName"),
            "status": c.get("status", {}).get("type", {}).get("description", ""),
            "state": c.get("status", {}).get("type", {}).get("state", ""),
        })
    return events


def fetch_match_lineups(event_id: str) -> Optional[MatchLineups]:
    """Fetch lineups for a single ESPN event.

    Returns None if lineups aren't available yet (rosters empty).
    """
    url = ESPN_SUMMARY.format(event_id=event_id)
    data = _http_get(url)
    if not data:
        return None
    rosters = data.get("rosters", [])
    if len(rosters) < 2:
        return None  # lineups not announced yet

    # Build home/away TeamLineup objects
    def build_team_lineup(roster_entry: Dict) -> Optional[TeamLineup]:
        team_info = roster_entry.get("team", {})
        espn_abbrev = team_info.get("abbreviation", "")
        fifa_code = ESPN_TO_FIFA.get(espn_abbrev, espn_abbrev)
        formation = roster_entry.get("formation", "?")
        roster_list = roster_entry.get("roster", [])
        starters: List[StarterPlayer] = []
        for p in roster_list:
            if not p.get("starter"):
                continue
            athlete = p.get("athlete", {})
            starters.append(StarterPlayer(
                name=athlete.get("displayName", "?"),
                jersey=str(p.get("jersey", "?")),
                position=p.get("position", {}).get("abbreviation", "?"),
                athlete_id=athlete.get("id", ""),
                formation_place=str(p.get("formationPlace", "?")),
            ))
        if len(starters) < 7:  # not a real lineup
            return None
        return TeamLineup(
            team_abbrev=espn_abbrev,
            fifa_code=fifa_code,
            formation=formation,
            starters=starters,
        )

    home = build_team_lineup(rosters[0])
    away = build_team_lineup(rosters[1])
    if home is None or away is None:
        return None
    return MatchLineups(
        event_id=event_id,
        date=data.get("header", {}).get("competitions", [{}])[0].get("date", ""),
        home=home,
        away=away,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def cache_path(event_id: str) -> Path:
    return CACHE_DIR / f"lineup_{event_id}.json"


def load_cached_lineup(event_id: str, max_age_hours: float = 1.0) -> Optional[MatchLineups]:
    """Load a cached lineup if it exists and is fresh."""
    p = cache_path(event_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        fetched_at = datetime.fromisoformat(data["fetched_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age > max_age_hours:
            return None
        # Reconstruct
        home_data = data["home"]
        away_data = data["away"]
        home = TeamLineup(
            team_abbrev=home_data["team_abbrev"],
            fifa_code=home_data["fifa_code"],
            formation=home_data["formation"],
            starters=[StarterPlayer(**s) for s in home_data["starters"]],
        )
        away = TeamLineup(
            team_abbrev=away_data["team_abbrev"],
            fifa_code=away_data["fifa_code"],
            formation=away_data["formation"],
            starters=[StarterPlayer(**s) for s in away_data["starters"]],
        )
        return MatchLineups(
            event_id=data["event_id"],
            date=data["date"],
            home=home, away=away,
            fetched_at=data["fetched_at"],
        )
    except Exception as e:
        print(f"  [WARN] cache load failed for {event_id}: {e}")
        return None


def save_cached_lineup(lineup: MatchLineups) -> None:
    p = cache_path(lineup.event_id)
    p.write_text(json.dumps(lineup.to_dict(), indent=2))


# --------------------------------------------------------------------------
# Fetch with cache
# --------------------------------------------------------------------------
def fetch_lineups_with_cache(event_id: str, force_refresh: bool = False) -> Optional[MatchLineups]:
    """Fetch lineups, using cache when fresh (<1h old)."""
    if not force_refresh:
        cached = load_cached_lineup(event_id, max_age_hours=1.0)
        if cached:
            print(f"  [CACHE] Using cached lineups for event {event_id} (age: "
                  f"{(datetime.now(timezone.utc) - datetime.fromisoformat(cached.fetched_at.replace('Z','+00:00'))).total_seconds()/60:.0f} min)")
            return cached
    print(f"  [FETCH] Pulling lineups for event {event_id} from ESPN ...")
    lineup = fetch_match_lineups(event_id)
    if lineup:
        save_cached_lineup(lineup)
        print(f"    ✓ Got lineups: {lineup.home.fifa_code} ({len(lineup.home.starters)} starters, "
              f"formation {lineup.home.formation}) vs {lineup.away.fifa_code} "
              f"({len(lineup.away.starters)} starters, formation {lineup.away.formation})")
    else:
        print(f"    ✗ Lineups not yet available (match may be >1h away or ESPN didn't publish)")
    return lineup


# --------------------------------------------------------------------------
# Map ESPN starters to squads.json player IDs and recompute star_power
# --------------------------------------------------------------------------
def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    # lowercase, strip accents, remove punctuation
    n = name.lower()
    # remove accent marks
    n = n.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    n = n.replace("ñ", "n").replace("ü", "u").replace("ç", "c").replace("ã", "a").replace("ö", "o")
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def map_starters_to_squad(starters: List[StarterPlayer],
                          squad_players: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Match each ESPN starter to a squads.json player.

    Returns:
        (matched_starters, unmatched_names)
    """
    # Build lookup tables from squads.json
    by_normalized_name: Dict[str, Dict] = {}
    by_jersey: Dict[str, Dict] = {}
    for p in squad_players:
        norm = _normalize_name(p.get("name", ""))
        if norm:
            by_normalized_name[norm] = p
        # also map by last name
        last = _normalize_name(p.get("name", "").split()[-1]) if p.get("name") else ""
        if last and last not in by_normalized_name:
            by_normalized_name[last] = p
        jersey = str(p.get("no", ""))
        if jersey:
            by_jersey[jersey] = p

    matched: List[Dict] = []
    unmatched: List[str] = []
    for s in starters:
        # Try exact normalized name first
        norm = _normalize_name(s.name)
        if norm in by_normalized_name:
            matched.append(by_normalized_name[norm])
            continue
        # Try last name
        last = _normalize_name(s.name.split()[-1]) if s.name else ""
        if last and last in by_normalized_name:
            matched.append(by_normalized_name[last])
            continue
        # Try jersey + position
        if s.jersey in by_jersey:
            candidate = by_jersey[s.jersey]
            # Sanity check: position should match (GK->GK, etc.)
            espn_pos = s.position.upper()
            squad_pos = candidate.get("pos", "").upper()
            pos_match = (
                (espn_pos == "G" and squad_pos == "GK") or
                (espn_pos in {"CD-L", "CD-R", "LB", "RB", "DF"} and squad_pos == "DF") or
                (espn_pos in {"CM", "LM", "RM", "MF"} and squad_pos == "MF") or
                (espn_pos in {"F", "LF", "RF", "FW"} and squad_pos == "FW")
            )
            if pos_match:
                matched.append(candidate)
                continue
        # No match
        unmatched.append(s.name)
    return matched, unmatched


def compute_lineup_star_power(starters: List[StarterPlayer],
                              squad_features: SquadFeatures,
                              squad_players: List[Dict]) -> Tuple[float, Dict[str, Any]]:
    """Recompute star_power using ONLY the 11 starters.

    Returns:
        (new_star_power, debug_info)
    """
    matched, unmatched = map_starters_to_squad(starters, squad_players)
    if len(matched) < 7:
        # Too few matches; fall back to full squad star_power
        return squad_features.star_power, {
            "fallback": True,
            "matched_count": len(matched),
            "unmatched": unmatched,
            "reason": "Too few starters matched to squad; using full-squad star_power",
        }
    # Recompute star_power metrics on the 11 starters only
    top5_count = sum(1 for p in matched if p.get("clubNat") in TOP5_LEAGUES)
    wc_goals = sum(p.get("wcGoals", 0) for p in matched)
    wc_apps = sum(p.get("wcApps", 0) for p in matched)
    squad_size = len(matched)
    top5_ratio = top5_count / squad_size if squad_size > 0 else 0
    wc_goals_norm = min(wc_goals / 10.0, 1.0)   # 10+ WC goals from starters = max
    wc_apps_norm = min(wc_apps / 50.0, 1.0)      # 50+ WC apps from starters = max
    new_star_power = 0.5 * top5_ratio + 0.25 * wc_goals_norm + 0.25 * wc_apps_norm
    return new_star_power, {
        "fallback": False,
        "matched_count": len(matched),
        "unmatched": unmatched,
        "starters_top5_count": top5_count,
        "starters_wc_goals": wc_goals,
        "starters_wc_apps": wc_apps,
        "new_star_power": new_star_power,
        "old_star_power": squad_features.star_power,
        "delta": new_star_power - squad_features.star_power,
    }


# --------------------------------------------------------------------------
# Top-level: fetch lineups for all upcoming R16 matches and adjust star_power
# --------------------------------------------------------------------------
def fetch_all_upcoming_lineups(target_dates: List[str] = None) -> Dict[str, MatchLineups]:
    """Fetch lineups for all upcoming WC matches.

    Args:
        target_dates: list of YYYYMMDD strings to fetch. If None, auto-detect
                      upcoming R16 dates (July 4-7, 2026).
    Returns:
        Dict mapping FIFA_code_pair (e.g. "PAR-FRA") to MatchLineups.
    """
    if target_dates is None:
        # R16 dates: July 4-7, 2026
        target_dates = ["20260704", "20260705", "20260706", "20260707"]

    print(f"\n=== Fetching ESPN lineups for dates: {target_dates} ===")
    all_events = []
    for d in target_dates:
        events = get_scoreboard_events(d)
        all_events.extend(events)
        print(f"  Date {d}: {len(events)} events")

    if not all_events:
        print("  No events found. ESPN may be unavailable.")
        return {}

    print(f"\n  Total events to check: {len(all_events)}")
    lineups_by_match: Dict[str, MatchLineups] = {}
    for ev in all_events:
        event_id = ev["id"]
        home_ab = ev["home_abbrev"]
        away_ab = ev["away_abbrev"]
        if not home_ab or not away_ab:
            continue
        home_fifa = ESPN_TO_FIFA.get(home_ab, home_ab)
        away_fifa = ESPN_TO_FIFA.get(away_ab, away_ab)
        match_key = f"{home_fifa}-{away_fifa}"
        print(f"\n  Event {event_id}: {home_fifa} vs {away_fifa} ({ev['status']})")
        lineup = fetch_lineups_with_cache(event_id)
        if lineup:
            lineups_by_match[match_key] = lineup
        else:
            print(f"    (no lineups yet — match may be >1h away)")

    return lineups_by_match


# --------------------------------------------------------------------------
# CLI for testing
# --------------------------------------------------------------------------
def main():
    """CLI entry: fetch lineups for all upcoming R16 matches and print summary."""
    print("="*70)
    print("ESPN Lineup Fetcher — WC 2026 R16")
    print("="*70)

    lineups = fetch_all_upcoming_lineups()

    if not lineups:
        print("\n✗ No lineups available yet. ESPN typically publishes lineups ~1 hour")
        print("  before kickoff. Try again closer to match time.")
        print("\nFor testing, here are the lineups from the last completed R32 match (COL vs GHA):")
        # Test on a completed match
        test_lineup = fetch_lineups_with_cache("760501", force_refresh=True)
        if test_lineup:
            print(f"\n=== Sample lineup (COL vs GHA, event 760501) ===")
            print(f"Formation: {test_lineup.home.formation} (COL) vs {test_lineup.away.formation} (GHA)")
            print(f"\nCOL starting XI:")
            for s in test_lineup.home.starters:
                print(f"  #{s.jersey:>2} {s.position:<5} {s.name}")
            print(f"\nGHA starting XI:")
            for s in test_lineup.away.starters:
                print(f"  #{s.jersey:>2} {s.position:<5} {s.name}")
        return

    print(f"\n=== Got lineups for {len(lineups)} matches ===")
    for key, lineup in lineups.items():
        print(f"\n{key} (event {lineup.event_id}):")
        print(f"  {lineup.home.fifa_code} formation: {lineup.home.formation}")
        print(f"  {lineup.away.fifa_code} formation: {lineup.away.formation}")
        print(f"  {lineup.home.fifa_code} starters:")
        for s in lineup.home.starters:
            print(f"    #{s.jersey:>2} {s.position:<5} {s.name}")
        print(f"  {lineup.away.fifa_code} starters:")
        for s in lineup.away.starters:
            print(f"    #{s.jersey:>2} {s.position:<5} {s.name}")

    # Test the star_power recalculation on the fetched lineups
    print("\n=== Testing star_power recalculation ===")
    squads = load_squad_features()
    squads_raw = json.loads(SQUADS_JSON.read_text())
    for key, lineup in lineups.items():
        print(f"\n{key}:")
        for team_lineup in [lineup.home, lineup.away]:
            fifa_code = team_lineup.fifa_code
            sq_features = squads.get(fifa_code)
            sq_players = squads_raw.get(fifa_code, {}).get("players", [])
            if not sq_features or not sq_players:
                print(f"  {fifa_code}: squad data not found")
                continue
            new_sp, debug = compute_lineup_star_power(
                team_lineup.starters, sq_features, sq_players
            )
            print(f"  {fifa_code}: old star_power={debug['old_star_power']:.3f} → "
                  f"new={debug['new_star_power']:.3f} (Δ={debug['delta']:+.3f})")
            print(f"    Matched {debug['matched_count']}/11 starters | "
                  f"top5={debug.get('starters_top5_count', '?')} | "
                  f"wc_goals={debug.get('starters_wc_goals', '?')} | "
                  f"wc_apps={debug.get('starters_wc_apps', '?')}")
            if debug.get("unmatched"):
                print(f"    Unmatched: {debug['unmatched']}")


if __name__ == "__main__":
    main()
