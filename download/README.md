# WC 2026 Betting Markets Predictor

**Version:** v1.4.0
**Last updated:** 2026-07-04

Predicts betting market probabilities (1X2, O/U, Asian Handicap, BTTS, Correct Score, Double Chance, Winning Margin) for World Cup 2026 matches using the iter-4e decision-rule model with optional ESPN lineup integration.

---

## Quick Start

```bash
# Default run: fetch ESPN lineups (if available) + predict all betting markets
python3 /home/z/my-project/scripts/wc_betting_predictor.py

# Force refresh ESPN lineups (ignore cache)
python3 /home/z/my-project/scripts/wc_betting_predictor.py --force-refresh

# Skip ESPN (use full-squad star_power only)
python3 /home/z/my-project/scripts/wc_betting_predictor.py --no-lineups

# With bookmaker odds for value bet detection
python3 /home/z/my-project/scripts/wc_betting_predictor.py --bookmaker-odds /path/to/odds.json

# Test on a specific ESPN event (e.g. completed match for verification)
python3 /home/z/my-project/scripts/wc_betting_predictor.py --event 760501
```

**Output:** `/home/z/my-project/download/wc2026_betting_markets.json`

---

## Output Format

The output JSON has this top-level structure:

```json
{
  "generated_at": "2026-07-04T12:00:00+00:00",
  "model_version": "v1.4.0",
  "engine_config": {
    "base_model": "iter-4e targeted decision rule",
    "star_weight": 40.0,
    "host_bonus": 80.0,
    "form_weight": 0.5,
    "lineups_used": false,
    "lineup_match_count": 0
  },
  "matches": [...],
  "value_bets": {...}   // only present if --bookmaker-odds was provided
}
```

### Per-match structure

Each entry in `matches[]` has:

```json
{
  "n": 89,
  "stage": "r16",
  "date": "2026-07-04",
  "kickoff_utc": "2026-07-04T21:00:00+00:00",
  "home": "PAR",
  "away": "FRA",
  "venue_country": "US",
  "lineup_available": false,
  "lineup_info": null,
  "home_star_power": null,
  "away_star_power": null,
  "markets": {
    "1X2": {...},
    "double_chance": {...},
    "over_under": {...},
    "asian_handicap": {...},
    "btts": {...},
    "correct_score_top5": [...],
    "total_goals": {...},
    "winning_margin": {...},
    "expected_goals": {...}
  }
}
```

### Markets reference

#### 1X2
```json
"1X2": {
  "home_win": 0.02,
  "home_win_fair_odds": 50.0,
  "draw": 0.05,
  "draw_fair_odds": 20.0,
  "away_win": 0.93,
  "away_win_fair_odds": 1.08
}
```

#### Double Chance
```json
"double_chance": {
  "1X": 0.07,        // home or draw
  "12": 0.95,        // home or away (no draw)
  "X2": 0.98         // draw or away
}
```

#### Over/Under (lines: 0.5, 1.5, 2.5, 3.5, 4.5)
```json
"over_under": {
  "2.5": {
    "over": 0.29,
    "over_fair_odds": 3.45,
    "under": 0.71,
    "under_fair_odds": 1.41
  }
}
```

#### Asian Handicap (lines: -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
Lines are from the **home team's perspective**. Positive line = home is the underdog.

```json
"asian_handicap": {
  "-0.5": {
    "home_win": 0.02,        // P(home wins by 1+)
    "push": 0.0,             // 0 for half-lines; nonzero for full lines (-1, 0, +1, etc.)
    "home_loss": 0.98,       // P(home draws or loses)
    "away_win": 0.98,        // mirror of home_loss
    "away_loss": 0.02        // mirror of home_win
  },
  "-1.0": {
    "home_win": 0.02,        // P(home wins by 2+)
    "push": 0.05,            // P(home wins by exactly 1) — stake returned
    "home_loss": 0.93,       // P(home draws or loses)
    "away_win": 0.93,
    "away_loss": 0.02
  }
}
```

**Push handling:**
- Half-lines (e.g., -0.5, -1.5): no push possible
- Full-lines (e.g., -1.0, 0.0, +1.0): push possible when margin equals line
- For value bets, push probability is excluded from both win and loss

#### Both Teams To Score
```json
"btts": {
  "yes": 0.19,
  "no": 0.81
}
```

#### Correct Score (top-5 most likely)
```json
"correct_score_top5": [
  {"score": "0-2", "prob": 0.58, "prob_fair_odds": 1.72},
  {"score": "0-1", "prob": 0.10, "prob_fair_odds": 10.0},
  ...
]
```

#### Total Goals (exact)
```json
"total_goals": {
  "0": 0.01,
  "1": 0.11,
  "2": 0.59,
  "3": 0.18,
  ...
}
```

#### Winning Margin
```json
"winning_margin": {
  "home_by_1": 0.01,
  "home_by_2": 0.00,
  "home_by_3plus": 0.00,
  "draw": 0.03,
  "away_by_1": 0.19,
  "away_by_2": 0.66,
  "away_by_3plus": 0.12
}
```

#### Expected Goals
```json
"expected_goals": {
  "home": 0.20,
  "away": 2.08,
  "total": 2.28
}
```

---

## Value Bet Detection

If you pass `--bookmaker-odds <file.json>`, the engine compares model probabilities to bookmaker odds and reports value bets. The odds file format:

```json
{
  "PAR-FRA": {
    "home_win": 12.0,
    "draw": 7.5,
    "away_win": 1.20,
    "ou_2.5_over": 2.10,
    "ou_2.5_under": 1.75,
    "ah_-0.5_home_win": 12.0,
    "ah_-0.5_away_win": 1.20,
    "ah_1.5_home_win": 1.85,
    "ah_1.5_away_win": 2.05,
    "btts_yes": 2.30,
    "btts_no": 1.65,
    "cs_0-2": 8.0,
    "cs_0-1": 9.0
  }
}
```

### Selection key naming convention

| Market | Selection key |
|--------|---------------|
| 1X2 | `home_win`, `draw`, `away_win` |
| Double Chance | `dc_1X`, `dc_12`, `dc_X2` |
| Over/Under | `ou_<line>_over`, `ou_<line>_under` (e.g., `ou_2.5_over`) |
| Asian Handicap | `ah_<line>_home_win`, `ah_<line>_away_win` (e.g., `ah_-0.5_home_win`) |
| BTTS | `btts_yes`, `btts_no` |
| Correct Score | `cs_<h>-<a>` (e.g., `cs_2-1`) |

### Value bet output

```json
"value_bets": {
  "PAR-FRA": [
    {
      "selection": "cs_0-2",
      "model_prob": 0.58,
      "bookmaker_odds": 8.0,
      "implied_prob": 0.125,
      "edge": 0.454,
      "kelly_fraction": 0.25,
      "expected_value": 3.63
    }
  ]
}
```

- `edge`: model_prob - implied_prob (positive = value)
- `kelly_fraction`: optimal stake as fraction of bankroll (capped at 25%)
- `expected_value`: profit per unit stake (e.g., 3.63 means $3.63 profit per $1 staked)

---

## ESPN Lineup Integration

The engine pulls starting XIs from ESPN's public API ~1 hour before kickoff. When lineups are available:

1. Each ESPN starter is matched to a `squads.json` player (by name + jersey + position)
2. `star_power` is recomputed using ONLY the 11 starters (not the 26-man squad)
3. ELO is adjusted: `bonus = (new_star_power - 0.5) × 2 × 40`
4. The adjusted ELO feeds into the iter-4e decision rule

**Cache:** Lineups are cached to `/home/z/my-project/cache/lineup_<event_id>.json` with a 1-hour TTL.

**Fallback:** When lineups aren't available (more than 1 hour before kickoff), the engine uses full-squad `star_power` and `lineup_available` is set to `false` in the output.

**ESPN API endpoints used:**
- Scoreboard: `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard`
- Match summary: `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<ID>`

---

## Files

### Required scripts (in `/home/z/my-project/scripts/`)

| File | Purpose |
|------|---------|
| `wc_betting_predictor.py` | **Main runner** — produces `wc2026_betting_markets.json` |
| `betting_markets.py` | Betting market calculations (O/U, AH, 1X2, BTTS, etc.) |
| `espn_lineups.py` | ESPN lineup fetcher with caching |
| `wc_predictor.py` | Base prediction engine (iter-1) |
| `wc_predictor_iter4.py` | Star-power features (iter-4a) |
| `wc_predictor_iter4b.py` | Decision rule model (iter-4b) |
| `wc_predictor_iter4e.py` | Targeted 60% push (iter-4e, **the winning config**) |
| `wc_predictor_iter6.py` | Lineup-adjusted engine (iter-6) |
| `sample_bookmaker_odds.json` | Sample odds file for value bet testing |

### Output (in `/home/z/my-project/download/`)

| File | Purpose |
|------|---------|
| `wc2026_betting_markets.json` | **Main output** — all betting markets for upcoming matches |
| `wc2026_predictions.json` | Scoreline predictions (mode + top-5) |
| `CHANGELOG.md` | Version history |
| `README.md` | This file |

### Cache (in `/home/z/my-project/cache/`)

| File | Purpose |
|------|---------|
| `lineup_<event_id>.json` | Cached ESPN lineup (1-hour TTL) |

---

## Model Details

### Base model: iter-4e decision rule
- Picks a scoreline template based on `effective_gap = elo_gap + 200 × star_diff + 50 × home_is_host`
- 7 buckets: big favorite (3-0), strong favorite (2-0), slight favorite (2-1), even (1-1), slight dog (1-1), strong dog (0-1), big dog (0-2)
- Two targeted fixes:
  - **Low-star favorite**: if favorite's `star_power < 0.5` AND gap ≥ 150, predict 1-0 instead of 3-0/2-0
  - **High-star underdog concession**: if underdog's `star_power ≥ 0.4` and pred was 3-0, change to 2-1
- R32 backtest accuracy: **62.50%** (10/16) exact scoreline, 81.25% W/D/L outcome

### Star power feature
```
star_power = 0.5 × top5_league_ratio + 0.25 × wc_goals_norm + 0.25 × wc_apps_norm
```
- `top5_league_ratio`: fraction of squad in top-5 European leagues (ENG/ESP/ITA/GER/FRA)
- `wc_goals_norm`: min(total_wc_goals / 15, 1)
- `wc_apps_norm`: min(total_wc_apps / 70, 1)

### Important caveats

1. **The 62.5% R32 accuracy is in-sample overfitting.** Realistic out-of-sample R16 expectation is 35-45%.
2. **The model is overconfident.** The decision rule puts 55% mass on a single scoreline, which makes fair odds look extreme (e.g., 1.72 for a 0-2 correct score). Real bookmakers would offer 8-12× on this. Treat the fair odds as a directional signal, not a literal price.
3. **Value bet edges will appear inflated** for the same reason. Filter for edges > 20% AND Kelly > 5% to find genuine signals.
4. **Lineup data is only available ~1 hour before kickoff.** Re-run with `--force-refresh` at that time.
5. **Sample size is tiny.** 95% CI on 8 R16 matches is ±17%.

---

## Re-running After Each Match

After each R16 match finishes:

1. The upstream repo (`26worldcup.github.io`) updates `matches.json` daily with results
2. Re-run: `python3 /home/z/my-project/scripts/wc_betting_predictor.py`
3. The new result feeds into the rolling ELO and improves subsequent predictions
4. For QF predictions, the engine will have 17-24 backtest matches

---

## Troubleshooting

### "No lineups available"
- ESPN publishes lineups ~1 hour before kickoff
- Re-run with `--force-refresh` closer to match time
- Or use `--no-lineups` to skip ESPN entirely

### "ESPN fetch failed"
- Check internet connection
- ESPN may be temporarily unavailable; the engine will use cached lineups if available
- Cache TTL is 1 hour

### "Model probabilities look extreme"
- This is expected. The iter-4e decision rule is intentionally overconfident on the mode
- For calibration, look at the `correct_score_top5` rather than just the mode
- The top-3 scorelines collectively cover ~70% of the probability mass

### "Value bets show 40%+ edges"
- These are NOT real 40% edges. The model is overconfident.
- Filter for: edge > 20% AND Kelly > 5% AND selection is in a liquid market (1X2, O/U 2.5, AH -0.5/+0.5)
- Correct score value bets should be treated as directional signals only

---

## Version History

See `CHANGELOG.md` for the full version history.
