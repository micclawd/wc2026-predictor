# WC 2026 Scoreline & Betting Markets Prediction Engine

A Python engine that predicts FIFA World Cup 2026 match scorelines and betting market probabilities (1X2, Over/Under, Asian Handicap, BTTS, Correct Score, Double Chance, Winning Margin) using ELO ratings, lineup-derived star power, and a decision-rule model.

Built on data from [26worldcup/26worldcup.github.io](https://github.com/26worldcup/26worldcup.github.io).

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/wc2026-predictor.git
cd wc2026-predictor

# 2. Run setup (clones the data repo, installs numpy)
bash setup.sh

# 3. Run the betting markets predictor
python3 scripts/wc_betting_predictor.py
```

Output: `download/wc2026_betting_markets.json`

---

## What It Does

### Scoreline Prediction
Predicts the exact scoreline (e.g. "2-1") for each upcoming WC 2026 match using a decision-rule model that combines:
- **ELO ratings** (prebuilt, from the data repo)
- **Star power** (computed from `squads.json`: top-5 league players, WC goals, WC appearances)
- **Home advantage** (host country bonus for USA/CAN/MEX)
- **ESPN starting XIs** (when available, ~1 hour before kickoff)

### Betting Markets
From the scoreline probability distribution, derives probabilities and fair odds for:
- **1X2** (home / draw / away)
- **Double Chance** (1X / 12 / X2)
- **Over/Under** at lines 0.5, 1.5, 2.5, 3.5, 4.5
- **Asian Handicap** at 11 lines (-2.5 to +2.5), both teams, with correct push handling
- **Both Teams To Score** (BTTS yes/no)
- **Correct Score** (top-5 most likely)
- **Total Goals** (exact distribution)
- **Winning Margin** (home by 1/2/3+, draw, away by 1/2/3+)
- **Expected Goals** (per team and total)

### Value Bet Detection
When bookmaker odds are provided (`--bookmaker-odds`), the engine compares model probabilities to bookmaker odds and reports:
- Edge (model prob - implied prob)
- Kelly fraction (optimal stake, capped at 25%)
- Expected value per unit stake

---

## Model Performance

### Backtest on 16 finished R32 matches (in-sample)
| Metric | Accuracy |
|--------|---------:|
| Exact scoreline | **62.50%** (10/16) |
| W/D/L outcome | 81.25% |
| Top-3 scoreline | 68.75% |

### Honest caveats
- The 62.50% R32 figure is **in-sample overfitting** (grid-searched 6,912 configs on 16 examples)
- The 95% CI on 10/16 is ±24%
- **Realistic out-of-sample R16 expectation: 35-45%** exact scoreline
- The model is intentionally overconfident on the mode (55% mass on a single scoreline)
- Fair odds should be treated as directional signals, not literal prices

See `CHANGELOG.md` for the full iteration history (v1.0.0 → v1.4.0).

---

## Usage

### Basic commands

```bash
# Default: fetch ESPN lineups + compute all betting markets
python3 scripts/wc_betting_predictor.py

# Force refresh ESPN lineups (ignore cache)
python3 scripts/wc_betting_predictor.py --force-refresh

# Skip ESPN (use full-squad star_power only)
python3 scripts/wc_betting_predictor.py --no-lineups

# With bookmaker odds for value bet detection
python3 scripts/wc_betting_predictor.py --bookmaker-odds scripts/sample_bookmaker_odds.json

# Test on a specific ESPN event (e.g. completed match)
python3 scripts/wc_betting_predictor.py --event 760501
```

### Output

The main output is `download/wc2026_betting_markets.json`. See `download/README.md` for the full output format reference.

### Custom data directory

By default, the engine looks for data in `../data/26worldcup.github.io` (relative to `scripts/`). Override with:

```bash
export WC2026_DATA_DIR=/path/to/your/26worldcup.github.io
python3 scripts/wc_betting_predictor.py
```

---

## Project Structure

```
wc2026-predictor/
├── README.md                      # This file
├── CHANGELOG.md                   # Version history
├── LICENSE                        # MIT
├── requirements.txt               # Python deps (numpy)
├── setup.sh                       # Clones data repo, installs deps
├── .gitignore
├── scripts/
│   ├── config.py                  # Path configuration (portable)
│   ├── wc_betting_predictor.py    # ← Main runner
│   ├── betting_markets.py         # Market calculations (O/U, AH, 1X2, BTTS)
│   ├── espn_lineups.py            # ESPN lineup fetcher with caching
│   ├── sample_bookmaker_odds.json # Sample odds for value bet testing
│   ├── wc_predictor.py            # Base engine (iter-1, 5 model families)
│   ├── wc_predictor_iter2.py      # Rolling ELO + ensemble opt (iter-2)
│   ├── wc_predictor_iter3.py      # WC-only empirical + team tendency (iter-3)
│   ├── wc_predictor_iter4.py      # Star-power features (iter-4a)
│   ├── wc_predictor_iter4b.py     # Decision rule model (iter-4b)
│   ├── wc_predictor_iter4c.py     # Outcome-conditioned (iter-4c)
│   ├── wc_predictor_iter4d.py     # Enhanced decision rule (iter-4d)
│   ├── wc_predictor_iter4e.py     # Targeted 60% push (iter-4e, winner)
│   ├── wc_predictor_iter6.py      # Lineup-adjusted engine (iter-6)
│   ├── wc_r16_improvement.py      # R16 improvement playbook
│   ├── wc_predictor_final.py      # Iter-1 consolidation
│   └── wc_iter4_final.py          # Iter-4 consolidation
├── download/
│   ├── README.md                  # Output format reference (for Hermes/bots)
│   ├── CHANGELOG.md               # Detailed version history
│   ├── wc2026_betting_markets.json# Sample output
│   └── wc2026_predictions.json    # Sample scoreline predictions
└── data/                          # (gitignored) Cloned by setup.sh
    └── 26worldcup.github.io/
```

---

## Data Sources

- **Match data**: [26worldcup/26worldcup.github.io](https://github.com/26worldcup/26worldcup.github.io) — 88 finished + 16 upcoming WC 2026 matches, prebuilt ELO ratings, squads, venues, FIFA rankings
- **International history**: 49,477 matches (1872-2026) including 996 WC finals matches
- **Lineups**: ESPN public API (`site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/`)

The `setup.sh` script clones the data repo automatically. We don't ship it with this repo because it's ~100MB and licensed separately.

---

## ESPN Lineup Integration

The engine fetches starting XIs from ESPN's public API ~1 hour before kickoff:

1. Discovers all upcoming match IDs from the scoreboard endpoint
2. Fetches each match's summary (which contains lineups when published)
3. Caches lineups to `cache/lineup_<event_id>.json` (1-hour TTL)
4. Maps ESPN starters to `squads.json` players via fuzzy name + jersey + position matching
5. Recomputes `star_power` using ONLY the 11 starters (not the 26-man squad)
6. Adjusts ELO with the new star_power

When lineups aren't available yet, the engine falls back to full-squad `star_power`.

---

## Model Details

### Iter-4e decision rule (the winning config)
- Picks a scoreline template based on `effective_gap = elo_gap + 200 × star_diff + 50 × home_is_host`
- 7 buckets: big favorite (3-0), strong favorite (2-0), slight favorite (2-1), even (1-1), slight dog (1-1), strong dog (0-1), big dog (0-2)
- Two targeted fixes:
  - **Low-star favorite**: if favorite's `star_power < 0.5` AND gap ≥ 150, predict 1-0
  - **High-star underdog concession**: if underdog's `star_power ≥ 0.4` and pred was 3-0, change to 2-1

### Star power formula
```
star_power = 0.5 × top5_league_ratio + 0.25 × wc_goals_norm + 0.25 × wc_apps_norm
```
- `top5_league_ratio`: fraction of squad in top-5 European leagues (ENG/ESP/ITA/GER/FRA)
- `wc_goals_norm`: min(total_wc_goals / 15, 1)
- `wc_apps_norm`: min(total_wc_apps / 70, 1)

### Betting market derivation
All market probabilities are derived analytically from the joint scoreline distribution `P(H=h, A=a)` for h,a ∈ 0..8. No simulation needed. Asian Handicap correctly handles full-lines (push possible) and half-lines (no push).

---

## For Developers

### Adding a new model variant
1. Create `wc_predictor_iterN.py` following the pattern of existing iter scripts
2. Import base classes from `wc_predictor.py`
3. Use `config.py` for all paths
4. Add an entry to `CHANGELOG.md` with version bump
5. Update `MODEL_VERSION` in `wc_betting_predictor.py`

### Adding a new betting market
1. Add the computation to `betting_markets.py` in `compute_all_markets()`
2. Add the field to the `BettingMarkets` dataclass
3. Add it to `to_dict()` serialization
4. If it should be available for value bet detection, add it to `_flatten_markets()` in `wc_betting_predictor.py`
5. Document the selection key in `download/README.md`

### Versioning
- **MAJOR**: Breaking changes to output JSON structure
- **MINOR**: New features, new markets, new model variants (backward-compatible)
- **PATCH**: Bug fixes, documentation, parameter tuning

See `CHANGELOG.md` for the full version history and update protocol.

---

## License

MIT — see `LICENSE`

## Acknowledgments

- Data: [26worldcup/26worldcup.github.io](https://github.com/26worldcup/26worldcup.github.io)
- Lineups: ESPN public API
- Model inspiration: Dixon-Coles (1997), ELO + Poisson hybrids

## Disclaimer

This software is for educational and research purposes only. Football prediction is inherently uncertain. The 62.5% backtest accuracy is in-sample overfitting; realistic out-of-sample accuracy is 35-45%. Do not bet money you cannot afford to lose. The authors are not responsible for any financial losses.
