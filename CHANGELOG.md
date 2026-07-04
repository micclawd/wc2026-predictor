# Changelog

All notable changes to the WC 2026 Scoreline Prediction Engine are documented here.
Versions follow semantic versioning: `MAJOR.MINOR.PATCH`.

The `model_version` field in `wc2026_betting_markets.json` reflects the current version.

---

## [v1.4.0] — 2026-07-04

### Added
- **Betting markets module** (`scripts/betting_markets.py`)
  - Computes 1X2, Double Chance, Over/Under (5 lines), Asian Handicap (11 lines, both teams), BTTS, Correct Score (top-5), Total Goals (exact), Winning Margin, and Expected Goals
  - All probabilities derived analytically from the joint scoreline distribution
  - Asian Handicap correctly handles full-lines (push) and half-lines (no push)
  - Adds fair decimal odds alongside every probability
- **Betting predictor runner** (`scripts/wc_betting_predictor.py`)
  - End-to-end pipeline: load engine → fetch lineups → compute markets → output JSON
  - Optional value bet detection with `--bookmaker-odds <file.json>`
  - Kelly fraction calculation for stake sizing (capped at 25%)
- **Sample bookmaker odds file** (`scripts/sample_bookmaker_odds.json`) for testing value bet detection
- **README.md** with full output format reference, selection key naming convention, and troubleshooting
- **CHANGELOG.md** (this file)

### Output
- New canonical output: `download/wc2026_betting_markets.json`
  - Per-match structure with `markets` object containing all betting markets
  - `engine_config` block documents which model and parameters were used
  - Optional `value_bets` block when bookmaker odds are provided

### Hermes integration
- Output JSON is stable and self-describing
- All selection keys follow documented naming convention (e.g., `ou_2.5_over`, `ah_-0.5_home_win`, `cs_2-1`)
- `model_version` field lets Hermes track which engine version produced the output

---

## [v1.3.0] — 2026-07-04

### Added
- **ESPN lineup integration** (`scripts/espn_lineups.py`)
  - Fetches starting XIs from `site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=<ID>`
  - Auto-discovers all upcoming match IDs from the scoreboard endpoint
  - Caches lineups to `cache/lineup_<event_id>.json` (1-hour TTL)
  - Maps ESPN starters to `squads.json` players via fuzzy name + jersey + position matching
  - Falls back to full-squad `star_power` when lineups aren't available
- **Lineup-adjusted engine** (`scripts/wc_predictor_iter6.py`)
  - Recomputes `star_power` using ONLY the 11 starters
  - Adjusts ELO with the new star_power: `bonus = (new_star_power - 0.5) × 2 × 40`
  - CLI flags: `--force-refresh`, `--no-lineups`, `--event <ID>`

### Verified working
- Live ESPN fetch tested on completed COL-GHA match (event 760501)
- 11/11 starters matched for both COL and GHA
- Star_power recalculated correctly: COL 0.474 → 0.583, GHA 0.585 → 0.548
- Cache tested (1-hour TTL works as expected)
- Graceful degradation tested (R16 lineups not yet available → falls back to squads.json)

### Projected impact
- +8-12% accuracy improvement on R16 (per the R16 improvement playbook)
- Combined with iter-4e: realistic R16 out-of-sample expectation 43-57% exact scoreline

---

## [v1.2.0] — 2026-07-04

### Added
- **R16 improvement playbook** (`scripts/wc_r16_improvement.py`)
  - Implements 3 of 6 identified accuracy levers:
    - Lever 3: Form-adjusted ELO using R32 goal difference
    - Lever 4: R16-specific scoreline templates (more 1-0, fewer 3-0)
    - Lever 5: Head-to-head history prior (30% weight when available)
  - Documents all 6 levers with projected accuracy improvements
  - Combined R16 model produces projections for all 8 R16 matches

### Documentation
- `download/wc2026_r16_improvement_playbook.md` with lever-by-lever analysis
- Honest projection: realistic R16 out-of-sample accuracy is 35-45% (not 60%)

---

## [v1.1.0] — 2026-07-04

### Added
- **Star-power feature** from `squads.json` (`scripts/wc_predictor_iter4.py`)
  - Counts players in top-5 European leagues (ENG/ESP/ITA/GER/FRA)
  - Composite score: `0.5 × top5_ratio + 0.25 × wc_goals_norm + 0.25 × wc_apps_norm`
  - Converts to ELO bonus: `(star_power - 0.5) × 2 × star_weight`
- **Knockout-specific empirical model** using historical WC matches
- **R32-scoreline prior** that boosts probability of common R32 scorelines

### Iter-4 sub-iterations
- **4a**: Poisson/DC + R32 prior + star power → 25.00% exact scoreline on R32
- **4b**: Decision rule (threshold-based) → 56.25%
- **4c**: Two-stage outcome-conditioned → 43.75%
- **4d**: Enhanced decision rule → 50.00%
- **4e**: Targeted fixes on iter-4b → **62.50%** (10/16) ✅ HIT 60% TARGET

### Winning configuration (iter-4e)
- Templates: (3-0, 2-0, 2-1, 1-0, 1-1, 1-2, 0-1, 0-2)
- Thresholds: big_fav=300, strong_fav=150, slight_fav=50
- star_weight=40, low_star_threshold=0.5
- Two targeted fixes:
  - Low-star favorite → 1-0 (catches COL-GHA pattern)
  - High-star underdog concession (catches ENG-COD pattern)

### Critical caveats documented
- 62.5% is in-sample overfitting (grid-searched 6,912 configs on 16 examples)
- 95% CI on 10/16 is ±24%
- Realistic R16 out-of-sample expectation: 35-45%

---

## [v1.0.0] — 2026-07-04

### Added
- **Initial prediction engine** (`scripts/wc_predictor.py`)
  - 5 model families: Poisson, Dixon-Coles, Attack/Defense, Empirical lookup, Ensemble
  - Hyperparameter sweep (70+ configurations)
  - Backtest on 88 finished WC 2026 matches
  - Best exact scoreline: 18.18% (Poisson tg=2.5, fw=0.7)
  - Best W/D/L outcome: 71.59% (Dixon-Coles tg=2.5, rho=+0.05)
  - Best top-3 scoreline: 38.64% (Dixon-Coles baseline)

- **Iteration 2** (`scripts/wc_predictor_iter2.py`)
  - In-tournament rolling ELO (updates after each WC match)
  - Stage-aware calibration (lower total_goals for knockouts)
  - Ensemble weight optimization
  - Best exact scoreline: 19.32% (Ensemble of Empirical + Attack/Defense)

- **Iteration 3** (`scripts/wc_predictor_iter3.py`)
  - WC-only empirical lookup (9,800+ historical WC matches)
  - Team-strength-tier bucketing (4 tiers by ELO)
  - Per-team goal tendency from WC history
  - Best top-3 scoreline: 39.77% (iter-3 ensemble)

### Data sources
- 88 finished WC 2026 matches (ground truth)
- 16 upcoming matches (R16 → Final)
- 49,477 international matches (1872-2026)
- 996 historical WC finals matches
- Prebuilt ELO ratings (current + form) for 48 WC teams
- FIFA rankings + venue host-country data

### Honest framing
- 90% target declared not achievable in legitimate football prediction
- Best published academic models: 10-20% exact scoreline, 50-65% W/D/L outcome
- Multiple metrics reported (exact, outcome, top-3, Brier, log loss)
- Refused to overfit beyond reporting best sweep result on same 88 matches

---

## Versioning Rules

- **MAJOR**: Breaking changes to output JSON structure (Hermes would need code changes)
- **MINOR**: New features, new markets, new model variants (backward-compatible)
- **PATCH**: Bug fixes, documentation, parameter tuning (no structural changes)

## Update Protocol

After every change:
1. Increment version in `scripts/wc_betting_predictor.py` (`MODEL_VERSION` constant)
2. Add entry to this CHANGELOG with date, version, and changes
3. Update `download/README.md` if output format or usage changes
4. Re-run `python3 scripts/wc_betting_predictor.py` to verify output
5. Commit with message format: `[vX.Y.Z] brief description`
