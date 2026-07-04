# Stake.com Integration

This document explains how to integrate the WC 2026 predictor with Stake.com for autonomous bet placement.

## ⚠️ Critical: Kasada Bot Protection

Stake.com uses **Kasada** for bot protection. You cannot simply pass a session token — every request requires 4 dynamically-generated headers (`x-kpsdk-cd`, `x-kpsdk-ct`, `x-kpsdk-h`, `x-kpsdk-v`) that prove the request came from a real browser. These headers expire after ~10-30 minutes.

**Three options for handling Kasada:**

| Option | Autonomy | Cost | Setup |
|--------|----------|------|-------|
| Manual session harvest | Low (refresh every 15 min) | Free | Browser devtools |
| Browser automation (Playwright) | High | Free (CPU cost) | `pip install playwright && playwright install chromium` |
| Third-party solver (ZenRows, Capsolver) | High | $30-100/month | API key |

The default implementation (`ManualSessionProvider`) uses option 1. For true autonomy, implement `BrowserSessionProvider` (already stubbed in `stake_api.py`) or wire in a solver service.

---

## Setup

### 1. Create your session file

```bash
cd scripts
cp stake_session.example.json stake_session.json
```

### 2. Harvest your Stake session

1. Open https://stake.com in your browser and log in
2. Open devtools (F12) → Network tab
3. Navigate to any sports page (e.g. https://stake.com/sports)
4. Find any request to `/_api/graphql` in the Network tab
5. Right-click → Copy → Copy as cURL
6. Extract these values:
   - `session` cookie → `session_token`
   - `x-kpsdk-cd`, `x-kpsdk-ct`, `x-kpsdk-h`, `x-kpsdk-v` headers → `kpsdk_headers`
   - `cf_clearance` cookie → `cf_clearance`
7. Paste them into `stake_session.json`

### 3. Find fixture slugs

For each match you want to bet on, find its Stake.com URL:
1. Browse to https://stake.com/sports/soccer/fifa_world
2. Click on each match
3. Copy the slug from the URL (the part after the last `/`)

Example: `https://stake.com/sports/soccer/fifa_world/canada-vs-morocco/46602331-w73-w75` → slug is `46602331-w73-w75`

Update `fixtures.json`:
```json
{
  "CAN-MAR": "46602331-w73-w75",
  "PAR-FRA": "..."
}
```

### 4. Test the integration

```bash
# Dry run — validate markets and show what would be placed
python3 scripts/stake_bet_placer.py --dry-run

# Live run — actually place bets
python3 scripts/stake_bet_placer.py

# Multibet mode (combine across matches)
python3 scripts/stake_bet_placer.py --mode multibet
```

---

## How It Works

### Workflow

```
wc_betting_predictor.py
  ↓ produces
download/wc2026_betting_markets.json
  ↓ read by
stake_bet_placer.py
  ↓ for each match
  ├── 1. Look up fixture slug in fixtures.json
  ├── 2. Fetch Stake markets via FixtureIndex query
  ├── 3. Find outcome IDs for value bet selections
  ├── 4. Check Stake odds vs model probability
  ├── 5. Filter by: edge ≥ 5%, kelly ≥ 3%, odds ≥ 1.50
  ├── 6. Calculate stake (quarter-Kelly, capped at 5% per match)
  ├── 7. Place bet via BetSlipFooter_CustomSportBet mutation
  └── 8. Log to bets_log.jsonl
```

### Bet Sizing Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Min edge | 5% | Filters model noise |
| Min Kelly | 3% | Avoids tiny bets |
| Min odds | 1.50 | No sub-1.50 favorites (juice eats edge) |
| Max selections per match | 3 | Avoids correlated parlay risk |
| Max exposure per match | 5% of bankroll | Risk management |
| Max total exposure | 25% of bankroll | Survival floor |
| Kelly fraction | 0.25 (quarter-Kelly) | Conservative; full-Kelly is too volatile |

### Selection Key Mapping

The model produces selection keys like `ou_2.5_over`, `ah_-0.5_home_win`. These map to Stake market names:

| Model key | Stake market | Stake selection |
|-----------|--------------|-----------------|
| `home_win` | 1X2 | Home |
| `draw` | 1X2 | Draw |
| `away_win` | 1X2 | Away |
| `dc_1X` | Double Chance | Home/Draw |
| `ou_2.5_over` | Total Goals | Over (specifier: 2.5) |
| `ah_-0.5_home_win` | Asian Handicap | Home (specifier: -0.5) |
| `btts_yes` | Both Teams to Score | Yes |

Full mapping in `stake_bet_placer.py` → `SELECTION_TO_STAKE`.

### Singles vs Multibets

- **Singles** (default): each value bet is placed independently. Safer — one loss doesn't affect others.
- **Multibet**: combines value bets across DIFFERENT matches into one bet. Higher combined odds, but all selections must win. **Never combines selections from the same match** (correlated — if you bet "MAR to win" and "BTTS no" on CAN-MAR, they're not independent).

Use multibet mode only when you have 2+ matches with value bets (e.g. QF round with 4 matches).

---

## Files

| File | Purpose |
|------|---------|
| `scripts/stake_api.py` | GraphQL client with pluggable Kasada provider |
| `scripts/stake_bet_placer.py` | High-level bet placement workflow |
| `scripts/stake_session.example.json` | Template for session config (copy to `stake_session.json`) |
| `fixtures.json` | Match → fixture slug mapping (update before each round) |
| `bets_log.jsonl` | Append-only log of placed bets (auto-created) |

**Important:** `stake_session.json` is in `.gitignore` and will NEVER be committed. Don't share it.

---

## API Reference

### `StakeAPIClient`

```python
from stake_api import StakeAPIClient, ManualSessionProvider

provider = ManualSessionProvider.from_file("stake_session.json")
client = StakeAPIClient(provider, verbose=True)

# Fetch markets for a fixture
fixture = client.get_fixture_markets("46602331-w73-w75")

# Find an outcome
outcome = client.find_outcome(fixture, "Total Goals", "Over", specifier="2.5")
# → Outcome(id="...", name="Over 2.5", odds=1.85, active=True, ...)

# Validate a multibet
result = client.validate_multibet([id1, id2, id3])
# → {"odds": 3.388, "compatible": True, "conflicts": []}

# Place a single bet
bet = client.place_single_bet(amount=1.0, currency="usdt", outcome_id=id1)
# → BetResult(id="...", amount=1.0, potential_multiplier=1.85, ...)

# Place a multibet
bet = client.place_multibet(amount=1.0, currency="usdt", outcome_ids=[id1, id2, id3])
```

### `KasadaHeaderProvider` (pluggable)

```python
from stake_api import ManualSessionProvider, BrowserSessionProvider

# Option 1: manual (default, requires periodic refresh)
provider = ManualSessionProvider.from_file("stake_session.json")

# Option 2: browser automation (autonomous, requires Playwright)
provider = BrowserSessionProvider(session_token="...", headless=True)

# Option 3: implement your own (e.g. third-party solver)
class MySolverProvider(KasadaHeaderProvider):
    def get_headers(self) -> dict:
        # call your solver service
        return {"x-kpsdk-cd": "...", "x-kpsdk-ct": "...", ...}
```

---

## Autonomous Operation

For Hermes (or any bot) to run fully autonomously:

1. **Use `BrowserSessionProvider`** instead of `ManualSessionProvider`
   - Requires Playwright: `pip install playwright && playwright install chromium`
   - Auto-harvests Kasada headers every 15 min
   - ~50 MB extra RAM for the browser

2. **Pre-populate `fixtures.json`** before each round
   - Browse Stake.com manually once per round
   - Or implement a search endpoint (Stake may have one — reverse-engineer it)

3. **Schedule the cron**
   ```bash
   # 15 min before kickoff: refresh lineups + place bets
   */15 * * * * cd /path/to/wc2026-predictor && python3 scripts/wc_betting_predictor.py --force-refresh && python3 scripts/stake_bet_placer.py
   ```

4. **Monitor `bets_log.jsonl`** for settled bets and update bankroll

---

## Troubleshooting

### "403 Forbidden — Kasada headers likely invalid"
- Your `stake_session.json` headers expired
- Re-harvest from browser devtools
- Or switch to `BrowserSessionProvider`

### "No fixture slug found for CAN-MAR"
- `fixtures.json` is missing the mapping
- Browse Stake.com, find the match URL, copy the slug

### "No value bets found"
- Either no edges cleared the 5% threshold (normal)
- Or Stake odds differ significantly from your model (check `wc2026_betting_markets.json`)

### "Multibet has conflicts"
- You're trying to combine selections from the same match
- The placer should prevent this automatically — file a bug if it doesn't

### "429 Too Many Requests"
- Rate limited — backoff is automatic (2s, 4s, 8s)
- If persistent, increase `_rate_limit_delay` in `stake_api.py`

---

## Disclaimer

This integration is for educational purposes. Automated betting carries significant risk:
- The model has 95% CI of ±24% on accuracy
- Kasada headers may fail at any time, blocking bet placement
- Stake may close your account for bot usage (check their ToS)
- Never bet money you cannot afford to lose

The authors are not responsible for any financial losses.
