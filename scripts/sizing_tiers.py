# WC 2026 — bet sizing calibration tiers.
# Updated 2026-07-04 based on Michael's post-CAN-MAR directive.
#
# Start: tier 0 (validation, 1.67% per bet, 5% max exposure, 1/4 Kelly)
# Bump to tier 1 after iter-5 if: 24-match backtest ≥50% exact + Brier <0.20
# Bump to tier 2 after iter-6 if: 28-match backtest ≥45% exact
# Hard floor: 25% total bankroll across open bets (survival)

# Tier 0: validation (CAN-MAR, R16 first night)
TIER_0 = {
    "name": "validation",
    "max_exposure_pct": 0.05,        # 5% per match
    "kelly_fraction": 0.25,          # 1/4 Kelly
    "max_stake_pct": 0.0167,         # 1.67% per bet cap
    "min_edge": 0.05,
    "min_kelly": 0.03,
    "min_odds": 1.50,
    "max_selections": 4,             # raised 3→4 on 2026-07-04 to allow CS leg
    "global_exposure_cap": 0.25,     # hard floor across all open bets
}

# Tier 1: post-iter-5 (24 matches, ≥50% exact, Brier <0.20)
TIER_1 = {
    "name": "tier_1",
    "max_exposure_pct": 0.10,        # 10% per match
    "kelly_fraction": 0.50,          # 1/2 Kelly
    "max_stake_pct": 0.03,           # 3% per bet cap
    "min_edge": 0.05,
    "min_kelly": 0.03,
    "min_odds": 1.50,
    "max_selections": 4,
    "global_exposure_cap": 0.25,
}

# Tier 2: post-iter-6 (28 matches, ≥45% exact)
TIER_2 = {
    "name": "tier_2",
    "max_exposure_pct": 0.15,        # 15% per match
    "kelly_fraction": 1.0,           # full Kelly
    "max_stake_pct": 0.08,           # 8% per bet cap
    "min_edge": 0.05,
    "min_kelly": 0.03,
    "min_odds": 1.50,
    "max_selections": 5,
    "global_exposure_cap": 0.25,
}

CURRENT_TIER = TIER_0  # updated by the iter-5/6 retrain runners

# Decision template for post-match report (Michael, 2026-07-04)
POST_MATCH_REPORT_TEMPLATE = """
{match} result: {actual_score}
Bets: {n_placed} placed, {n_won} won, {n_lost} lost
P&L: {pnl:+.2f}
Model calibration: predicted {home} win {p_home}%, actual {outcome}
Pre-edge: {pre_edges}
Post-edge (if {home} won): {post_edges}
Action for {next_match}: {action}
"""
