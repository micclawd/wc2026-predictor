#!/usr/bin/env python3
"""
Betting Markets Module — WC 2026
================================

Converts a scoreline probability distribution P(H=h, A=a) into probabilities
for all major betting markets:

- 1X2 (Home / Draw / Away)
- Double Chance (1X / 12 / X2)
- Over/Under (0.5, 1.5, 2.5, 3.5, 4.5)
- Asian Handicap (full and half lines, both teams)
- Both Teams To Score (BTTS Yes/No)
- Correct Score (top-N most likely)
- Total Goals (exact)
- Winning Margin (home by 1/2/3+, away by 1/2/3+)

All probabilities are derived analytically from the joint distribution —
no simulation needed. Asian Handicap handles full-lines (push possible) and
half-lines (no push) correctly.

Usage:
    from betting_markets import compute_all_markets
    markets = compute_all_markets(score_dist, home_code, away_code)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Core data structures
# --------------------------------------------------------------------------
@dataclass
class BettingMarkets:
    """All betting market probabilities for a single match."""
    home: str
    away: str

    # 1X2
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0

    # Double chance
    double_chance_1X: float = 0.0   # home or draw
    double_chance_12: float = 0.0   # home or away
    double_chance_X2: float = 0.0   # draw or away

    # Over/Under (key: line, value: P(over))
    over_under: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g. {"2.5": {"over": 0.55, "under": 0.45}}

    # Asian Handicap (key: line, value: P(home win)/push/loss from home perspective)
    asian_handicap: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g. {"-0.5": {"home_win": 0.65, "push": 0.0, "home_loss": 0.35}}

    # BTTS
    btts_yes: float = 0.0
    btts_no: float = 0.0

    # Correct score top-5
    correct_score_top5: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"score": "2-1", "prob": 0.15}, ...]

    # Total goals exact
    total_goals: Dict[str, float] = field(default_factory=dict)
    # e.g. {"0": 0.05, "1": 0.15, "2": 0.25, ...}

    # Winning margin
    winning_margin: Dict[str, float] = field(default_factory=dict)
    # e.g. {"home_by_1": 0.20, "home_by_2": 0.15, ..., "away_by_1": 0.10, ...}

    # Expected goals (for reference)
    expected_goals_home: float = 0.0
    expected_goals_away: float = 0.0
    expected_total_goals: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home": self.home,
            "away": self.away,
            "markets": {
                "1X2": {
                    "home_win": round(self.home_win, 4),
                    "draw": round(self.draw, 4),
                    "away_win": round(self.away_win, 4),
                },
                "double_chance": {
                    "1X": round(self.double_chance_1X, 4),
                    "12": round(self.double_chance_12, 4),
                    "X2": round(self.double_chance_X2, 4),
                },
                "over_under": {
                    line: {k: round(v, 4) for k, v in probs.items()}
                    for line, probs in self.over_under.items()
                },
                "asian_handicap": {
                    line: {k: round(v, 4) for k, v in probs.items()}
                    for line, probs in self.asian_handicap.items()
                },
                "btts": {
                    "yes": round(self.btts_yes, 4),
                    "no": round(self.btts_no, 4),
                },
                "correct_score_top5": [
                    {"score": s["score"], "prob": round(s["prob"], 4)}
                    for s in self.correct_score_top5
                ],
                "total_goals": {
                    k: round(v, 4) for k, v in self.total_goals.items()
                },
                "winning_margin": {
                    k: round(v, 4) for k, v in self.winning_margin.items()
                },
                "expected_goals": {
                    "home": round(self.expected_goals_home, 3),
                    "away": round(self.expected_goals_away, 3),
                    "total": round(self.expected_total_goals, 3),
                },
            },
        }


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
def compute_all_markets(probs: np.ndarray, home: str, away: str,
                        max_goals: int = 8) -> BettingMarkets:
    """Compute all betting market probabilities from a scoreline distribution.

    Args:
        probs: 2D numpy array of shape (max_goals+1, max_goals+1) where
               probs[h, a] = P(home scores h, away scores a). Must sum to ~1.
        home: home team code (e.g. "PAR")
        away: away team code (e.g. "FRA")
        max_goals: maximum goals per team in the distribution

    Returns:
        BettingMarkets object with all probabilities filled in.
    """
    # Safety: renormalize
    probs = probs.copy()
    total = probs.sum()
    if total > 0:
        probs /= total

    n = max_goals + 1  # grid size

    # --- 1X2 ---
    # Home win: h > a (lower triangle, excluding diagonal)
    home_win = float(np.sum(np.tril(probs, -1)))
    draw = float(np.sum(np.diag(probs)))
    away_win = float(np.sum(np.triu(probs, 1)))

    # --- Double chance ---
    dc_1X = home_win + draw
    dc_12 = home_win + away_win
    dc_X2 = draw + away_win

    # --- Over/Under ---
    # P(total > line) = sum of probs[h,a] for h+a > line
    ou_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    over_under: Dict[str, Dict[str, float]] = {}
    for line in ou_lines:
        # total goals = h + a
        # P(over) = sum probs[h,a] where h+a > line
        over_prob = 0.0
        under_prob = 0.0
        for h in range(n):
            for a in range(n):
                tg = h + a
                if tg > line:
                    over_prob += probs[h, a]
                else:
                    under_prob += probs[h, a]
        over_under[str(line)] = {"over": over_prob, "under": under_prob}

    # --- Asian Handicap ---
    # For each line, compute P(home wins bet), P(push), P(home loses bet)
    # from the home-team perspective.
    # Adjusted margin = h - a + line
    #   > 0 → home wins bet
    #   = 0 → push (only possible for full lines like -1, -2, 0, +1, +2)
    #   < 0 → home loses bet
    ah_lines = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    asian_handicap: Dict[str, Dict[str, float]] = {}
    for line in ah_lines:
        home_bet_win = 0.0
        push = 0.0
        home_bet_loss = 0.0
        for h in range(n):
            for a in range(n):
                adj_margin = (h - a) + line
                if adj_margin > 0.001:
                    home_bet_win += probs[h, a]
                elif adj_margin < -0.001:
                    home_bet_loss += probs[h, a]
                else:
                    push += probs[h, a]
        # away perspective is mirror
        asian_handicap[str(line)] = {
            "home_win": home_bet_win,
            "push": push,
            "home_loss": home_bet_loss,
            # away win prob = home loss prob (with push reflected)
            "away_win": home_bet_loss,
            "away_loss": home_bet_win,
        }

    # --- BTTS ---
    # P(BTTS yes) = sum probs[h,a] for h>=1 AND a>=1
    btts_yes = 0.0
    for h in range(1, n):
        for a in range(1, n):
            btts_yes += probs[h, a]
    btts_no = 1.0 - btts_yes

    # --- Correct score top-5 ---
    flat = []
    for h in range(n):
        for a in range(n):
            flat.append((f"{h}-{a}", float(probs[h, a])))
    flat.sort(key=lambda x: x[1], reverse=True)
    correct_score_top5 = [{"score": s, "prob": p} for s, p in flat[:5]]

    # --- Total goals exact ---
    total_goals: Dict[str, float] = {}
    for tg in range(0, 2 * max_goals + 1):
        p = 0.0
        for h in range(n):
            a = tg - h
            if 0 <= a < n:
                p += probs[h, a]
        if p > 0.0001:  # skip negligible
            total_goals[str(tg)] = p

    # --- Winning margin ---
    # home_by_1, home_by_2, home_by_3+, draw, away_by_1, away_by_2, away_by_3+
    winning_margin: Dict[str, float] = {
        "home_by_1": 0.0, "home_by_2": 0.0, "home_by_3plus": 0.0,
        "draw": 0.0,
        "away_by_1": 0.0, "away_by_2": 0.0, "away_by_3plus": 0.0,
    }
    for h in range(n):
        for a in range(n):
            diff = h - a
            if diff == 0:
                winning_margin["draw"] += probs[h, a]
            elif diff == 1:
                winning_margin["home_by_1"] += probs[h, a]
            elif diff == 2:
                winning_margin["home_by_2"] += probs[h, a]
            elif diff >= 3:
                winning_margin["home_by_3plus"] += probs[h, a]
            elif diff == -1:
                winning_margin["away_by_1"] += probs[h, a]
            elif diff == -2:
                winning_margin["away_by_2"] += probs[h, a]
            elif diff <= -3:
                winning_margin["away_by_3plus"] += probs[h, a]

    # --- Expected goals ---
    eg_h = float(sum(h * np.sum(probs[h, :]) for h in range(n)))
    eg_a = float(sum(a * np.sum(probs[:, a]) for a in range(n)))

    return BettingMarkets(
        home=home, away=away,
        home_win=home_win, draw=draw, away_win=away_win,
        double_chance_1X=dc_1X, double_chance_12=dc_12, double_chance_X2=dc_X2,
        over_under=over_under,
        asian_handicap=asian_handicap,
        btts_yes=btts_yes, btts_no=btts_no,
        correct_score_top5=correct_score_top5,
        total_goals=total_goals,
        winning_margin=winning_margin,
        expected_goals_home=eg_h,
        expected_goals_away=eg_a,
        expected_total_goals=eg_h + eg_a,
    )


# --------------------------------------------------------------------------
# Implied odds / fair odds
# --------------------------------------------------------------------------
def to_implied_odds(prob: float) -> float:
    """Convert probability to fair decimal odds (no margin)."""
    if prob <= 0:
        return 0.0
    return 1.0 / prob


def add_fair_odds(markets_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add 'fair_odds' alongside every probability in the markets dict."""
    def add_odds(obj):
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                if isinstance(v, (int, float)) and 0 < v < 1:
                    new[k] = v
                    new[f"{k}_fair_odds"] = round(to_implied_odds(v), 2)
                else:
                    new[k] = add_odds(v)
            return new
        if isinstance(obj, list):
            return [add_odds(item) for item in obj]
        return obj
    return add_odds(markets_dict)


# --------------------------------------------------------------------------
# Value bet detection
# --------------------------------------------------------------------------
def find_value_bets(model_probs: Dict[str, float],
                    bookmaker_odds: Dict[str, float]) -> List[Dict[str, Any]]:
    """Identify value bets where model probability > implied bookmaker probability.

    Args:
        model_probs: {selection: probability}, e.g. {"over_2.5": 0.55}
        bookmaker_odds: {selection: decimal_odds}, e.g. {"over_2.5": 1.90}

    Returns:
        List of value bets with edge and Kelly fraction.
    """
    value_bets = []
    for sel, p in model_probs.items():
        odds = bookmaker_odds.get(sel)
        if odds is None or odds <= 1.0:
            continue
        implied_p = 1.0 / odds
        edge = p - implied_p
        if edge > 0.01:  # at least 1% edge
            # Kelly fraction: f = (p * odds - 1) / (odds - 1)
            kelly = (p * odds - 1) / (odds - 1) if odds > 1 else 0
            kelly = max(0, min(kelly, 0.25))  # cap at 25% bankroll
            value_bets.append({
                "selection": sel,
                "model_prob": round(p, 4),
                "bookmaker_odds": odds,
                "implied_prob": round(implied_p, 4),
                "edge": round(edge, 4),
                "kelly_fraction": round(kelly, 4),
                "expected_value": round(p * odds - 1, 4),
            })
    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


# --------------------------------------------------------------------------
# CLI for testing
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity test using a Poisson(1.5, 1.1) distribution
    import math
    max_g = 8
    lh, la = 1.5, 1.1
    h_range = np.arange(max_g + 1)
    a_range = np.arange(max_g + 1)
    ph = np.exp(-lh) * (lh ** h_range) / np.array([math.factorial(h) for h in h_range])
    pa = np.exp(-la) * (la ** a_range) / np.array([math.factorial(a) for a in a_range])
    probs = np.outer(ph, pa)
    probs /= probs.sum()

    markets = compute_all_markets(probs, "HOME", "AWAY", max_g)
    out = markets.to_dict()
    out_with_odds = add_fair_odds(out)

    import json
    print(json.dumps(out_with_odds, indent=2))
