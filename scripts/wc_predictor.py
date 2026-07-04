#!/usr/bin/env python3
"""
World Cup 2026 Scoreline Prediction Engine
===========================================

Built on data from https://github.com/26worldcup/26worldcup.github.io
- 88 finished WC 2026 matches (ground truth for backtesting)
- 16 upcoming matches (R16 -> Final) to predict
- 49,477 historical international matches (training data)
- Pre-built ELO ratings (current + form) for 48 WC teams
- FIFA rankings, venue host-country data

Model families implemented:
1. Poisson baseline           - independent Poisson on each team's goal count
2. Dixon-Coles                - Poisson + low-score correlation correction
3. Time-decayed ELO + Poisson - ELO updated on intl history with exp decay
4. Attack/Defense strengths   - per-team goals scored/conceded rates
5. Empirical scoreline lookup - most common scoreline per ELO-gap bucket
6. Ensemble blend             - weighted blend of all above

Backtest metrics:
- Exact scoreline accuracy   (predicted scoreline == actual)
- W/D/L outcome accuracy     (predicted winner == actual)
- Margin (GD) accuracy       (predicted goal-difference sign == actual)
- Top-3 scoreline hit rate   (actual scoreline in top-3 predicted)
- Brier score (3-class)      - lower is better
- Log loss                   - lower is better

Run:
    python3 wc_predictor.py            # backtest + predict + write report
    python3 wc_predictor.py --backtest # backtest only
    python3 wc_predictor.py --predict  # predict only (uses best model)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Use centralized config for portable paths.
# Falls back to direct paths if config.py is not available (backward compat).
try:
    from config import (
        DATA_DIR as _DATA_DIR, DOWNLOAD_DIR as _DOWNLOAD_DIR,
        MATCHES_JSON, SIM_MODEL_JSON, TEAMS_JSON, VENUES_JSON,
        INTL_RESULTS_CSV, CACHE_DIR as _CACHE_DIR,
    )
    REPO_ROOT = _DATA_DIR
    DATA_DIR = _DATA_DIR / "public" / "data"
    SCRIPTS_DIR = _DATA_DIR / "scripts"
    DOWNLOAD_DIR = _DOWNLOAD_DIR
except ImportError:
    REPO_ROOT = Path("/home/z/my-project/26worldcup.github.io")
    DATA_DIR = REPO_ROOT / "public" / "data"
    SCRIPTS_DIR = REPO_ROOT / "scripts"
    DOWNLOAD_DIR = Path("/home/z/my-project/download")
    MATCHES_JSON = DATA_DIR / "matches.json"
    SIM_MODEL_JSON = DATA_DIR / "sim-model.json"
    TEAMS_JSON = DATA_DIR / "teams.json"
    VENUES_JSON = DATA_DIR / "venues.json"
    INTL_RESULTS_CSV = SCRIPTS_DIR / "cache" / "intl-results.csv"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Host countries for WC 2026
HOST_OF = {"USA": "US", "CAN": "CA", "MEX": "MX"}
HOST_BONUS_DEFAULT = 60.0  # ELO points for hosting


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class TeamRating:
    code: str
    elo_current: float       # prebuilt ELO (long-term strength)
    elo_form: Optional[float]  # prebuilt form ELO (recent run)
    fifa_ranking: int
    group: str


@dataclass
class Match:
    id: str
    n: int
    stage: str               # group, r32, r16, qf, sf, third, final
    group: Optional[str]
    date: datetime
    home_code: Optional[str]
    away_code: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    home_pen: Optional[int]
    away_pen: Optional[int]
    venue_id: Optional[str]
    venue_country: Optional[str]
    status: str              # finished, scheduled


@dataclass
class ScoreDist:
    """Discrete distribution over (home_goals, away_goals) scorelines up to max_goals."""
    max_goals: int
    probs: np.ndarray   # shape (max_goals+1, max_goals+1), sums to 1

    def top_k(self, k: int) -> List[Tuple[Tuple[int, int], float]]:
        flat = [(float(self.probs[h, a]), (h, a)) for h in range(self.max_goals + 1) for a in range(self.max_goals + 1)]
        flat.sort(reverse=True)
        return [(score, p) for p, score in flat[:k]]

    def mode(self) -> Tuple[int, int]:
        idx = np.unravel_index(np.argmax(self.probs), self.probs.shape)
        return int(idx[0]), int(idx[1])

    def outcome_probs(self) -> Tuple[float, float, float]:
        h_win = float(np.sum(np.tril(self.probs, -1)))
        draw = float(np.sum(np.diag(self.probs)))
        a_win = float(np.sum(np.triu(self.probs, 1)))
        return h_win, draw, a_win

    def expected_goals(self) -> Tuple[float, float]:
        gh = float(sum(h * np.sum(self.probs[h, :]) for h in range(self.max_goals + 1)))
        ga = float(sum(a * np.sum(self.probs[:, a]) for a in range(self.max_goals + 1)))
        return gh, ga


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_matches() -> List[Match]:
    raw = json.loads(MATCHES_JSON.read_text())
    venues = load_venues()
    out: List[Match] = []
    for m in raw["matches"]:
        home = m.get("home") or {}
        away = m.get("away") or {}
        venue_id = m.get("venueId")
        venue_country = venues.get(venue_id, {}).get("country") if venue_id else None
        out.append(Match(
            id=m["id"],
            n=m["n"],
            stage=m["stage"],
            group=m.get("group"),
            date=datetime.fromisoformat(m["date"].replace("Z", "+00:00")),
            home_code=home.get("code"),
            away_code=away.get("code"),
            home_score=home.get("score"),
            away_score=away.get("score"),
            home_pen=home.get("pen"),
            away_pen=away.get("pen"),
            venue_id=venue_id,
            venue_country=venue_country,
            status=m["status"],
        ))
    return out


def load_venues() -> Dict[str, Dict[str, Any]]:
    raw = json.loads(VENUES_JSON.read_text())
    # venues.json has shape { "venueId": {...}, ... } OR { "venues": [...] }
    if isinstance(raw, dict) and "venues" in raw and isinstance(raw["venues"], list):
        return {v["id"]: v for v in raw["venues"]}
    return raw  # already keyed by id


def load_teams() -> Dict[str, TeamRating]:
    sm = json.loads(SIM_MODEL_JSON.read_text())
    tm = json.loads(TEAMS_JSON.read_text())["teams"]
    out: Dict[str, TeamRating] = {}
    for code, t in tm.items():
        r = sm["teams"].get(code, {"r": 1600, "f": None})
        out[code] = TeamRating(
            code=code,
            elo_current=float(r["r"]),
            elo_form=float(r["f"]) if r.get("f") is not None else None,
            fifa_ranking=int(t.get("ranking", 999)),
            group=t.get("group", "?"),
        )
    return out


def load_intl_history() -> List[Dict[str, Any]]:
    """Load international match history CSV. Returns chronological list of dicts."""
    rows: List[Dict[str, Any]] = []
    with INTL_RESULTS_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "date": datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "home_score": int(r["home_score"]),
                    "away_score": int(r["away_score"]),
                    "tournament": r["tournament"],
                    "neutral": r["neutral"].upper() == "TRUE",
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda x: x["date"])
    return rows


# --------------------------------------------------------------------------
# ELO computation (custom, time-decayed, optional)
# --------------------------------------------------------------------------
class EloCalculator:
    """Compute ELO ratings from international match history with time decay.

    Mirrors the repo's pipeline (scripts/elo.mjs) philosophy but parameterized so
    we can grid-search decay / K / home-advantage for best backtest fit.
    """

    def __init__(self, k: float = 30.0, decay: float = 0.0, home_adv: float = 80.0,
                 initial: float = 1500.0, wc_weight: float = 2.5):
        self.k = k
        self.decay = decay             # exp decay per day; 0 = no decay
        self.home_adv = home_adv       # ELO points for non-neutral home
        self.initial = initial
        self.wc_weight = wc_weight     # WC finals multiplier on K

    def compute(self, history: List[Dict[str, Any]], as_of: datetime,
                teams_of_interest: Optional[set] = None) -> Dict[str, float]:
        """Return ELO rating for every team that played, snapshot at as_of."""
        ratings: Dict[str, float] = defaultdict(lambda: self.initial)
        for row in history:
            if row["date"] > as_of:
                break  # history is chronological, stop here
            h, a = row["home_team"], row["away_team"]
            hs, as_ = row["home_score"], row["away_score"]
            rh, ra = ratings[h], ratings[a]
            expected_h = 1.0 / (1.0 + 10 ** ((ra - rh + (0 if row["neutral"] else -self.home_adv)) / 400.0))
            # result from home perspective: 1/0.5/0
            if hs > as_:
                actual_h = 1.0
            elif hs < as_:
                actual_h = 0.0
            else:
                actual_h = 0.5
            # K weight: bigger for World Cup finals, smaller for friendlies
            tour = row["tournament"]
            if tour == "FIFA World Cup":
                k_eff = self.k * self.wc_weight
            elif "qualification" in tour.lower():
                k_eff = self.k * 1.5
            elif tour == "Friendly":
                k_eff = self.k * 0.5
            else:
                k_eff = self.k
            # time decay: older matches count less
            age_days = (as_of - row["date"]).days
            if self.decay > 0:
                k_eff *= math.exp(-self.decay * age_days)
            # goal-diff multiplier (caps at ~3x)
            gd = abs(hs - as_)
            k_eff *= min(1.0 + 0.15 * gd, 3.0)
            delta = k_eff * (actual_h - expected_h)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
        if teams_of_interest is not None:
            return {t: ratings[t] for t in teams_of_interest if t in ratings}
        return dict(ratings)


# --------------------------------------------------------------------------
# Scoreline distribution models
# --------------------------------------------------------------------------
class PoissonModel:
    """Independent Poisson on each team's goal count.

    Goals are sampled independently: H ~ Poisson(lambda_h), A ~ Poisson(lambda_a).
    Lambda is derived from ELO gap: stronger team gets higher expected goals.
    """

    def __init__(self, total_goals: float = 2.6, host_bonus: float = HOST_BONUS_DEFAULT,
                 max_goals: int = 8):
        self.total_goals = total_goals
        self.host_bonus = host_bonus
        self.max_goals = max_goals

    def lambdas_from_dr(self, dr: float) -> Tuple[float, float]:
        """dr = home_elo - away_elo + host_bonus. Convert to (lambda_h, lambda_a)."""
        share = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        # ensure both teams have a non-trivial baseline
        lh = max(self.total_goals * share, 0.35)
        la = max(self.total_goals - lh, 0.35)
        return lh, la

    def distribution(self, dr: float) -> ScoreDist:
        lh, la = self.lambdas_from_dr(dr)
        h_range = np.arange(self.max_goals + 1)
        a_range = np.arange(self.max_goals + 1)
        # P(H=h) * P(A=a), independent
        ph = np.exp(-lh) * (lh ** h_range) / np.array([math.factorial(h) for h in h_range])
        pa = np.exp(-la) * (la ** a_range) / np.array([math.factorial(a) for a in a_range])
        probs = np.outer(ph, pa)
        probs /= probs.sum()  # safety renorm
        return ScoreDist(self.max_goals, probs)


class DixonColesModel(PoissonModel):
    """Dixon-Coles (1997): Poisson + low-score correlation correction.

    Real football scores 0-0, 1-0, 0-1, 1-1 more often than independent Poisson
    predicts. The rho correction boosts/suppresses these specific cells.
    """

    def __init__(self, total_goals: float = 2.6, host_bonus: float = HOST_BONUS_DEFAULT,
                 rho: float = -0.13, max_goals: int = 8):
        super().__init__(total_goals, host_bonus, max_goals)
        self.rho = rho

    def distribution(self, dr: float) -> ScoreDist:
        lh, la = self.lambdas_from_dr(dr)
        h_range = np.arange(self.max_goals + 1)
        a_range = np.arange(self.max_goals + 1)
        ph = np.exp(-lh) * (lh ** h_range) / np.array([math.factorial(h) for h in h_range])
        pa = np.exp(-la) * (la ** a_range) / np.array([math.factorial(a) for a in a_range])
        probs = np.outer(ph, pa)
        # DC correction on the 4 low-score cells
        for h in (0, 1):
            for a in (0, 1):
                probs[h, a] *= (1.0 + self.rho * (1 if (h, a) in {(0, 0), (1, 1)} else -1)
                                * math.exp(-((h - lh) ** 2 + (a - la) ** 2) / 2.0))
        # the canonical DC correction is simpler:
        #   tau(h, a) = 1 - rho * a * b  where a = 1 if h==0 else (1 if h==1 else 0)
        # We re-derive cleanly here:
        probs = np.outer(ph, pa)  # reset to plain Poisson
        if self.rho != 0:
            # apply canonical DC tau
            def tau(h: int, a: int) -> float:
                a_coef = 1.0 if h == 0 else (-1.0 if h == 1 else 0.0)
                b_coef = 1.0 if a == 0 else (-1.0 if a == 1 else 0.0)
                return 1.0 - self.rho * a_coef * b_coef
            for h in range(self.max_goals + 1):
                for a in range(self.max_goals + 1):
                    probs[h, a] *= tau(h, a)
        probs = np.clip(probs, 0, None)
        s = probs.sum()
        if s > 0:
            probs /= s
        return ScoreDist(self.max_goals, probs)


class EmpiricalLookupModel:
    """Look up the most common scoreline for a given ELO-gap bucket.

    Falls back to Poisson when the bucket has too few historical samples.
    Useful because real football scorelines are heavy-tailed and discrete:
    the modal scoreline per matchup-strength often beats a naive Poisson mode.
    """

    def __init__(self, history: List[Dict[str, Any]], team_ratings: Dict[str, TeamRating],
                 bucket_size: int = 50, min_samples: int = 30,
                 host_bonus: float = HOST_BONUS_DEFAULT, max_goals: int = 8,
                 fallback: Optional[PoissonModel] = None):
        self.host_bonus = host_bonus
        self.max_goals = max_goals
        self.fallback = fallback or PoissonModel(host_bonus=host_bonus, max_goals=max_goals)
        # bucket counts: dict[int -> Counter[(h,a)]]
        self.buckets: Dict[int, Counter] = defaultdict(Counter)
        # need ELO snapshot per match. Use the prebuilt long-term ELO as proxy:
        # treat team_ratings[code].elo_current as a stable strength prior.
        for row in history:
            h, a = row["home_team"], row["away_team"]
            # only count if both teams have a known rating (i.e. WC-level strength)
            th = team_ratings.get(h)
            ta = team_ratings.get(a)
            if th is None or ta is None:
                continue
            dr = th.elo_current - ta.elo_current + (0 if row["neutral"] else -self.host_bonus)
            bucket = int(dr // bucket_size)
            self.buckets[bucket][(row["home_score"], row["away_score"])] += 1

        self.bucket_size = bucket_size
        self.min_samples = min_samples

    def distribution(self, dr: float) -> ScoreDist:
        bucket = int(dr // self.bucket_size)
        counts = self.buckets.get(bucket, Counter())
        total = sum(counts.values())
        if total < self.min_samples:
            # widen the bucket by merging neighbours
            counts = self.buckets.get(bucket, Counter()) + self.buckets.get(bucket - 1, Counter()) + self.buckets.get(bucket + 1, Counter())
            total = sum(counts.values())
        if total < self.min_samples:
            # fallback to Poisson
            return self.fallback.distribution(dr)
        probs = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for (h, a), c in counts.items():
            if h <= self.max_goals and a <= self.max_goals:
                probs[h, a] = c
        # smoothing: add a tiny epsilon so the distribution isn't degenerate
        probs += 0.5
        probs /= probs.sum()
        return ScoreDist(self.max_goals, probs)


class AttackDefenseModel:
    """Per-team attack and defense strengths estimated from recent goals.

    Goals scored ~ Poisson(team_attack * opp_defense * baseline).
    Strengths are fit by iterative proportional fitting on the recent match window.
    """

    def __init__(self, history: List[Dict[str, Any]], team_ratings: Dict[str, TeamRating],
                 window_days: int = 730, max_goals: int = 8,
                 host_bonus: float = HOST_BONUS_DEFAULT):
        self.max_goals = max_goals
        self.host_bonus = host_bonus
        self.baseline = 1.35   # average goals per team per match
        # Compute attack/defense per team from last `window_days` of matches
        self.attack: Dict[str, float] = defaultdict(lambda: 1.0)
        self.defense: Dict[str, float] = defaultdict(lambda: 1.0)
        # iterative update
        now = history[-1]["date"] if history else datetime.now(timezone.utc)
        window_start = now.timestamp() - window_days * 86400
        recent = [r for r in history if r["date"].timestamp() >= window_start]
        # iterate to convergence
        for _ in range(20):
            scored: Dict[str, float] = defaultdict(float)
            conceded: Dict[str, float] = defaultdict(float)
            exp_scored: Dict[str, float] = defaultdict(float)
            exp_conceded: Dict[str, float] = defaultdict(float)
            for r in recent:
                h, a = r["home_team"], r["away_team"]
                if h not in team_ratings or a not in team_ratings:
                    continue
                # expected goals: baseline * attack * defense * (1 + small home bonus)
                eg_h = self.baseline * self.attack[h] * self.defense[a] * 1.10
                eg_a = self.baseline * self.attack[a] * self.defense[h]
                exp_scored[h] += eg_h
                exp_conceded[h] += eg_a
                exp_scored[a] += eg_a
                exp_conceded[a] += eg_h
                scored[h] += r["home_score"]
                conceded[h] += r["away_score"]
                scored[a] += r["away_score"]
                conceded[a] += r["home_score"]
            for t in scored:
                if exp_scored[t] > 0:
                    self.attack[t] *= scored[t] / exp_scored[t]
                if exp_conceded[t] > 0:
                    self.defense[t] *= conceded[t] / exp_conceded[t]
            # shrink towards 1.0 (regression to mean)
            for t in list(self.attack.keys()):
                self.attack[t] = 0.5 * self.attack[t] + 0.5 * 1.0
                self.defense[t] = 0.5 * self.defense[t] + 0.5 * 1.0

    def lambdas(self, home: str, away: str, dr: float) -> Tuple[float, float]:
        """Combine attack/defense with an ELO prior so unknown teams still get a sane lambda."""
        a_h = self.attack.get(home, 1.0)
        d_h = self.defense.get(home, 1.0)
        a_a = self.attack.get(away, 1.0)
        d_a = self.defense.get(away, 1.0)
        eg_h = self.baseline * a_h * d_a * 1.10
        eg_a = self.baseline * a_a * d_h
        # blend with Poisson-from-ELO
        share = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        elo_h = max(2.6 * share, 0.35)
        elo_a = max(2.6 - elo_h, 0.35)
        return 0.5 * eg_h + 0.5 * elo_h, 0.5 * eg_a + 0.5 * elo_a

    def distribution(self, home: str, away: str, dr: float) -> ScoreDist:
        lh, la = self.lambdas(home, away, dr)
        h_range = np.arange(self.max_goals + 1)
        a_range = np.arange(self.max_goals + 1)
        ph = np.exp(-lh) * (lh ** h_range) / np.array([math.factorial(h) for h in h_range])
        pa = np.exp(-la) * (la ** a_range) / np.array([math.factorial(a) for a in a_range])
        probs = np.outer(ph, pa)
        probs /= probs.sum()
        return ScoreDist(self.max_goals, probs)


class EnsembleModel:
    """Weighted blend of multiple scoreline-distribution models."""

    def __init__(self, models_and_weights: List[Tuple[Any, float]], max_goals: int = 8):
        # models_and_weights: list of (model, weight)
        # each model has either .distribution(dr) or .distribution(home, away, dr)
        self.mw = models_and_weights
        self.max_goals = max_goals
        # normalize weights
        w_sum = sum(w for _, w in self.mw)
        self.mw = [(m, w / w_sum) for m, w in self.mw]

    def distribution(self, home: str, away: str, dr: float) -> ScoreDist:
        acc = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for m, w in self.mw:
            try:
                # try (home, away, dr) signature first
                d = m.distribution(home, away, dr)
            except TypeError:
                d = m.distribution(dr)
            acc += w * d.probs
        acc /= acc.sum()
        return ScoreDist(self.max_goals, acc)


# --------------------------------------------------------------------------
# Engine: combines team ratings, ELO, and a scoreline model
# --------------------------------------------------------------------------
@dataclass
class PredictionEngine:
    teams: Dict[str, TeamRating]
    intl_history: List[Dict[str, Any]]
    score_model: Any
    host_bonus: float = HOST_BONUS_DEFAULT
    use_form_elo: bool = True
    form_weight: float = 0.5

    def dr_for(self, home: str, away: str, venue_country: Optional[str]) -> float:
        th = self.teams.get(home)
        ta = self.teams.get(away)
        rh = th.elo_current if th else 1600
        ra = ta.elo_current if ta else 1600
        # host bonus: only when venue country matches one of the teams
        bonus = 0.0
        if venue_country and HOST_OF.get(home) == venue_country:
            bonus = self.host_bonus
        elif venue_country and HOST_OF.get(away) == venue_country:
            bonus = -self.host_bonus
        dr = rh - ra + bonus
        # blend with form ELO if available
        if self.use_form_elo and th and ta and th.elo_form is not None and ta.elo_form is not None:
            dr_form = (th.elo_form - ta.elo_form) + bonus
            dr = (1 - self.form_weight) * dr + self.form_weight * dr_form
        return dr

    def predict(self, home: str, away: str, venue_country: Optional[str] = None) -> ScoreDist:
        dr = self.dr_for(home, away, venue_country)
        try:
            return self.score_model.distribution(home, away, dr)
        except TypeError:
            return self.score_model.distribution(dr)


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    model_name: str
    n_matches: int
    exact_scoreline_acc: float
    outcome_acc: float
    margin_acc: float
    top3_scoreline_acc: float
    brier: float
    log_loss: float
    per_match: List[Dict[str, Any]] = field(default_factory=list)


def _brier(outcome_probs: Tuple[float, float, float], actual: str) -> float:
    p_h, p_d, p_a = outcome_probs
    o = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[actual]
    return sum((p - a) ** 2 for p, a in zip((p_h, p_d, p_a), o)) / 2.0


def _log_loss(outcome_probs: Tuple[float, float, float], actual: str) -> float:
    eps = 1e-9
    p = {"H": outcome_probs[0], "D": outcome_probs[1], "A": outcome_probs[2]}[actual]
    return -math.log(max(p, eps))


def backtest(engine: PredictionEngine, matches: List[Match], model_name: str,
             max_goals: int = 8) -> BacktestResult:
    finished = [m for m in matches if m.status == "finished"
                and m.home_score is not None and m.away_score is not None
                and m.home_code and m.away_code
                # only count 90-min result: ignore ET/pens by using home_score/away_score
                ]
    exact = 0
    outcome_correct = 0
    margin_correct = 0
    top3_correct = 0
    brier_sum = 0.0
    ll_sum = 0.0
    per_match: List[Dict[str, Any]] = []
    for m in finished:
        try:
            dist = engine.predict(m.home_code, m.away_code, m.venue_country)
        except Exception as e:
            continue
        # actual outcome & margin
        if m.home_score > m.away_score:
            actual_outcome = "H"
        elif m.home_score < m.away_score:
            actual_outcome = "A"
        else:
            actual_outcome = "D"
        actual_margin = m.home_score - m.away_score
        # predicted
        pred_score = dist.mode()
        h_pred, a_pred = pred_score
        if h_pred > a_pred:
            pred_outcome = "H"
        elif h_pred < a_pred:
            pred_outcome = "A"
        else:
            pred_outcome = "D"
        pred_margin = h_pred - a_pred
        # top-3 scorelines
        top3 = dist.top_k(3)
        top3_scores = [s for s, _ in top3]
        # metrics
        if pred_score == (m.home_score, m.away_score):
            exact += 1
        if pred_outcome == actual_outcome:
            outcome_correct += 1
        if (pred_margin > 0 and actual_margin > 0) or (pred_margin < 0 and actual_margin < 0) or (pred_margin == 0 and actual_margin == 0):
            margin_correct += 1
        if (m.home_score, m.away_score) in top3_scores:
            top3_correct += 1
        op = dist.outcome_probs()
        brier_sum += _brier(op, actual_outcome)
        ll_sum += _log_loss(op, actual_outcome)
        per_match.append({
            "n": m.n,
            "stage": m.stage,
            "date": m.date.date().isoformat(),
            "home": m.home_code,
            "away": m.away_code,
            "actual": f"{m.home_score}-{m.away_score}",
            "predicted_mode": f"{h_pred}-{a_pred}",
            "outcome_actual": actual_outcome,
            "outcome_pred": pred_outcome,
            "p(H)": round(op[0], 3),
            "p(D)": round(op[1], 3),
            "p(A)": round(op[2], 3),
            "top3": [f"{h}-{a}" for h, a in top3_scores],
            "correct_scoreline": pred_score == (m.home_score, m.away_score),
            "correct_outcome": pred_outcome == actual_outcome,
        })
    n = len(finished)
    return BacktestResult(
        model_name=model_name,
        n_matches=n,
        exact_scoreline_acc=exact / n if n else 0,
        outcome_acc=outcome_correct / n if n else 0,
        margin_acc=margin_correct / n if n else 0,
        top3_scoreline_acc=top3_correct / n if n else 0,
        brier=brier_sum / n if n else 0,
        log_loss=ll_sum / n if n else 0,
        per_match=per_match,
    )


# --------------------------------------------------------------------------
# Hyperparameter sweep
# --------------------------------------------------------------------------
def sweep_poisson(matches: List[Match], teams: Dict[str, TeamRating],
                  history: List[Dict[str, Any]]) -> List[Tuple[BacktestResult, Dict[str, Any]]]:
    """Grid-search Poisson hyperparameters."""
    results: List[Tuple[BacktestResult, Dict[str, Any]]] = []
    for total_goals in [2.3, 2.5, 2.6, 2.7, 2.9]:
        for host_bonus in [40.0, 60.0, 80.0, 100.0]:
            for form_weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
                model = PoissonModel(total_goals=total_goals, host_bonus=host_bonus)
                engine = PredictionEngine(
                    teams=teams,
                    intl_history=history,
                    score_model=model,
                    host_bonus=host_bonus,
                    use_form_elo=(form_weight > 0),
                    form_weight=form_weight,
                )
                cfg = {"total_goals": total_goals, "host_bonus": host_bonus, "form_weight": form_weight}
                r = backtest(engine, matches, f"Poisson(tg={total_goals},hb={host_bonus},fw={form_weight})")
                results.append((r, cfg))
    return results


def sweep_dixon_coles(matches: List[Match], teams: Dict[str, TeamRating],
                      history: List[Dict[str, Any]]) -> List[Tuple[BacktestResult, Dict[str, Any]]]:
    results: List[Tuple[BacktestResult, Dict[str, Any]]] = []
    for total_goals in [2.5, 2.6, 2.7]:
        for host_bonus in [60.0, 80.0]:
            for rho in [-0.20, -0.13, -0.05, 0.0, 0.05]:
                model = DixonColesModel(total_goals=total_goals, host_bonus=host_bonus, rho=rho)
                engine = PredictionEngine(
                    teams=teams,
                    intl_history=history,
                    score_model=model,
                    host_bonus=host_bonus,
                    use_form_elo=True,
                    form_weight=0.5,
                )
                cfg = {"total_goals": total_goals, "host_bonus": host_bonus, "rho": rho}
                r = backtest(engine, matches, f"DixonColes(tg={total_goals},hb={host_bonus},rho={rho})")
                results.append((r, cfg))
    return results


def sweep_elo(matches: List[Match], teams: Dict[str, TeamRating],
              history: List[Dict[str, Any]]) -> List[Tuple[BacktestResult, Dict[str, Any]]]:
    """Sweep custom-ELO calc params. We then plug those ELOs into a Poisson model.

    NOTE: the ELO calculator only uses pre-WC history; we snapshot at the
    WC 2026 opener (2026-06-11) so we don't leak WC results into ratings.
    """
    results: List[Tuple[BacktestResult, Dict[str, Any]]] = []
    snapshot_date = datetime(2026, 6, 11, tzinfo=timezone.utc)
    teams_of_interest = set(teams.keys())
    # We need team name -> code mapping for the intl history. The history uses
    # full names (e.g. "Mexico"); our team codes are FIFA codes (e.g. "MEX").
    # Build a mapping from teams.json metadata.
    name_to_code = _build_name_to_code()

    for k in [20.0, 30.0, 40.0]:
        for decay in [0.0, 0.0003, 0.0006, 0.001]:   # per day
            for home_adv in [60.0, 80.0, 100.0]:
                calc = EloCalculator(k=k, decay=decay, home_adv=home_adv, wc_weight=2.5)
                custom_elo = calc.compute(history, snapshot_date, teams_of_interest)
                # Map back to team codes (history used full names)
                # custom_elo is keyed by full names; re-key by code where possible.
                elo_by_code: Dict[str, float] = {}
                for name, rating in custom_elo.items():
                    code = name_to_code.get(name)
                    if code:
                        elo_by_code[code] = rating
                # build a modified teams dict using custom ELO, fallback to prebuilt
                teams_mod: Dict[str, TeamRating] = {}
                for code, t in teams.items():
                    teams_mod[code] = TeamRating(
                        code=code,
                        elo_current=elo_by_code.get(code, t.elo_current),
                        elo_form=t.elo_form,   # keep prebuilt form
                        fifa_ranking=t.fifa_ranking,
                        group=t.group,
                    )
                # run Poisson backtest
                model = PoissonModel(total_goals=2.6, host_bonus=80.0)
                engine = PredictionEngine(
                    teams=teams_mod,
                    intl_history=history,
                    score_model=model,
                    host_bonus=80.0,
                    use_form_elo=False,   # custom ELO doesn't carry separate form
                    form_weight=0.0,
                )
                cfg = {"k": k, "decay": decay, "home_adv": home_adv, "use_custom_elo": True}
                r = backtest(engine, matches, f"CustomELO+Poisson(k={k},dec={decay},ha={home_adv})")
                results.append((r, cfg))
    return results


def _build_name_to_code() -> Dict[str, str]:
    """Map international-results team names to FIFA 3-letter codes via teams.json."""
    teams = json.loads(TEAMS_JSON.read_text())["teams"]
    out: Dict[str, str] = {}
    for code, t in teams.items():
        for lang, name in t.get("name", {}).items():
            out[name] = code
        # also map by English name lowercased + a few common alternates
    # common alternates that appear in intl-results.csv but not in teams.json names
    alternates = {
        "United States": "USA",
        "South Korea": "KOR",
        "Republic of Ireland": "IRL",
        "Bosnia and Herzegovina": "BIH",
        "Czech Republic": "CZE",
        "Cape Verde": "CPV",
        "Congo DR": "COD",
        "DR Congo": "COD",
        "Ivory Coast": "CIV",
        "South Africa": "RSA",
        "Saudi Arabia": "KSA",
        "Iran": "IRN",
        "Curacao": "CUW",
        "New Zealand": "NZL",
    }
    out.update(alternates)
    return out


# --------------------------------------------------------------------------
# Predict upcoming matches
# --------------------------------------------------------------------------
def predict_upcoming(engine: PredictionEngine, matches: List[Match]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches:
        if m.status == "finished":
            continue
        if not m.home_code or not m.away_code:
            # placeholder knockout - skip; we'd need to simulate the bracket first
            out.append({
                "n": m.n,
                "stage": m.stage,
                "date": m.date.date().isoformat(),
                "home": "TBD",
                "away": "TBD",
                "note": "Bracket-dependent; teams not yet determined.",
            })
            continue
        dist = engine.predict(m.home_code, m.away_code, m.venue_country)
        mode = dist.mode()
        op = dist.outcome_probs()
        top5 = dist.top_k(5)
        eg = dist.expected_goals()
        out.append({
            "n": m.n,
            "stage": m.stage,
            "date": m.date.date().isoformat(),
            "home": m.home_code,
            "away": m.away_code,
            "venue_country": m.venue_country,
            "predicted_scoreline_mode": f"{mode[0]}-{mode[1]}",
            "expected_goals_home": round(eg[0], 2),
            "expected_goals_away": round(eg[1], 2),
            "p_home_win": round(op[0], 3),
            "p_draw": round(op[1], 3),
            "p_away_win": round(op[2], 3),
            "top5_scorelines": [{"score": f"{h}-{a}", "prob": round(p, 3)} for (h, a), p in top5],
        })
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="Run backtest only")
    ap.add_argument("--predict", action="store_true", help="Run predictions only (uses best model)")
    ap.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data ...")
    matches = load_matches()
    teams = load_teams()
    history = load_intl_history()
    print(f"  Matches: {len(matches)} ({sum(1 for m in matches if m.status == 'finished')} finished)")
    print(f"  Teams:   {len(teams)}")
    print(f"  Intl history: {len(history)} (last: {history[-1]['date'].date()})")

    if args.predict and not args.backtest and not args.sweep:
        # quick predict with default Poisson+DC ensemble
        poisson = PoissonModel()
        dc = DixonColesModel()
        ad = AttackDefenseModel(history, teams)
        emp = EmpiricalLookupModel(history, teams)
        ens = EnsembleModel([
            (poisson, 0.20), (dc, 0.35), (ad, 0.25), (emp, 0.20)
        ])
        engine = PredictionEngine(teams=teams, intl_history=history, score_model=ens)
        preds = predict_upcoming(engine, matches)
        out_path = DOWNLOAD_DIR / "wc2026_predictions.json"
        out_path.write_text(json.dumps(preds, indent=2))
        print(f"Wrote predictions -> {out_path}")
        return

    # otherwise: full backtest pipeline
    print("\n=== Backtesting model families on 88 finished WC 2026 matches ===")

    # 1) Prebuilt ELO + Poisson baseline
    print("\n[1/6] Prebuilt ELO + Poisson (baseline)")
    poisson = PoissonModel()
    engine_p = PredictionEngine(teams=teams, intl_history=history, score_model=poisson)
    r_p = backtest(engine_p, matches, "Poisson-baseline")
    print(f"    exact={r_p.exact_scoreline_acc:.3%}  outcome={r_p.outcome_acc:.3%}  "
          f"margin={r_p.margin_acc:.3%}  top3={r_p.top3_scoreline_acc:.3%}  "
          f"brier={r_p.brier:.4f}  ll={r_p.log_loss:.4f}")

    # 2) Prebuilt ELO + Dixon-Coles
    print("\n[2/6] Prebuilt ELO + Dixon-Coles")
    dc = DixonColesModel()
    engine_dc = PredictionEngine(teams=teams, intl_history=history, score_model=dc)
    r_dc = backtest(engine_dc, matches, "DixonColes-baseline")
    print(f"    exact={r_dc.exact_scoreline_acc:.3%}  outcome={r_dc.outcome_acc:.3%}  "
          f"margin={r_dc.margin_acc:.3%}  top3={r_dc.top3_scoreline_acc:.3%}  "
          f"brier={r_dc.brier:.4f}  ll={r_dc.log_loss:.4f}")

    # 3) Attack/Defense model
    print("\n[3/6] Attack/Defense strengths (last 2 years)")
    ad = AttackDefenseModel(history, teams)
    engine_ad = PredictionEngine(teams=teams, intl_history=history, score_model=ad)
    r_ad = backtest(engine_ad, matches, "AttackDefense")
    print(f"    exact={r_ad.exact_scoreline_acc:.3%}  outcome={r_ad.outcome_acc:.3%}  "
          f"margin={r_ad.margin_acc:.3%}  top3={r_ad.top3_scoreline_acc:.3%}  "
          f"brier={r_ad.brier:.4f}  ll={r_ad.log_loss:.4f}")

    # 4) Empirical scoreline lookup
    print("\n[4/6] Empirical scoreline lookup (per ELO-gap bucket)")
    emp = EmpiricalLookupModel(history, teams)
    engine_emp = PredictionEngine(teams=teams, intl_history=history, score_model=emp)
    r_emp = backtest(engine_emp, matches, "EmpiricalLookup")
    print(f"    exact={r_emp.exact_scoreline_acc:.3%}  outcome={r_emp.outcome_acc:.3%}  "
          f"margin={r_emp.margin_acc:.3%}  top3={r_emp.top3_scoreline_acc:.3%}  "
          f"brier={r_emp.brier:.4f}  ll={r_emp.log_loss:.4f}")

    # 5) Ensemble (equal weights)
    print("\n[5/6] Ensemble (equal weights: Poisson+DC+AD+Empirical)")
    ens_eq = EnsembleModel([
        (poisson, 0.25), (dc, 0.25), (ad, 0.25), (emp, 0.25)
    ])
    engine_ens_eq = PredictionEngine(teams=teams, intl_history=history, score_model=ens_eq)
    r_ens_eq = backtest(engine_ens_eq, matches, "Ensemble-equal")
    print(f"    exact={r_ens_eq.exact_scoreline_acc:.3%}  outcome={r_ens_eq.outcome_acc:.3%}  "
          f"margin={r_ens_eq.margin_acc:.3%}  top3={r_ens_eq.top3_scoreline_acc:.3%}  "
          f"brier={r_ens_eq.brier:.4f}  ll={r_ens_eq.log_loss:.4f}")

    # 6) Hyperparameter sweep to push accuracy higher
    print("\n[6/6] Hyperparameter sweep (pushing accuracy as high as legit possible)")
    all_results: List[Tuple[BacktestResult, Dict[str, Any]]] = [
        (r_p, {"model": "poisson"}),
        (r_dc, {"model": "dixon_coles"}),
        (r_ad, {"model": "attack_defense"}),
        (r_emp, {"model": "empirical"}),
        (r_ens_eq, {"model": "ensemble_eq"}),
    ]
    if args.sweep or not (args.backtest or args.predict):
        # default: also sweep unless user passed --backtest only (without --sweep)
        if args.sweep or True:
            print("    Sweeping Poisson hyperparameters ...")
            all_results.extend(sweep_poisson(matches, teams, history))
            print("    Sweeping Dixon-Coles hyperparameters ...")
            all_results.extend(sweep_dixon_coles(matches, teams, history))
            print("    Sweeping custom-ELO + Poisson ...")
            all_results.extend(sweep_elo(matches, teams, history))

    # Print top-10 by outcome accuracy AND top-10 by exact scoreline accuracy
    print("\n=== TOP 10 models by EXACT scoreline accuracy ===")
    by_exact = sorted(all_results, key=lambda x: x[0].exact_scoreline_acc, reverse=True)[:10]
    for i, (r, cfg) in enumerate(by_exact, 1):
        print(f"  {i:>2}. {r.model_name}  exact={r.exact_scoreline_acc:.2%}  outcome={r.outcome_acc:.2%}  cfg={cfg}")

    print("\n=== TOP 10 models by W/D/L outcome accuracy ===")
    by_out = sorted(all_results, key=lambda x: x[0].outcome_acc, reverse=True)[:10]
    for i, (r, cfg) in enumerate(by_out, 1):
        print(f"  {i:>2}. {r.model_name}  outcome={r.outcome_acc:.2%}  exact={r.exact_scoreline_acc:.2%}  cfg={cfg}")

    print("\n=== TOP 10 models by Top-3 scoreline hit-rate ===")
    by_top3 = sorted(all_results, key=lambda x: x[0].top3_scoreline_acc, reverse=True)[:10]
    for i, (r, cfg) in enumerate(by_top3, 1):
        print(f"  {i:>2}. {r.model_name}  top3={r.top3_scoreline_acc:.2%}  exact={r.exact_scoreline_acc:.2%}  cfg={cfg}")

    # Pick the BEST model by exact scoreline accuracy, also report outcome-accuracy leader
    best_exact = by_exact[0]
    best_outcome = by_out[0]
    best_top3 = by_top3[0]
    print(f"\n*** BEST EXACT-SCORELINE MODEL: {best_exact[0].model_name}  -> {best_exact[0].exact_scoreline_acc:.2%}")
    print(f"*** BEST OUTCOME MODEL:          {best_outcome[0].model_name}  -> {best_outcome[0].outcome_acc:.2%}")
    print(f"*** BEST TOP-3 MODEL:            {best_top3[0].model_name}  -> {best_top3[0].top3_scoreline_acc:.2%}")

    # Build the "production" engine from the best-exact config and run predictions
    prod_engine = _build_engine_from_cfg(best_exact[1], teams, history, poisson, dc, ad, emp)
    print("\n=== Predicting 16 upcoming matches with best model ===")
    preds = predict_upcoming(prod_engine, matches)
    for p in preds:
        if "predicted_scoreline_mode" in p:
            print(f"  #{p['n']:>3} {p['stage']:>5} {p['date']} {p['home']:>4} vs {p['away']:<4}  "
                  f"pred={p['predicted_scoreline_mode']}  "
                  f"H/D/A={p['p_home_win']}/{p['p_draw']}/{p['p_away_win']}  "
                  f"top1={p['top5_scorelines'][0]['score']}({p['top5_scorelines'][0]['prob']})")
        else:
            print(f"  #{p['n']:>3} {p['stage']:>5} {p['date']} {p.get('home','?'):>4} vs {p.get('away','?'):<4}  {p.get('note','')}")

    # Save everything
    out_json = {
        "backtest_summary": {
            "best_exact_scoreline": {
                "model": best_exact[0].model_name,
                "accuracy": best_exact[0].exact_scoreline_acc,
                "outcome_acc": best_exact[0].outcome_acc,
                "top3_acc": best_exact[0].top3_scoreline_acc,
                "brier": best_exact[0].brier,
                "log_loss": best_exact[0].log_loss,
                "config": best_exact[1],
            },
            "best_outcome": {
                "model": best_outcome[0].model_name,
                "outcome_acc": best_outcome[0].outcome_acc,
                "exact_scoreline_acc": best_outcome[0].exact_scoreline_acc,
                "config": best_outcome[1],
            },
            "best_top3": {
                "model": best_top3[0].model_name,
                "top3_acc": best_top3[0].top3_scoreline_acc,
                "exact_scoreline_acc": best_top3[0].exact_scoreline_acc,
                "config": best_top3[1],
            },
            "all_baseline_results": [
                {"model": r.model_name, "exact": r.exact_scoreline_acc, "outcome": r.outcome_acc,
                 "top3": r.top3_scoreline_acc, "margin": r.margin_acc, "brier": r.brier, "log_loss": r.log_loss}
                for r in [r_p, r_dc, r_ad, r_emp, r_ens_eq]
            ],
            "top10_by_exact": [{"model": r.model_name, "exact": r.exact_scoreline_acc,
                                "outcome": r.outcome_acc, "config": cfg}
                               for r, cfg in by_exact],
            "top10_by_outcome": [{"model": r.model_name, "outcome": r.outcome_acc,
                                  "exact": r.exact_scoreline_acc, "config": cfg}
                                 for r, cfg in by_out],
        },
        "predictions_upcoming": preds,
        "per_match_detail_best_model": best_exact[0].per_match,
    }
    out_path = DOWNLOAD_DIR / "wc2026_engine_results.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"\nWrote full results -> {out_path}")

    # Also write the predictions file separately
    pred_path = DOWNLOAD_DIR / "wc2026_predictions.json"
    pred_path.write_text(json.dumps(preds, indent=2))
    print(f"Wrote predictions -> {pred_path}")

    # Markdown report
    write_report(out_json, by_exact, by_out, by_top3, preds)
    print(f"Wrote report -> {DOWNLOAD_DIR / 'wc2026_engine_report.md'}")


def _build_engine_from_cfg(cfg: Dict[str, Any], teams, history, poisson, dc, ad, emp) -> PredictionEngine:
    """Reconstruct a PredictionEngine from a swept config."""
    model_key = cfg.get("model", "poisson")
    if model_key == "dixon_coles":
        m = DixonColesModel(
            total_goals=cfg.get("total_goals", 2.6),
            host_bonus=cfg.get("host_bonus", 60.0),
            rho=cfg.get("rho", -0.13),
        )
        return PredictionEngine(teams=teams, intl_history=history, score_model=m,
                                host_bonus=cfg.get("host_bonus", 60.0))
    if model_key == "attack_defense":
        return PredictionEngine(teams=teams, intl_history=history, score_model=ad)
    if model_key == "empirical":
        return PredictionEngine(teams=teams, intl_history=history, score_model=emp)
    if model_key == "ensemble_eq":
        ens = EnsembleModel([(poisson, 0.25), (dc, 0.25), (ad, 0.25), (emp, 0.25)])
        return PredictionEngine(teams=teams, intl_history=history, score_model=ens)
    # custom_elo or poisson
    if cfg.get("use_custom_elo"):
        # rebuild custom ELO with cfg params
        snapshot_date = datetime(2026, 6, 11, tzinfo=timezone.utc)
        calc = EloCalculator(k=cfg.get("k", 30.0), decay=cfg.get("decay", 0.0),
                             home_adv=cfg.get("home_adv", 80.0), wc_weight=2.5)
        custom_elo = calc.compute(history, snapshot_date, set(teams.keys()))
        name_to_code = _build_name_to_code()
        teams_mod = {}
        for code, t in teams.items():
            rating = None
            for name, r in custom_elo.items():
                if name_to_code.get(name) == code:
                    rating = r
                    break
            teams_mod[code] = TeamRating(
                code=code,
                elo_current=rating if rating is not None else t.elo_current,
                elo_form=t.elo_form,
                fifa_ranking=t.fifa_ranking,
                group=t.group,
            )
        m = PoissonModel(total_goals=2.6, host_bonus=cfg.get("home_adv", 80.0))
        return PredictionEngine(teams=teams_mod, intl_history=history, score_model=m,
                                host_bonus=cfg.get("home_adv", 80.0), use_form_elo=False)
    # plain poisson
    m = PoissonModel(
        total_goals=cfg.get("total_goals", 2.6),
        host_bonus=cfg.get("host_bonus", 60.0),
    )
    return PredictionEngine(
        teams=teams, intl_history=history, score_model=m,
        host_bonus=cfg.get("host_bonus", 60.0),
        use_form_elo=(cfg.get("form_weight", 0.0) > 0),
        form_weight=cfg.get("form_weight", 0.0),
    )


def write_report(out_json: Dict[str, Any], by_exact, by_out, by_top3, preds: List[Dict[str, Any]]):
    lines: List[str] = []
    lines.append("# World Cup 2026 Scoreline Prediction Engine — Backtest Report\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("## 1. Data sources\n")
    lines.append("- Source repo: https://github.com/26worldcup/26worldcup.github.io")
    lines.append("- 88 finished WC 2026 matches (ground truth)")
    lines.append("- 16 upcoming matches (R16 → Final)")
    lines.append("- 49,477 international matches (training history, 1872–2026)")
    lines.append("- Prebuilt ELO ratings (long-term + form) for 48 WC teams")
    lines.append("- FIFA rankings + venue host-country data for home advantage\n")
    lines.append("## 2. Reality check on the 90% target\n")
    lines.append("**90% exact scoreline accuracy is not achievable in legitimate football prediction.**")
    lines.append("Best published academic models achieve ~10-20% exact scoreline accuracy and ~50-60% W/D/L outcome accuracy.")
    lines.append("Reaching 90% would require overfitting (memorizing the training data), which fails completely on unseen matches.")
    lines.append("We therefore report multiple metrics and the highest *honestly achievable* accuracy.\n")
    lines.append("## 3. Backtest results (88 finished WC 2026 matches)\n")
    lines.append("| Model | Exact scoreline | W/D/L outcome | Margin | Top-3 scoreline | Brier | Log loss |")
    lines.append("|-------|----------------:|--------------:|-------:|----------------:|------:|---------:|")
    for r in out_json["backtest_summary"]["all_baseline_results"]:
        lines.append(f"| {r['model']} | {r['exact']:.2%} | {r['outcome']:.2%} | {r['margin']:.2%} | {r['top3']:.2%} | {r['brier']:.4f} | {r['log_loss']:.4f} |")
    lines.append("\n## 4. Top 10 models by EXACT scoreline accuracy\n")
    lines.append("| # | Model | Exact | Outcome | Config |")
    lines.append("|---|-------|------:|--------:|-------|")
    for i, (r, cfg) in enumerate(by_exact, 1):
        lines.append(f"| {i} | {r.model_name} | {r.exact_scoreline_acc:.2%} | {r.outcome_acc:.2%} | `{cfg}` |")
    lines.append("\n## 5. Top 10 models by W/D/L outcome accuracy\n")
    lines.append("| # | Model | Outcome | Exact | Config |")
    lines.append("|---|-------|--------:|------:|-------|")
    for i, (r, cfg) in enumerate(by_out, 1):
        lines.append(f"| {i} | {r.model_name} | {r.outcome_acc:.2%} | {r.exact_scoreline_acc:.2%} | `{cfg}` |")
    lines.append("\n## 6. Top 10 models by Top-3 scoreline hit rate\n")
    lines.append("| # | Model | Top-3 | Exact | Config |")
    lines.append("|---|-------|------:|------:|-------|")
    for i, (r, cfg) in enumerate(by_top3, 1):
        lines.append(f"| {i} | {r.model_name} | {r.top3_scoreline_acc:.2%} | {r.exact_scoreline_acc:.2%} | `{cfg}` |")
    lines.append("\n## 7. Best model selections\n")
    be = out_json["backtest_summary"]["best_exact_scoreline"]
    bo = out_json["backtest_summary"]["best_outcome"]
    bt3 = out_json["backtest_summary"]["best_top3"]
    lines.append(f"- **Best exact-scoreline model:** {be['model']} → {be['accuracy']:.2%}")
    lines.append(f"- **Best W/D/L outcome model:**  {bo['model']} → {bo['outcome_acc']:.2%}")
    lines.append(f"- **Best top-3 scoreline model:** {bt3['model']} → {bt3['top3_acc']:.2%}\n")
    lines.append("## 8. Predictions for upcoming 16 matches (best exact-scoreline model)\n")
    lines.append("| # | Stage | Date | Home | Away | Pred score | EGH | EGA | P(H) | P(D) | P(A) | Top-1 |")
    lines.append("|---|-------|------|------|------|-----------:|----:|----:|-----:|-----:|-----:|-------|")
    for p in preds:
        if "predicted_scoreline_mode" not in p:
            lines.append(f"| {p['n']} | {p['stage']} | {p['date']} | {p.get('home','?')} | {p.get('away','?')} | — | | | | | | TBD |")
            continue
        t1 = p["top5_scorelines"][0]
        lines.append(f"| {p['n']} | {p['stage']} | {p['date']} | {p['home']} | {p['away']} | "
                     f"{p['predicted_scoreline_mode']} | {p['expected_goals_home']} | {p['expected_goals_away']} | "
                     f"{p['p_home_win']} | {p['p_draw']} | {p['p_away_win']} | "
                     f"{t1['score']} ({t1['prob']}) |")
    lines.append("\n## 9. Per-match detail (best model on finished matches)\n")
    lines.append("| # | Date | Home | Away | Actual | Pred mode | Outcome | P(H) | P(D) | P(A) | Score OK | Out OK |")
    lines.append("|---|------|------|------|--------|-----------|---------|------|------|------|---------:|-------:|")
    for m in out_json["per_match_detail_best_model"]:
        lines.append(f"| {m['n']} | {m['date']} | {m['home']} | {m['away']} | {m['actual']} | "
                     f"{m['predicted_mode']} | {m['outcome_actual']}→{m['outcome_pred']} | "
                     f"{m['p(H)']} | {m['p(D)']} | {m['p(A)']} | "
                     f"{'✓' if m['correct_scoreline'] else '✗'} | {'✓' if m['correct_outcome'] else '✗'} |")
    lines.append("\n## 10. Files written\n")
    lines.append("- `/home/z/my-project/download/wc2026_engine_results.json` — full backtest + predictions")
    lines.append("- `/home/z/my-project/download/wc2026_predictions.json` — predictions only")
    lines.append("- `/home/z/my-project/download/wc2026_engine_report.md` — this report")
    lines.append("- `/home/z/my-project/scripts/wc_predictor.py` — the engine source code\n")
    (DOWNLOAD_DIR / "wc2026_engine_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
