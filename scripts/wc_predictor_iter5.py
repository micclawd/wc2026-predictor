#!/usr/bin/env python3
"""
WC 2026 — iter-5 retrain after R16.

Reuses iter-4e TargetedDecisionRuleModel. Backtests on 16 R32 + 8 R16 = 24 matches.
Grid-searches same hyperparameters as iter-4e (5 dims × ~5 values = ~3000 configs).

Picks config with best exact_scoreline_acc on 24 matches.
Writes download/wc2026_iter5_results.json.

Usage:
    python3 wc_predictor_iter5.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DOWNLOAD_DIR
from wc_predictor import (
    HOST_BONUS_DEFAULT, Match, TeamRating, load_matches, load_teams,
)
from wc_predictor_iter4b import DecisionRuleEngine
from wc_predictor_iter4e import TargetedDecisionRuleModel
from wc_predictor_iter6 import adjust_elo_with_lineups
from wc_predictor_iter4 import load_squad_features
from wc_r16_improvement import form_adjusted_elo


@dataclass
class Iter5Result:
    config: Dict[str, Any]
    exact_scoreline_acc: float
    top3_scoreline_acc: float
    outcome_acc: float
    brier: float
    log_loss: float

    def to_dict(self):
        return {
            "config": self.config,
            "exact_scoreline_acc": self.exact_scoreline_acc,
            "top3_scoreline_acc": self.top3_scoreline_acc,
            "outcome_acc": self.outcome_acc,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }


def backtest_one_config(
    matches: List[Match],
    teams_seed: Dict[str, TeamRating],
    squads,
    config: Dict[str, Any],
) -> Iter5Result:
    """Run a single config through 24 backtest matches."""
    finished = [m for m in matches if m.status == "finished"
                and m.home_score is not None and m.away_score is not None
                and m.home_code and m.away_code]
    if not finished:
        return Iter5Result(config, 0, 0, 0, 1, 1)

    # Apply form-adjusted ELO once (config-independent)
    teams_form = form_adjusted_elo(matches, teams_seed,
                                    k_form=config.get("k_form", 25.0),
                                    gd_multiplier=config.get("gd_multiplier", 0.3),
                                    blend=config.get("blend", 0.7))

    # Apply lineup adjustment with given star_weight
    teams_adj, _ = adjust_elo_with_lineups(teams_form, squads, {}, star_weight=config["star_weight"])

    # Build engine
    model = TargetedDecisionRuleModel(
        big_fav=config["big_fav"],
        strong_fav=config["strong_fav"],
        slight_fav=config["slight_fav"],
        low_star_threshold=config["low_star_threshold"],
        enable_low_star_fix=True,
        enable_high_underdog_concession=True,
        high_underdog_threshold=config["high_underdog_threshold"],
    )
    engine = DecisionRuleEngine(
        teams=teams_adj, intl_history={},
        score_model=model, host_bonus=80.0,
        use_form_elo=True, form_weight=0.5,
        squads=squads,
    )

    exact = 0
    top3 = 0
    outcome = 0
    brier_sum = 0.0
    ll_sum = 0.0
    n = 0

    for m in finished:
        try:
            dist = engine.predict(m.home_code, m.away_code, m.venue_country)
        except Exception:
            continue
        actual_score = f"{m.home_score}-{m.away_score}"
        # Outcome: H/D/A
        if m.home_score > m.away_score:
            actual_outcome = "H"
        elif m.home_score < m.away_score:
            actual_outcome = "A"
        else:
            actual_outcome = "D"
        # Top scoreline
        top_score = max(dist.probs.items(), key=lambda x: x[1])[0]
        if top_score == actual_score:
            exact += 1
        # Top-3
        top3_scores = [s for s, _ in sorted(dist.probs.items(), key=lambda x: -x[1])[:3]]
        if actual_score in top3_scores:
            top3 += 1
        # Outcome
        h_prob = sum(p for s, p in dist.probs.items() if int(s.split("-")[0]) > int(s.split("-")[1]))
        a_prob = sum(p for s, p in dist.probs.items() if int(s.split("-")[0]) < int(s.split("-")[1]))
        d_prob = sum(p for s, p in dist.probs.items() if int(s.split("-")[0]) == int(s.split("-")[1]))
        probs_norm = {"H": h_prob, "D": d_prob, "A": a_prob}
        pred_outcome = max(probs_norm.items(), key=lambda x: x[1])[0]
        if pred_outcome == actual_outcome:
            outcome += 1
        # Brier
        actual_oh = {"H": 1, "D": 0, "A": 0}[actual_outcome]
        brier_sum += (h_prob - actual_oh) ** 2 + (d_prob - 0) ** 2 + (a_prob - (1 - actual_oh)) ** 2
        # Log loss
        p_actual = max(probs_norm[actual_outcome], 1e-9)
        ll_sum += -np.log(p_actual)
        n += 1

    if n == 0:
        return Iter5Result(config, 0, 0, 0, 1, 1)

    return Iter5Result(
        config=config,
        exact_scoreline_acc=exact / n,
        top3_scoreline_acc=top3 / n,
        outcome_acc=outcome / n,
        brier=brier_sum / n,
        log_loss=ll_sum / n,
    )


def grid_search(matches, teams, squads) -> Iter5Result:
    """Grid search the iter-4e hyperparameter space."""
    grid = {
        "big_fav": [200, 250, 300, 350, 400],
        "strong_fav": [100, 150, 200, 250],
        "slight_fav": [25, 50, 75, 100],
        "low_star_threshold": [0.3, 0.4, 0.5, 0.6],
        "high_underdog_threshold": [0.3, 0.4, 0.5, 0.6],
        "star_weight": [20, 30, 40, 50, 60],
    }
    keys = list(grid.keys())
    values = list(grid.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"Grid search: {total} configs")

    best: Optional[Iter5Result] = None
    results: List[Iter5Result] = []
    for i, combo in enumerate(itertools.product(*values)):
        if i % 200 == 0:
            print(f"  [{i}/{total}] ...")
        config = dict(zip(keys, combo))
        try:
            r = backtest_one_config(matches, teams, squads, config)
        except Exception as e:
            print(f"  config {config} failed: {e}")
            continue
        results.append(r)
        if best is None or r.exact_scoreline_acc > best.exact_scoreline_acc:
            best = r
            print(f"  [{i}/{total}] new best: {r.exact_scoreline_acc*100:.1f}% exact "
                  f"(top3 {r.top3_scoreline_acc*100:.1f}%, outcome {r.outcome_acc*100:.1f}%)")

    print(f"\nBest config: {best.exact_scoreline_acc*100:.1f}% exact")
    return best, results


def main():
    print("=" * 70)
    print("WC 2026 — iter-5 retrain (R16 complete, 24 matches available)")
    print("=" * 70)

    print("\n[1/3] Loading data ...")
    matches = load_matches()
    teams = load_teams()
    squads = load_squad_features()
    n_finished = sum(1 for m in matches if m.status == "finished")
    print(f"  Matches: {len(matches)} ({n_finished} finished)")

    if n_finished < 24:
        print(f"  ⚠ Only {n_finished} finished matches; expected 24 (16 R32 + 8 R16)")
        print(f"  Proceeding with available matches; results may be noisy")

    print("\n[2/3] Grid search (~3000 configs) ...")
    best, all_results = grid_search(matches, teams, squads)

    print("\n[3/3] Writing results ...")
    out = {
        "model_version": "v1.5.0",
        "n_matches": n_finished,
        "best": best.to_dict(),
        "top_10": sorted([r.to_dict() for r in all_results],
                         key=lambda r: -r["exact_scoreline_acc"])[:10],
    }
    out_path = DOWNLOAD_DIR / "wc2026_iter5_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ Wrote {out_path}")
    print(f"\n=== iter-5 best ===")
    print(f"  config: {best.config}")
    print(f"  exact_scoreline_acc: {best.exact_scoreline_acc*100:.1f}%")
    print(f"  top3_scoreline_acc:  {best.top3_scoreline_acc*100:.1f}%")
    print(f"  outcome_acc:         {best.outcome_acc*100:.1f}%")
    print(f"  brier:               {best.brier:.3f}")
    print(f"  log_loss:            {best.log_loss:.3f}")


if __name__ == "__main__":
    main()
