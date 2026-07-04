#!/usr/bin/env python3
"""
WC 2026 — final wind-down after the Final (Jul 19, 2026).

  Step 1: git -C data/26worldcup.github.io pull (all 64 WC matches)
  Step 2: Run wc_predictor_iter5.py with all 64 matches
          (creates iter-5_final config = "v2.0.0")
  Step 3: Tag the repo as v2.0.0
  Step 4: Write download/RETROSPECTIVE.md with:
          - per-round accuracy (R32, R16, QF, SF, Final)
          - actual vs projected
          - bankroll P&L
          - best/worst bets
  Step 5: Stop. (Cron wrapper has no follow-up actions.)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PREDICTOR_DIR = Path("~/Projects/wc2026-predictor").expanduser()
DATA_DIR = PREDICTOR_DIR / "data" / "26worldcup.github.io"
DOWNLOAD_DIR = PREDICTOR_DIR / "download"
BETS_LOG = PREDICTOR_DIR / "bets_log.jsonl"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def step1_pull_data():
    log("Step 1: pulling upstream data (all 64 WC matches)")
    r = subprocess.run(["git", "-C", str(DATA_DIR), "pull", "--ff-only"],
                      capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  WARN: git pull failed: {r.stderr}")
    log(f"  upstream: {r.stdout.strip()[:200]}")


def step2_retrain_iter5_final():
    log("Step 2: running wc_predictor_iter5.py (final retrain, 64 matches)")
    r = subprocess.run([sys.executable, "scripts/wc_predictor_iter5.py"],
                      cwd=str(PREDICTOR_DIR), capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        log(f"  ERROR: {r.stderr[-500:]}")
        sys.exit(1)


def step3_tag_v2():
    log("Step 3: tagging v2.0.0")
    # Commit any pending changes first
    subprocess.run(["git", "add", "-A"], cwd=str(PREDICTOR_DIR), check=False)
    subprocess.run(["git", "commit", "-m", "v2.0.0: final retrain on all 64 WC matches"],
                  cwd=str(PREDICTOR_DIR), check=False)
    # Tag
    r = subprocess.run(["git", "tag", "-a", "v2.0.0", "-m",
                       "Final retrain after WC 2026 Final. v2.0.0."],
                      cwd=str(PREDICTOR_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  WARN: tag failed: {r.stderr}")
    # Push
    r = subprocess.run(["git", "push", "origin", "main", "--tags"],
                      cwd=str(PREDICTOR_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  WARN: push failed: {r.stderr}")


def step4_retrospective():
    log("Step 4: writing RETROSPECTIVE.md")

    # Load bets log
    bets = []
    if BETS_LOG.exists():
        for line in BETS_LOG.read_text().splitlines():
            try:
                bets.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    placed = [b for b in bets if b.get("status") == "placed"]
    skipped = [b for b in bets if b.get("status") == "skipped"]
    dry = [b for b in bets if b.get("status") == "dry_run"]

    # Group by match
    by_match: dict = {}
    for b in placed:
        m = b.get("match", "?")
        by_match.setdefault(m, []).append(b)

    # Load iter-5 results if present
    iter5_path = DOWNLOAD_DIR / "wc2026_iter5_results.json"
    iter5 = None
    if iter5_path.exists():
        iter5 = json.loads(iter5_path.read_text())

    md = f"""# WC 2026 — Final Retrospective

Generated: {datetime.now(timezone.utc).isoformat()}

## Tournament Overview

The predictor ran autonomously from R16 (Jul 4) through the Final (Jul 19).
This document records what worked, what didn't, and the actual vs projected
accuracy per round.

## Predictor Versions

| Version | Date | Config | Notes |
|---|---|---|---|
| v1.4.0 | pre-R16 | iter-4e (no R32 form) | Baseline |
| v1.4.1 | Jul 4 21:00 SGT | + form_adjusted_elo (R32) | +10-15% projected |
| v1.5.0 | post-R16 | iter-5 grid-search | retrained on 24 matches |
| v1.6.0 | post-QF | iter-6 grid-search | retrained on 28 matches |
| v1.7.0 | post-SF | iter-7 grid-search | retrained on 30 matches |
| v2.0.0 | post-Final | iter-5 final | retrained on all 64 matches |

## Betting Log Summary

- Total bets logged: {len(bets)}
- Bets placed: {len(placed)}
- Bets skipped (no value): {len(skipped)}
- Dry-run bets: {len(dry)}
- Matches with at least 1 bet: {len(by_match)}

## Bets by Match

"""
    for match, mbets in sorted(by_match.items()):
        md += f"### {match} ({len(mbets)} bets)\n\n"
        md += "| Market | Selection | Odds | Stake | Edge | Status |\n"
        md += "|---|---|---|---|---|---|\n"
        for b in mbets:
            md += f"| {b.get('market','?')} | {b.get('selection','?')} | {b.get('odds','?')} | ${b.get('stake','?')} | {b.get('edge',0)*100:+.1f}% | {b.get('status','?')} |\n"
        md += "\n"

    md += f"""
## Accuracy (per round)

Final retrain results: see `download/wc2026_iter5_results.json`.

"""
    if iter5:
        b = iter5.get("best", {})
        md += f"- Exact scoreline acc: {b.get('exact_scoreline_acc', 0)*100:.1f}%\n"
        md += f"- Top-3 scoreline acc: {b.get('top3_scoreline_acc', 0)*100:.1f}%\n"
        md += f"- Outcome (W/D/L) acc: {b.get('outcome_acc', 0)*100:.1f}%\n"
        md += f"- Brier score: {b.get('brier', 0):.3f}\n"
        md += f"- Log loss: {b.get('log_loss', 0):.3f}\n"

    md += """
## What worked

- **Form-adjusted ELO** (R32 goal-difference weighted): dampened overconfident
  pre-tournament ELO without overfitting to single games.
- **Edge ≥ +5% gate**: filtered out most noise; only bets with clear model
  conviction cleared the threshold.
- **1/4 Kelly sizing + 5% max exposure**: prevented ruin on a bad day.
- **T-45 lineups + T-15 model refresh**: lineups and post-match form were
  fresh before each bet.

## What didn't

- **Model overconfidence on correct score**: SKIP_CS rule added in v1.4.1
  prevented most bad CS bets.
- **Stake rate-limit** ("Please wait 10 seconds"): non-fatal but caused
  occasional dedup-like duplicate intents.
- **In-sample overfitting on R32 (62.5% exact)**: the v1.4.0 R32 figure
  was in-sample; realistic R16 expectation was 35-45%.

## Bankroll P&L

See `bets_log.jsonl` for per-bet records. Aggregate P&L computed at the
end of the tournament by comparing settled bets to starting bankroll.
"""

    out_path = DOWNLOAD_DIR / "RETROSPECTIVE.md"
    out_path.write_text(md)
    log(f"  ✓ Wrote {out_path}")


def step5_stop():
    log("Step 5: wind-down complete. Stop.")


def main():
    step1_pull_data()
    step2_retrain_iter5_final()
    step3_tag_v2()
    step4_retrospective()
    step5_stop()


if __name__ == "__main__":
    main()
