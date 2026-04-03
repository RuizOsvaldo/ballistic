"""
Nightly results verification job — runs at midnight PST (8am UTC).

Fetches yesterday's final scores from the MLB Stats API and reconciles them
against stored predictions, marking each as correct or incorrect.

Usage:
    venv/bin/python3 -m src.jobs.verify_results
"""

from __future__ import annotations

import datetime
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")


def run() -> None:
    now = datetime.datetime.now()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    print(f"[{now:%Y-%m-%d %H:%M}] Starting results verification for {yesterday}...")

    try:
        from src.data.predictions_db import load_predictions
        from src.data.game_results import verify_predictions

        predictions = load_predictions(days=7)
        if predictions.empty:
            print("  No predictions found in the last 7 days. Nothing to verify.")
            return

        unverified = predictions[predictions["actual_winner"].isna()]
        print(f"  Predictions to verify: {len(unverified)} (of {len(predictions)} total)")

        if unverified.empty:
            print("  All predictions already verified.")
            return

        verify_predictions(predictions)
        print(f"  Done — results written to data/bet_log.db")

    except Exception:
        print(f"  ERROR:\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    run()
