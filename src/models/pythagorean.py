"""Pythagorean win expectation model (Bill James)."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPONENT = 2.0  # Classic Bill James exponent (Pythagorean); 1.83 is empirically better but 2 is canonical


def pythagorean_win_pct(runs_scored: float, runs_allowed: float, exponent: float = EXPONENT) -> float:
    """
    Compute expected win percentage from runs scored and allowed.
    Formula: RS^exp / (RS^exp + RA^exp)
    """
    if runs_allowed == 0:
        return 1.0
    rs_e = runs_scored ** exponent
    ra_e = runs_allowed ** exponent
    return rs_e / (rs_e + ra_e)


def compute_pythagorean(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add Pythagorean columns to a team stats DataFrame.

    Input columns required: runs_scored, runs_allowed, win_pct
    Added columns:
      - pyth_win_pct   : expected W% from RS/RA
      - pyth_deviation : actual_win_pct - pyth_win_pct
      - pyth_signal    : "Overperforming" | "Underperforming" | "On Track"
    """
    df = team_stats.copy()

    df["pyth_win_pct"] = df.apply(
        lambda r: pythagorean_win_pct(r["runs_scored"], r["runs_allowed"]),
        axis=1,
    )
    df["pyth_deviation"] = (df["win_pct"] - df["pyth_win_pct"]).round(4)

    def _signal(dev: float) -> str:
        if dev > 0.05:
            return "Overperforming"
        if dev < -0.05:
            return "Underperforming"
        return "On Track"

    df["pyth_signal"] = df["pyth_deviation"].apply(_signal)
    df["pyth_win_pct"] = df["pyth_win_pct"].round(4)
    return df
