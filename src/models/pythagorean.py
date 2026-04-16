"""Pythagorean win expectation model (Bill James)."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPONENT = 1.83  # Empirically derived exponent (Pythagenpat); more accurate than the classic 2.0


def log5_probability(team_a_wpct: float, team_b_wpct: float) -> float:
    """
    Bill James Log5 formula: probability that team A beats team B in a single game.

    Formula: P = (A - A*B) / (A + B - 2*A*B)
    Correctly handles equal teams (→ 0.5) and asymmetric matchups.
    Falls back to 0.5 on degenerate inputs (both 0 or both 1).
    """
    denom = team_a_wpct + team_b_wpct - 2.0 * team_a_wpct * team_b_wpct
    if denom == 0.0:
        return 0.5
    return (team_a_wpct - team_a_wpct * team_b_wpct) / denom


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


MIN_GAMES_FOR_REGRESSION = 20   # Don't apply shrinkage until a team has played this many games
SHRINKAGE_K = 30                # At G games played, weight = G / (G + K); at 30G = 50%, 81G = 73%


def regress_rs_ra(
    rs: float,
    ra: float,
    games: int,
    league_avg_rs_pg: float,
    league_avg_ra_pg: float,
) -> tuple[float, float]:
    """
    Shrink a team's RS/RA toward the league mean using a games-played weight.

    Returns adjusted (rs, ra) as season totals (not per-game) for use in
    pythagorean_win_pct.  Below MIN_GAMES_FOR_REGRESSION the raw values are
    returned unchanged — not enough data to trust the regression yet.
    """
    if games < MIN_GAMES_FOR_REGRESSION or games == 0:
        return rs, ra

    rs_pg = rs / games
    ra_pg = ra / games
    weight = games / (games + SHRINKAGE_K)

    adj_rs_pg = weight * rs_pg + (1.0 - weight) * league_avg_rs_pg
    adj_ra_pg = weight * ra_pg + (1.0 - weight) * league_avg_ra_pg

    # Return as totals so pythagorean_win_pct receives consistent input
    return adj_rs_pg * games, adj_ra_pg * games


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
