"""Preseason win total projection model (Joe Peta methodology)."""

from __future__ import annotations

import pandas as pd

from src.models.pythagorean import pythagorean_win_pct

SEASON_GAMES = 162
MIN_EDGE_WINS = 2.0  # Flag as bet when projection diverges from Vegas by this many wins


def project_team_wins(
    prior_runs_scored: float,
    prior_runs_allowed: float,
    prior_games: int = 162,
    war_adjustment: float = 0.0,
) -> float:
    """
    Project a team's win total for the coming season using Peta's methodology:
    1. Compute prior season Pythagorean win %
    2. Apply small WAR-based roster adjustment
    3. Apply mild regression toward .500 (all teams drift toward mean)
    4. Scale to 162 games

    Parameters
    ----------
    prior_runs_scored : float
        Team's total runs scored in prior season
    prior_runs_allowed : float
        Team's total runs allowed in prior season
    prior_games : int
        Number of games played in prior season (handles shortened seasons)
    war_adjustment : float
        Net WAR change from roster moves (positive = better, negative = worse)
        Roughly 2 WAR ≈ 2 additional wins over a season
    """
    pyth_pct = pythagorean_win_pct(prior_runs_scored, prior_runs_allowed)

    # Regress 20% toward .500 — teams don't fully repeat prior performance
    regressed_pct = pyth_pct * 0.80 + 0.500 * 0.20

    # WAR adjustment: each full WAR ≈ 1 additional win over 162 games
    war_win_adj = war_adjustment / SEASON_GAMES * SEASON_GAMES

    projected_wins = regressed_pct * SEASON_GAMES + war_win_adj
    return round(projected_wins, 1)


def compute_preseason_projections(
    prior_team_stats: pd.DataFrame,
    vegas_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute preseason win projections for all teams and compare to Vegas lines.

    Parameters
    ----------
    prior_team_stats : pd.DataFrame
        Prior season team stats. Required columns: team, runs_scored, runs_allowed, wins
        Optional: war_adjustment (net WAR change from offseason moves)
    vegas_lines : pd.DataFrame | None
        Vegas win total O/U lines. Required columns: team, vegas_total
        If None, projection table is returned without edge calculation.

    Returns
    -------
    DataFrame with columns:
        team, prior_wins, prior_pyth_pct, prior_run_diff,
        projected_wins, vegas_total (if provided),
        edge_wins, bet_direction, confidence, signal_strength
    """
    df = prior_team_stats.copy()
    results = []

    for _, row in df.iterrows():
        team = row["team"]
        rs = row.get("runs_scored", 0)
        ra = row.get("runs_allowed", 0)
        prior_wins = row.get("wins", 0)
        war_adj = row.get("war_adjustment", 0.0)

        pyth_pct = pythagorean_win_pct(rs, ra)
        projected = project_team_wins(rs, ra, war_adjustment=war_adj)
        prior_run_diff = int(rs - ra) if rs and ra else 0

        result = {
            "team": team,
            "prior_wins": prior_wins,
            "prior_pyth_pct": round(pyth_pct, 4),
            "prior_pyth_wins": round(pyth_pct * SEASON_GAMES, 1),
            "prior_run_diff": prior_run_diff,
            "projected_wins": projected,
        }
        results.append(result)

    out = pd.DataFrame(results)

    if vegas_lines is not None and not vegas_lines.empty:
        out = out.merge(vegas_lines[["team", "vegas_total"]], on="team", how="left")
        out["edge_wins"] = (out["projected_wins"] - out["vegas_total"]).round(1)

        def _direction(edge):
            if edge >= MIN_EDGE_WINS:
                return "OVER"
            if edge <= -MIN_EDGE_WINS:
                return "UNDER"
            return "PASS"

        def _strength(edge):
            abs_edge = abs(edge)
            if abs_edge >= 5:
                return "High"
            if abs_edge >= MIN_EDGE_WINS:
                return "Medium"
            return "Low"

        out["bet_direction"] = out["edge_wins"].apply(_direction)
        out["signal_strength"] = out["edge_wins"].apply(_strength)
    else:
        out["vegas_total"] = None
        out["edge_wins"] = None
        out["bet_direction"] = "N/A"
        out["signal_strength"] = "N/A"

    return out.sort_values("projected_wins", ascending=False).reset_index(drop=True)
