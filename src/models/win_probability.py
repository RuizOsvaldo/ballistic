"""Composite win probability model combining Pythagorean base + pitcher FIP + home field."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pythagorean import pythagorean_win_pct

HOME_FIELD_ADJ = 0.04      # ~4% home field advantage
FIP_ADJ_PER_POINT = 0.03   # ~3% win prob per FIP point difference from league average
MIN_WIN_PROB = 0.05
MAX_WIN_PROB = 0.95


def _clamp(value: float, lo: float = MIN_WIN_PROB, hi: float = MAX_WIN_PROB) -> float:
    return max(lo, min(hi, value))


def compute_league_avg_fip(pitcher_df: pd.DataFrame) -> float:
    """Compute IP-weighted league average FIP."""
    if pitcher_df.empty or "fip" not in pitcher_df.columns:
        return 4.00  # fallback
    if "ip" in pitcher_df.columns:
        total_ip = pitcher_df["ip"].sum()
        if total_ip > 0:
            return (pitcher_df["fip"] * pitcher_df["ip"]).sum() / total_ip
    return pitcher_df["fip"].mean()


def game_win_probability(
    home_rs: float,
    home_ra: float,
    away_rs: float,
    away_ra: float,
    home_starter_fip: float | None,
    away_starter_fip: float | None,
    league_avg_fip: float,
) -> tuple[float, float]:
    """
    Compute (home_win_prob, away_win_prob) for a single game.

    Steps:
    1. Average each team's Pythagorean W% as a base
    2. Adjust for starting pitcher FIP vs. league average
    3. Add home field advantage
    4. Normalise to sum to 1.0
    """
    home_base = pythagorean_win_pct(home_rs, home_ra)
    away_base = pythagorean_win_pct(away_rs, away_ra)

    # Pitcher adjustments (better FIP = better for that team's chance of winning)
    home_pitch_adj = 0.0
    if home_starter_fip is not None:
        home_pitch_adj = (league_avg_fip - home_starter_fip) * FIP_ADJ_PER_POINT

    away_pitch_adj = 0.0
    if away_starter_fip is not None:
        away_pitch_adj = (league_avg_fip - away_starter_fip) * FIP_ADJ_PER_POINT

    home_prob = home_base + home_pitch_adj + HOME_FIELD_ADJ - away_pitch_adj
    away_prob = away_base + away_pitch_adj - home_pitch_adj

    # Normalise
    total = home_prob + away_prob
    if total > 0:
        home_prob /= total
        away_prob /= total

    return _clamp(home_prob), _clamp(away_prob)


def compute_win_probabilities(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute model win probabilities for a DataFrame of games.

    games_df required columns: home_team, away_team
    Optional columns: home_starter, away_starter (pitcher names)

    Returns games_df with added columns:
      home_model_prob, away_model_prob
    """
    df = games_df.copy()
    league_avg_fip = compute_league_avg_fip(pitcher_df)

    # Build lookup dicts
    team_stats = team_stats_df.set_index("team")
    pitcher_stats = {}
    if not pitcher_df.empty and "name" in pitcher_df.columns:
        pitcher_stats = pitcher_df.set_index("name")["fip"].to_dict()

    home_probs, away_probs = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        try:
            h = team_stats.loc[home]
            a = team_stats.loc[away]
            home_rs, home_ra = h["runs_scored"], h["runs_allowed"]
            away_rs, away_ra = a["runs_scored"], a["runs_allowed"]
        except KeyError:
            home_probs.append(np.nan)
            away_probs.append(np.nan)
            continue

        home_fip = pitcher_stats.get(row.get("home_starter")) if "home_starter" in row else None
        away_fip = pitcher_stats.get(row.get("away_starter")) if "away_starter" in row else None

        hp, ap = game_win_probability(
            home_rs, home_ra, away_rs, away_ra,
            home_fip, away_fip, league_avg_fip,
        )
        home_probs.append(round(hp, 4))
        away_probs.append(round(ap, 4))

    df["home_model_prob"] = home_probs
    df["away_model_prob"] = away_probs
    return df
