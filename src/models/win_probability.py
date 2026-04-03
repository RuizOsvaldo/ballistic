"""Composite win probability model combining Pythagorean base + pitcher FIP + home field."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pythagorean import pythagorean_win_pct
from src.data.ballpark import get_park_factor

HOME_FIELD_ADJ    = 0.04   # ~4% home field advantage
FIP_ADJ_PER_POINT = 0.03   # ~3% win prob per FIP point difference from league average
# Bullpen adjustment: each run of bullpen FIP above/below league avg shifts
# game total by ~0.15 runs (bullpen faces ~3 innings per game on average)
BULLPEN_IP_SHARE  = 3.0 / 9.0   # ~33% of innings pitched by bullpen
MIN_WIN_PROB = 0.30
MAX_WIN_PROB = 0.70


def _clamp(value: float, lo: float = MIN_WIN_PROB, hi: float = MAX_WIN_PROB) -> float:
    return max(lo, min(hi, value))


def compute_league_avg_fip(pitcher_df: pd.DataFrame) -> float:
    """Compute IP-weighted league average FIP."""
    if pitcher_df.empty or "fip" not in pitcher_df.columns:
        return 4.00
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
    1. Pythagorean W% from season RS/RA as base
    2. Adjust for starting pitcher FIP vs. league average
    3. Add home field advantage
    4. Normalise to sum to 1.0
    """
    home_base = pythagorean_win_pct(home_rs, home_ra)
    away_base = pythagorean_win_pct(away_rs, away_ra)

    home_pitch_adj = 0.0
    if home_starter_fip is not None:
        home_pitch_adj = (league_avg_fip - home_starter_fip) * FIP_ADJ_PER_POINT

    away_pitch_adj = 0.0
    if away_starter_fip is not None:
        away_pitch_adj = (league_avg_fip - away_starter_fip) * FIP_ADJ_PER_POINT

    home_prob = home_base + home_pitch_adj + HOME_FIELD_ADJ - away_pitch_adj
    away_prob = away_base + away_pitch_adj - home_pitch_adj

    total = home_prob + away_prob
    if total > 0:
        home_prob /= total
        away_prob /= total

    return _clamp(home_prob), _clamp(away_prob)


def compute_win_probabilities(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    bullpen_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute model win probabilities and projected game totals.

    Added columns: home_model_prob, away_model_prob, proj_total
    proj_total incorporates:
      - Season RS/RA per game for each team
      - Park factor for the home ballpark
      - Bullpen FIP adjustment (if bullpen_df provided)
    """
    df = games_df.copy()
    league_avg_fip = compute_league_avg_fip(pitcher_df)

    team_stats = team_stats_df.set_index("team")
    pitcher_stats: dict[str, float] = {}
    if not pitcher_df.empty and "name" in pitcher_df.columns:
        pitcher_stats = pitcher_df.set_index("name")["fip"].to_dict()

    # Build bullpen FIP lookup {team: bullpen_fip}
    league_avg_bullpen_fip = 4.20
    bullpen_fip: dict[str, float] = {}
    if bullpen_df is not None and not bullpen_df.empty and "bullpen_fip" in bullpen_df.columns:
        valid = bullpen_df.dropna(subset=["bullpen_fip"])
        if not valid.empty:
            league_avg_bullpen_fip = valid["bullpen_fip"].mean()
            bullpen_fip = valid.set_index("team")["bullpen_fip"].to_dict()

    home_probs, away_probs, proj_totals = [], [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        try:
            h = team_stats.loc[home]
            a = team_stats.loc[away]
            home_rs, home_ra = float(h["runs_scored"]), float(h["runs_allowed"])
            away_rs, away_ra = float(a["runs_scored"]), float(a["runs_allowed"])
        except KeyError:
            home_probs.append(np.nan)
            away_probs.append(np.nan)
            proj_totals.append(None)
            continue

        home_fip = pitcher_stats.get(row.get("home_starter")) if "home_starter" in row else None
        away_fip = pitcher_stats.get(row.get("away_starter")) if "away_starter" in row else None

        hp, ap = game_win_probability(
            home_rs, home_ra, away_rs, away_ra,
            home_fip, away_fip, league_avg_fip,
        )
        home_probs.append(round(hp, 4))
        away_probs.append(round(ap, 4))

        # ── Projected game total ──────────────────────────────────────────
        try:
            h_w = float(np.nan_to_num(h.get("wins", 0), nan=0.0))
            h_l = float(np.nan_to_num(h.get("losses", 0), nan=0.0))
            a_w = float(np.nan_to_num(a.get("wins", 0), nan=0.0))
            a_l = float(np.nan_to_num(a.get("losses", 0), nan=0.0))
            h_g = max(h_w + h_l, 1.0)
            a_g = max(a_w + a_l, 1.0)

            # Base: blend each team's offense rate with opponent's defense rate
            home_exp = (home_rs / h_g + away_ra / a_g) / 2
            away_exp = (away_rs / a_g + home_ra / h_g) / 2
            raw_total = home_exp + away_exp

            # Park factor adjustment (Coors adds runs, Petco/Miami reduce them)
            park_factor = get_park_factor(home)
            park_adj = raw_total * (park_factor - 1.0)

            # Bullpen FIP adjustment — each team's bullpen covers ~1/3 of innings
            # Better bullpen (lower FIP) → fewer runs; worse bullpen → more runs
            h_bp_fip = bullpen_fip.get(home, league_avg_bullpen_fip)
            a_bp_fip = bullpen_fip.get(away, league_avg_bullpen_fip)
            bullpen_adj = (
                (h_bp_fip - league_avg_bullpen_fip) * BULLPEN_IP_SHARE * 0.5
                + (a_bp_fip - league_avg_bullpen_fip) * BULLPEN_IP_SHARE * 0.5
            )

            pt = round(raw_total + park_adj + bullpen_adj, 2)
            proj_totals.append(pt if np.isfinite(pt) else None)
        except Exception:
            proj_totals.append(None)

    df["home_model_prob"] = home_probs
    df["away_model_prob"] = away_probs
    df["proj_total"] = proj_totals
    return df
