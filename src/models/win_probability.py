"""Composite win probability model combining Pythagorean base + pitcher FIP + home field."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.pythagorean import pythagorean_win_pct, log5_probability, regress_rs_ra
from src.data.ballpark import get_park_factor

HOME_FIELD_ADJ    = 0.04   # ~4% home field advantage
FIP_ADJ_PER_POINT = 0.03   # ~3% win prob per FIP point difference from league average
# Bullpen adjustment: each run of bullpen FIP above/below league avg shifts
# game total by ~0.15 runs (bullpen faces ~3 innings per game on average)
BULLPEN_IP_SHARE  = 3.0 / 9.0   # ~33% of innings pitched by bullpen
# FIP → projected runs: empirical ~0.30 runs per game per FIP point vs. league avg
FIP_RUNS_PER_POINT = 0.30
MIN_WIN_PROB = 0.30
MAX_WIN_PROB = 0.70
# Lineup matchup: OPS scale factor — 0.050 OPS above average → +0.15 effective FIP penalty
LINEUP_OPS_SCALE  = 3.0
LEAGUE_AVG_OPS    = 0.720   # MLB historical baseline; updated each season


def _clamp(value: float, lo: float = MIN_WIN_PROB, hi: float = MAX_WIN_PROB) -> float:
    return max(lo, min(hi, value))


def lineup_matchup_fip_adjustment(
    lineup_df: pd.DataFrame,
    batter_stats_df: pd.DataFrame,
    team_name: str,
) -> float:
    """
    Return a FIP delta representing how the opposing lineup's quality affects
    this pitcher's expected performance.

    A stronger-than-average lineup (OPS > LEAGUE_AVG_OPS) increases the
    pitcher's effective FIP; a weaker lineup decreases it.

    Returns 0.0 when lineup has not been posted or batter data is unavailable.
    """
    if lineup_df.empty or batter_stats_df.empty:
        return 0.0

    team_lineup = lineup_df[lineup_df["team"] == team_name]
    if team_lineup.empty:
        return 0.0

    # Join lineup players to batter stats on name
    merged = team_lineup.merge(
        batter_stats_df[["name", "obp", "slg"]].dropna(subset=["obp", "slg"]),
        left_on="player_name",
        right_on="name",
        how="inner",
    )
    if merged.empty:
        return 0.0

    merged["ops"] = merged["obp"] + merged["slg"]
    lineup_avg_ops = merged["ops"].mean()

    return (lineup_avg_ops - LEAGUE_AVG_OPS) * LINEUP_OPS_SCALE


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
    1. Log5 head-to-head probability from each team's Pythagorean W%
    2. Apply FIP deltas for starting pitchers vs. league average
    3. Add home field advantage
    4. Renormalise to sum to 1.0 and clamp to [0.30, 0.70]
    """
    home_pyth = pythagorean_win_pct(home_rs, home_ra)
    away_pyth = pythagorean_win_pct(away_rs, away_ra)

    # Log5 gives the true head-to-head win probability given each team's overall quality
    home_log5 = log5_probability(home_pyth, away_pyth)

    home_pitch_adj = 0.0
    if home_starter_fip is not None:
        home_pitch_adj = (league_avg_fip - home_starter_fip) * FIP_ADJ_PER_POINT

    away_pitch_adj = 0.0
    if away_starter_fip is not None:
        away_pitch_adj = (league_avg_fip - away_starter_fip) * FIP_ADJ_PER_POINT

    home_prob = home_log5 + home_pitch_adj + HOME_FIELD_ADJ - away_pitch_adj
    away_prob = (1.0 - home_log5) + away_pitch_adj - home_pitch_adj

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
    lineup_df: pd.DataFrame | None = None,
    batter_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute model win probabilities and projected game totals.

    Added columns: home_model_prob, away_model_prob, proj_total
    proj_total incorporates:
      - Season RS/RA per game for each team (regression-to-mean applied after 20 games)
      - Park factor for the home ballpark
      - Bullpen FIP adjustment (if bullpen_df provided)
      - Lineup quality matchup adjustment on starting pitchers (if lineup_df + batter_stats_df provided)
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

    # Compute league avg RS/G and RA/G from current team data for regression
    _ts = team_stats_df.copy()
    _ts["_g"] = _ts.get("wins", pd.Series(0, index=_ts.index)).fillna(0) + \
                _ts.get("losses", pd.Series(0, index=_ts.index)).fillna(0)
    _valid = _ts[_ts["_g"] > 0]
    if not _valid.empty:
        league_avg_rs_pg = (_valid["runs_scored"] / _valid["_g"]).mean()
        league_avg_ra_pg = (_valid["runs_allowed"] / _valid["_g"]).mean()
    else:
        league_avg_rs_pg = 4.5
        league_avg_ra_pg = 4.5

    home_probs, away_probs = [], []
    proj_home_runs_list, proj_away_runs_list, proj_totals = [], [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        try:
            h = team_stats.loc[home]
            a = team_stats.loc[away]
            home_rs, home_ra = float(h["runs_scored"]), float(h["runs_allowed"])
            away_rs, away_ra = float(a["runs_scored"]), float(a["runs_allowed"])

            # Apply shrinkage regression toward league mean, per-team
            h_games = int(float(h.get("wins", 0) or 0) + float(h.get("losses", 0) or 0))
            a_games = int(float(a.get("wins", 0) or 0) + float(a.get("losses", 0) or 0))
            home_rs, home_ra = regress_rs_ra(home_rs, home_ra, h_games, league_avg_rs_pg, league_avg_ra_pg)
            away_rs, away_ra = regress_rs_ra(away_rs, away_ra, a_games, league_avg_rs_pg, league_avg_ra_pg)
        except KeyError:
            home_probs.append(np.nan)
            away_probs.append(np.nan)
            proj_home_runs_list.append(None)
            proj_away_runs_list.append(None)
            proj_totals.append(None)
            continue

        home_fip = pitcher_stats.get(row.get("home_starter")) if "home_starter" in row else None
        away_fip = pitcher_stats.get(row.get("away_starter")) if "away_starter" in row else None

        # Lineup quality matchup: adjust each starter's effective FIP for the opposing lineup
        _ld = lineup_df if lineup_df is not None else pd.DataFrame()
        _bd = batter_stats_df if batter_stats_df is not None else pd.DataFrame()
        if home_fip is not None:
            home_fip = home_fip + lineup_matchup_fip_adjustment(_ld, _bd, away)
        if away_fip is not None:
            away_fip = away_fip + lineup_matchup_fip_adjustment(_ld, _bd, home)

        hp, ap = game_win_probability(
            home_rs, home_ra, away_rs, away_ra,
            home_fip, away_fip, league_avg_fip,
        )
        home_probs.append(round(hp, 4))
        away_probs.append(round(ap, 4))

        # ── Per-team projected runs ───────────────────────────────────────
        # Formula:
        #   base = average of team's offense rate and opponent's defense rate
        #   park_adj = base * (park_factor - 1)  — same park affects both teams equally
        #   starter_adj = (opposing_starter_FIP - league_avg) * FIP_RUNS_PER_POINT
        #     (higher opposing FIP → that team allows more runs → scoring team benefits)
        #   bullpen_adj = (opposing_bullpen_FIP - league_avg) * BULLPEN_IP_SHARE * FIP_RUNS_PER_POINT
        #     (home bullpen faces away batters, so affects proj_away_runs; vice versa)
        try:
            h_w = float(np.nan_to_num(h.get("wins", 0), nan=0.0))
            h_l = float(np.nan_to_num(h.get("losses", 0), nan=0.0))
            a_w = float(np.nan_to_num(a.get("wins", 0), nan=0.0))
            a_l = float(np.nan_to_num(a.get("losses", 0), nan=0.0))
            h_g = max(h_w + h_l, 1.0)
            a_g = max(a_w + a_l, 1.0)

            park_factor = get_park_factor(home)

            # Offensive base for each team (blend own R/G with opponent's RA/G)
            home_base = (home_rs / h_g + away_ra / a_g) / 2
            away_base = (away_rs / a_g + home_ra / h_g) / 2

            # Park factor applies equally to both teams
            home_park = home_base * (park_factor - 1.0)
            away_park = away_base * (park_factor - 1.0)

            # Starting pitcher adjustment (opposing SP's FIP affects this team's runs)
            #   away FIP above league avg → home team scores more
            home_sp_adj = (away_fip - league_avg_fip) * FIP_RUNS_PER_POINT if away_fip is not None else 0.0
            away_sp_adj = (home_fip - league_avg_fip) * FIP_RUNS_PER_POINT if home_fip is not None else 0.0

            # Bullpen adjustment (opposing bullpen's FIP affects this team's runs)
            h_bp_fip = bullpen_fip.get(home, league_avg_bullpen_fip)
            a_bp_fip = bullpen_fip.get(away, league_avg_bullpen_fip)
            home_bp_adj = (a_bp_fip - league_avg_bullpen_fip) * BULLPEN_IP_SHARE * FIP_RUNS_PER_POINT
            away_bp_adj = (h_bp_fip - league_avg_bullpen_fip) * BULLPEN_IP_SHARE * FIP_RUNS_PER_POINT

            phr = home_base + home_park + home_sp_adj + home_bp_adj
            par = away_base + away_park + away_sp_adj + away_bp_adj
            pt  = phr + par

            proj_home_runs_list.append(round(phr, 2) if np.isfinite(phr) else None)
            proj_away_runs_list.append(round(par, 2) if np.isfinite(par) else None)
            proj_totals.append(round(pt, 2) if np.isfinite(pt) else None)
        except Exception:
            proj_home_runs_list.append(None)
            proj_away_runs_list.append(None)
            proj_totals.append(None)

    df["home_model_prob"]  = home_probs
    df["away_model_prob"]  = away_probs
    df["proj_home_runs"]   = proj_home_runs_list
    df["proj_away_runs"]   = proj_away_runs_list
    df["proj_total"]       = proj_totals
    return df


def get_formula_state(team_stats_df: pd.DataFrame) -> dict:
    """
    Detect which formula mode is active based on how many games teams have played.

    Returns a dict with:
      - state   : "EARLY_SEASON" | "REGRESSION_ACTIVE"
      - min_games : int — fewest games played by any team
      - message : plain-language explanation for the in-app banner
    """
    from src.models.pythagorean import MIN_GAMES_FOR_REGRESSION

    if team_stats_df.empty:
        return {
            "state": "EARLY_SEASON",
            "min_games": 0,
            "message": (
                "Early Season Mode — waiting for game data. "
                "Win probabilities will activate once teams have played games."
            ),
        }

    games_played = (
        team_stats_df.get("wins", pd.Series(dtype=float)).fillna(0)
        + team_stats_df.get("losses", pd.Series(dtype=float)).fillna(0)
    ).astype(int)
    min_games = int(games_played.min())

    if min_games < MIN_GAMES_FOR_REGRESSION:
        teams_below = int((games_played < MIN_GAMES_FOR_REGRESSION).sum())
        return {
            "state": "EARLY_SEASON",
            "min_games": min_games,
            "message": (
                f"**Early Season Mode** — {teams_below} team(s) have played fewer than "
                f"{MIN_GAMES_FOR_REGRESSION} games. "
                f"Win probabilities use raw Pythagorean RS/RA without regression to the mean. "
                f"Model accuracy improves significantly once all teams reach {MIN_GAMES_FOR_REGRESSION} games. "
                f"Log5 head-to-head probability and lineup matchup adjustments are active."
            ),
        }

    return {
        "state": "REGRESSION_ACTIVE",
        "min_games": min_games,
        "message": (
            f"**Full Model Active** — all teams have played {min_games}+ games. "
            f"Win probabilities now use **Log5** head-to-head probability + "
            f"**shrinkage regression** (blending {MIN_GAMES_FOR_REGRESSION}+ game RS/RA toward the "
            f"league mean, weight = G / (G + 30)) + **FIP pitcher adjustment** + "
            f"**lineup quality matchup** (opposing lineup OPS vs. league average). "
            f"This is the full Bill James / Peta methodology."
        ),
    }
