"""NFL player prop projections — QB, RB, WR/TE."""

from __future__ import annotations

import pandas as pd

# Regression weights toward season average (1.0 = no regression, 0.0 = full regression)
REGRESSION_WEIGHTS = {
    "pass_yds": 0.85,
    "completions": 0.85,
    "pass_tds": 0.75,   # TDs are volatile — regress more
    "interceptions": 0.70,
    "rush_yds": 0.82,
    "rec_yds": 0.80,
    "receptions": 0.82,
    "rec_tds": 0.65,
}

# Minimum edge % to flag a bet
MIN_PROP_EDGE_PCT = 5.0

# League average defensive EPA/play (neutral baseline)
LEAGUE_AVG_DEF_EPA = -0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matchup_factor(opp_def_epa: float | None, position: str = "pass") -> float:
    """
    Return a multiplier based on opponent defensive EPA/play.

    Better defense (more negative def_epa) reduces projections.
    opp_def_epa: opponent's defensive EPA/play from nfl_stats (negative = good D)
    """
    if opp_def_epa is None:
        return 1.0
    # League avg ~= -0.02; each 0.1 deviation ≈ 5% impact
    delta = opp_def_epa - LEAGUE_AVG_DEF_EPA
    return max(0.70, min(1.30, 1.0 + delta * 0.5))


def _regress(projection: float, season_avg: float, weight: float) -> float:
    return round(projection * weight + season_avg * (1 - weight), 1)


def _edge(projection: float, line: float) -> float:
    """
    Compute edge % for an over/under prop.
    Positive = over is value, negative = under is value.
    """
    if line <= 0:
        return 0.0
    return round((projection - line) / line * 100, 1)


# ---------------------------------------------------------------------------
# QB Props
# ---------------------------------------------------------------------------

def project_passing_yards(
    season_avg_pass_yds: float,
    opp_def_epa: float | None = None,
    game_script_factor: float = 1.0,
) -> float:
    """Project QB passing yards in a specific game."""
    mf = _matchup_factor(opp_def_epa, "pass")
    raw = season_avg_pass_yds * mf * game_script_factor
    return _regress(raw, season_avg_pass_yds, REGRESSION_WEIGHTS["pass_yds"])


def project_completions(
    season_avg_comp: float,
    opp_def_epa: float | None = None,
) -> float:
    mf = _matchup_factor(opp_def_epa, "pass")
    raw = season_avg_comp * mf
    return _regress(raw, season_avg_comp, REGRESSION_WEIGHTS["completions"])


def project_pass_tds(
    season_avg_pass_tds: float,
    opp_def_epa: float | None = None,
) -> float:
    mf = _matchup_factor(opp_def_epa, "pass")
    raw = season_avg_pass_tds * mf
    return _regress(raw, season_avg_pass_tds, REGRESSION_WEIGHTS["pass_tds"])


def project_interceptions(
    season_avg_ints: float,
    opp_def_epa: float | None = None,
) -> float:
    """More negative opponent def_epa (better D) → more pressure → more INTs."""
    if opp_def_epa is None:
        factor = 1.0
    else:
        delta = LEAGUE_AVG_DEF_EPA - opp_def_epa   # inverted: better D increases INTs
        factor = max(0.70, min(1.40, 1.0 + delta * 0.4))
    raw = season_avg_ints * factor
    return _regress(raw, season_avg_ints, REGRESSION_WEIGHTS["interceptions"])


# ---------------------------------------------------------------------------
# RB Props
# ---------------------------------------------------------------------------

def project_rushing_yards(
    season_avg_rush_yds: float,
    opp_def_epa: float | None = None,
    carries_vs_season: float = 1.0,
) -> float:
    """
    Project RB rushing yards.
    carries_vs_season: ratio of projected carries to season-avg carries (default 1.0 = same).
    """
    mf = _matchup_factor(opp_def_epa, "rush")
    raw = season_avg_rush_yds * mf * carries_vs_season
    return _regress(raw, season_avg_rush_yds, REGRESSION_WEIGHTS["rush_yds"])


def project_rb_receiving_yards(
    season_avg_rec_yds: float,
    opp_def_epa: float | None = None,
) -> float:
    mf = _matchup_factor(opp_def_epa, "pass")
    raw = season_avg_rec_yds * mf
    return _regress(raw, season_avg_rec_yds, REGRESSION_WEIGHTS["rec_yds"])


# ---------------------------------------------------------------------------
# WR / TE Props
# ---------------------------------------------------------------------------

def project_receiving_yards(
    season_avg_rec_yds: float,
    air_yards_share: float | None = None,
    opp_def_epa: float | None = None,
) -> float:
    """
    Project WR/TE receiving yards.
    air_yards_share: fraction of team air yards targeted to this player (0-1).
    Higher air_yards_share → adjust projection upward (opportunity signal).
    """
    mf = _matchup_factor(opp_def_epa, "pass")

    # Air yards adjustment: league avg air yards share per WR ≈ 0.20
    air_adj = 1.0
    if air_yards_share is not None and air_yards_share > 0:
        air_adj = max(0.8, min(1.3, air_yards_share / 0.20))

    raw = season_avg_rec_yds * mf * air_adj
    return _regress(raw, season_avg_rec_yds, REGRESSION_WEIGHTS["rec_yds"])


def project_receptions(
    season_avg_rec: float,
    target_share: float | None = None,
    opp_def_epa: float | None = None,
) -> float:
    mf = _matchup_factor(opp_def_epa, "pass")
    tgt_adj = 1.0
    if target_share is not None and target_share > 0:
        tgt_adj = max(0.8, min(1.3, target_share / 0.20))
    raw = season_avg_rec * mf * tgt_adj
    return _regress(raw, season_avg_rec, REGRESSION_WEIGHTS["receptions"])


# ---------------------------------------------------------------------------
# Full player prop card
# ---------------------------------------------------------------------------

def build_qb_prop_card(
    player_name: str,
    team: str,
    season_stats: dict,
    prop_lines: dict,
    opp_def_epa: float | None = None,
) -> dict:
    """
    Build a full QB prop card with projections and edges.

    season_stats keys: pass_yds_pg, comp_pg, pass_tds_pg, int_pg
    prop_lines keys: pass_yds, completions, pass_tds, interceptions (all optional)
    """
    proj_pass_yds = project_passing_yards(season_stats.get("pass_yds_pg", 0), opp_def_epa)
    proj_comp = project_completions(season_stats.get("comp_pg", 0), opp_def_epa)
    proj_tds = project_pass_tds(season_stats.get("pass_tds_pg", 0), opp_def_epa)
    proj_ints = project_interceptions(season_stats.get("int_pg", 0), opp_def_epa)

    return {
        "player_name": player_name,
        "team": team,
        "position": "QB",
        "opp_def_epa": opp_def_epa,
        "proj_pass_yds": proj_pass_yds,
        "line_pass_yds": prop_lines.get("pass_yds"),
        "edge_pass_yds": _edge(proj_pass_yds, prop_lines.get("pass_yds", 0)) if prop_lines.get("pass_yds") else None,
        "proj_completions": proj_comp,
        "line_completions": prop_lines.get("completions"),
        "edge_completions": _edge(proj_comp, prop_lines.get("completions", 0)) if prop_lines.get("completions") else None,
        "proj_pass_tds": proj_tds,
        "line_pass_tds": prop_lines.get("pass_tds"),
        "edge_pass_tds": _edge(proj_tds, prop_lines.get("pass_tds", 0)) if prop_lines.get("pass_tds") else None,
        "proj_ints": proj_ints,
        "line_ints": prop_lines.get("interceptions"),
        "edge_ints": _edge(proj_ints, prop_lines.get("interceptions", 0)) if prop_lines.get("interceptions") else None,
    }


def build_rb_prop_card(
    player_name: str,
    team: str,
    season_stats: dict,
    prop_lines: dict,
    opp_def_epa: float | None = None,
) -> dict:
    """
    season_stats keys: rush_yds_pg, rec_yds_pg
    prop_lines keys: rush_yds, rec_yds
    """
    proj_rush = project_rushing_yards(season_stats.get("rush_yds_pg", 0), opp_def_epa)
    proj_rec = project_rb_receiving_yards(season_stats.get("rec_yds_pg", 0), opp_def_epa)

    return {
        "player_name": player_name,
        "team": team,
        "position": "RB",
        "opp_def_epa": opp_def_epa,
        "proj_rush_yds": proj_rush,
        "line_rush_yds": prop_lines.get("rush_yds"),
        "edge_rush_yds": _edge(proj_rush, prop_lines.get("rush_yds", 0)) if prop_lines.get("rush_yds") else None,
        "proj_rec_yds": proj_rec,
        "line_rec_yds": prop_lines.get("rec_yds"),
        "edge_rec_yds": _edge(proj_rec, prop_lines.get("rec_yds", 0)) if prop_lines.get("rec_yds") else None,
    }


def build_wr_te_prop_card(
    player_name: str,
    team: str,
    position: str,
    season_stats: dict,
    prop_lines: dict,
    opp_def_epa: float | None = None,
) -> dict:
    """
    season_stats keys: rec_yds_pg, rec_pg, air_yards_share, target_share
    prop_lines keys: rec_yds, receptions
    """
    proj_rec_yds = project_receiving_yards(
        season_stats.get("rec_yds_pg", 0),
        season_stats.get("air_yards_share"),
        opp_def_epa,
    )
    proj_rec = project_receptions(
        season_stats.get("rec_pg", 0),
        season_stats.get("target_share"),
        opp_def_epa,
    )

    return {
        "player_name": player_name,
        "team": team,
        "position": position,
        "opp_def_epa": opp_def_epa,
        "air_yards_share": season_stats.get("air_yards_share"),
        "target_share": season_stats.get("target_share"),
        "proj_rec_yds": proj_rec_yds,
        "line_rec_yds": prop_lines.get("rec_yds"),
        "edge_rec_yds": _edge(proj_rec_yds, prop_lines.get("rec_yds", 0)) if prop_lines.get("rec_yds") else None,
        "proj_receptions": proj_rec,
        "line_receptions": prop_lines.get("receptions"),
        "edge_receptions": _edge(proj_rec, prop_lines.get("receptions", 0)) if prop_lines.get("receptions") else None,
    }
