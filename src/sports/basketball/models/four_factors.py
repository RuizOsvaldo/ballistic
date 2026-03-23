"""NBA Four Factors matchup analysis and pace-adjusted game total model."""

from __future__ import annotations

import pandas as pd

# Dean Oliver Four Factor weights
FACTOR_WEIGHTS = {
    "efg_pct": 0.40,
    "tov_pct": 0.25,
    "oreb_pct": 0.20,
    "ft_rate": 0.15,
}

LEAGUE_AVG_ORTG = 113.0     # approximate current league average
LEAGUE_AVG_PACE = 99.0


def compute_matchup_four_factors(
    home_stats: dict,
    away_stats: dict,
) -> dict:
    """
    Compare four factors for a specific matchup.

    Returns a dict summarizing each factor's edge direction and magnitude.
    """
    factors = {}

    # eFG% — higher is better offensively; lower allowed is better defensively
    home_efg_edge = home_stats.get("efg_pct", 0.5) - away_stats.get("opp_efg_pct", 0.5)
    away_efg_edge = away_stats.get("efg_pct", 0.5) - home_stats.get("opp_efg_pct", 0.5)
    factors["efg_edge_home"] = round(home_efg_edge, 4)
    factors["efg_edge_away"] = round(away_efg_edge, 4)

    # Turnover rate — lower is better offensively; higher forced is better defensively
    home_tov_edge = away_stats.get("tov_pct", 0.14) - home_stats.get("tov_pct", 0.14)
    away_tov_edge = home_stats.get("tov_pct", 0.14) - away_stats.get("tov_pct", 0.14)
    factors["tov_edge_home"] = round(home_tov_edge, 4)
    factors["tov_edge_away"] = round(away_tov_edge, 4)

    # OREB% — higher is better
    home_oreb_edge = home_stats.get("oreb_pct", 0.25) - away_stats.get("oreb_pct", 0.25)
    factors["oreb_edge_home"] = round(home_oreb_edge, 4)
    factors["oreb_edge_away"] = round(-home_oreb_edge, 4)

    # FT rate — higher is better
    home_ft_edge = home_stats.get("ft_rate", 0.22) - away_stats.get("ft_rate", 0.22)
    factors["ft_edge_home"] = round(home_ft_edge, 4)
    factors["ft_edge_away"] = round(-home_ft_edge, 4)

    # Composite four-factor score (weighted sum of edges)
    home_score = (
        home_efg_edge * FACTOR_WEIGHTS["efg_pct"]
        + home_tov_edge * FACTOR_WEIGHTS["tov_pct"]
        + home_oreb_edge * FACTOR_WEIGHTS["oreb_pct"]
        + home_ft_edge * FACTOR_WEIGHTS["ft_rate"]
    )
    factors["home_four_factor_score"] = round(home_score, 4)
    factors["away_four_factor_score"] = round(-home_score, 4)
    factors["four_factor_edge"] = "HOME" if home_score > 0 else "AWAY"

    return factors


def project_game_total(
    home_ortg: float,
    away_ortg: float,
    home_drtg: float,
    away_drtg: float,
    home_pace: float,
    away_pace: float,
) -> dict:
    """
    Project combined game total using pace and efficiency.

    Formula:
      avg_pace = (home_pace + away_pace) / 2
      home_pts = avg_pace * ((home_ortg + away_drtg) / 2) / 100
      away_pts = avg_pace * ((away_ortg + home_drtg) / 2) / 100
      total = home_pts + away_pts
    """
    avg_pace = (home_pace + away_pace) / 2

    # Each team's projected pts — average of own ORTG and opponent's DRTG
    home_pts = avg_pace * ((home_ortg + away_drtg) / 2) / 100
    away_pts = avg_pace * ((away_ortg + home_drtg) / 2) / 100

    return {
        "projected_home_pts": round(home_pts, 1),
        "projected_away_pts": round(away_pts, 1),
        "projected_total": round(home_pts + away_pts, 1),
        "avg_pace": round(avg_pace, 1),
    }


def compute_total_edge(projected_total: float, posted_line: float) -> dict:
    """
    Compute over/under edge for a game total.

    Returns edge %, direction, and recommendation.
    Minimum 2-point difference to flag a bet.
    """
    diff = projected_total - posted_line
    pct = abs(diff) / posted_line * 100

    if abs(diff) < 2.0:
        return {"direction": "PASS", "edge_pts": round(diff, 1), "edge_pct": round(pct, 2)}

    direction = "OVER" if diff > 0 else "UNDER"
    return {
        "direction": direction,
        "edge_pts": round(diff, 1),
        "edge_pct": round(pct, 2),
    }
