"""NBA player prop projections — pts, reb, ast, PRA, 3PM."""

from __future__ import annotations

import pandas as pd

# Regression to mean weights — more stable stats get less regression
REGRESSION_WEIGHTS = {
    "pts": 0.85,        # Points: fairly stable with usage
    "reb": 0.80,        # Rebounds: stable
    "ast": 0.80,        # Assists: stable
    "three_pct": 0.70,  # 3P%: volatile, regress 30% toward career/league avg
    "ts_pct": 0.85,     # TS%: reasonably stable
}

LEAGUE_AVG_THREE_PCT = 0.36
LEAGUE_AVG_PACE = 99.0
MIN_PROP_EDGE_PCT = 5.0   # Minimum edge % to flag a bet


# ---------------------------------------------------------------------------
# Pace adjustment
# ---------------------------------------------------------------------------

def pace_factor(game_pace: float, player_team_pace: float) -> float:
    """
    Compute multiplicative pace factor for a player's counting stats.
    Game pace relative to the player's normal team pace affects opportunities.
    """
    if player_team_pace <= 0:
        return 1.0
    return game_pace / player_team_pace


# ---------------------------------------------------------------------------
# Individual stat projections
# ---------------------------------------------------------------------------

def project_points(
    season_avg_pts: float,
    usg_pct: float,
    opp_drtg: float,
    game_pace: float,
    player_team_pace: float,
    league_avg_drtg: float = 113.0,
) -> float:
    """
    Project points for a player in a specific game.

    Adjustments:
    - Opponent DRTG relative to league average (better defense = fewer points)
    - Pace factor (more possessions = more opportunities)
    """
    matchup_factor = max(opp_drtg, 90.0) / league_avg_drtg
    pace_adj = pace_factor(game_pace, player_team_pace)
    projection = season_avg_pts * matchup_factor * pace_adj

    # Regress toward season average
    return round(projection * REGRESSION_WEIGHTS["pts"] + season_avg_pts * (1 - REGRESSION_WEIGHTS["pts"]), 1)


def project_rebounds(
    season_avg_reb: float,
    game_pace: float,
    player_team_pace: float,
    opp_oreb_pct: float = 0.25,
    player_oreb_pct: float = 0.05,
) -> float:
    """
    Project rebounds. Pace and opponent offensive rebounding rate matter.
    Higher opp OREB% means more contested boards (fewer for this team's players).
    """
    pace_adj = pace_factor(game_pace, player_team_pace)
    oreb_penalty = max(0.85, 1.0 - (opp_oreb_pct - 0.25) * 0.5)
    projection = season_avg_reb * pace_adj * oreb_penalty
    return round(projection * REGRESSION_WEIGHTS["reb"] + season_avg_reb * (1 - REGRESSION_WEIGHTS["reb"]), 1)


def project_assists(
    season_avg_ast: float,
    game_pace: float,
    player_team_pace: float,
    opp_tov_pct: float = 0.14,
) -> float:
    """
    Project assists. Pace-adjusted with opponent turnover forcing as context.
    Teams that force more turnovers reduce opponent assist opportunities.
    """
    pace_adj = pace_factor(game_pace, player_team_pace)
    # Higher opp TOV forced → fewer possessions to convert → slight ast reduction
    tov_factor = max(0.90, 1.0 - (opp_tov_pct - 0.14) * 0.3)
    projection = season_avg_ast * pace_adj * tov_factor
    return round(projection * REGRESSION_WEIGHTS["ast"] + season_avg_ast * (1 - REGRESSION_WEIGHTS["ast"]), 1)


def project_three_pm(
    season_avg_three_pa: float,
    season_three_pct: float,
    opp_three_pct_allowed: float = 0.36,
    game_pace: float = LEAGUE_AVG_PACE,
    player_team_pace: float = LEAGUE_AVG_PACE,
) -> float:
    """
    Project 3-pointers made.

    3P% is volatile — regress toward league average.
    Opponent 3P% allowed affects the projection.
    """
    # Regress 3P% toward league average
    regressed_pct = (
        season_three_pct * REGRESSION_WEIGHTS["three_pct"]
        + LEAGUE_AVG_THREE_PCT * (1 - REGRESSION_WEIGHTS["three_pct"])
    )
    # Adjust for opponent defense
    defense_factor = opp_three_pct_allowed / LEAGUE_AVG_THREE_PCT
    adjusted_pct = regressed_pct * defense_factor

    pace_adj = pace_factor(game_pace, player_team_pace)
    attempts = season_avg_three_pa * pace_adj
    return round(attempts * adjusted_pct, 1)


def project_pra(pts: float, reb: float, ast: float) -> float:
    """
    Project Points + Rebounds + Assists (PRA).

    Small negative correlation adjustment applied (players who score more
    tend to get fewer rebounds, and vice versa).
    """
    raw = pts + reb + ast
    # Minor correlation discount — typically -0.5 to -1 point
    correlation_adj = -0.3
    return round(raw + correlation_adj, 1)


# ---------------------------------------------------------------------------
# Prop edge calculator
# ---------------------------------------------------------------------------

def compute_prop_edge(
    model_projection: float,
    prop_line: float,
    bet_direction: str,
) -> dict:
    """
    Compute edge for an NBA player prop.

    Parameters
    ----------
    model_projection : float
        Model's projected stat value
    prop_line : float
        Sportsbook's posted line
    bet_direction : str
        "OVER" or "UNDER"

    Returns
    -------
    dict with: model_projection, prop_line, bet_direction, edge_pct,
               implied_prob, recommendation
    """
    diff = model_projection - prop_line

    if bet_direction == "OVER":
        edge_pct = (diff / prop_line) * 100 if prop_line > 0 else 0.0
    else:
        edge_pct = (-diff / prop_line) * 100 if prop_line > 0 else 0.0

    # Approximate implied probability from edge (simplified)
    implied_prob = 0.5 + (edge_pct / 200)
    implied_prob = max(0.05, min(0.95, implied_prob))

    recommendation = "BET" if edge_pct >= MIN_PROP_EDGE_PCT else "PASS"

    return {
        "model_projection": model_projection,
        "prop_line": prop_line,
        "bet_direction": bet_direction,
        "edge_pct": round(edge_pct, 2),
        "implied_prob": round(implied_prob, 4),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Full player prop suite for one player/game
# ---------------------------------------------------------------------------

def compute_all_props(
    player_row: dict,
    game_context: dict,
) -> dict:
    """
    Compute all prop projections for a player in a specific game.

    player_row keys: season_avg_pts, season_avg_reb, season_avg_ast,
                     season_avg_three_pa, season_three_pct, usg_pct, team_pace
    game_context keys: game_pace, opp_drtg, opp_oreb_pct, opp_tov_pct,
                       opp_three_pct_allowed
    """
    pts = project_points(
        season_avg_pts=player_row.get("pts", 0),
        usg_pct=player_row.get("usg_pct", 0.2),
        opp_drtg=game_context.get("opp_drtg", 113.0),
        game_pace=game_context.get("game_pace", LEAGUE_AVG_PACE),
        player_team_pace=player_row.get("team_pace", LEAGUE_AVG_PACE),
    )
    reb = project_rebounds(
        season_avg_reb=player_row.get("reb", 0),
        game_pace=game_context.get("game_pace", LEAGUE_AVG_PACE),
        player_team_pace=player_row.get("team_pace", LEAGUE_AVG_PACE),
        opp_oreb_pct=game_context.get("opp_oreb_pct", 0.25),
    )
    ast = project_assists(
        season_avg_ast=player_row.get("ast", 0),
        game_pace=game_context.get("game_pace", LEAGUE_AVG_PACE),
        player_team_pace=player_row.get("team_pace", LEAGUE_AVG_PACE),
        opp_tov_pct=game_context.get("opp_tov_pct", 0.14),
    )
    three_pm = project_three_pm(
        season_avg_three_pa=player_row.get("three_pa", 0),
        season_three_pct=player_row.get("three_pct", LEAGUE_AVG_THREE_PCT),
        opp_three_pct_allowed=game_context.get("opp_three_pct_allowed", LEAGUE_AVG_THREE_PCT),
        game_pace=game_context.get("game_pace", LEAGUE_AVG_PACE),
        player_team_pace=player_row.get("team_pace", LEAGUE_AVG_PACE),
    )
    pra = project_pra(pts, reb, ast)

    return {
        "pts_projection": pts,
        "reb_projection": reb,
        "ast_projection": ast,
        "three_pm_projection": three_pm,
        "pra_projection": pra,
    }
