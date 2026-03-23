"""NFL rest and schedule adjustments — bye week, short week, home field, travel."""

from __future__ import annotations

import pandas as pd

# Spread-point adjustments
HOME_FIELD_PTS = 2.5          # Standard NFL home field advantage
BYE_WEEK_BONUS_PTS = 2.5      # Bye week rest advantage vs. normal opponent
SHORT_WEEK_PENALTY_PTS = -2.0  # Thursday night (short rest) vs. rested opponent
WEST_COAST_PENALTY_PTS = -1.5  # West Coast team playing at 1pm ET (body clock)

# NFL team timezone groupings (abbreviation → timezone bucket)
WEST_COAST_TEAMS = {
    "LAR", "LAC", "SF", "SEA", "ARI",   # Pacific
    "LV",                                 # Mountain
}

# Days-rest thresholds
SHORT_WEEK_DAYS = 4   # ≤4 days rest = short week (Thursday game)
BYE_WEEK_DAYS = 13    # ≥13 days rest = coming off bye


def classify_rest(days_rest: int | None) -> str:
    """Return 'Bye', 'Short Week', or 'Normal' based on days since last game."""
    if days_rest is None:
        return "Normal"
    if days_rest >= BYE_WEEK_DAYS:
        return "Bye"
    if days_rest <= SHORT_WEEK_DAYS:
        return "Short Week"
    return "Normal"


def rest_adjustment_pts(rest_type: str) -> float:
    """
    Return the spread-point adjustment for a team's rest situation.

    Positive = team benefits; negative = team is disadvantaged.
    The caller subtracts the opponent's adjustment to get net effect.
    """
    if rest_type == "Bye":
        return BYE_WEEK_BONUS_PTS
    if rest_type == "Short Week":
        return SHORT_WEEK_PENALTY_PTS
    return 0.0


def is_west_coast_early_kickoff(team: str, kickoff_hour_et: int | None) -> bool:
    """Return True if a West Coast team is playing at 1pm ET (body clock disadvantage)."""
    if team not in WEST_COAST_TEAMS:
        return False
    if kickoff_hour_et is None:
        return False
    return kickoff_hour_et <= 13


def compute_rest_adjustments(
    games_df: pd.DataFrame,
    home_days_rest: dict[str, int] | None = None,
    away_days_rest: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Add rest-based spread adjustments to a games DataFrame.

    games_df required columns: home_team, away_team
    Optional dicts: {team_abbr → days_rest_int}

    Added columns:
      - home_rest_type     : "Bye" | "Short Week" | "Normal"
      - away_rest_type     : same
      - home_rest_adj_pts  : spread-point bonus/penalty for home team
      - away_rest_adj_pts  : spread-point bonus/penalty for away team
      - rest_net_pts       : home_rest_adj - away_rest_adj (positive = home benefits)
      - rest_mismatch      : True if one team is on bye and other is on short week
    """
    df = games_df.copy()
    home_days = home_days_rest or {}
    away_days = away_days_rest or {}

    home_rest_types, away_rest_types = [], []
    home_rest_adjs, away_rest_adjs = [], []
    rest_nets, mismatches = [], []

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")

        hrt = classify_rest(home_days.get(home))
        art = classify_rest(away_days.get(away))
        h_adj = rest_adjustment_pts(hrt)
        a_adj = rest_adjustment_pts(art)

        home_rest_types.append(hrt)
        away_rest_types.append(art)
        home_rest_adjs.append(h_adj)
        away_rest_adjs.append(a_adj)
        rest_nets.append(round(h_adj - a_adj, 1))
        mismatches.append(
            (hrt == "Bye" and art == "Short Week") or
            (hrt == "Short Week" and art == "Bye")
        )

    df["home_rest_type"] = home_rest_types
    df["away_rest_type"] = away_rest_types
    df["home_rest_adj_pts"] = home_rest_adjs
    df["away_rest_adj_pts"] = away_rest_adjs
    df["rest_net_pts"] = rest_nets
    df["rest_mismatch"] = mismatches
    return df


def build_rest_df(days_rest_map: dict[str, int]) -> pd.DataFrame:
    """
    Convert a {team → days_rest} dict to a DataFrame for use in epa.compute_nfl_win_probabilities.

    Returns DataFrame with columns: team, rest_adjustment (spread points)
    """
    rows = []
    for team, days in days_rest_map.items():
        rest_type = classify_rest(days)
        rows.append({
            "team": team,
            "rest_type": rest_type,
            "rest_adjustment": rest_adjustment_pts(rest_type),
        })
    return pd.DataFrame(rows)
