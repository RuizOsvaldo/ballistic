"""NBA rest and schedule analysis — back-to-back detection and rest adjustments."""

from __future__ import annotations

import datetime
import pandas as pd

# Net rating adjustments per rest situation (in net rating points)
REST_ADJUSTMENTS = {
    "back_to_back": -2.5,       # Second night of back-to-back
    "three_in_four": -1.5,      # Third game in four nights
    "rested_7plus": -0.5,       # Slight rust after 7+ days off
    "normal": 0.0,              # 1-3 days rest — standard
}

MIN_EDGE_REST_PTS = 2.0   # Only flag rest mismatch if net adjustment >= 2 pts


def classify_rest(last_game_date: datetime.date | None, today: datetime.date | None = None) -> str:
    """Classify a team's rest situation given their last game date."""
    today = today or datetime.date.today()

    if last_game_date is None:
        return "normal"

    days_rest = (today - last_game_date).days

    if days_rest == 1:
        return "back_to_back"
    if days_rest == 2:
        return "three_in_four"
    if days_rest >= 7:
        return "rested_7plus"
    return "normal"


def get_rest_adjustment(rest_type: str) -> float:
    """Return net rating adjustment (in points) for a given rest type."""
    return REST_ADJUSTMENTS.get(rest_type, 0.0)


def compute_game_rest_context(
    home_last_game: datetime.date | None,
    away_last_game: datetime.date | None,
    game_date: datetime.date | None = None,
) -> dict:
    """
    Compute rest context for a single game.

    Returns:
      - home_rest_type, away_rest_type
      - home_rest_adj, away_rest_adj (net rating points)
      - rest_mismatch (bool)
      - rest_edge_team ("HOME" | "AWAY" | "NONE")
      - net_rest_adj (home_adj - away_adj, positive = home advantage)
    """
    game_date = game_date or datetime.date.today()

    home_rest = classify_rest(home_last_game, game_date)
    away_rest = classify_rest(away_last_game, game_date)

    home_adj = get_rest_adjustment(home_rest)
    away_adj = get_rest_adjustment(away_rest)
    net_adj = home_adj - away_adj

    mismatch = abs(net_adj) >= MIN_EDGE_REST_PTS

    if net_adj > MIN_EDGE_REST_PTS:
        edge_team = "HOME"
    elif net_adj < -MIN_EDGE_REST_PTS:
        edge_team = "AWAY"
    else:
        edge_team = "NONE"

    return {
        "home_rest_type": home_rest,
        "away_rest_type": away_rest,
        "home_rest_adj": home_adj,
        "away_rest_adj": away_adj,
        "net_rest_adj": round(net_adj, 2),
        "rest_mismatch": mismatch,
        "rest_edge_team": edge_team,
    }


def compute_rest_adjustments_df(
    games_df: pd.DataFrame,
    team_last_game: dict[str, datetime.date],
) -> pd.DataFrame:
    """
    Add rest adjustment columns to a games DataFrame.

    games_df required columns: home_team, away_team
    team_last_game: {team_name: last_game_date}

    Added columns: home_rest_type, away_rest_type, home_rest_adj,
                   away_rest_adj, net_rest_adj, rest_mismatch, rest_edge_team
    """
    df = games_df.copy()
    rows = []

    for _, row in df.iterrows():
        home_last = team_last_game.get(row["home_team"])
        away_last = team_last_game.get(row["away_team"])
        ctx = compute_game_rest_context(home_last, away_last)
        rows.append(ctx)

    rest_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), rest_df], axis=1)


def get_team_last_game_dates(team_game_logs: dict[str, pd.DataFrame]) -> dict[str, datetime.date]:
    """
    Extract the most recent game date for each team from their game logs.

    team_game_logs: {team_name: DataFrame with 'game_date' column}
    Returns: {team_name: last_game_date}
    """
    result = {}
    for team, log_df in team_game_logs.items():
        if log_df.empty or "game_date" not in log_df.columns:
            continue
        last = pd.to_datetime(log_df["game_date"]).max()
        if pd.notna(last):
            result[team] = last.date()
    return result
