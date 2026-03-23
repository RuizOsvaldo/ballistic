"""NBA composite win probability — net rating + rest + home court."""

from __future__ import annotations

import pandas as pd

from src.sports.basketball.models.net_rating import compute_nba_win_probabilities
from src.sports.basketball.models.rest_schedule import compute_rest_adjustments_df
from src.models.kelly import compute_kelly_for_games


def build_nba_games_with_edge(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    team_last_game_dates: dict,
    odds_df: pd.DataFrame,
    bankroll: float = 1000.0,
) -> pd.DataFrame:
    """
    Full pipeline: games + team stats + rest + odds → edge table.

    1. Compute rest adjustments for each game
    2. Compute win probabilities
    3. Merge with odds (implied probs)
    4. Compute Kelly edge and stakes

    Returns enriched DataFrame ready for the dashboard.
    """
    if games_df.empty or team_stats_df.empty:
        return pd.DataFrame()

    # Step 1: rest adjustments
    rest_df = compute_rest_adjustments_df(games_df, team_last_game_dates)

    # Build per-team rest adjustment dict for win prob model
    rest_adj_rows = []
    for _, row in rest_df.iterrows():
        rest_adj_rows.append({"team_name": row["home_team"], "rest_adjustment": row.get("home_rest_adj", 0.0)})
        rest_adj_rows.append({"team_name": row["away_team"], "rest_adjustment": row.get("away_rest_adj", 0.0)})
    rest_adj_df = pd.DataFrame(rest_adj_rows).drop_duplicates("team_name") if rest_adj_rows else None

    # Step 2: win probabilities
    probs_df = compute_nba_win_probabilities(rest_df, team_stats_df, rest_adj_df)

    # Step 3: merge odds
    if not odds_df.empty and "home_team" in odds_df.columns:
        merged = probs_df.merge(
            odds_df[["home_team", "away_team", "home_odds", "away_odds",
                     "home_implied_prob", "away_implied_prob"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        merged = probs_df.copy()
        for col in ["home_odds", "away_odds", "home_implied_prob", "away_implied_prob"]:
            merged[col] = float("nan")

    # Step 4: Kelly
    merged = merged.dropna(subset=["home_model_prob", "away_model_prob"])
    if merged.empty:
        return merged

    return compute_kelly_for_games(merged, bankroll=bankroll)
