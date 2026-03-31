"""Fetch completed MLB game scores from the MLB Stats API (free, no key required)."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import requests

from src.data.cache import cached

MLB_API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 10

# Odds API full name → MLB Stats API team name mapping
TEAM_NAME_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
}


def get_games_for_date(date: datetime.date) -> pd.DataFrame:
    """
    Return completed game scores for a given date.

    Returns DataFrame with columns:
      game_date, home_team, away_team, home_score, away_score, winner, status
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_results_{date_str}"
    # Only cache completed dates (not today — scores may still be coming in)
    ttl = 24.0 if date < datetime.date.today() else 0.5

    def fetch() -> pd.DataFrame:
        url = f"{MLB_API}/schedule"
        params = {
            "sportId": 1,
            "date": date_str,
            "hydrate": "linescore",
        }
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    continue
                teams = game.get("teams", {})
                home = teams.get("home", {})
                away = teams.get("away", {})
                home_name = home.get("team", {}).get("name", "")
                away_name = away.get("team", {}).get("name", "")
                home_score = home.get("score")
                away_score = away.get("score")
                if home_score is None or away_score is None:
                    continue
                winner = home_name if home_score > away_score else away_name
                rows.append({
                    "game_date": date_str,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "winner": winner,
                    "status": status,
                })
        return pd.DataFrame(rows)

    return cached(cache_key, fetch, ttl_hours=ttl)


def get_probable_starters(date: datetime.date) -> pd.DataFrame:
    """
    Fetch probable starting pitchers for a given date from the MLB Stats API.

    Returns DataFrame with columns:
      home_team, away_team, home_starter, away_starter,
      home_starter_announced, away_starter_announced
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_starters_{date_str}"

    def fetch() -> pd.DataFrame:
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                home_info = teams.get("home", {})
                away_info = teams.get("away", {})
                home_team = home_info.get("team", {}).get("name", "")
                away_team = away_info.get("team", {}).get("name", "")
                home_pitcher = home_info.get("probablePitcher", {})
                away_pitcher = away_info.get("probablePitcher", {})
                rows.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_starter": home_pitcher.get("fullName") if home_pitcher else None,
                    "away_starter": away_pitcher.get("fullName") if away_pitcher else None,
                    "home_starter_announced": bool(home_pitcher),
                    "away_starter_announced": bool(away_pitcher),
                })
        return pd.DataFrame(rows)

    # Short TTL — starters get announced throughout the day
    return cached(cache_key, fetch, ttl_hours=1.0)


def get_today_lineups(date: datetime.date) -> pd.DataFrame:
    """
    Fetch confirmed starting lineups from the MLB Stats API.

    Returns DataFrame with columns: team, player_name, batting_position
    Returns empty DataFrame if lineups have not been posted yet.
    Lineups are typically posted 1–2 hours before first pitch.
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_lineups_{date_str}"

    def fetch() -> pd.DataFrame:
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={"sportId": 1, "date": date_str, "hydrate": "lineups"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                home_name = teams.get("home", {}).get("team", {}).get("name", "")
                away_name = teams.get("away", {}).get("team", {}).get("name", "")

                # Try game-level lineups (newer API response format)
                game_lineups = game.get("lineups", {})
                if game_lineups:
                    for side_key, team_name in [("homePlayers", home_name), ("awayPlayers", away_name)]:
                        for i, player in enumerate(game_lineups.get(side_key, []), 1):
                            name = player.get("fullName") or player.get("name", "")
                            if name:
                                rows.append({"team": team_name, "player_name": name, "batting_position": i})
                else:
                    # Try team-level lineups (older format)
                    for side, team_name in [("home", home_name), ("away", away_name)]:
                        team_info = teams.get(side, {})
                        batting_order = team_info.get("lineups", {}).get("battingOrder", [])
                        for i, player in enumerate(batting_order, 1):
                            name = player.get("fullName") or player.get("name", "")
                            if name:
                                rows.append({"team": team_name, "player_name": name, "batting_position": i})

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return cached(cache_key, fetch, ttl_hours=0.5)


def get_results_for_date_range(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Fetch and combine results for a range of dates."""
    all_rows = []
    current = start
    while current <= end:
        df = get_games_for_date(current)
        if not df.empty:
            all_rows.append(df)
        current += datetime.timedelta(days=1)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def verify_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unverified prediction, fetch the actual game result and mark correct/incorrect.
    Returns updated predictions_df with actual_winner and correct columns filled in.
    """
    if predictions_df.empty:
        return predictions_df

    df = predictions_df.copy()
    today = datetime.date.today()

    unverified = df[df["actual_winner"].isna()].copy()
    if unverified.empty:
        return df

    # Only verify past games (not today's)
    unverified["_date"] = pd.to_datetime(unverified["prediction_date"]).dt.date
    past = unverified[unverified["_date"] < today]

    if past.empty:
        return df

    # Fetch results for all relevant dates
    min_date = past["_date"].min()
    results = get_results_for_date_range(min_date, today - datetime.timedelta(days=1))

    if results.empty:
        return df

    verified_at = datetime.datetime.now().isoformat()

    for idx, pred_row in past.iterrows():
        game_date = str(pred_row["prediction_date"])
        home = pred_row["home_team"]
        away = pred_row["away_team"]

        # Try to match game result
        match = results[
            (results["game_date"] == game_date) &
            (results["home_team"].str.contains(home.split()[-1], case=False, na=False) |
             results["away_team"].str.contains(away.split()[-1], case=False, na=False))
        ]

        if match.empty:
            continue

        result_row = match.iloc[0]
        actual_winner = result_row["winner"]
        predicted_winner = pred_row["predicted_winner"]
        correct = int(predicted_winner in actual_winner or actual_winner in predicted_winner)

        df.at[idx, "actual_winner"] = actual_winner
        df.at[idx, "correct"] = correct
        df.at[idx, "verified_at"] = verified_at

        # Persist to DB
        from src.data.predictions_db import update_result
        update_result(game_date, home, away, actual_winner, bool(correct), verified_at)

    df = df.drop(columns=["_date"], errors="ignore")
    return df
