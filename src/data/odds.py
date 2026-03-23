"""The Odds API client — fetches moneylines, totals, and player props."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.cache import cached

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
REGIONS = "us"

# Sport keys
SPORT_MLB = "baseball_mlb"
SPORT_NBA = "basketball_nba"
SPORT_NFL = "americanfootball_nfl"

# Prop market keys available on The Odds API
MLB_PROP_MARKETS = [
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "pitcher_strikeouts",
    "pitcher_outs",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_key():
    if not ODDS_API_KEY:
        raise EnvironmentError(
            "ODDS_API_KEY is not set. Copy .env.example to .env and add your key."
        )


def american_to_implied_prob(american: int) -> float:
    """Convert American moneyline odds to implied win probability (no vig)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def implied_prob_to_american(prob: float) -> int:
    """Convert win probability back to American odds (approximate)."""
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Normalise implied probs to remove the bookmaker's overround."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


# ---------------------------------------------------------------------------
# Moneyline fetching (h2h)
# ---------------------------------------------------------------------------

def _fetch_moneylines(sport: str) -> pd.DataFrame:
    _require_key()
    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": "h2h",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return _parse_moneylines(resp.json())


def _parse_moneylines(data: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")

        best_home_odds: int | None = None
        best_away_odds: int | None = None

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if outcome.get("name") == home:
                        if best_home_odds is None or price > best_home_odds:
                            best_home_odds = price
                    elif outcome.get("name") == away:
                        if best_away_odds is None or price > best_away_odds:
                            best_away_odds = price

        if best_home_odds is None or best_away_odds is None:
            continue

        raw_home_prob = american_to_implied_prob(best_home_odds)
        raw_away_prob = american_to_implied_prob(best_away_odds)
        home_prob, away_prob = remove_vig(raw_home_prob, raw_away_prob)

        rows.append({
            "game_id": game.get("id", ""),
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "home_odds": best_home_odds,
            "away_odds": best_away_odds,
            "home_implied_prob": round(home_prob, 4),
            "away_implied_prob": round(away_prob, 4),
        })

    return pd.DataFrame(rows)


def get_mlb_odds() -> pd.DataFrame:
    """Return upcoming MLB games with best moneyline odds and vig-removed implied probs."""
    return cached("mlb_odds", lambda: _fetch_moneylines(SPORT_MLB), ttl_hours=2.0)


def get_mlb_odds_no_cache() -> pd.DataFrame:
    """Bypass cache — always fetch fresh MLB odds."""
    return _fetch_moneylines(SPORT_MLB)


# ---------------------------------------------------------------------------
# Totals (over/under) fetching
# ---------------------------------------------------------------------------

def _fetch_totals(sport: str) -> pd.DataFrame:
    _require_key()
    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": "totals",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return _parse_totals(resp.json())


def _parse_totals(data: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")
        total_line = None
        over_odds = None
        under_odds = None

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        if total_line is None:
                            total_line = outcome.get("point")
                        over_odds = outcome.get("price")
                    elif outcome.get("name") == "Under":
                        under_odds = outcome.get("price")
            if total_line is not None:
                break

        if total_line is None:
            continue

        rows.append({
            "game_id": game.get("id", ""),
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "total_line": total_line,
            "over_odds": over_odds,
            "under_odds": under_odds,
        })

    return pd.DataFrame(rows)


def get_mlb_totals() -> pd.DataFrame:
    """Return upcoming MLB games with totals (over/under) lines."""
    return cached("mlb_totals", lambda: _fetch_totals(SPORT_MLB), ttl_hours=2.0)


# ---------------------------------------------------------------------------
# Player props fetching
# ---------------------------------------------------------------------------

def _fetch_player_props(sport: str, event_id: str, markets: list[str]) -> pd.DataFrame:
    """Fetch player props for a specific game event."""
    _require_key()
    url = f"{BASE_URL}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return _parse_player_props(resp.json())


def _parse_player_props(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    home = data.get("home_team", "")
    away = data.get("away_team", "")
    commence = data.get("commence_time", "")

    for bookmaker in data.get("bookmakers", []):
        book_name = bookmaker.get("title", "")
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "bookmaker": book_name,
                    "market": market_key,
                    "player_name": outcome.get("description", outcome.get("name", "")),
                    "bet_name": outcome.get("name", ""),
                    "prop_line": outcome.get("point"),
                    "odds": outcome.get("price"),
                })

    return pd.DataFrame(rows)


def get_mlb_player_props(event_id: str) -> pd.DataFrame:
    """
    Fetch MLB player props for a specific game by event ID.
    Returns props for batter hits, total bases, HR, and pitcher strikeouts.
    """
    cache_key = f"mlb_props_{event_id}"
    return cached(cache_key, lambda: _fetch_player_props(SPORT_MLB, event_id, MLB_PROP_MARKETS), ttl_hours=2.0)


def get_best_prop_lines(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    From a raw props DataFrame (multiple bookmakers), return best available
    line per player per market (best over odds and best under odds).
    """
    if props_df.empty:
        return props_df

    over_df = props_df[props_df["bet_name"] == "Over"].copy()
    under_df = props_df[props_df["bet_name"] == "Under"].copy()

    best_over = over_df.sort_values("odds", ascending=False).groupby(
        ["player_name", "market", "prop_line"], as_index=False
    ).first()
    best_under = under_df.sort_values("odds", ascending=False).groupby(
        ["player_name", "market", "prop_line"], as_index=False
    ).first()

    combined = pd.concat([best_over, best_under], ignore_index=True)
    return combined.sort_values(["player_name", "market"])
