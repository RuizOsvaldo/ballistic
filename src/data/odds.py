"""The Odds API client — fetches moneylines, totals, and player props."""

from __future__ import annotations

import datetime
import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

from src.data.cache import cached

load_dotenv(find_dotenv(usecwd=True))

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
# Full game markets — h2h, spreads, totals, F5 ML, F5 total, 1st inning
# ---------------------------------------------------------------------------

# Markets to request in a single API call
_MLB_GAME_MARKETS = "h2h,spreads,totals"


def _best_price(outcomes: list[dict], name: str) -> int | None:
    """Return the best (highest) American odds for a named outcome across bookmakers."""
    best = None
    for o in outcomes:
        if o.get("name") == name:
            p = o.get("price")
            if p is not None and (best is None or p > best):
                best = p
    return best


def _best_point(outcomes: list[dict], name: str) -> float | None:
    """Return the point (line) for a named outcome — just take first seen."""
    for o in outcomes:
        if o.get("name") == name and o.get("point") is not None:
            return o.get("point")
    return None


def _fetch_full_markets(sport: str) -> pd.DataFrame:
    _require_key()
    resp = requests.get(
        f"{BASE_URL}/sports/{sport}/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": REGIONS,
            "markets": _MLB_GAME_MARKETS,
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return _parse_full_markets(resp.json())


def _parse_full_markets(data: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")

        # Aggregate best odds per market key across all bookmakers
        market_outcomes: dict[str, list[dict]] = {}
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key", "")
                market_outcomes.setdefault(key, []).extend(market.get("outcomes", []))

        # ── Moneyline (h2h) ──
        h2h = market_outcomes.get("h2h", [])
        home_ml = _best_price(h2h, home)
        away_ml = _best_price(h2h, away)
        if home_ml is None or away_ml is None:
            continue

        raw_home = american_to_implied_prob(home_ml)
        raw_away = american_to_implied_prob(away_ml)
        home_prob, away_prob = remove_vig(raw_home, raw_away)

        # ── Run line / spread ──
        sp = market_outcomes.get("spreads", [])
        home_rl_point = _best_point(sp, home)
        away_rl_point = _best_point(sp, away)
        home_rl_odds = _best_price(sp, home)
        away_rl_odds = _best_price(sp, away)

        # ── Game total ──
        tot = market_outcomes.get("totals", [])
        total_line = _best_point(tot, "Over")
        over_odds = _best_price(tot, "Over")
        under_odds = _best_price(tot, "Under")

        rows.append({
            "game_id": game.get("id", ""),
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            # Moneyline
            "home_odds": home_ml,
            "away_odds": away_ml,
            "home_implied_prob": round(home_prob, 4),
            "away_implied_prob": round(away_prob, 4),
            # Run line
            "home_rl": home_rl_point,
            "away_rl": away_rl_point,
            "home_rl_odds": home_rl_odds,
            "away_rl_odds": away_rl_odds,
            # Game total
            "total_line": total_line,
            "over_odds": over_odds,
            "under_odds": under_odds,
        })

    return pd.DataFrame(rows)


def get_mlb_odds() -> pd.DataFrame:
    """
    Return today's MLB games with full market data:
    moneyline, run line, game total, F5 ML, F5 total, 1st inning ML.
    """
    def fetch_todays():
        df = _fetch_full_markets(SPORT_MLB)
        if df.empty or "commence_time" not in df.columns:
            return df
        today = datetime.date.today()
        df["_date"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert("America/New_York").dt.date
        df = df[df["_date"] == today].drop(columns=["_date"]).reset_index(drop=True)
        return df
    return cached("mlb_odds", fetch_todays, ttl_hours=2.0)


def get_mlb_odds_no_cache() -> pd.DataFrame:
    """Bypass cache — always fetch fresh MLB odds."""
    return _fetch_full_markets(SPORT_MLB)


# ---------------------------------------------------------------------------
# Legacy totals getter (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def get_mlb_totals() -> pd.DataFrame:
    """Return today's game totals. Prefer get_mlb_odds() which includes totals."""
    df = get_mlb_odds()
    cols = ["game_id", "commence_time", "home_team", "away_team",
            "total_line", "over_odds", "under_odds"]
    return df[[c for c in cols if c in df.columns]]


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


def get_all_batter_hits_props(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch batter_hits props for all today's games and return a clean DataFrame.

    Parameters
    ----------
    games_df : DataFrame with game_id, home_team, away_team columns (from get_mlb_odds)

    Returns
    -------
    DataFrame with: player_name, home_team, away_team, prop_line, over_odds, under_odds
    """
    if games_df.empty or "game_id" not in games_df.columns:
        return pd.DataFrame()

    all_props: list[pd.DataFrame] = []
    for _, game in games_df.iterrows():
        try:
            raw = get_mlb_player_props(str(game["game_id"]))
            if raw.empty:
                continue
            hits = raw[raw["market"] == "batter_hits"].copy()
            if hits.empty:
                continue
            all_props.append(hits)
        except Exception:
            continue

    if not all_props:
        return pd.DataFrame()

    combined = pd.concat(all_props, ignore_index=True)
    best = get_best_prop_lines(combined)
    if best.empty:
        return pd.DataFrame()

    # Pivot over/under into one row per player per line
    over_rows = best[best["bet_name"] == "Over"][
        ["player_name", "home_team", "away_team", "prop_line", "odds"]
    ].rename(columns={"odds": "over_odds"})
    under_rows = best[best["bet_name"] == "Under"][
        ["player_name", "home_team", "away_team", "prop_line", "odds"]
    ].rename(columns={"odds": "under_odds"})

    merged = over_rows.merge(
        under_rows[["player_name", "home_team", "away_team", "prop_line", "under_odds"]],
        on=["player_name", "home_team", "away_team", "prop_line"],
        how="outer",
    )
    return merged.dropna(subset=["player_name", "prop_line"]).reset_index(drop=True)


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
