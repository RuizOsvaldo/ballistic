"""ESPN odds client — fetches moneylines, run lines, and totals. No API key required."""

from __future__ import annotations

import datetime

import pandas as pd
import requests

from src.data.cache import cached

_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
_ODDS_URL = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{event_id}/competitions/{event_id}/odds"
_MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
_TIMEOUT = 10

# ESPN status names that mean the game is not being played today
_SKIP_STATUSES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED", "STATUS_RAIN_DELAY"}


# ---------------------------------------------------------------------------
# Utility functions (unchanged — used by models and dashboard)
# ---------------------------------------------------------------------------

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
# ESPN fetch helpers
# ---------------------------------------------------------------------------

def _parse_american(value) -> int | None:
    """Parse an American odds value from ESPN (int, float, or string like '+175')."""
    if value is None:
        return None
    try:
        return int(float(str(value).replace("+", "")))
    except (ValueError, TypeError):
        return None


def _fetch_espn_odds(event_id: str) -> dict | None:
    """Fetch the first odds provider entry for a single ESPN event. Returns None on failure."""
    url = _ODDS_URL.format(event_id=event_id)
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return items[0] if items else None
    except Exception:
        return None


def _fetch_espn_scoreboard() -> list[dict]:
    """Return today's ESPN event dicts, excluding postponed or cancelled games."""
    resp = requests.get(_SCOREBOARD_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    events = resp.json().get("events", [])
    active = []
    for event in events:
        comp = event.get("competitions", [{}])[0]
        status_name = comp.get("status", {}).get("type", {}).get("name", "")
        if status_name not in _SKIP_STATUSES:
            active.append(event)
    return active


def _fetch_mlb_scheduled_games(date: datetime.date) -> set[tuple[str, str]]:
    """
    Return a set of (away_team_name, home_team_name) for games on the MLB Stats API
    schedule for the given date. Used to cross-validate ESPN odds.
    Only includes games with status Preview, Scheduled, Live, or Final (not Postponed).
    """
    try:
        resp = requests.get(
            _MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": date.strftime("%Y-%m-%d")},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return set()

    pairs: set[tuple[str, str]] = set()
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            # Skip postponed/suspended games from MLB API too
            detail = game.get("status", {}).get("detailedState", "")
            if "Postponed" in detail or "Suspended" in detail or "Cancelled" in detail:
                continue
            teams = game.get("teams", {})
            home = teams.get("home", {}).get("team", {}).get("name", "")
            away = teams.get("away", {}).get("team", {}).get("name", "")
            if home and away:
                pairs.add((away, home))
    return pairs


# ---------------------------------------------------------------------------
# Core game odds builder
# ---------------------------------------------------------------------------

def _build_game_row(event: dict, odds: dict) -> dict | None:
    """
    Build a single game row from an ESPN event dict and its odds dict.
    Returns None if moneylines are unavailable.
    """
    comp = event["competitions"][0]
    home_team = away_team = ""
    for c in comp.get("competitors", []):
        if c["homeAway"] == "home":
            home_team = c["team"]["displayName"]
        else:
            away_team = c["team"]["displayName"]

    if not home_team or not away_team:
        return None

    home_odds_data = odds.get("homeTeamOdds", {})
    away_odds_data = odds.get("awayTeamOdds", {})

    home_ml = _parse_american(home_odds_data.get("moneyLine"))
    away_ml = _parse_american(away_odds_data.get("moneyLine"))
    if home_ml is None or away_ml is None:
        return None

    raw_home = american_to_implied_prob(home_ml)
    raw_away = american_to_implied_prob(away_ml)
    home_prob, away_prob = remove_vig(raw_home, raw_away)

    # Run line from current point spread
    home_current = home_odds_data.get("current", {})
    away_current = away_odds_data.get("current", {})

    home_rl_str = home_current.get("pointSpread", {}).get("alternateDisplayValue")
    away_rl_str = away_current.get("pointSpread", {}).get("alternateDisplayValue")

    def _parse_rl_point(s) -> float | None:
        if not s:
            return None
        try:
            return float(str(s).replace("+", ""))
        except (ValueError, TypeError):
            return None

    home_rl = _parse_rl_point(home_rl_str)
    # Derive away_rl as the mirror of home_rl — ESPN occasionally returns both with the
    # same sign or wrong direction. Run lines are always ±1.5 and mirror each other.
    if home_rl is not None:
        away_rl = round(-home_rl, 1)
    else:
        away_rl = _parse_rl_point(away_rl_str)

    home_rl_odds = _parse_american(home_current.get("spread", {}).get("american"))
    away_rl_odds = _parse_american(away_current.get("spread", {}).get("american"))

    # Game total
    total_line_raw = odds.get("overUnder")
    try:
        total_line = float(total_line_raw) if total_line_raw is not None else None
    except (ValueError, TypeError):
        total_line = None

    over_odds = _parse_american(odds.get("overOdds"))
    under_odds = _parse_american(odds.get("underOdds"))

    return {
        "game_id": event["id"],
        "commence_time": event.get("date", ""),
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": home_ml,
        "away_odds": away_ml,
        "home_implied_prob": round(home_prob, 4),
        "away_implied_prob": round(away_prob, 4),
        "home_rl": home_rl,
        "away_rl": away_rl,
        "home_rl_odds": home_rl_odds,
        "away_rl_odds": away_rl_odds,
        "total_line": total_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_odds_rows(events: list[dict]) -> list[dict]:
    """Fetch odds for each event and return valid rows, cross-validated against MLB schedule."""
    mlb_games = _fetch_mlb_scheduled_games(datetime.date.today())
    rows = []
    for event in events:
        event_id = event.get("id", "")
        odds = _fetch_espn_odds(event_id)
        if odds is None:
            continue
        row = _build_game_row(event, odds)
        if row is None:
            continue
        # Drop the game if neither team pairing appears in today's MLB schedule.
        # This catches games ESPN lists that are not actually scheduled today.
        if mlb_games:
            home = row["home_team"]
            away = row["away_team"]
            match = any(
                (home_part in home or home in home_part) and
                (away_part in away or away in away_part)
                for away_part, home_part in mlb_games
            )
            if not match:
                continue
        rows.append(row)
    return rows


def get_mlb_odds() -> pd.DataFrame:
    """
    Return today's MLB games with moneyline, run line, and game total from ESPN.
    Postponed/cancelled games are excluded. Cross-validated against the MLB Stats API.
    No API key required. Cached for 2 hours.
    """
    def fetch():
        events = _fetch_espn_scoreboard()
        return pd.DataFrame(_build_odds_rows(events))

    return cached("mlb_odds", fetch, ttl_hours=2.0)


def get_mlb_odds_no_cache() -> pd.DataFrame:
    """Bypass cache — always fetch fresh MLB odds from ESPN."""
    events = _fetch_espn_scoreboard()
    return pd.DataFrame(_build_odds_rows(events))


def get_mlb_totals() -> pd.DataFrame:
    """Return today's game totals."""
    df = get_mlb_odds()
    cols = ["game_id", "commence_time", "home_team", "away_team",
            "total_line", "over_odds", "under_odds"]
    return df[[c for c in cols if c in df.columns]]


def get_mlb_player_props(event_id: str) -> pd.DataFrame:
    """Player props are not available from ESPN. Returns empty DataFrame."""
    return pd.DataFrame()


def get_all_batter_hits_props(games_df: pd.DataFrame) -> pd.DataFrame:
    """Player props are not available from ESPN. Returns empty DataFrame."""
    return pd.DataFrame()


def get_best_prop_lines(props_df: pd.DataFrame) -> pd.DataFrame:
    """No-op — props not available from ESPN."""
    return pd.DataFrame()
