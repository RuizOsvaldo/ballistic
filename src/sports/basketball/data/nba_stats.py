"""NBA data layer — nba_api wrappers for team stats, player stats, and schedule."""

from __future__ import annotations

import time
import datetime
import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguedashplayerstats,
    scoreboardv2,
    teamgamelog,
)
from nba_api.stats.static import teams as nba_teams_static

from src.data.cache import cached

# nba_api is rate-limited at stats.nba.com — space out calls
_NBA_SLEEP = 0.6


def _sleep():
    time.sleep(_NBA_SLEEP)


def _current_season() -> str:
    """Return NBA season string e.g. '2024-25'."""
    now = datetime.datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Team stats
# ---------------------------------------------------------------------------

def get_nba_team_stats(season: str | None = None) -> pd.DataFrame:
    """
    Return team stats including net rating, four factors, pace, and W-L.

    Columns: team_id, team_name, team_abbrev, w, l, win_pct,
             ortg, drtg, net_rtg, pace,
             efg_pct, opp_efg_pct, tov_pct, opp_tov_pct,
             oreb_pct, ft_rate, opp_ft_rate
    """
    season = season or _current_season()

    def fetch():
        _sleep()
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
        )
        df = endpoint.get_data_frames()[0]

        rename = {
            "TEAM_ID": "team_id",
            "TEAM_NAME": "team_name",
            "W": "w",
            "L": "l",
            "W_PCT": "win_pct",
            "OFF_RATING": "ortg",
            "DEF_RATING": "drtg",
            "NET_RATING": "net_rtg",
            "PACE": "pace",
            "EFG_PCT": "efg_pct",
            "OPP_EFG_PCT": "opp_efg_pct",
            "TM_TOV_PCT": "tov_pct",
            "OPP_TOV_PCT": "opp_tov_pct",
            "OREB_PCT": "oreb_pct",
            "FTA_RATE": "ft_rate",
            "OPP_FTA_RATE": "opp_ft_rate",
        }
        df = df.rename(columns=rename)

        # Add abbreviation from static team data
        abbrev_map = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
        df["team_abbrev"] = df["team_id"].map(abbrev_map)

        cols = [
            "team_id", "team_name", "team_abbrev", "w", "l", "win_pct",
            "ortg", "drtg", "net_rtg", "pace",
            "efg_pct", "opp_efg_pct", "tov_pct", "opp_tov_pct",
            "oreb_pct", "ft_rate", "opp_ft_rate",
        ]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    return cached(f"nba_team_stats_{season}", fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Player stats
# ---------------------------------------------------------------------------

def get_nba_player_stats(season: str | None = None, min_minutes: int = 15) -> pd.DataFrame:
    """
    Return player stats: season averages + rolling context.

    Columns: player_id, player_name, team_abbrev, min, pts, reb, ast,
             usg_pct, ts_pct, three_pm, three_pa, three_pct,
             ortg, drtg, w_pct (team)
    """
    season = season or _current_season()

    def fetch():
        _sleep()
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
        )
        adv = endpoint.get_data_frames()[0]

        _sleep()
        endpoint2 = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_simple="PerGame",
        )
        base = endpoint2.get_data_frames()[0]

        adv_rename = {
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "TEAM_ABBREVIATION": "team_abbrev",
            "MIN": "min",
            "USG_PCT": "usg_pct",
            "TS_PCT": "ts_pct",
            "OFF_RATING": "ortg",
            "DEF_RATING": "drtg",
        }
        base_rename = {
            "PLAYER_ID": "player_id",
            "PTS": "pts",
            "REB": "reb",
            "AST": "ast",
            "FG3M": "three_pm",
            "FG3A": "three_pa",
            "FG3_PCT": "three_pct",
        }

        adv = adv.rename(columns=adv_rename)
        base = base.rename(columns=base_rename)

        adv_cols = ["player_id", "player_name", "team_abbrev", "min", "usg_pct", "ts_pct", "ortg", "drtg"]
        base_cols = ["player_id", "pts", "reb", "ast", "three_pm", "three_pa", "three_pct"]

        adv = adv[[c for c in adv_cols if c in adv.columns]]
        base = base[[c for c in base_cols if c in base.columns]]

        df = adv.merge(base, on="player_id", how="left")
        df = df[df["min"] >= min_minutes].copy()
        return df.reset_index(drop=True)

    return cached(f"nba_player_stats_{season}", fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Team game log — for rolling averages and rest calculation
# ---------------------------------------------------------------------------

def get_team_game_log(team_id: int, season: str | None = None, last_n: int = 15) -> pd.DataFrame:
    """
    Return last N games for a team.

    Columns: game_id, game_date, matchup, wl, pts, opp_pts, net_pts
    """
    season = season or _current_season()

    def fetch():
        _sleep()
        endpoint = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
        )
        df = endpoint.get_data_frames()[0]
        rename = {
            "Game_ID": "game_id",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "wl",
            "PTS": "pts",
        }
        df = df.rename(columns=rename)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date", ascending=False).head(last_n)

        # Compute opponent points from matchup context if available
        if "pts" in df.columns:
            df["net_pts"] = df["pts"].diff(-1).fillna(0)

        cols = ["game_id", "game_date", "matchup", "wl", "pts"]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    cache_key = f"nba_game_log_{team_id}_{season}"
    return cached(cache_key, fetch, ttl_hours=2.0)


# ---------------------------------------------------------------------------
# Today's schedule
# ---------------------------------------------------------------------------

def get_todays_nba_games() -> pd.DataFrame:
    """
    Return today's NBA games.

    Columns: game_id, home_team_id, home_team, away_team_id, away_team,
             game_time, arena
    """
    today = datetime.datetime.now().strftime("%m/%d/%Y")

    def fetch():
        _sleep()
        board = scoreboardv2.ScoreboardV2(game_date=today)
        games = board.get_data_frames()[0]

        team_map = {t["id"]: t["full_name"] for t in nba_teams_static.get_teams()}

        rename = {
            "GAME_ID": "game_id",
            "HOME_TEAM_ID": "home_team_id",
            "VISITOR_TEAM_ID": "away_team_id",
            "GAME_STATUS_TEXT": "game_time",
            "ARENA_NAME": "arena",
        }
        games = games.rename(columns=rename)
        games["home_team"] = games["home_team_id"].map(team_map).fillna("Unknown")
        games["away_team"] = games["away_team_id"].map(team_map).fillna("Unknown")

        cols = ["game_id", "home_team_id", "home_team", "away_team_id", "away_team", "game_time"]
        available = [c for c in cols if c in games.columns]
        return games[available].copy()

    return cached(f"nba_today_{today.replace('/', '_')}", fetch, ttl_hours=2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_team_id_map() -> dict[str, int]:
    """Return {team_name: team_id} map from nba_api static data."""
    return {t["full_name"]: t["id"] for t in nba_teams_static.get_teams()}


def get_team_abbrev_map() -> dict[str, str]:
    """Return {team_name: abbreviation} map."""
    return {t["full_name"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
