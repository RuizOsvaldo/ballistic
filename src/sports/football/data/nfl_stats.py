"""NFL data layer — nfl_data_py wrappers for team EPA, player stats, and schedule."""

from __future__ import annotations

import datetime
import warnings
import pandas as pd
import nfl_data_py as nfl

from src.data.cache import cached

warnings.filterwarnings("ignore", category=FutureWarning)


def _current_season() -> int:
    now = datetime.datetime.now()
    return now.year if now.month >= 9 else now.year - 1


# ---------------------------------------------------------------------------
# Team EPA stats — computed from play-by-play
# ---------------------------------------------------------------------------

def get_nfl_team_epa(season: int | None = None) -> pd.DataFrame:
    """
    Return team offensive and defensive EPA/play for the season.

    Columns: team, off_epa, def_epa, off_pass_epa, off_rush_epa,
             off_success_rate, def_success_rate, epa_composite,
             implied_win_pct, games
    """
    season = season or _current_season()

    def fetch():
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "play_type", "epa", "posteam", "defteam",
                "game_id", "week", "pass_attempt", "rush_attempt", "success",
            ],
        )
        # Filter to actual plays
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
        pbp = pbp[pbp["epa"].notna()].copy()

        # Offensive EPA/play
        off = (
            pbp.groupby("posteam")
            .agg(
                off_epa=("epa", "mean"),
                off_pass_epa=("epa", lambda x: x[pbp.loc[x.index, "pass_attempt"] == 1].mean() if len(x) > 0 else 0),
                off_rush_epa=("epa", lambda x: x[pbp.loc[x.index, "rush_attempt"] == 1].mean() if len(x) > 0 else 0),
                off_success_rate=("success", "mean"),
                games=("game_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

        # Defensive EPA/play (negative = good defense)
        def_stats = (
            pbp.groupby("defteam")
            .agg(
                def_epa=("epa", "mean"),
                def_success_rate=("success", "mean"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )

        df = off.merge(def_stats, on="team", how="outer")

        # Composite efficiency: off_epa minus def_epa_allowed
        # Good defense has negative def_epa, so subtracting a negative number boosts the composite
        df["epa_composite"] = (df["off_epa"] - df["def_epa"]).round(4)
        df["off_epa"] = df["off_epa"].round(4)
        df["def_epa"] = df["def_epa"].round(4)
        df["off_success_rate"] = df["off_success_rate"].round(4)
        df["def_success_rate"] = df["def_success_rate"].round(4)

        return df.sort_values("epa_composite", ascending=False).reset_index(drop=True)

    return cached(f"nfl_team_epa_{season}", fetch, ttl_hours=12.0)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_nfl_schedule(season: int | None = None) -> pd.DataFrame:
    """
    Return NFL schedule with spread lines, totals, stadium type, and weather.

    Columns: game_id, week, gameday, home_team, away_team,
             spread_line, total_line, roof, surface,
             temp, wind, home_score, away_score
    """
    season = season or _current_season()

    def fetch():
        df = nfl.import_schedules([season])
        cols = [
            "game_id", "week", "gameday", "home_team", "away_team",
            "spread_line", "total_line", "roof", "surface",
            "temp", "wind", "home_score", "away_score",
        ]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        if "gameday" in df.columns:
            df["gameday"] = pd.to_datetime(df["gameday"])
        return df

    return cached(f"nfl_schedule_{season}", fetch, ttl_hours=2.0)


def get_upcoming_nfl_games(season: int | None = None) -> pd.DataFrame:
    """Return games that haven't been played yet (no home_score)."""
    schedule = get_nfl_schedule(season)
    if schedule.empty:
        return schedule
    if "home_score" in schedule.columns:
        return schedule[schedule["home_score"].isna()].copy()
    return schedule


def get_current_week_games(season: int | None = None) -> pd.DataFrame:
    """Return games for the current NFL week."""
    upcoming = get_upcoming_nfl_games(season)
    if upcoming.empty or "week" not in upcoming.columns:
        return upcoming
    min_week = upcoming["week"].min()
    return upcoming[upcoming["week"] == min_week].copy()


# ---------------------------------------------------------------------------
# Player weekly stats
# ---------------------------------------------------------------------------

def get_nfl_player_stats(season: int | None = None) -> pd.DataFrame:
    """
    Return seasonal player stats (summed from weekly).

    Columns: player_id, player_name, position, recent_team,
             games, completions, attempts, passing_yards, passing_tds,
             interceptions, carries, rushing_yards, rushing_tds,
             receptions, targets, receiving_yards, receiving_tds,
             air_yards, target_share, epa (total)
    """
    season = season or _current_season()

    def fetch():
        weekly = nfl.import_weekly_data([season])

        group_cols = [c for c in ["player_id", "player_name", "position", "recent_team"] if c in weekly.columns]
        sum_cols = [
            c for c in [
                "completions", "attempts", "passing_yards", "passing_tds",
                "interceptions", "carries", "rushing_yards", "rushing_tds",
                "receptions", "targets", "receiving_yards", "receiving_tds",
                "target_share", "air_yards",
            ]
            if c in weekly.columns
        ]

        df = weekly.groupby(group_cols)[sum_cols].sum().reset_index()
        df["games"] = weekly.groupby(group_cols[0])["week"].nunique().values[:len(df)] if "week" in weekly.columns else 1

        # Per-game averages for key stats
        for col in ["passing_yards", "rushing_yards", "receiving_yards"]:
            if col in df.columns:
                df[f"{col}_pg"] = (df[col] / df["games"].clip(lower=1)).round(1)

        return df

    return cached(f"nfl_player_stats_{season}", fetch, ttl_hours=6.0)


def get_nfl_snap_counts(season: int | None = None) -> pd.DataFrame:
    """Return snap count data for usage context."""
    season = season or _current_season()

    def fetch():
        df = nfl.import_snap_counts([season])
        group_cols = [c for c in ["player_id", "player_name", "position", "team"] if c in df.columns]
        if not group_cols:
            return df
        sum_cols = [c for c in ["offense_snaps", "offense_pct"] if c in df.columns]
        return df.groupby(group_cols)[sum_cols].mean().reset_index()

    return cached(f"nfl_snaps_{season}", fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Win totals (preseason market)
# ---------------------------------------------------------------------------

def get_nfl_win_totals(season: int | None = None) -> pd.DataFrame:
    """Return preseason Vegas win total lines per team."""
    season = season or _current_season()

    def fetch():
        try:
            df = nfl.import_win_totals([season])
            cols = [c for c in ["team", "season", "wins", "away_wins", "home_wins"] if c in df.columns]
            return df[cols].copy() if cols else df
        except Exception:
            return pd.DataFrame()

    return cached(f"nfl_win_totals_{season}", fetch, ttl_hours=24.0)
