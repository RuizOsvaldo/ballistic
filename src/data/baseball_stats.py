"""pybaseball wrappers for team and pitcher stats — includes extended metrics."""

from __future__ import annotations

import pandas as pd
import pybaseball as pb

from src.data.cache import cached

pb.cache.enable()


# ---------------------------------------------------------------------------
# Team stats
# ---------------------------------------------------------------------------

def get_team_batting(season: int) -> pd.DataFrame:
    """Return team batting totals: team, G, R (runs scored)."""
    def fetch():
        df = pb.team_batting(season)
        df = df.rename(columns={"Team": "team", "R": "runs_scored", "G": "games"})
        return df[["team", "games", "runs_scored"]].copy()

    return cached(f"team_batting_{season}", fetch, ttl_hours=6.0)


def get_team_pitching(season: int) -> pd.DataFrame:
    """Return team pitching totals: team, RA, ERA, FIP."""
    def fetch():
        df = pb.team_pitching(season)
        df = df.rename(columns={
            "Team": "team",
            "R": "runs_allowed",
            "ERA": "era",
            "FIP": "fip",
        })
        cols = ["team", "runs_allowed", "era", "fip"]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    return cached(f"team_pitching_{season}", fetch, ttl_hours=6.0)


def get_team_records(season: int) -> pd.DataFrame:
    """Return wins, losses, win pct for all teams."""
    def fetch():
        frames = pb.standings(season)
        rows = []
        for div_df in frames:
            for _, row in div_df.iterrows():
                try:
                    w = int(row["W"])
                    l = int(row["L"])
                    rows.append({
                        "team": row["Tm"],
                        "wins": w,
                        "losses": l,
                        "win_pct": w / (w + l) if (w + l) > 0 else 0.0,
                    })
                except (KeyError, ValueError):
                    continue
        return pd.DataFrame(rows)

    return cached(f"team_records_{season}", fetch, ttl_hours=6.0)


def get_team_stats(season: int) -> pd.DataFrame:
    """
    Merge batting, pitching, and records into a single team stats DataFrame.
    Columns: team, games, wins, losses, win_pct, runs_scored, runs_allowed,
             run_diff, era, fip
    """
    batting = get_team_batting(season)
    pitching = get_team_pitching(season)
    records = get_team_records(season)

    df = records.merge(batting, on="team", how="left")
    df = df.merge(pitching, on="team", how="left")
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    return df


# ---------------------------------------------------------------------------
# Pitcher stats — includes xFIP, SIERA, K%, BB%
# ---------------------------------------------------------------------------

def get_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Return qualified starter stats via FanGraphs pitching leaderboard.
    Columns: name, team, ip, era, fip, xfip, siera, babip, k_pct, bb_pct, hr9, whiff_pct
    """
    def fetch():
        df = pb.pitching_stats(season, qual=1)
        rename = {
            "Name": "name",
            "Team": "team",
            "IP": "ip",
            "ERA": "era",
            "FIP": "fip",
            "xFIP": "xfip",
            "SIERA": "siera",
            "BABIP": "babip",
            "K%": "k_pct",
            "BB%": "bb_pct",
            "HR/9": "hr9",
            "Whiff%": "whiff_pct",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
        }
        df = df.rename(columns=rename)
        cols = ["name", "team", "ip", "era", "fip", "xfip", "siera",
                "babip", "k_pct", "bb_pct", "hr9", "whiff_pct", "gb_pct", "fb_pct"]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    return cached(f"pitcher_stats_{season}", fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Batter stats — wRC+, BABIP, K%, BB%, sprint speed
# ---------------------------------------------------------------------------

def get_batter_stats(season: int, min_pa: int = 100) -> pd.DataFrame:
    """
    Return qualified batter stats via FanGraphs batting leaderboard.
    Columns: name, team, pa, avg, obp, slg, wrc_plus, babip, k_pct, bb_pct,
             hard_hit_pct, barrel_pct, exit_velocity
    """
    def fetch():
        df = pb.batting_stats(season, qual=min_pa)
        rename = {
            "Name": "name",
            "Team": "team",
            "PA": "pa",
            "AVG": "avg",
            "OBP": "obp",
            "SLG": "slg",
            "wRC+": "wrc_plus",
            "BABIP": "babip",
            "K%": "k_pct",
            "BB%": "bb_pct",
            "Hard%": "hard_hit_pct",
            "Barrel%": "barrel_pct",
            "EV": "exit_velocity",
            "HR": "hr",
            "H": "hits",
            "AB": "ab",
            "SB": "sb",
        }
        df = df.rename(columns=rename)
        cols = ["name", "team", "pa", "avg", "obp", "slg", "wrc_plus",
                "babip", "k_pct", "bb_pct", "hard_hit_pct", "barrel_pct",
                "exit_velocity", "hr", "hits", "ab", "sb"]
        available = [c for c in cols if c in df.columns]
        return df[available].copy()

    return cached(f"batter_stats_{season}", fetch, ttl_hours=6.0)
