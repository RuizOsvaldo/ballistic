"""Team and pitcher stats — MLB Stats API primary, pybaseball/FanGraphs for advanced metrics."""

from __future__ import annotations

import requests
import pandas as pd
import pybaseball as pb

from src.data.cache import cached

pb.cache.enable()

MLB_API = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Team stats — MLB Stats API (free, no key, no scraping)
# ---------------------------------------------------------------------------

def get_team_batting(season: int) -> pd.DataFrame:
    """Return team batting totals from MLB Stats API: team_id, team, games, runs_scored."""
    def fetch():
        try:
            resp = requests.get(
                f"{MLB_API}/teams/stats",
                params={"season": season, "sportId": 1, "stats": "season", "group": "hitting"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            splits = resp.json()["stats"][0]["splits"]
            rows = []
            for s in splits:
                stat = s.get("stat", {})
                rows.append({
                    "team_id": s["team"]["id"],
                    "team": s["team"]["name"],
                    "games": stat.get("gamesPlayed", 0),
                    "runs_scored": stat.get("runs", 0),
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    return cached(f"team_batting_{season}", fetch, ttl_hours=6.0)


def get_team_pitching(season: int) -> pd.DataFrame:
    """Return team pitching totals from MLB Stats API: team_id, runs_allowed, era."""
    def fetch():
        try:
            resp = requests.get(
                f"{MLB_API}/teams/stats",
                params={"season": season, "sportId": 1, "stats": "season", "group": "pitching"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            splits = resp.json()["stats"][0]["splits"]
            rows = []
            for s in splits:
                stat = s.get("stat", {})
                era_str = stat.get("era", "0.00")
                try:
                    era = float(era_str)
                except (ValueError, TypeError):
                    era = None
                rows.append({
                    "team_id": s["team"]["id"],
                    "runs_allowed": stat.get("runs", 0),
                    "era": era,
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    return cached(f"team_pitching_{season}", fetch, ttl_hours=6.0)


def get_team_records(season: int) -> pd.DataFrame:
    """Return wins, losses, win pct from MLB Stats API standings."""
    def fetch():
        try:
            resp = requests.get(
                f"{MLB_API}/standings",
                params={"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = []
            for record in data.get("records", []):
                for tr in record.get("teamRecords", []):
                    w = tr.get("wins", 0)
                    l = tr.get("losses", 0)
                    rows.append({
                        "team_id": tr["team"]["id"],
                        "wins": w,
                        "losses": l,
                        "win_pct": w / (w + l) if (w + l) > 0 else 0.0,
                    })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    return cached(f"team_records_{season}", fetch, ttl_hours=6.0)


# FanGraphs abbreviation → full team name (matches Odds API + pybaseball standings)
_ABBR_TO_FULL: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATH": "Athletics",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}


def get_team_stats(season: int) -> pd.DataFrame:
    """
    Merge batting, pitching, and records into a single team stats DataFrame.
    Columns: team, games, wins, losses, win_pct, runs_scored, runs_allowed,
             run_diff, era

    Merges on team_id (numeric) to avoid name-format mismatches between
    standings (short names) and stats endpoints (full names).
    """
    batting  = get_team_batting(season)
    pitching = get_team_pitching(season)
    records  = get_team_records(season)

    if records.empty or batting.empty:
        return pd.DataFrame()

    # batting has full team names + team_id; use it as the base for canonical names
    df = batting.merge(records, on="team_id", how="left")
    df = df.merge(pitching, on="team_id", how="left")
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    # Drop team_id from final output
    df = df.drop(columns=["team_id"], errors="ignore")
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

def get_batter_stats(season: int, min_pa: int = 10) -> pd.DataFrame:
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
