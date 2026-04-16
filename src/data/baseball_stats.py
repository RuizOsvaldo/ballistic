"""Team and pitcher stats — MLB Stats API for all team and individual stats."""

from __future__ import annotations

import requests
import pandas as pd

from src.data.cache import cached

MLB_API = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10


def _fetch_handedness(player_ids: list[int]) -> dict[int, dict]:
    """Batch-fetch pitchHand and batSide for a list of MLB player IDs."""
    if not player_ids:
        return {}
    try:
        resp = requests.get(
            f"{MLB_API}/people",
            params={"personIds": ",".join(str(i) for i in player_ids)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return {
            p["id"]: {
                "pitch_hand": p.get("pitchHand", {}).get("code", "R"),
                "bat_side": p.get("batSide", {}).get("code", "R"),
            }
            for p in resp.json().get("people", [])
        }
    except Exception:
        return {}


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
# Historical team stats — multi-season, optional date range for quarterly view
# ---------------------------------------------------------------------------

def get_historical_team_stats(season: int, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """
    Return per-game team stats for a given season (and optional date range).
    Columns: team, team_id, games, hits_pg, hr_pg, k_pg, runs_pg, runs_allowed_pg,
             run_diff_pg, win_pct, wins, losses
    Uses byDateRange stat type when dates are provided, season otherwise.
    """
    stat_type = "byDateRange" if start_date and end_date else "season"

    def fetch():
        params_base = {"season": season, "sportId": 1, "stats": stat_type}
        if start_date and end_date:
            params_base["startDate"] = start_date
            params_base["endDate"] = end_date

        bat_resp = requests.get(f"{MLB_API}/teams/stats", params={**params_base, "group": "hitting"}, timeout=_TIMEOUT)
        bat_resp.raise_for_status()
        pit_resp = requests.get(f"{MLB_API}/teams/stats", params={**params_base, "group": "pitching"}, timeout=_TIMEOUT)
        pit_resp.raise_for_status()

        # Standings at end_date if quarterly, otherwise current
        std_params = {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason", "hydrate": "team"}
        if end_date:
            std_params["date"] = end_date
        std_resp = requests.get(f"{MLB_API}/standings", params=std_params, timeout=_TIMEOUT)
        std_resp.raise_for_status()

        bat_rows = {}
        for s in bat_resp.json()["stats"][0]["splits"]:
            tid = s["team"]["id"]
            st = s["stat"]
            g = int(st.get("gamesPlayed", 0) or 0)
            bat_rows[tid] = {
                "team": s["team"]["name"],
                "team_id": tid,
                "games": g,
                "hits": int(st.get("hits", 0) or 0),
                "hr": int(st.get("homeRuns", 0) or 0),
                "k": int(st.get("strikeOuts", 0) or 0),
                "runs": int(st.get("runs", 0) or 0),
            }

        pit_rows = {}
        for s in pit_resp.json()["stats"][0]["splits"]:
            tid = s["team"]["id"]
            pit_rows[tid] = {"runs_allowed": int(s["stat"].get("runs", 0) or 0)}

        std_rows = {}
        for division in std_resp.json().get("records", []):
            for tr in division.get("teamRecords", []):
                tid = tr["team"]["id"]
                w = tr.get("wins", 0)
                l = tr.get("losses", 0)
                std_rows[tid] = {"wins": w, "losses": l, "run_diff": tr.get("runDifferential", 0)}

        rows = []
        for tid, bat in bat_rows.items():
            g = max(bat["games"], 1)
            pit = pit_rows.get(tid, {})
            std = std_rows.get(tid, {})
            ra = pit.get("runs_allowed", 0)
            w = std.get("wins", 0)
            l = std.get("losses", 0)
            rows.append({
                "team": bat["team"],
                "team_id": tid,
                "games": bat["games"],
                "wins": w,
                "losses": l,
                "win_pct": round(w / (w + l), 3) if (w + l) > 0 else 0.0,
                "hits_pg": round(bat["hits"] / g, 2),
                "hr_pg": round(bat["hr"] / g, 2),
                "k_pg": round(bat["k"] / g, 2),
                "runs_pg": round(bat["runs"] / g, 2),
                "runs_allowed_pg": round(ra / g, 2),
                "run_diff_pg": round((bat["runs"] - ra) / g, 2),
            })
        return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

    cache_key = f"hist_team_{season}_{start_date or 'full'}_{end_date or 'full'}"
    return cached(cache_key, fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Pitcher stats — includes xFIP, SIERA, K%, BB%
# ---------------------------------------------------------------------------

def get_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Return pitcher stats via MLB Stats API with computed FIP, BABIP, K%, BB%.
    Columns: name, team, ip, era, fip, babip, k_pct, bb_pct
    xfip is not available from MLB API; models fall back to fip when xfip is None.
    FIP constant 3.15 is a reasonable approximation for 2024-2026 seasons.
    """
    FIP_CONSTANT = 3.15

    def fetch():
        resp = requests.get(
            f"{MLB_API}/stats",
            params={"stats": "season", "group": "pitching", "season": season,
                    "sportId": 1, "limit": 500},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
        rows = []
        for s in splits:
            stat = s.get("stat", {})
            player = s.get("player", {})
            team = s.get("team", {})

            ip = float(stat.get("inningsPitched", 0) or 0)
            hr = int(stat.get("homeRuns", 0) or 0)
            bb = int(stat.get("baseOnBalls", 0) or 0)
            hbp = int(stat.get("hitBatsmen", 0) or 0)
            k = int(stat.get("strikeOuts", 0) or 0)
            h = int(stat.get("hits", 0) or 0)
            bf = int(stat.get("battersFaced", 0) or 0)
            ab = int(stat.get("atBats", 0) or 0)  # opponent at-bats against pitcher
            sf = int(stat.get("sacFlies", 0) or 0)

            era_str = stat.get("era", "0.00")
            try:
                era = float(era_str)
            except (ValueError, TypeError):
                era = None

            fip = round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT, 3) if ip > 0 else None

            # BABIP allowed: (H - HR) / (AB - K - HR + SF)
            denom = ab - k - hr + sf
            babip = round((h - hr) / denom, 3) if denom > 0 and h >= hr else None

            k_pct = round(k / bf, 4) if bf > 0 else None
            bb_pct = round(bb / bf, 4) if bf > 0 else None

            rows.append({
                "_player_id": player.get("id"),
                "name": player.get("fullName", ""),
                "team": team.get("name", ""),
                "ip": ip,
                "era": era,
                "fip": fip,
                "xfip": None,
                "babip": babip,
                "k_pct": k_pct,
                "bb_pct": bb_pct,
            })

        hand_map = _fetch_handedness([r["_player_id"] for r in rows if r["_player_id"]])
        for r in rows:
            r["pitch_hand"] = hand_map.get(r.pop("_player_id"), {}).get("pitch_hand", "R")

        return pd.DataFrame(rows)

    return cached(f"pitcher_stats_{season}", fetch, ttl_hours=6.0)


# ---------------------------------------------------------------------------
# Batter stats — wRC+, BABIP, K%, BB%, sprint speed
# ---------------------------------------------------------------------------

def get_batter_stats(season: int, min_pa: int = 10) -> pd.DataFrame:
    """
    Return batter stats via MLB Stats API with computed BABIP, K%, BB%.
    Columns: name, team, pa, ab, avg, obp, slg, babip, k_pct, bb_pct, hr, hits, sb
    wrc_plus is not available from MLB API; displays as None where shown.
    """
    def fetch():
        resp = requests.get(
            f"{MLB_API}/stats",
            params={"stats": "season", "group": "hitting", "season": season,
                    "sportId": 1, "limit": 1000},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
        rows = []
        for s in splits:
            stat = s.get("stat", {})
            player = s.get("player", {})
            team = s.get("team", {})

            pa = int(stat.get("plateAppearances", 0) or 0)
            if pa < min_pa:
                continue

            ab = int(stat.get("atBats", 0) or 0)
            h = int(stat.get("hits", 0) or 0)
            hr = int(stat.get("homeRuns", 0) or 0)
            k = int(stat.get("strikeOuts", 0) or 0)
            bb = int(stat.get("baseOnBalls", 0) or 0)
            sf = int(stat.get("sacFlies", 0) or 0)
            sb = int(stat.get("stolenBases", 0) or 0)

            try:
                avg = float(stat.get("avg", ".000"))
            except (ValueError, TypeError):
                avg = None
            try:
                obp = float(stat.get("obp", ".000"))
            except (ValueError, TypeError):
                obp = None
            try:
                slg = float(stat.get("slg", ".000"))
            except (ValueError, TypeError):
                slg = None

            # babip is provided directly by the MLB Stats API
            try:
                babip = round(float(stat.get("babip", "0") or 0), 3) if stat.get("babip") else None
            except (ValueError, TypeError):
                babip = None

            k_pct = round(k / pa, 4) if pa > 0 else None
            bb_pct = round(bb / pa, 4) if pa > 0 else None

            rows.append({
                "_player_id": player.get("id"),
                "name": player.get("fullName", ""),
                "team": team.get("name", ""),
                "pa": pa,
                "ab": ab,
                "avg": avg,
                "obp": obp,
                "slg": slg,
                "wrc_plus": None,
                "babip": babip,
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "hr": hr,
                "hits": h,
                "sb": sb,
            })

        hand_map = _fetch_handedness([r["_player_id"] for r in rows if r["_player_id"]])
        for r in rows:
            r["bat_side"] = hand_map.get(r.pop("_player_id"), {}).get("bat_side", "R")

        return pd.DataFrame(rows)

    return cached(f"batter_stats_{season}", fetch, ttl_hours=6.0)
