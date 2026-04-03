"""
Park factors and bullpen stats.

Park factors: 2024/2025 FanGraphs runs-based park factor (100 = neutral).
Values above 100 = hitter-friendly (more runs), below 100 = pitcher-friendly.
These change very slowly year-to-year so a static table is appropriate.

Bullpen stats: fetched from FanGraphs via pybaseball — relievers only
(GS == 0 or Relief-IP > Start-IP), aggregated by team.
"""

from __future__ import annotations

import pandas as pd
import pybaseball as pb

from src.data.cache import cached
from src.data.baseball_stats import _ABBR_TO_FULL

# ---------------------------------------------------------------------------
# Park factors (FanGraphs 2024, runs-based, 3-year regressed)
# ---------------------------------------------------------------------------

PARK_FACTORS: dict[str, float] = {
    "Arizona Diamondbacks": 99,
    "Athletics":            97,
    "Atlanta Braves":       100,
    "Baltimore Orioles":    97,
    "Boston Red Sox":       102,
    "Chicago Cubs":         102,
    "Chicago White Sox":    97,
    "Cincinnati Reds":      103,
    "Cleveland Guardians":  97,
    "Colorado Rockies":     115,   # Coors — most extreme in baseball
    "Detroit Tigers":       97,
    "Houston Astros":       96,
    "Kansas City Royals":   100,
    "Los Angeles Angels":   98,
    "Los Angeles Dodgers":  97,
    "Miami Marlins":        93,    # LoanDepot Park — very pitcher-friendly
    "Milwaukee Brewers":    97,
    "Minnesota Twins":      101,
    "New York Mets":        99,
    "New York Yankees":     102,
    "Philadelphia Phillies":101,
    "Pittsburgh Pirates":   98,
    "San Diego Padres":     94,    # Petco — second most pitcher-friendly
    "San Francisco Giants": 96,
    "Seattle Mariners":     95,
    "St. Louis Cardinals":  98,
    "Tampa Bay Rays":       98,
    "Texas Rangers":        100,
    "Toronto Blue Jays":    101,
    "Washington Nationals": 99,
}


def get_park_factor(home_team: str) -> float:
    """Return the park factor for the home team (100 = neutral). Defaults to 100."""
    return PARK_FACTORS.get(home_team, 100) / 100.0


# ---------------------------------------------------------------------------
# Bullpen stats
# ---------------------------------------------------------------------------

def get_bullpen_stats(season: int) -> pd.DataFrame:
    """
    Return aggregated bullpen stats per team.

    A pitcher is classified as a reliever when their Relief-IP > Start-IP.
    Returns DataFrame with columns:
      team, bullpen_era, bullpen_fip, bullpen_k_pct, bullpen_bb_pct, bullpen_ip
    """
    def fetch() -> pd.DataFrame:
        try:
            df = pb.pitching_stats(season, qual=1)
        except Exception:
            return pd.DataFrame()

        # Classify relievers: GS == 0 is the most reliable signal early in the season.
        # For players with mixed roles mid-season, further require Relief-IP > Start-IP.
        if "GS" in df.columns:
            pure_relievers = df[df["GS"] == 0].copy()
            if "Relief-IP" in df.columns and "Start-IP" in df.columns:
                mixed = df[(df["GS"] > 0) & (df["Relief-IP"] > df["Start-IP"])].copy()
                relievers = pd.concat([pure_relievers, mixed], ignore_index=True)
            else:
                relievers = pure_relievers
        elif "Relief-IP" in df.columns and "Start-IP" in df.columns:
            relievers = df[df["Relief-IP"] > df["Start-IP"]].copy()
        else:
            relievers = df.copy()

        if relievers.empty:
            return pd.DataFrame()

        # Translate team abbreviations to full names
        relievers["team"] = relievers["Team"].map(lambda t: _ABBR_TO_FULL.get(t, t))

        # Weighted aggregate by IP
        rows = []
        for team, grp in relievers.groupby("team"):
            ip = grp["IP"].sum()
            if ip == 0:
                continue
            era = (grp["ERA"] * grp["IP"]).sum() / ip if "ERA" in grp else None
            fip = (grp["FIP"] * grp["IP"]).sum() / ip if "FIP" in grp else None
            k_pct = (grp["K%"] * grp["IP"]).sum() / ip if "K%" in grp else None
            bb_pct = (grp["BB%"] * grp["IP"]).sum() / ip if "BB%" in grp else None
            rows.append({
                "team":          team,
                "bullpen_era":   round(era, 2)   if era   is not None else None,
                "bullpen_fip":   round(fip, 2)   if fip   is not None else None,
                "bullpen_k_pct": round(k_pct, 3) if k_pct is not None else None,
                "bullpen_bb_pct":round(bb_pct, 3)if bb_pct is not None else None,
                "bullpen_ip":    round(ip, 1),
            })

        return pd.DataFrame(rows)

    return cached(f"bullpen_stats_{season}", fetch, ttl_hours=6.0)
