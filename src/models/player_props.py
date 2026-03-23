"""MLB player prop projection models."""

from __future__ import annotations

import pandas as pd

LEAGUE_AVG_BABIP = 0.300
MIN_PROP_EDGE_PCT = 5.0  # % edge threshold to flag a prop bet


# ---------------------------------------------------------------------------
# Pitcher prop models
# ---------------------------------------------------------------------------

def project_pitcher_strikeouts(
    pitcher_k_pct: float,
    pitcher_ip_per_start: float,
    opponent_k_pct: float | None = None,
    umpire_zone_factor: float = 1.0,
) -> float:
    """
    Project strikeouts for a starting pitcher.

    Parameters
    ----------
    pitcher_k_pct : float
        Pitcher's season K% (fraction, e.g. 0.28 = 28%)
    pitcher_ip_per_start : float
        Average innings pitched per start (proxy for batters faced)
    opponent_k_pct : float | None
        Opposing team's strikeout rate as batters. If None uses league avg (0.228)
    umpire_zone_factor : float
        Multiplier for umpire strike zone tendency (>1 = wider zone, more K)
    """
    league_avg_batter_k = 0.228
    opp_k = opponent_k_pct if opponent_k_pct is not None else league_avg_batter_k

    # Blended K rate: 60% pitcher skill, 40% batter tendency
    blended_k_rate = pitcher_k_pct * 0.60 + opp_k * 0.40

    # Batters faced ≈ IP * 4.3 (league average)
    batters_faced = pitcher_ip_per_start * 4.3

    projected_k = blended_k_rate * batters_faced * umpire_zone_factor
    return round(projected_k, 2)


def project_pitcher_earned_runs(
    pitcher_fip: float,
    pitcher_xfip: float | None,
    innings_projected: float,
    park_factor: float = 1.0,
) -> float:
    """
    Project earned runs for a starting pitcher using FIP/xFIP blended ERA estimate.
    """
    # Use xFIP if available; it normalizes HR rate and is more stable
    true_talent_era = pitcher_xfip if pitcher_xfip is not None else pitcher_fip

    # Blend slightly toward FIP for current-season weighting
    if pitcher_xfip is not None:
        blended_era = pitcher_fip * 0.40 + pitcher_xfip * 0.60
    else:
        blended_era = pitcher_fip

    # ERA is per 9 innings
    projected_er = (blended_era / 9.0) * innings_projected * park_factor
    return round(projected_er, 2)


# ---------------------------------------------------------------------------
# Batter prop models
# ---------------------------------------------------------------------------

def project_batter_hits(
    batter_babip: float,
    at_bats_projected: float,
    batter_k_pct: float,
    pitcher_babip_allowed: float | None = None,
) -> float:
    """
    Project hits for a batter using BABIP regression.

    The key insight: if a batter's BABIP is significantly above or below .300,
    their actual hit rate is inflating or deflating their expected hits.
    """
    # Blend batter BABIP with league average (regression to mean)
    regressed_babip = batter_babip * 0.60 + LEAGUE_AVG_BABIP * 0.40

    # If pitcher BABIP allowed is known, factor in slightly
    if pitcher_babip_allowed is not None:
        regressed_babip = regressed_babip * 0.70 + pitcher_babip_allowed * 0.30

    # Balls in play = AB * (1 - K%) * (1 - HR rate approx 0.04)
    balls_in_play = at_bats_projected * (1 - batter_k_pct) * 0.96

    projected_hits = regressed_babip * balls_in_play
    return round(projected_hits, 2)


def project_batter_total_bases(
    batter_slg: float,
    at_bats_projected: float,
    park_hr_factor: float = 1.0,
) -> float:
    """
    Project total bases using slugging percentage.

    SLG = total bases / AB, so TB = SLG * AB
    """
    # Apply mild park factor adjustment for HR component (~30% of SLG)
    adjusted_slg = batter_slg * (1 + (park_hr_factor - 1) * 0.3)
    projected_tb = adjusted_slg * at_bats_projected
    return round(projected_tb, 2)


def project_batter_home_runs(
    batter_barrel_pct: float,
    batter_fb_pct: float,
    at_bats_projected: float,
    park_hr_factor: float = 1.0,
) -> float:
    """
    Project home runs using barrel rate and fly ball tendency.
    Roughly: HR ≈ FB% * HR/FB_rate * AB, where HR/FB ≈ barrel% * 0.55
    """
    hr_per_fb = batter_barrel_pct * 0.55  # empirical scaling
    fly_balls = at_bats_projected * batter_fb_pct
    projected_hr = fly_balls * hr_per_fb * park_hr_factor
    return round(projected_hr, 2)


# ---------------------------------------------------------------------------
# Prop edge calculation
# ---------------------------------------------------------------------------

def compute_prop_edge(
    model_projection: float,
    prop_line: float,
    bet_direction: str,  # "OVER" or "UNDER"
    juice: int = -110,
) -> dict:
    """
    Compute edge for a player prop bet.

    Parameters
    ----------
    model_projection : float
        Model's projected stat value
    prop_line : float
        Sportsbook's over/under line
    bet_direction : str
        "OVER" or "UNDER"
    juice : int
        American odds on the prop (default -110)

    Returns
    -------
    dict with: model_proj, prop_line, bet_direction, edge_pct, recommendation
    """
    # Rough normal distribution assumption: model projection is mean
    # edge is approximated as distance from line relative to expected variance
    distance = model_projection - prop_line

    if bet_direction == "OVER":
        edge_raw = distance / max(abs(prop_line), 0.5)
    else:
        edge_raw = -distance / max(abs(prop_line), 0.5)

    edge_pct = round(edge_raw * 100, 2)

    # Convert juice to implied probability
    if juice < 0:
        implied_prob = abs(juice) / (abs(juice) + 100)
    else:
        implied_prob = 100 / (juice + 100)

    return {
        "model_projection": round(model_projection, 2),
        "prop_line": prop_line,
        "bet_direction": bet_direction,
        "edge_pct": edge_pct,
        "implied_prob": round(implied_prob, 4),
        "recommendation": "BET" if edge_pct >= MIN_PROP_EDGE_PCT else "PASS",
    }


# ---------------------------------------------------------------------------
# Batch prop evaluation
# ---------------------------------------------------------------------------

def evaluate_pitcher_k_props(
    pitcher_df: pd.DataFrame,
    prop_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate strikeout props for a list of pitchers.

    pitcher_df required columns: name, team, k_pct, ip
    prop_lines required columns: name, k_line (sportsbook strikeout O/U)

    Returns DataFrame with projection vs. line edge for each pitcher.
    """
    merged = pitcher_df.merge(prop_lines, on="name", how="inner")
    results = []

    for _, row in merged.iterrows():
        # Average IP per start: use total IP, assume 30 starts as baseline
        ip_per_start = row.get("ip", 150) / 30

        proj_k = project_pitcher_strikeouts(
            pitcher_k_pct=row.get("k_pct", 0.22),
            pitcher_ip_per_start=ip_per_start,
        )
        line = row["k_line"]
        direction = "OVER" if proj_k > line else "UNDER"
        edge = compute_prop_edge(proj_k, line, direction)

        results.append({
            "name": row["name"],
            "team": row["team"],
            "prop_type": "Strikeouts",
            "model_projection": proj_k,
            "prop_line": line,
            "bet_direction": direction,
            "edge_pct": edge["edge_pct"],
            "recommendation": edge["recommendation"],
            "k_pct": row.get("k_pct"),
            "babip": row.get("babip"),
            "xfip": row.get("xfip"),
        })

    return pd.DataFrame(results).sort_values("edge_pct", ascending=False)


def evaluate_batter_hit_props(
    batter_df: pd.DataFrame,
    prop_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate hits props for a list of batters using BABIP regression.

    batter_df required columns: name, team, babip, k_pct, ab (or pa)
    prop_lines required columns: name, hits_line
    """
    merged = batter_df.merge(prop_lines, on="name", how="inner")
    results = []

    for _, row in merged.iterrows():
        ab_per_game = row.get("ab", 500) / 140  # approx per game

        proj_hits = project_batter_hits(
            batter_babip=row.get("babip", LEAGUE_AVG_BABIP),
            at_bats_projected=ab_per_game,
            batter_k_pct=row.get("k_pct", 0.22),
        )
        line = row["hits_line"]
        direction = "OVER" if proj_hits > line else "UNDER"
        edge = compute_prop_edge(proj_hits, line, direction)

        results.append({
            "name": row["name"],
            "team": row["team"],
            "prop_type": "Hits",
            "model_projection": proj_hits,
            "prop_line": line,
            "bet_direction": direction,
            "edge_pct": edge["edge_pct"],
            "recommendation": edge["recommendation"],
            "babip": row.get("babip"),
            "wrc_plus": row.get("wrc_plus"),
        })

    return pd.DataFrame(results).sort_values("edge_pct", ascending=False)
