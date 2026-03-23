"""NFL EPA model — composite efficiency, regression signals, and spread equivalent."""

from __future__ import annotations

import math
import pandas as pd

# EPA/play to spread-points conversion factor
# 1 EPA/play differential ≈ 14 spread points over a full game
EPA_TO_SPREAD = 14.0

# Win probability logistic scale: 7-point spread ≈ 73% win prob
SPREAD_TO_WIN_SCALE = 7.0

# Thresholds for regression signals
EPA_SIGNAL_THRESHOLD = 0.05    # ±0.05 composite deviation from implied
EPA_HIGH_THRESHOLD = 0.10


def epa_composite_to_spread(composite_diff: float) -> float:
    """
    Convert EPA composite differential to spread equivalent.

    composite_diff = home_epa_composite - away_epa_composite (+ rest/home adj already included)
    Positive = home team is better.
    """
    return composite_diff * EPA_TO_SPREAD


def spread_to_win_prob(spread_equivalent: float) -> float:
    """
    Convert spread equivalent to win probability using logistic transform.

    spread_equivalent = points home team is favored by (positive = home favored)
    At spread = 7 points: P(win) ≈ 73%
    """
    return 1 / (1 + math.exp(-spread_equivalent / SPREAD_TO_WIN_SCALE))


def compute_implied_epa_composite(win_pct: float) -> float:
    """Back-calculate EPA composite implied by a team's actual win percentage."""
    if win_pct <= 0:
        return -0.30
    if win_pct >= 1:
        return 0.30
    spread_impl = -SPREAD_TO_WIN_SCALE * math.log((1 / win_pct) - 1)
    return spread_impl / EPA_TO_SPREAD


def compute_epa_signals(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EPA regression signals to team DataFrame.

    Required columns: epa_composite, off_epa, def_epa
    Optional: win_pct (to compute deviation from record)

    Added columns:
      - epa_signal       : "Overvalued" | "Undervalued" | "On Track"
      - epa_severity     : "High" | "Medium" | "Low" | "None"
      - epa_direction    : "Likely to decline" | "Likely to improve" | "Stable"
      - implied_epa      : EPA composite implied by win_pct
      - epa_deviation    : epa_composite - implied_epa
    """
    df = team_df.copy()

    if "win_pct" in df.columns:
        df["implied_epa"] = df["win_pct"].apply(compute_implied_epa_composite).round(4)
        df["epa_deviation"] = (df["epa_composite"] - df["implied_epa"]).round(4)
    else:
        df["implied_epa"] = 0.0
        df["epa_deviation"] = df["epa_composite"]

    def _signal(dev: float):
        abs_dev = abs(dev)
        if abs_dev < EPA_SIGNAL_THRESHOLD:
            return pd.Series({"epa_signal": "On Track", "epa_severity": "None", "epa_direction": "Stable"})
        sev = "High" if abs_dev >= EPA_HIGH_THRESHOLD else "Medium"
        if dev > 0:
            return pd.Series({"epa_signal": "Undervalued", "epa_severity": sev, "epa_direction": "Likely to improve"})
        return pd.Series({"epa_signal": "Overvalued", "epa_severity": sev, "epa_direction": "Likely to decline"})

    signals = df["epa_deviation"].apply(_signal)
    return pd.concat([df, signals], axis=1)


def compute_nfl_win_probabilities(
    games_df: pd.DataFrame,
    team_epa_df: pd.DataFrame,
    rest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute model win probabilities for NFL games.

    games_df required columns: home_team, away_team
    team_epa_df required columns: team, epa_composite
    rest_df optional: team, rest_adjustment (spread points)

    Added columns: home_model_prob, away_model_prob, spread_equivalent
    """
    df = games_df.copy()
    epa_map = team_epa_df.set_index("team")["epa_composite"].to_dict() if "team" in team_epa_df.columns else {}

    rest_adj: dict[str, float] = {}
    if rest_df is not None and not rest_df.empty and "team" in rest_df.columns:
        rest_adj = rest_df.set_index("team")["rest_adjustment"].to_dict()

    home_probs, away_probs, spreads = [], [], []

    for _, row in df.iterrows():
        home = row.get("home_team", "")
        away = row.get("away_team", "")

        home_epa = epa_map.get(home)
        away_epa = epa_map.get(away)

        if home_epa is None or away_epa is None:
            home_probs.append(float("nan"))
            away_probs.append(float("nan"))
            spreads.append(float("nan"))
            continue

        # EPA composite diff + home field + rest
        from src.sports.football.models.rest_schedule import HOME_FIELD_PTS
        composite_diff = home_epa - away_epa
        rest_net = rest_adj.get(home, 0.0) - rest_adj.get(away, 0.0)

        spread_eq = epa_composite_to_spread(composite_diff) + HOME_FIELD_PTS + rest_net
        hp = spread_to_win_prob(spread_eq)
        hp = max(0.05, min(0.95, hp))

        home_probs.append(round(hp, 4))
        away_probs.append(round(1 - hp, 4))
        spreads.append(round(-spread_eq, 2))   # negative = home team favored (standard convention)

    df["home_model_prob"] = home_probs
    df["away_model_prob"] = away_probs
    df["spread_equivalent"] = spreads
    return df
