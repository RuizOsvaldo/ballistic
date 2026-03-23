"""NBA net rating model — team quality signal and win probability base."""

from __future__ import annotations

import math
import pandas as pd

# Net rating deviation threshold for regression signal (points)
NET_RTG_SIGNAL_THRESHOLD = 5.0   # ±5 net rtg pts divergence from implied W% = flag
NET_RTG_HIGH_THRESHOLD = 10.0

# Home court advantage in net rating points
HOME_COURT_ADJ = 3.0

# Logistic scale factor: each net rating point ≈ 2.7% win prob
LOGISTIC_SCALE = 0.10


def net_rtg_to_win_prob(net_rtg_diff: float) -> float:
    """
    Convert net rating differential to win probability using logistic transform.

    net_rtg_diff = home_net_rtg - away_net_rtg (+ home court already included)
    Each point of net rating ≈ 2.7% win probability at the midpoint.
    """
    return 1 / (1 + math.exp(-net_rtg_diff * LOGISTIC_SCALE))


def compute_implied_net_rtg(win_pct: float) -> float:
    """Back-calculate the net rating implied by a team's actual win percentage."""
    if win_pct <= 0:
        return -20.0
    if win_pct >= 1:
        return 20.0
    return math.log(win_pct / (1 - win_pct)) / LOGISTIC_SCALE


def compute_net_rating_signals(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add net rating deviation signals to team DataFrame.

    Required columns: net_rtg, win_pct
    Added columns:
      - implied_net_rtg     : net rating implied by actual W%
      - net_rtg_deviation   : actual net_rtg - implied_net_rtg
      - net_rtg_signal      : "Overperforming" | "Underperforming" | "On Track"
      - net_rtg_severity    : "High" | "Medium" | "Low" | "None"
      - net_rtg_direction   : "Likely to decline" | "Likely to improve" | "Stable"
    """
    df = team_df.copy()

    df["implied_net_rtg"] = df["win_pct"].apply(compute_implied_net_rtg).round(2)
    df["net_rtg_deviation"] = (df["net_rtg"] - df["implied_net_rtg"]).round(2)

    def _signal(dev: float):
        abs_dev = abs(dev)
        if abs_dev < NET_RTG_SIGNAL_THRESHOLD:
            return pd.Series({
                "net_rtg_signal": "On Track",
                "net_rtg_severity": "None",
                "net_rtg_direction": "Stable",
            })
        sev = "High" if abs_dev >= NET_RTG_HIGH_THRESHOLD else "Medium"
        if dev > 0:
            # Actual net_rtg > implied → team better than record → likely to improve
            return pd.Series({
                "net_rtg_signal": "Undervalued",
                "net_rtg_severity": sev,
                "net_rtg_direction": "Likely to improve",
            })
        else:
            # Actual net_rtg < implied → team worse than record → likely to decline
            return pd.Series({
                "net_rtg_signal": "Overvalued",
                "net_rtg_severity": sev,
                "net_rtg_direction": "Likely to decline",
            })

    signals = df["net_rtg_deviation"].apply(_signal)
    df = pd.concat([df, signals], axis=1)
    return df


def compute_nba_win_probabilities(
    games_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    rest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute model win probabilities for NBA games.

    games_df required columns: home_team, away_team
    team_stats_df required columns: team_name, net_rtg
    rest_df optional: team_name, rest_adjustment (net rating points)

    Added columns: home_model_prob, away_model_prob, net_rtg_diff
    """
    df = games_df.copy()
    team_stats = team_stats_df.set_index("team_name")

    rest_adj: dict[str, float] = {}
    if rest_df is not None and not rest_df.empty:
        rest_adj = rest_df.set_index("team_name")["rest_adjustment"].to_dict()

    home_probs, away_probs, diffs = [], [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        try:
            home_rtg = team_stats.loc[home, "net_rtg"]
            away_rtg = team_stats.loc[away, "net_rtg"]
        except KeyError:
            home_probs.append(float("nan"))
            away_probs.append(float("nan"))
            diffs.append(float("nan"))
            continue

        diff = (home_rtg + HOME_COURT_ADJ) - away_rtg
        diff += rest_adj.get(home, 0.0) - rest_adj.get(away, 0.0)

        hp = net_rtg_to_win_prob(diff)
        hp = max(0.05, min(0.95, hp))
        ap = 1 - hp

        home_probs.append(round(hp, 4))
        away_probs.append(round(ap, 4))
        diffs.append(round(diff, 2))

    df["home_model_prob"] = home_probs
    df["away_model_prob"] = away_probs
    df["net_rtg_diff"] = diffs
    return df
