"""Regression signal detection: BABIP deviation and FIP-ERA gap."""

from __future__ import annotations

import pandas as pd

# BABIP thresholds
BABIP_HIGH = 0.320   # Above this: pitcher is giving up too many hits on contact → likely lucky (ERA will rise)
BABIP_LOW = 0.275    # Below this: pitcher has been fortunate → ERA likely to rise or BABIP to normalise

# FIP-ERA gap threshold
FIP_ERA_GAP_THRESHOLD = 0.75  # FIP - ERA > 0.75 → ERA is artificially low, expect regression up

LEAGUE_AVG_BABIP = 0.300


def _severity(value: float, low: float, mid: float) -> str:
    """Map an absolute deviation to Low / Medium / High severity."""
    if abs(value) >= mid:
        return "High"
    if abs(value) >= low:
        return "Medium"
    return "Low"


def compute_pitcher_signals(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regression signal columns to a pitcher stats DataFrame.

    Required columns: era, fip, babip
    Added columns:
      - fip_era_gap        : fip - era  (positive = ERA artificially low)
      - babip_deviation    : babip - 0.300
      - fip_signal         : signal label for FIP-ERA gap
      - babip_signal       : signal label for BABIP
      - regression_signal  : combined worst-case signal
      - signal_severity    : "High" | "Medium" | "Low" | "None"
      - signal_direction   : "ERA likely UP" | "ERA likely DOWN" | "Stable"
    """
    df = pitcher_df.copy()

    df["fip_era_gap"] = (df["fip"] - df["era"]).round(3)
    df["babip_deviation"] = (df["babip"] - LEAGUE_AVG_BABIP).round(3)

    def _pitcher_signal(row):
        gap = row["fip_era_gap"]
        babip_dev = row["babip_deviation"]
        signals = []

        if gap >= FIP_ERA_GAP_THRESHOLD:
            sev = _severity(gap, FIP_ERA_GAP_THRESHOLD, FIP_ERA_GAP_THRESHOLD * 1.5)
            signals.append((sev, "ERA likely UP", f"FIP-ERA gap +{gap:.2f}"))

        if row["babip"] > BABIP_HIGH:
            sev = _severity(babip_dev, 0.020, 0.040)
            signals.append((sev, "ERA likely UP", f"BABIP high ({row['babip']:.3f})"))
        elif row["babip"] < BABIP_LOW:
            sev = _severity(abs(babip_dev), 0.020, 0.040)
            signals.append((sev, "ERA likely DOWN", f"BABIP low ({row['babip']:.3f}), pitcher may sustain"))

        if not signals:
            return pd.Series({
                "regression_signal": "None",
                "signal_severity": "None",
                "signal_direction": "Stable",
                "signal_notes": "",
            })

        order = {"High": 3, "Medium": 2, "Low": 1}
        worst = max(signals, key=lambda x: order[x[0]])
        notes = "; ".join(s[2] for s in signals)
        return pd.Series({
            "regression_signal": worst[2],
            "signal_severity": worst[0],
            "signal_direction": worst[1],
            "signal_notes": notes,
        })

    signal_cols = df.apply(_pitcher_signal, axis=1)
    df = pd.concat([df, signal_cols], axis=1)
    return df


def compute_team_signals(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team-level regression signal based on Pythagorean deviation.
    Requires pyth_deviation column (from pythagorean.py).

    Added columns:
      - team_signal_severity : "High" | "Medium" | "Low" | "None"
      - team_signal_direction: "Likely to decline" | "Likely to improve" | "Stable"
    """
    df = team_df.copy()

    def _team_signal(dev: float):
        if dev > 0.05:
            sev = _severity(dev, 0.05, 0.10)
            return pd.Series({"team_signal_severity": sev, "team_signal_direction": "Likely to decline"})
        if dev < -0.05:
            sev = _severity(abs(dev), 0.05, 0.10)
            return pd.Series({"team_signal_severity": sev, "team_signal_direction": "Likely to improve"})
        return pd.Series({"team_signal_severity": "None", "team_signal_direction": "Stable"})

    signal_cols = df["pyth_deviation"].apply(_team_signal)
    df = pd.concat([df, signal_cols], axis=1)
    return df
