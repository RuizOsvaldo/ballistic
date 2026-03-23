"""Model calibration analytics — measures prediction accuracy and ROI by signal type.

Usage
-----
Pass the settled rows from the bet log (outcome == Win or Loss) to the functions below.
The bet log must include model_prob (float 0-1) and signal_type (str) columns.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Probability buckets for calibration (lower bound, upper bound, label)
PROB_BUCKETS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 1.00, "70%+"),
]


def compute_calibration_table(settled: pd.DataFrame) -> pd.DataFrame:
    """
    Group settled bets by model_prob bucket and compare predicted vs actual win rate.

    Parameters
    ----------
    settled : DataFrame with columns: model_prob (float 0-1), outcome ('Win'/'Loss'),
              edge_pct (float), pnl (float), stake (float)

    Returns
    -------
    DataFrame with columns:
      bucket, bets, expected_win_rate, actual_win_rate, calibration_error,
      total_staked, total_pnl, roi_pct
    """
    if settled.empty or "model_prob" not in settled.columns:
        return pd.DataFrame()

    rows = []
    for lo, hi, label in PROB_BUCKETS:
        mask = (settled["model_prob"] >= lo) & (settled["model_prob"] < hi)
        bucket_df = settled[mask]
        if bucket_df.empty:
            continue
        n = len(bucket_df)
        midpoint = (lo + hi) / 2
        actual_wr = (bucket_df["outcome"] == "Win").mean()
        staked = bucket_df["stake"].sum()
        pnl = bucket_df["pnl"].sum()
        roi = (pnl / staked * 100) if staked > 0 else 0.0
        rows.append({
            "bucket": label,
            "bets": n,
            "expected_win_rate": round(midpoint, 3),
            "actual_win_rate": round(actual_wr, 3),
            "calibration_error": round(actual_wr - midpoint, 3),
            "total_staked": round(staked, 2),
            "total_pnl": round(pnl, 2),
            "roi_pct": round(roi, 1),
        })

    return pd.DataFrame(rows)


def compute_signal_roi(settled: pd.DataFrame) -> pd.DataFrame:
    """
    Break down win rate and ROI by signal type.

    Parameters
    ----------
    settled : DataFrame with columns: signal_type (str), outcome, stake, pnl

    Returns
    -------
    DataFrame with columns: signal_type, bets, win_rate, total_staked, total_pnl, roi_pct
    """
    if settled.empty or "signal_type" not in settled.columns:
        return pd.DataFrame()

    rows = []
    for sig_type, group in settled.groupby("signal_type"):
        n = len(group)
        wr = (group["outcome"] == "Win").mean()
        staked = group["stake"].sum()
        pnl = group["pnl"].sum()
        roi = (pnl / staked * 100) if staked > 0 else 0.0
        rows.append({
            "signal_type": sig_type,
            "bets": n,
            "win_rate": round(wr, 3),
            "total_staked": round(staked, 2),
            "total_pnl": round(pnl, 2),
            "roi_pct": round(roi, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("roi_pct", ascending=False).reset_index(drop=True)
    return df


def compute_edge_vs_outcome(settled: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """
    Bin bets by edge_pct and show actual win rate vs expected edge.
    Reveals whether higher-edge bets actually win more often.

    Parameters
    ----------
    settled : DataFrame with columns: edge_pct (float %), outcome, stake, pnl

    Returns
    -------
    DataFrame with columns: edge_bucket, bets, avg_edge, actual_win_rate, roi_pct
    """
    if settled.empty or "edge_pct" not in settled.columns:
        return pd.DataFrame()

    df = settled.copy()
    try:
        df["edge_bucket"] = pd.cut(df["edge_pct"], bins=bins, precision=1)
    except Exception:
        return pd.DataFrame()

    rows = []
    for bucket, group in df.groupby("edge_bucket", observed=True):
        n = len(group)
        avg_edge = group["edge_pct"].mean()
        wr = (group["outcome"] == "Win").mean()
        staked = group["stake"].sum()
        pnl = group["pnl"].sum()
        roi = (pnl / staked * 100) if staked > 0 else 0.0
        rows.append({
            "edge_bucket": str(bucket),
            "bets": n,
            "avg_edge_pct": round(avg_edge, 1),
            "actual_win_rate": round(wr, 3),
            "roi_pct": round(roi, 1),
        })

    return pd.DataFrame(rows)


def recommend_threshold_adjustments(calibration_df: pd.DataFrame) -> list[str]:
    """
    Given a calibration table, return human-readable tuning suggestions.

    Returns a list of suggestion strings (empty list if insufficient data).
    """
    if calibration_df.empty:
        return []

    suggestions = []
    for _, row in calibration_df.iterrows():
        err = row["calibration_error"]
        bucket = row["bucket"]
        n = row["bets"]
        if n < 5:
            continue  # not enough data
        if err < -0.05:
            suggestions.append(
                f"{bucket}: model overestimates by {abs(err):.1%} "
                f"(predicted {row['expected_win_rate']:.0%}, actual {row['actual_win_rate']:.0%}) "
                f"— consider reducing FIP_ADJ_PER_POINT or HOME_FIELD_ADJ."
            )
        elif err > 0.05:
            suggestions.append(
                f"{bucket}: model underestimates by {err:.1%} "
                f"(predicted {row['expected_win_rate']:.0%}, actual {row['actual_win_rate']:.0%}) "
                f"— model is conservative here; edge threshold may be too high."
            )

    if not suggestions:
        suggestions.append("Calibration looks reasonable — no major adjustments recommended yet.")

    return suggestions
