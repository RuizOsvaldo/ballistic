"""Tests for src/models/calibration.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.calibration import (
    compute_calibration_table,
    compute_edge_vs_outcome,
    compute_signal_roi,
    recommend_threshold_adjustments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settled(rows: list[dict]) -> pd.DataFrame:
    defaults = {"stake": 50.0, "pnl": 0.0, "edge_pct": 5.0, "signal_type": "None"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# compute_calibration_table
# ---------------------------------------------------------------------------

def test_calibration_table_empty():
    result = compute_calibration_table(pd.DataFrame())
    assert result.empty


def test_calibration_table_missing_column():
    df = pd.DataFrame({"outcome": ["Win", "Loss"]})
    result = compute_calibration_table(df)
    assert result.empty


def test_calibration_table_basic():
    # 6 bets in 55-60% bucket: all wins → actual_win_rate = 1.0, expected ≈ 0.575
    rows = [
        {"model_prob": 0.57, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.58, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.56, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.59, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.55, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.57, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_calibration_table(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["bucket"] == "55-60%"
    assert row["bets"] == 6
    assert row["actual_win_rate"] == pytest.approx(1.0)
    assert row["calibration_error"] == pytest.approx(1.0 - 0.575, abs=0.01)


def test_calibration_table_multiple_buckets():
    rows = [
        {"model_prob": 0.52, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.52, "outcome": "Loss", "stake": 50.0, "pnl": -50.0},
        {"model_prob": 0.63, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"model_prob": 0.63, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_calibration_table(df)
    assert len(result) == 2
    buckets = result["bucket"].tolist()
    assert "50-55%" in buckets
    assert "60-65%" in buckets


def test_calibration_roi_computed():
    rows = [
        {"model_prob": 0.62, "outcome": "Win", "stake": 100.0, "pnl": 90.0},
        {"model_prob": 0.61, "outcome": "Loss", "stake": 100.0, "pnl": -100.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_calibration_table(df)
    row = result.iloc[0]
    # total pnl = -10, staked = 200 → roi = -5%
    assert row["roi_pct"] == pytest.approx(-5.0, abs=0.1)


# ---------------------------------------------------------------------------
# compute_signal_roi
# ---------------------------------------------------------------------------

def test_signal_roi_empty():
    result = compute_signal_roi(pd.DataFrame())
    assert result.empty


def test_signal_roi_missing_column():
    df = pd.DataFrame({"outcome": ["Win"]})
    result = compute_signal_roi(df)
    assert result.empty


def test_signal_roi_groups_correctly():
    rows = [
        {"signal_type": "BABIP", "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"signal_type": "BABIP", "outcome": "Loss", "stake": 50.0, "pnl": -50.0},
        {"signal_type": "FIP-ERA", "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"signal_type": "FIP-ERA", "outcome": "Win", "stake": 50.0, "pnl": 45.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_signal_roi(df)
    assert len(result) == 2
    fip_row = result[result["signal_type"] == "FIP-ERA"].iloc[0]
    assert fip_row["win_rate"] == pytest.approx(1.0)
    babip_row = result[result["signal_type"] == "BABIP"].iloc[0]
    assert babip_row["win_rate"] == pytest.approx(0.5)


def test_signal_roi_sorted_by_roi():
    rows = [
        {"signal_type": "A", "outcome": "Loss", "stake": 50.0, "pnl": -50.0},
        {"signal_type": "B", "outcome": "Win", "stake": 50.0, "pnl": 45.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_signal_roi(df)
    assert result.iloc[0]["signal_type"] == "B"


# ---------------------------------------------------------------------------
# compute_edge_vs_outcome
# ---------------------------------------------------------------------------

def test_edge_vs_outcome_empty():
    result = compute_edge_vs_outcome(pd.DataFrame())
    assert result.empty


def test_edge_vs_outcome_returns_rows():
    rows = [
        {"edge_pct": 3.0, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"edge_pct": 5.0, "outcome": "Loss", "stake": 50.0, "pnl": -50.0},
        {"edge_pct": 8.0, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
        {"edge_pct": 10.0, "outcome": "Win", "stake": 50.0, "pnl": 45.0},
    ]
    df = pd.DataFrame(rows)
    result = compute_edge_vs_outcome(df, bins=2)
    assert not result.empty
    assert "edge_bucket" in result.columns
    assert "actual_win_rate" in result.columns


# ---------------------------------------------------------------------------
# recommend_threshold_adjustments
# ---------------------------------------------------------------------------

def test_recommendations_empty_df():
    result = recommend_threshold_adjustments(pd.DataFrame())
    assert result == []


def test_recommendations_insufficient_data():
    # < 5 bets per bucket — should produce no suggestions
    cal_df = pd.DataFrame([{
        "bucket": "55-60%", "bets": 3,
        "expected_win_rate": 0.575, "actual_win_rate": 0.20,
        "calibration_error": -0.375,
    }])
    result = recommend_threshold_adjustments(cal_df)
    assert result == ["Calibration looks reasonable — no major adjustments recommended yet."]


def test_recommendations_overestimate():
    cal_df = pd.DataFrame([{
        "bucket": "60-65%", "bets": 10,
        "expected_win_rate": 0.625, "actual_win_rate": 0.40,
        "calibration_error": -0.225,
    }])
    result = recommend_threshold_adjustments(cal_df)
    assert len(result) == 1
    assert "overestimates" in result[0]
    assert "60-65%" in result[0]


def test_recommendations_underestimate():
    cal_df = pd.DataFrame([{
        "bucket": "55-60%", "bets": 8,
        "expected_win_rate": 0.575, "actual_win_rate": 0.80,
        "calibration_error": 0.225,
    }])
    result = recommend_threshold_adjustments(cal_df)
    assert len(result) == 1
    assert "underestimates" in result[0]


def test_recommendations_well_calibrated():
    cal_df = pd.DataFrame([{
        "bucket": "55-60%", "bets": 10,
        "expected_win_rate": 0.575, "actual_win_rate": 0.58,
        "calibration_error": 0.005,
    }])
    result = recommend_threshold_adjustments(cal_df)
    assert result == ["Calibration looks reasonable — no major adjustments recommended yet."]
