"""Unit tests for the Pythagorean win expectation model."""

import pandas as pd
import pytest

from src.models.pythagorean import pythagorean_win_pct, compute_pythagorean


def test_equal_runs_gives_500():
    assert pythagorean_win_pct(500, 500) == pytest.approx(0.5, abs=0.001)


def test_dominant_team_above_500():
    result = pythagorean_win_pct(900, 600)
    assert result > 0.5


def test_weak_team_below_500():
    result = pythagorean_win_pct(600, 900)
    assert result < 0.5


def test_no_runs_allowed_gives_perfect():
    assert pythagorean_win_pct(500, 0) == 1.0


def test_compute_pythagorean_adds_columns():
    df = pd.DataFrame({
        "team": ["NYY", "BOS"],
        "runs_scored": [800, 700],
        "runs_allowed": [650, 720],
        "win_pct": [0.580, 0.490],
        "wins": [94, 79],
        "losses": [68, 83],
    })
    result = compute_pythagorean(df)
    assert "pyth_win_pct" in result.columns
    assert "pyth_deviation" in result.columns
    assert "pyth_signal" in result.columns


def test_overperforming_signal():
    df = pd.DataFrame({
        "team": ["NYY"],
        "runs_scored": [700],
        "runs_allowed": [700],  # Pythagorean = 0.500
        "win_pct": [0.620],     # Actual much higher → overperforming
        "wins": [100],
        "losses": [62],
    })
    result = compute_pythagorean(df)
    assert result.iloc[0]["pyth_signal"] == "Overperforming"


def test_underperforming_signal():
    df = pd.DataFrame({
        "team": ["NYY"],
        "runs_scored": [800],
        "runs_allowed": [600],  # Pythagorean ~0.64
        "win_pct": [0.500],     # Actual much lower → underperforming
        "wins": [81],
        "losses": [81],
    })
    result = compute_pythagorean(df)
    assert result.iloc[0]["pyth_signal"] == "Underperforming"


def test_on_track_signal():
    df = pd.DataFrame({
        "team": ["NYY"],
        "runs_scored": [700],
        "runs_allowed": [700],
        "win_pct": [0.500],
        "wins": [81],
        "losses": [81],
    })
    result = compute_pythagorean(df)
    assert result.iloc[0]["pyth_signal"] == "On Track"
