"""Unit tests for the Pythagorean win expectation model."""

import pandas as pd
import pytest

from src.models.pythagorean import (
    pythagorean_win_pct,
    compute_pythagorean,
    log5_probability,
    regress_rs_ra,
    MIN_GAMES_FOR_REGRESSION,
    SHRINKAGE_K,
)


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


# ---------------------------------------------------------------------------
# Log5
# ---------------------------------------------------------------------------

def test_log5_equal_teams():
    assert log5_probability(0.5, 0.5) == pytest.approx(0.5, abs=0.001)


def test_log5_reference_value():
    # Bill James reference: .600 team vs .400 team → ~0.692
    assert log5_probability(0.6, 0.4) == pytest.approx(0.6923, abs=0.001)


def test_log5_asymmetric_strong_team():
    # .700 vs .300: strong team wins ~84%
    assert log5_probability(0.7, 0.3) == pytest.approx(0.8448, abs=0.001)


def test_log5_reversed_is_complement():
    p = log5_probability(0.6, 0.4)
    p_rev = log5_probability(0.4, 0.6)
    assert p + p_rev == pytest.approx(1.0, abs=0.001)


def test_log5_degenerate_returns_half():
    # Both zero or both one → denominator collapses → fallback 0.5
    assert log5_probability(0.0, 0.0) == pytest.approx(0.5, abs=0.001)
    assert log5_probability(1.0, 1.0) == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# Regression to mean
# ---------------------------------------------------------------------------

def test_regression_below_threshold_unchanged():
    rs, ra = regress_rs_ra(50, 30, MIN_GAMES_FOR_REGRESSION - 1, 4.5, 4.5)
    assert rs == pytest.approx(50, abs=0.01)
    assert ra == pytest.approx(30, abs=0.01)


def test_regression_at_threshold_weight():
    # At MIN_GAMES_FOR_REGRESSION games weight = G / (G + K)
    G = MIN_GAMES_FOR_REGRESSION
    rs_in, ra_in = 4.0 * G, 5.0 * G   # 4.0 and 5.0 R/G
    league_rs, league_ra = 4.5, 4.5
    rs_out, ra_out = regress_rs_ra(rs_in, ra_in, G, league_rs, league_ra)
    w = G / (G + SHRINKAGE_K)
    assert rs_out == pytest.approx((w * 4.0 + (1 - w) * league_rs) * G, abs=0.01)
    assert ra_out == pytest.approx((w * 5.0 + (1 - w) * league_ra) * G, abs=0.01)


def test_regression_162_games_weight():
    w = 162 / (162 + SHRINKAGE_K)
    assert w == pytest.approx(0.844, abs=0.005)


def test_regression_league_avg_team_unchanged():
    # A team already at league average should return the same values after regression
    G = 81
    league_avg = 4.5
    rs, ra = regress_rs_ra(league_avg * G, league_avg * G, G, league_avg, league_avg)
    assert rs == pytest.approx(league_avg * G, abs=0.01)
    assert ra == pytest.approx(league_avg * G, abs=0.01)


# ---------------------------------------------------------------------------
# Formula state
# ---------------------------------------------------------------------------

def test_formula_state_early_season():
    from src.models.win_probability import get_formula_state
    ts = pd.DataFrame({
        "team": ["NYY", "BOS"],
        "wins": [8, 22],
        "losses": [5, 10],
        "runs_scored": [70, 180],
        "runs_allowed": [60, 170],
    })
    state = get_formula_state(ts)
    assert state["state"] == "EARLY_SEASON"
    assert state["min_games"] == 13


def test_formula_state_regression_active():
    from src.models.win_probability import get_formula_state
    ts = pd.DataFrame({
        "team": ["NYY", "BOS"],
        "wins": [22, 25],
        "losses": [10, 12],
        "runs_scored": [200, 210],
        "runs_allowed": [180, 190],
    })
    state = get_formula_state(ts)
    assert state["state"] == "REGRESSION_ACTIVE"
    assert state["min_games"] >= MIN_GAMES_FOR_REGRESSION


def test_formula_state_empty_df():
    from src.models.win_probability import get_formula_state
    state = get_formula_state(pd.DataFrame())
    assert state["state"] == "EARLY_SEASON"
