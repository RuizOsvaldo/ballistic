"""Tests for NFL EPA model and rest/schedule adjustments."""

from __future__ import annotations

import math
import pandas as pd
import pytest

from src.sports.football.models.epa import (
    EPA_TO_SPREAD,
    SPREAD_TO_WIN_SCALE,
    compute_epa_signals,
    compute_implied_epa_composite,
    compute_nfl_win_probabilities,
    epa_composite_to_spread,
    spread_to_win_prob,
)
from src.sports.football.models.rest_schedule import (
    BYE_WEEK_BONUS_PTS,
    HOME_FIELD_PTS,
    SHORT_WEEK_PENALTY_PTS,
    build_rest_df,
    classify_rest,
    compute_rest_adjustments,
    rest_adjustment_pts,
)


# ---------------------------------------------------------------------------
# epa_composite_to_spread
# ---------------------------------------------------------------------------

def test_spread_positive_home_favored():
    result = epa_composite_to_spread(0.10)
    assert result == pytest.approx(0.10 * EPA_TO_SPREAD)


def test_spread_zero():
    assert epa_composite_to_spread(0.0) == pytest.approx(0.0)


def test_spread_negative_away_favored():
    result = epa_composite_to_spread(-0.10)
    assert result < 0


# ---------------------------------------------------------------------------
# spread_to_win_prob
# ---------------------------------------------------------------------------

def test_even_spread_is_50pct():
    assert spread_to_win_prob(0.0) == pytest.approx(0.5, abs=0.01)


def test_positive_spread_favors_home():
    assert spread_to_win_prob(7.0) > 0.5


def test_seven_pt_spread_is_approx_73pct():
    prob = spread_to_win_prob(7.0)
    assert 0.70 < prob < 0.76


def test_large_spread_approaches_1():
    assert spread_to_win_prob(40.0) > 0.98


def test_large_negative_spread_approaches_0():
    assert spread_to_win_prob(-40.0) < 0.02


# ---------------------------------------------------------------------------
# compute_implied_epa_composite
# ---------------------------------------------------------------------------

def test_implied_epa_50pct_is_zero():
    result = compute_implied_epa_composite(0.50)
    assert result == pytest.approx(0.0, abs=0.01)


def test_implied_epa_above_50pct_is_positive():
    assert compute_implied_epa_composite(0.60) > 0


def test_implied_epa_below_50pct_is_negative():
    assert compute_implied_epa_composite(0.40) < 0


def test_implied_epa_zero_clamps():
    result = compute_implied_epa_composite(0.0)
    assert result == pytest.approx(-0.30)


def test_implied_epa_one_clamps():
    result = compute_implied_epa_composite(1.0)
    assert result == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# compute_epa_signals
# ---------------------------------------------------------------------------

def _make_team_df(teams_data: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(teams_data)


def test_signals_stable_team():
    df = _make_team_df([{
        "team": "KC", "epa_composite": 0.10, "off_epa": 0.15, "def_epa": 0.05,
        "win_pct": 0.60,
    }])
    result = compute_epa_signals(df)
    # Implied EPA for 60% ≈ +0.10; deviation ≈ 0 → stable
    assert "epa_signal" in result.columns
    assert result.iloc[0]["epa_signal"] in ("On Track", "Undervalued", "Overvalued")


def test_signals_overvalued_team():
    """Team with 0% win rate but average EPA → large negative deviation → Overvalued."""
    df = _make_team_df([{
        "team": "NYG", "epa_composite": -0.20, "off_epa": -0.10, "def_epa": 0.10,
        "win_pct": 0.70,   # Winning way more than EPA suggests
    }])
    result = compute_epa_signals(df)
    assert result.iloc[0]["epa_signal"] == "Overvalued"
    assert result.iloc[0]["epa_direction"] == "Likely to decline"


def test_signals_undervalued_team():
    """Team with strong EPA but poor record → Undervalued."""
    df = _make_team_df([{
        "team": "DET", "epa_composite": 0.15, "off_epa": 0.20, "def_epa": 0.05,
        "win_pct": 0.30,   # Losing more than EPA suggests
    }])
    result = compute_epa_signals(df)
    assert result.iloc[0]["epa_signal"] == "Undervalued"
    assert result.iloc[0]["epa_direction"] == "Likely to improve"


def test_signals_no_win_pct_uses_composite():
    """Without win_pct, deviation = epa_composite itself."""
    df = _make_team_df([{
        "team": "SF", "epa_composite": 0.20, "off_epa": 0.25, "def_epa": 0.05,
    }])
    result = compute_epa_signals(df)
    assert result.iloc[0]["epa_deviation"] == pytest.approx(0.20, abs=0.01)


def test_signals_severity_high():
    df = _make_team_df([{
        "team": "TEN", "epa_composite": -0.25, "off_epa": -0.15, "def_epa": 0.10,
        "win_pct": 0.80,
    }])
    result = compute_epa_signals(df)
    assert result.iloc[0]["epa_severity"] == "High"


# ---------------------------------------------------------------------------
# compute_nfl_win_probabilities
# ---------------------------------------------------------------------------

def test_win_probs_sum_to_one():
    games = pd.DataFrame([{"home_team": "KC", "away_team": "BUF"}])
    epa = pd.DataFrame([
        {"team": "KC", "epa_composite": 0.10},
        {"team": "BUF", "epa_composite": 0.05},
    ])
    result = compute_nfl_win_probabilities(games, epa)
    row = result.iloc[0]
    assert row["home_model_prob"] + row["away_model_prob"] == pytest.approx(1.0, abs=0.01)


def test_home_field_advantage_baseline():
    """Even EPA teams — home team should win >50% due to home field."""
    games = pd.DataFrame([{"home_team": "NYG", "away_team": "NYJ"}])
    epa = pd.DataFrame([
        {"team": "NYG", "epa_composite": 0.0},
        {"team": "NYJ", "epa_composite": 0.0},
    ])
    result = compute_nfl_win_probabilities(games, epa)
    assert result.iloc[0]["home_model_prob"] > 0.5


def test_missing_team_returns_nan():
    games = pd.DataFrame([{"home_team": "UNKNOWN", "away_team": "KC"}])
    epa = pd.DataFrame([{"team": "KC", "epa_composite": 0.10}])
    result = compute_nfl_win_probabilities(games, epa)
    assert math.isnan(result.iloc[0]["home_model_prob"])


def test_probs_bounded():
    games = pd.DataFrame([{"home_team": "KC", "away_team": "BUF"}])
    epa = pd.DataFrame([
        {"team": "KC", "epa_composite": 1.0},
        {"team": "BUF", "epa_composite": -1.0},
    ])
    result = compute_nfl_win_probabilities(games, epa)
    assert result.iloc[0]["home_model_prob"] <= 0.95
    assert result.iloc[0]["away_model_prob"] >= 0.05


# ---------------------------------------------------------------------------
# rest_schedule
# ---------------------------------------------------------------------------

def test_classify_rest_bye():
    assert classify_rest(14) == "Bye"
    assert classify_rest(13) == "Bye"


def test_classify_rest_short_week():
    assert classify_rest(4) == "Short Week"
    assert classify_rest(3) == "Short Week"


def test_classify_rest_normal():
    assert classify_rest(7) == "Normal"
    assert classify_rest(6) == "Normal"


def test_classify_rest_none():
    assert classify_rest(None) == "Normal"


def test_rest_adjustment_bye():
    assert rest_adjustment_pts("Bye") == pytest.approx(BYE_WEEK_BONUS_PTS)


def test_rest_adjustment_short():
    assert rest_adjustment_pts("Short Week") == pytest.approx(SHORT_WEEK_PENALTY_PTS)


def test_rest_adjustment_normal():
    assert rest_adjustment_pts("Normal") == pytest.approx(0.0)


def test_compute_rest_adjustments_columns():
    games = pd.DataFrame([{"home_team": "KC", "away_team": "BUF"}])
    result = compute_rest_adjustments(games)
    for col in ["home_rest_type", "away_rest_type", "rest_net_pts", "rest_mismatch"]:
        assert col in result.columns


def test_compute_rest_adjustments_bye_vs_short():
    games = pd.DataFrame([{"home_team": "KC", "away_team": "BUF"}])
    result = compute_rest_adjustments(
        games,
        home_days_rest={"KC": 14},
        away_days_rest={"BUF": 4},
    )
    row = result.iloc[0]
    assert row["home_rest_type"] == "Bye"
    assert row["away_rest_type"] == "Short Week"
    assert row["rest_mismatch"] == True
    assert row["rest_net_pts"] == pytest.approx(BYE_WEEK_BONUS_PTS - SHORT_WEEK_PENALTY_PTS)


def test_build_rest_df():
    rest_map = {"KC": 14, "BUF": 4, "SF": 7}
    df = build_rest_df(rest_map)
    assert len(df) == 3
    kc = df[df["team"] == "KC"].iloc[0]
    assert kc["rest_type"] == "Bye"
    assert kc["rest_adjustment"] == pytest.approx(BYE_WEEK_BONUS_PTS)
