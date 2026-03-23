"""Tests for NFL weather model."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sports.football.models.weather import (
    PRECIPITATION_ADJ,
    add_weather_adjustments,
    compute_weather_adjustment,
    get_temp_adjustment,
    get_wind_adjustment,
    is_dome,
)


# ---------------------------------------------------------------------------
# get_wind_adjustment
# ---------------------------------------------------------------------------

def test_no_wind():
    assert get_wind_adjustment(0) == 0.0
    assert get_wind_adjustment(None) == 0.0


def test_wind_below_threshold():
    assert get_wind_adjustment(5) == 0.0


def test_wind_10mph():
    assert get_wind_adjustment(10) == pytest.approx(-1.5)


def test_wind_15mph():
    assert get_wind_adjustment(15) == pytest.approx(-3.0)


def test_wind_20mph():
    assert get_wind_adjustment(20) == pytest.approx(-5.0)


def test_wind_25mph():
    assert get_wind_adjustment(25) == pytest.approx(-7.0)


def test_wind_30mph_uses_25mph_bracket():
    assert get_wind_adjustment(30) == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# get_temp_adjustment
# ---------------------------------------------------------------------------

def test_mild_temp_no_adjustment():
    assert get_temp_adjustment(65) == 0.0
    assert get_temp_adjustment(None) == 0.0


def test_temp_40f():
    assert get_temp_adjustment(40) == pytest.approx(-1.0)


def test_temp_30f():
    assert get_temp_adjustment(30) == pytest.approx(-2.0)


def test_temp_20f():
    assert get_temp_adjustment(20) == pytest.approx(-3.0)


def test_temp_15f_uses_20f_bracket():
    assert get_temp_adjustment(15) == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# is_dome
# ---------------------------------------------------------------------------

def test_dome_detection():
    assert is_dome("dome") is True
    assert is_dome("closed") is True
    assert is_dome("Dome") is True


def test_outdoor_not_dome():
    assert is_dome("outdoor") is False
    assert is_dome("open") is False
    assert is_dome(None) is False


def test_retractable_not_dome():
    # retractable is not in DOME_ROOFS — it's outdoor when open
    assert is_dome("retractable") is False


# ---------------------------------------------------------------------------
# compute_weather_adjustment
# ---------------------------------------------------------------------------

def test_dome_game_zero_adjustment():
    result = compute_weather_adjustment(25, 20, True, roof="dome")
    assert result["total_adj"] == 0.0
    assert result["is_dome"] is True
    assert result["weather_flag"] is False


def test_no_weather_effects():
    result = compute_weather_adjustment(5, 65, False)
    assert result["total_adj"] == 0.0
    assert result["weather_flag"] is False


def test_wind_and_cold_combine():
    # temp 25°F falls in ≤30°F bracket → -2.0; wind 20mph → -5.0
    result = compute_weather_adjustment(wind_mph=20, temp_f=25, precipitation=False)
    assert result["wind_adj"] == pytest.approx(-5.0)
    assert result["temp_adj"] == pytest.approx(-2.0)
    assert result["total_adj"] == pytest.approx(-7.0)
    assert result["weather_flag"] is True


def test_precipitation_adds_adjustment():
    result = compute_weather_adjustment(wind_mph=5, temp_f=50, precipitation=True)
    assert result["precip_adj"] == pytest.approx(PRECIPITATION_ADJ)
    assert result["total_adj"] == pytest.approx(PRECIPITATION_ADJ)


def test_all_factors_combined():
    result = compute_weather_adjustment(wind_mph=25, temp_f=20, precipitation=True)
    expected = -7.0 + -3.0 + PRECIPITATION_ADJ
    assert result["total_adj"] == pytest.approx(expected)


def test_weather_flag_threshold():
    # Below 2 pts → no flag
    result = compute_weather_adjustment(wind_mph=5, temp_f=50, precipitation=False)
    assert result["weather_flag"] is False
    # At or above 2 pts → flag
    result2 = compute_weather_adjustment(wind_mph=15, temp_f=65, precipitation=False)
    assert result2["weather_flag"] is True


def test_summary_includes_conditions():
    result = compute_weather_adjustment(wind_mph=20, temp_f=25)
    assert "Wind" in result["summary"]
    assert "Temp" in result["summary"]


# ---------------------------------------------------------------------------
# add_weather_adjustments
# ---------------------------------------------------------------------------

def test_add_weather_adjustments_columns():
    df = pd.DataFrame([{
        "home_team": "GB", "away_team": "CHI",
        "wind": 18, "temp": 28, "roof": "outdoor",
        "total_line": 42.5,
    }])
    result = add_weather_adjustments(df)
    for col in ["weather_total_adj", "weather_flag", "weather_summary", "is_dome", "adjusted_total"]:
        assert col in result.columns


def test_add_weather_adjustments_dome_unchanged():
    df = pd.DataFrame([{
        "home_team": "LV", "away_team": "KC",
        "wind": 20, "temp": 30, "roof": "dome",
        "total_line": 50.5,
    }])
    result = add_weather_adjustments(df)
    assert result.iloc[0]["weather_total_adj"] == 0.0
    assert result.iloc[0]["adjusted_total"] == pytest.approx(50.5)


def test_add_weather_adjustments_reduces_total():
    df = pd.DataFrame([{
        "home_team": "GB", "away_team": "CHI",
        "wind": 20, "temp": 28, "roof": "outdoor",
        "total_line": 44.0,
    }])
    result = add_weather_adjustments(df)
    assert result.iloc[0]["adjusted_total"] < 44.0


def test_add_weather_adjustments_missing_columns():
    """Should not crash when wind/temp/roof are absent."""
    df = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])
    result = add_weather_adjustments(df)
    assert "weather_total_adj" in result.columns
    assert result.iloc[0]["weather_total_adj"] == 0.0
