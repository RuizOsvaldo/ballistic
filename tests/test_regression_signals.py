"""Unit tests for regression signal detection."""

import pandas as pd
import pytest

from src.models.regression_signals import compute_pitcher_signals, compute_team_signals


def _make_pitcher(era=3.50, fip=3.60, babip=0.300):
    return pd.DataFrame({
        "name": ["Test Pitcher"],
        "team": ["NYY"],
        "era": [era],
        "fip": [fip],
        "babip": [babip],
    })


def test_no_signal_for_average_pitcher():
    df = compute_pitcher_signals(_make_pitcher(era=3.50, fip=3.60, babip=0.300))
    assert df.iloc[0]["signal_severity"] == "None"
    assert df.iloc[0]["signal_direction"] == "Stable"


def test_high_fip_era_gap_signals_regression():
    # FIP much higher than ERA → ERA likely to rise
    df = compute_pitcher_signals(_make_pitcher(era=2.50, fip=4.10, babip=0.290))
    assert df.iloc[0]["fip_era_gap"] == pytest.approx(1.60, abs=0.01)
    assert df.iloc[0]["signal_severity"] in ("Medium", "High")
    assert df.iloc[0]["signal_direction"] == "ERA likely UP"


def test_high_babip_signals_regression():
    df = compute_pitcher_signals(_make_pitcher(era=3.20, fip=3.40, babip=0.340))
    assert df.iloc[0]["signal_direction"] == "ERA likely UP"


def test_low_babip_signals_positive():
    df = compute_pitcher_signals(_make_pitcher(era=3.80, fip=3.60, babip=0.260))
    assert df.iloc[0]["signal_direction"] == "ERA likely DOWN"


def test_fip_era_gap_column_computed():
    df = compute_pitcher_signals(_make_pitcher(era=3.00, fip=4.00))
    assert df.iloc[0]["fip_era_gap"] == pytest.approx(1.00, abs=0.01)


def test_team_signal_overperforming():
    df = pd.DataFrame({"team": ["NYY"], "pyth_deviation": [0.12]})
    result = compute_team_signals(df)
    assert result.iloc[0]["team_signal_direction"] == "Likely to decline"
    assert result.iloc[0]["team_signal_severity"] == "High"


def test_team_signal_underperforming():
    df = pd.DataFrame({"team": ["NYY"], "pyth_deviation": [-0.08]})
    result = compute_team_signals(df)
    assert result.iloc[0]["team_signal_direction"] == "Likely to improve"


def test_team_signal_stable():
    df = pd.DataFrame({"team": ["NYY"], "pyth_deviation": [0.02]})
    result = compute_team_signals(df)
    assert result.iloc[0]["team_signal_direction"] == "Stable"
