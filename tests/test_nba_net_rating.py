"""Unit tests for NBA net rating model."""

import pandas as pd
import pytest

from src.sports.basketball.models.net_rating import (
    net_rtg_to_win_prob,
    compute_implied_net_rtg,
    compute_net_rating_signals,
    compute_nba_win_probabilities,
    HOME_COURT_ADJ,
)


class TestNetRtgToWinProb:
    def test_zero_diff_near_50_pct(self):
        assert net_rtg_to_win_prob(0) == pytest.approx(0.5, abs=0.01)

    def test_positive_diff_above_50(self):
        assert net_rtg_to_win_prob(5) > 0.5

    def test_negative_diff_below_50(self):
        assert net_rtg_to_win_prob(-5) < 0.5

    def test_output_bounded(self):
        assert 0 < net_rtg_to_win_prob(50) < 1
        assert 0 < net_rtg_to_win_prob(-50) < 1

    def test_symmetric(self):
        pos = net_rtg_to_win_prob(10)
        neg = net_rtg_to_win_prob(-10)
        assert pos + neg == pytest.approx(1.0, abs=0.001)


class TestComputeImpliedNetRtg:
    def test_500_win_pct_implies_zero(self):
        assert compute_implied_net_rtg(0.5) == pytest.approx(0.0, abs=0.1)

    def test_high_win_pct_positive_net_rtg(self):
        assert compute_implied_net_rtg(0.70) > 0

    def test_low_win_pct_negative_net_rtg(self):
        assert compute_implied_net_rtg(0.30) < 0

    def test_extreme_win_pct_clamps(self):
        assert compute_implied_net_rtg(0.0) == pytest.approx(-20.0, abs=0.1)
        assert compute_implied_net_rtg(1.0) == pytest.approx(20.0, abs=0.1)


class TestComputeNetRatingSignals:
    def _make_df(self, net_rtg, win_pct):
        return pd.DataFrame([{"team_name": "LAL", "net_rtg": net_rtg, "win_pct": win_pct}])

    def test_adds_signal_columns(self):
        df = compute_net_rating_signals(self._make_df(5.0, 0.5))
        for col in ["implied_net_rtg", "net_rtg_deviation", "net_rtg_signal", "net_rtg_severity"]:
            assert col in df.columns

    def test_stable_when_small_deviation(self):
        df = compute_net_rating_signals(self._make_df(0.0, 0.5))
        assert df.iloc[0]["net_rtg_signal"] == "On Track"

    def test_undervalued_when_better_than_record(self):
        # Net rating = +10 but only 50% W% → team is better than their record
        df = compute_net_rating_signals(self._make_df(10.0, 0.5))
        assert df.iloc[0]["net_rtg_signal"] == "Undervalued"
        assert df.iloc[0]["net_rtg_direction"] == "Likely to improve"

    def test_overvalued_when_worse_than_record(self):
        # Net rating = -5 but 60% W% → team is worse than their record
        df = compute_net_rating_signals(self._make_df(-5.0, 0.6))
        assert df.iloc[0]["net_rtg_signal"] == "Overvalued"
        assert df.iloc[0]["net_rtg_direction"] == "Likely to decline"

    def test_high_severity_at_large_deviation(self):
        df = compute_net_rating_signals(self._make_df(15.0, 0.4))
        assert df.iloc[0]["net_rtg_severity"] == "High"


class TestComputeNBAWinProbabilities:
    def _make_games(self):
        return pd.DataFrame([{
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
        }])

    def _make_teams(self):
        return pd.DataFrame([
            {"team_name": "Boston Celtics", "net_rtg": 8.5},
            {"team_name": "Los Angeles Lakers", "net_rtg": 2.1},
        ])

    def test_adds_prob_columns(self):
        df = compute_nba_win_probabilities(self._make_games(), self._make_teams())
        assert "home_model_prob" in df.columns
        assert "away_model_prob" in df.columns

    def test_probs_sum_to_one(self):
        df = compute_nba_win_probabilities(self._make_games(), self._make_teams())
        row = df.iloc[0]
        assert row["home_model_prob"] + row["away_model_prob"] == pytest.approx(1.0, abs=0.001)

    def test_better_team_plus_home_court_wins_more(self):
        # Celtics have higher net rating AND home court
        df = compute_nba_win_probabilities(self._make_games(), self._make_teams())
        assert df.iloc[0]["home_model_prob"] > 0.5

    def test_missing_team_returns_nan(self):
        games = pd.DataFrame([{"home_team": "Unknown Team", "away_team": "Los Angeles Lakers"}])
        df = compute_nba_win_probabilities(games, self._make_teams())
        import math
        assert math.isnan(df.iloc[0]["home_model_prob"])
