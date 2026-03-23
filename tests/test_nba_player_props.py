"""Unit tests for NBA player prop projections."""

import pytest

from src.sports.basketball.models.player_props import (
    project_points,
    project_rebounds,
    project_assists,
    project_three_pm,
    project_pra,
    compute_prop_edge,
    pace_factor,
    LEAGUE_AVG_THREE_PCT,
    MIN_PROP_EDGE_PCT,
)


class TestPaceFactor:
    def test_same_pace_returns_one(self):
        assert pace_factor(100.0, 100.0) == pytest.approx(1.0, abs=0.001)

    def test_faster_game_above_one(self):
        assert pace_factor(105.0, 98.0) > 1.0

    def test_slower_game_below_one(self):
        assert pace_factor(92.0, 100.0) < 1.0

    def test_zero_team_pace_returns_one(self):
        assert pace_factor(100.0, 0.0) == 1.0


class TestProjectPoints:
    def test_worse_defense_more_points(self):
        easy = project_points(25.0, 0.28, opp_drtg=118.0, game_pace=99.0, player_team_pace=99.0)
        hard = project_points(25.0, 0.28, opp_drtg=105.0, game_pace=99.0, player_team_pace=99.0)
        assert easy > hard

    def test_faster_pace_more_points(self):
        slow = project_points(25.0, 0.28, opp_drtg=113.0, game_pace=92.0, player_team_pace=99.0)
        fast = project_points(25.0, 0.28, opp_drtg=113.0, game_pace=108.0, player_team_pace=99.0)
        assert fast > slow

    def test_output_positive(self):
        result = project_points(20.0, 0.25, 113.0, 99.0, 99.0)
        assert result > 0

    def test_regression_pulls_toward_season_avg(self):
        # With neutral matchup, projection should be close to season average
        result = project_points(20.0, 0.25, 113.0, 99.0, 99.0)
        assert 15.0 <= result <= 25.0


class TestProjectRebounds:
    def test_output_positive(self):
        assert project_rebounds(8.0, 99.0, 99.0) > 0

    def test_faster_pace_more_rebounds(self):
        slow = project_rebounds(8.0, 90.0, 99.0)
        fast = project_rebounds(8.0, 108.0, 99.0)
        assert fast > slow

    def test_high_opp_oreb_penalty(self):
        normal = project_rebounds(8.0, 99.0, 99.0, opp_oreb_pct=0.25)
        aggressive = project_rebounds(8.0, 99.0, 99.0, opp_oreb_pct=0.38)
        assert normal >= aggressive


class TestProjectAssists:
    def test_output_positive(self):
        assert project_assists(7.0, 99.0, 99.0) > 0

    def test_faster_pace_more_assists(self):
        slow = project_assists(7.0, 90.0, 99.0)
        fast = project_assists(7.0, 108.0, 99.0)
        assert fast > slow


class TestProjectThreePM:
    def test_higher_volume_more_makes(self):
        low = project_three_pm(3.0, 0.38, 0.36)
        high = project_three_pm(9.0, 0.38, 0.36)
        assert high > low

    def test_poor_defense_more_makes(self):
        good_def = project_three_pm(7.0, 0.38, opp_three_pct_allowed=0.32)
        bad_def = project_three_pm(7.0, 0.38, opp_three_pct_allowed=0.40)
        assert bad_def > good_def

    def test_regression_applied_to_hot_shooter(self):
        # Hot shooter (.46 3P%) should project less than raw rate
        raw_makes = 8.0 * 0.46
        projected = project_three_pm(8.0, 0.46, LEAGUE_AVG_THREE_PCT)
        assert projected < raw_makes

    def test_output_non_negative(self):
        assert project_three_pm(0.0, 0.33, 0.36) >= 0


class TestProjectPRA:
    def test_sum_of_components_with_adjustment(self):
        pra = project_pra(25.0, 8.0, 7.0)
        assert pra < 40.0   # should be slightly less than 40 due to correlation adj

    def test_higher_components_higher_pra(self):
        low = project_pra(15.0, 5.0, 4.0)
        high = project_pra(30.0, 10.0, 9.0)
        assert high > low


class TestComputePropEdge:
    def test_over_with_projection_above_line(self):
        result = compute_prop_edge(32.0, 28.5, "OVER")
        assert result["edge_pct"] > 0
        assert result["recommendation"] == "BET"

    def test_under_with_projection_below_line(self):
        result = compute_prop_edge(22.0, 27.5, "UNDER")
        assert result["edge_pct"] > 0
        assert result["recommendation"] == "BET"

    def test_wrong_direction_negative_edge(self):
        result = compute_prop_edge(20.0, 28.5, "OVER")
        assert result["edge_pct"] < 0

    def test_pass_when_small_edge(self):
        result = compute_prop_edge(28.6, 28.5, "OVER")
        assert result["recommendation"] == "PASS"

    def test_returns_required_keys(self):
        result = compute_prop_edge(30.0, 27.5, "OVER")
        assert set(result.keys()) == {
            "model_projection", "prop_line", "bet_direction",
            "edge_pct", "implied_prob", "recommendation"
        }

    def test_implied_prob_bounded(self):
        result = compute_prop_edge(100.0, 5.0, "OVER")
        assert 0.05 <= result["implied_prob"] <= 0.95
