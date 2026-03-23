"""Unit tests for preseason.py"""

import pandas as pd
import pytest

from src.models.preseason import project_team_wins, compute_preseason_projections, SEASON_GAMES, MIN_EDGE_WINS


class TestProjectTeamWins:
    def test_average_team_near_81_wins(self):
        # A perfectly average team (700 RS, 700 RA, no WAR adj)
        wins = project_team_wins(700, 700)
        assert wins == pytest.approx(81.0, abs=2.0)

    def test_strong_offense_more_wins(self):
        good = project_team_wins(850, 650)
        avg = project_team_wins(700, 700)
        assert good > avg

    def test_regression_to_mean(self):
        # An extreme team should be pulled toward 81 wins
        extreme = project_team_wins(950, 550)
        pyth_raw = (950**2 / (950**2 + 550**2)) * 162
        assert extreme < pyth_raw  # regression pulls it down

    def test_positive_war_adjustment_increases_wins(self):
        base = project_team_wins(700, 700, war_adjustment=0)
        improved = project_team_wins(700, 700, war_adjustment=5)
        assert improved > base

    def test_output_in_realistic_range(self):
        wins = project_team_wins(750, 650)
        assert 60 <= wins <= 110

    def test_output_is_rounded(self):
        wins = project_team_wins(700, 700)
        assert wins == round(wins, 1)


class TestComputePreseasonProjections:
    def _make_prior_stats(self):
        return pd.DataFrame([
            {"team": "NYY", "runs_scored": 850, "runs_allowed": 650, "wins": 95},
            {"team": "BOS", "runs_scored": 720, "runs_allowed": 710, "wins": 81},
            {"team": "BAL", "runs_scored": 630, "runs_allowed": 780, "wins": 65},
        ])

    def test_returns_all_teams(self):
        df = compute_preseason_projections(self._make_prior_stats())
        assert len(df) == 3

    def test_includes_projected_wins(self):
        df = compute_preseason_projections(self._make_prior_stats())
        assert "projected_wins" in df.columns

    def test_sorted_by_projected_wins_descending(self):
        df = compute_preseason_projections(self._make_prior_stats())
        wins = df["projected_wins"].tolist()
        assert wins == sorted(wins, reverse=True)

    def test_strong_team_projects_more_wins(self):
        df = compute_preseason_projections(self._make_prior_stats())
        nyy_wins = df[df["team"] == "NYY"]["projected_wins"].iloc[0]
        bal_wins = df[df["team"] == "BAL"]["projected_wins"].iloc[0]
        assert nyy_wins > bal_wins

    def test_with_vegas_lines_adds_edge_column(self):
        vegas = pd.DataFrame([
            {"team": "NYY", "vegas_total": 88.5},
            {"team": "BOS", "vegas_total": 81.5},
            {"team": "BAL", "vegas_total": 70.5},
        ])
        df = compute_preseason_projections(self._make_prior_stats(), vegas)
        assert "edge_wins" in df.columns
        assert "bet_direction" in df.columns

    def test_over_bet_when_projection_above_line(self):
        vegas = pd.DataFrame([{"team": "NYY", "vegas_total": 80.0}])  # well below projection
        df = compute_preseason_projections(
            self._make_prior_stats()[self._make_prior_stats()["team"] == "NYY"].reset_index(drop=True),
            vegas,
        )
        assert df.iloc[0]["bet_direction"] == "OVER"

    def test_under_bet_when_projection_below_line(self):
        vegas = pd.DataFrame([{"team": "BAL", "vegas_total": 90.0}])  # well above projection
        df = compute_preseason_projections(
            self._make_prior_stats()[self._make_prior_stats()["team"] == "BAL"].reset_index(drop=True),
            vegas,
        )
        assert df.iloc[0]["bet_direction"] == "UNDER"

    def test_pass_when_close_to_line(self):
        # Project ~81 wins, line at 81 → PASS
        stats = pd.DataFrame([{"team": "BOS", "runs_scored": 720, "runs_allowed": 710, "wins": 81}])
        vegas = pd.DataFrame([{"team": "BOS", "vegas_total": 81.0}])
        df = compute_preseason_projections(stats, vegas)
        assert df.iloc[0]["bet_direction"] == "PASS"
