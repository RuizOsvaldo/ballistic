"""Unit tests for win_probability.py"""

import pytest

from src.models.win_probability import game_win_probability, compute_league_avg_fip
import pandas as pd


class TestGameWinProbability:
    def test_probs_sum_to_one(self):
        hp, ap = game_win_probability(750, 650, 700, 700, None, None, 4.00)
        assert hp + ap == pytest.approx(1.0, abs=0.001)

    def test_home_field_advantage_favors_home(self):
        # Equal teams — home should win more than 50%
        hp, ap = game_win_probability(700, 700, 700, 700, None, None, 4.00)
        assert hp > 0.50

    def test_better_pitcher_improves_win_prob(self):
        # Home team has elite pitcher (FIP 2.50 vs league avg 4.00)
        hp_with_ace, _ = game_win_probability(700, 700, 700, 700, 2.50, None, 4.00)
        hp_avg, _ = game_win_probability(700, 700, 700, 700, None, None, 4.00)
        assert hp_with_ace > hp_avg

    def test_better_away_pitcher_hurts_home(self):
        hp_with_opp_ace, _ = game_win_probability(700, 700, 700, 700, None, 2.50, 4.00)
        hp_avg, _ = game_win_probability(700, 700, 700, 700, None, None, 4.00)
        assert hp_with_opp_ace < hp_avg

    def test_output_bounded(self):
        hp, ap = game_win_probability(1000, 100, 100, 1000, 1.50, 6.00, 4.00)
        assert 0.05 <= hp <= 0.95
        assert 0.05 <= ap <= 0.95

    def test_no_pitcher_info_uses_pythagorean_base(self):
        hp, ap = game_win_probability(800, 600, 600, 800, None, None, 4.00)
        # Home has better run differential — should win more
        assert hp > ap


class TestComputeLeagueAvgFip:
    def test_empty_df_returns_fallback(self):
        assert compute_league_avg_fip(pd.DataFrame()) == 4.00

    def test_ip_weighted_average(self):
        df = pd.DataFrame([
            {"fip": 3.00, "ip": 100},
            {"fip": 5.00, "ip": 100},
        ])
        avg = compute_league_avg_fip(df)
        assert avg == pytest.approx(4.00, abs=0.01)

    def test_higher_ip_pitcher_weighted_more(self):
        df = pd.DataFrame([
            {"fip": 2.00, "ip": 200},
            {"fip": 6.00, "ip": 50},
        ])
        avg = compute_league_avg_fip(df)
        # Weighted toward the 2.00 pitcher with more IP
        assert avg < 4.00
