"""Unit tests for win_probability.py"""

import pytest
import pandas as pd

from src.models.win_probability import (
    game_win_probability,
    compute_league_avg_fip,
    lineup_matchup_fip_adjustment,
    LEAGUE_AVG_OPS,
    LINEUP_OPS_SCALE,
)


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


# ---------------------------------------------------------------------------
# Lineup matchup FIP adjustment
# ---------------------------------------------------------------------------

def _make_lineup(team: str, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"team": [team] * len(names), "player_name": names})


def _make_batters(names: list[str], obp: float, slg: float) -> pd.DataFrame:
    return pd.DataFrame({"name": names, "obp": [obp] * len(names), "slg": [slg] * len(names)})


class TestLineupMatchupFipAdjustment:
    def test_league_avg_lineup_returns_zero(self):
        # OBP + SLG = LEAGUE_AVG_OPS → adjustment = 0
        obp = LEAGUE_AVG_OPS * 0.45   # rough split
        slg = LEAGUE_AVG_OPS * 0.55
        lineup = _make_lineup("NYY", ["A", "B", "C"])
        batters = _make_batters(["A", "B", "C"], obp, slg)
        adj = lineup_matchup_fip_adjustment(lineup, batters, "NYY")
        assert adj == pytest.approx(0.0, abs=0.01)

    def test_strong_lineup_positive_adjustment(self):
        # OPS 0.770 → (0.770 - 0.720) * 3.0 = +0.15
        lineup = _make_lineup("BOS", ["X", "Y", "Z"])
        batters = _make_batters(["X", "Y", "Z"], obp=0.360, slg=0.410)
        adj = lineup_matchup_fip_adjustment(lineup, batters, "BOS")
        assert adj == pytest.approx(0.15, abs=0.01)

    def test_weak_lineup_negative_adjustment(self):
        # OPS 0.620 → (0.620 - 0.720) * 3.0 = -0.30
        lineup = _make_lineup("MIA", ["P", "Q", "R"])
        batters = _make_batters(["P", "Q", "R"], obp=0.290, slg=0.330)
        adj = lineup_matchup_fip_adjustment(lineup, batters, "MIA")
        assert adj == pytest.approx(-0.30, abs=0.01)

    def test_empty_lineup_returns_zero(self):
        batters = _make_batters(["A"], obp=0.350, slg=0.450)
        adj = lineup_matchup_fip_adjustment(pd.DataFrame(), batters, "NYY")
        assert adj == 0.0

    def test_empty_batter_stats_returns_zero(self):
        lineup = _make_lineup("NYY", ["A"])
        adj = lineup_matchup_fip_adjustment(lineup, pd.DataFrame(), "NYY")
        assert adj == 0.0

    def test_no_matching_players_returns_zero(self):
        lineup = _make_lineup("NYY", ["Unknown Player"])
        batters = _make_batters(["Someone Else"], obp=0.350, slg=0.450)
        adj = lineup_matchup_fip_adjustment(lineup, batters, "NYY")
        assert adj == 0.0
