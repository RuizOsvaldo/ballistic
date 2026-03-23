"""Unit tests for player_props.py"""

import pytest

from src.models.player_props import (
    project_pitcher_strikeouts,
    project_batter_hits,
    project_batter_total_bases,
    project_batter_home_runs,
    compute_prop_edge,
    MIN_PROP_EDGE_PCT,
)


class TestProjectPitcherStrikeouts:
    def test_higher_k_pct_more_strikeouts(self):
        low = project_pitcher_strikeouts(0.18, 6.0)
        high = project_pitcher_strikeouts(0.32, 6.0)
        assert high > low

    def test_more_ip_more_strikeouts(self):
        short = project_pitcher_strikeouts(0.25, 5.0)
        long = project_pitcher_strikeouts(0.25, 7.0)
        assert long > short

    def test_umpire_factor_scales_output(self):
        base = project_pitcher_strikeouts(0.25, 6.0, umpire_zone_factor=1.0)
        wide = project_pitcher_strikeouts(0.25, 6.0, umpire_zone_factor=1.15)
        assert wide > base

    def test_output_positive(self):
        result = project_pitcher_strikeouts(0.25, 6.0)
        assert result > 0

    def test_realistic_range(self):
        # A solid starter (~25% K, 6 IP) should project 5-9 Ks
        result = project_pitcher_strikeouts(0.25, 6.0)
        assert 4.0 <= result <= 10.0


class TestProjectBatterHits:
    def test_high_babip_projects_more_hits(self):
        normal = project_batter_hits(0.300, 4.0, 0.22)
        high = project_batter_hits(0.380, 4.0, 0.22)
        assert high > normal

    def test_low_babip_projects_fewer_hits(self):
        normal = project_batter_hits(0.300, 4.0, 0.22)
        low = project_batter_hits(0.220, 4.0, 0.22)
        assert low < normal

    def test_higher_k_pct_reduces_hits(self):
        low_k = project_batter_hits(0.300, 4.0, 0.15)
        high_k = project_batter_hits(0.300, 4.0, 0.35)
        assert low_k > high_k

    def test_output_positive(self):
        result = project_batter_hits(0.300, 4.0, 0.22)
        assert result > 0

    def test_regression_toward_mean(self):
        # Extreme high BABIP should be regressed — projection should be below raw hit rate
        raw_hits = 0.380 * 4.0 * 0.78  # pure BABIP * AB * (1-K%)
        projected = project_batter_hits(0.380, 4.0, 0.22)
        assert projected < raw_hits


class TestProjectBatterTotalBases:
    def test_higher_slg_more_bases(self):
        low = project_batter_total_bases(0.350, 4.0)
        high = project_batter_total_bases(0.550, 4.0)
        assert high > low

    def test_park_factor_above_one_increases_tb(self):
        neutral = project_batter_total_bases(0.450, 4.0, park_hr_factor=1.0)
        hitter = project_batter_total_bases(0.450, 4.0, park_hr_factor=1.3)
        assert hitter > neutral


class TestComputePropEdge:
    def test_over_with_projection_above_line(self):
        result = compute_prop_edge(7.5, 6.5, "OVER")
        assert result["edge_pct"] > 0

    def test_under_with_projection_below_line(self):
        result = compute_prop_edge(5.5, 6.5, "UNDER")
        assert result["edge_pct"] > 0

    def test_wrong_direction_negative_edge(self):
        result = compute_prop_edge(5.0, 7.0, "OVER")
        assert result["edge_pct"] < 0

    def test_bet_recommendation_above_threshold(self):
        # Create a large edge
        result = compute_prop_edge(10.0, 6.5, "OVER")
        assert result["recommendation"] == "BET"

    def test_pass_recommendation_below_threshold(self):
        # Small edge
        result = compute_prop_edge(6.6, 6.5, "OVER")
        assert result["recommendation"] == "PASS"

    def test_returns_required_keys(self):
        result = compute_prop_edge(7.0, 6.5, "OVER")
        assert set(result.keys()) == {
            "model_projection", "prop_line", "bet_direction",
            "edge_pct", "implied_prob", "recommendation"
        }
