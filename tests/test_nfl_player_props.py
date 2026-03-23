"""Tests for NFL player prop projections."""

from __future__ import annotations

import pytest

from src.sports.football.models.player_props import (
    MIN_PROP_EDGE_PCT,
    build_qb_prop_card,
    build_rb_prop_card,
    build_wr_te_prop_card,
    project_interceptions,
    project_pass_tds,
    project_passing_yards,
    project_rb_receiving_yards,
    project_receiving_yards,
    project_receptions,
    project_rushing_yards,
    _edge,
    _matchup_factor,
)


# ---------------------------------------------------------------------------
# _matchup_factor
# ---------------------------------------------------------------------------

def test_matchup_factor_neutral():
    # opp_def_epa == LEAGUE_AVG_DEF_EPA → factor ≈ 1.0
    result = _matchup_factor(-0.02)
    assert result == pytest.approx(1.0, abs=0.02)


def test_matchup_factor_bad_defense():
    # Positive def_epa = bad defense → factor > 1.0 (more production expected)
    assert _matchup_factor(0.10) > 1.0


def test_matchup_factor_great_defense():
    # Very negative def_epa = great defense → factor < 1.0
    assert _matchup_factor(-0.30) < 1.0


def test_matchup_factor_none():
    assert _matchup_factor(None) == pytest.approx(1.0)


def test_matchup_factor_clamped():
    assert _matchup_factor(5.0) <= 1.30
    assert _matchup_factor(-5.0) >= 0.70


# ---------------------------------------------------------------------------
# _edge
# ---------------------------------------------------------------------------

def test_edge_over_value():
    assert _edge(280.0, 250.0) == pytest.approx(12.0, abs=0.1)


def test_edge_under_value():
    assert _edge(230.0, 250.0) == pytest.approx(-8.0, abs=0.1)


def test_edge_zero_line():
    assert _edge(100.0, 0) == 0.0


# ---------------------------------------------------------------------------
# project_passing_yards
# ---------------------------------------------------------------------------

def test_pass_yds_neutral_defense():
    result = project_passing_yards(270.0, opp_def_epa=-0.02)
    assert 240.0 < result < 290.0


def test_pass_yds_bad_defense_increases():
    neutral = project_passing_yards(270.0, opp_def_epa=-0.02)
    bad_d = project_passing_yards(270.0, opp_def_epa=0.15)
    assert bad_d > neutral


def test_pass_yds_great_defense_decreases():
    neutral = project_passing_yards(270.0, opp_def_epa=-0.02)
    great_d = project_passing_yards(270.0, opp_def_epa=-0.25)
    assert great_d < neutral


def test_pass_yds_regression_toward_mean():
    """Extreme projection should be pulled back toward season average."""
    # With very bad defense, projection shouldn't be too far from season avg
    proj = project_passing_yards(270.0, opp_def_epa=1.0)
    assert proj < 270.0 * 1.4   # Regression caps the upside


# ---------------------------------------------------------------------------
# project_rushing_yards
# ---------------------------------------------------------------------------

def test_rush_yds_baseline():
    result = project_rushing_yards(85.0)
    assert 70.0 < result < 100.0


def test_rush_yds_carry_ratio():
    baseline = project_rushing_yards(85.0)
    more_carries = project_rushing_yards(85.0, carries_vs_season=1.3)
    assert more_carries > baseline


# ---------------------------------------------------------------------------
# project_receiving_yards (WR/TE)
# ---------------------------------------------------------------------------

def test_rec_yds_air_yards_high():
    baseline = project_receiving_yards(65.0)
    high_air = project_receiving_yards(65.0, air_yards_share=0.35)
    assert high_air > baseline


def test_rec_yds_air_yards_low():
    baseline = project_receiving_yards(65.0)
    low_air = project_receiving_yards(65.0, air_yards_share=0.10)
    assert low_air < baseline


def test_rec_yds_no_air_yards():
    result = project_receiving_yards(65.0)
    assert 50.0 < result < 80.0


# ---------------------------------------------------------------------------
# project_receptions
# ---------------------------------------------------------------------------

def test_receptions_high_target_share():
    baseline = project_receptions(5.5)
    high_tgt = project_receptions(5.5, target_share=0.35)
    assert high_tgt > baseline


# ---------------------------------------------------------------------------
# project_interceptions
# ---------------------------------------------------------------------------

def test_interceptions_great_defense_increases_ints():
    baseline = project_interceptions(1.2, opp_def_epa=-0.02)
    great_d = project_interceptions(1.2, opp_def_epa=-0.25)
    assert great_d > baseline


def test_interceptions_bad_defense_decreases_ints():
    baseline = project_interceptions(1.2, opp_def_epa=-0.02)
    bad_d = project_interceptions(1.2, opp_def_epa=0.15)
    assert bad_d < baseline


# ---------------------------------------------------------------------------
# build_qb_prop_card
# ---------------------------------------------------------------------------

def test_qb_prop_card_structure():
    card = build_qb_prop_card(
        player_name="P. Mahomes",
        team="KC",
        season_stats={"pass_yds_pg": 290.0, "comp_pg": 25.0, "pass_tds_pg": 2.2, "int_pg": 0.5},
        prop_lines={"pass_yds": 275.5, "completions": 24.5, "pass_tds": 2.5},
        opp_def_epa=-0.05,
    )
    assert card["player_name"] == "P. Mahomes"
    assert card["position"] == "QB"
    assert card["proj_pass_yds"] is not None
    assert card["edge_pass_yds"] is not None
    assert card["line_pass_tds"] == 2.5


def test_qb_prop_card_missing_lines():
    card = build_qb_prop_card(
        player_name="L. Jackson",
        team="BAL",
        season_stats={"pass_yds_pg": 250.0, "comp_pg": 20.0, "pass_tds_pg": 1.8, "int_pg": 0.4},
        prop_lines={},
    )
    assert card["edge_pass_yds"] is None
    assert card["line_pass_yds"] is None


# ---------------------------------------------------------------------------
# build_rb_prop_card
# ---------------------------------------------------------------------------

def test_rb_prop_card_structure():
    card = build_rb_prop_card(
        player_name="D. Henry",
        team="TEN",
        season_stats={"rush_yds_pg": 95.0, "rec_yds_pg": 22.0},
        prop_lines={"rush_yds": 88.5},
        opp_def_epa=0.05,
    )
    assert card["position"] == "RB"
    assert card["proj_rush_yds"] is not None
    assert card["edge_rush_yds"] is not None
    assert card["line_rec_yds"] is None


# ---------------------------------------------------------------------------
# build_wr_te_prop_card
# ---------------------------------------------------------------------------

def test_wr_prop_card_structure():
    card = build_wr_te_prop_card(
        player_name="T. Hill",
        team="MIA",
        position="WR",
        season_stats={
            "rec_yds_pg": 85.0, "rec_pg": 6.5,
            "air_yards_share": 0.30, "target_share": 0.28,
        },
        prop_lines={"rec_yds": 79.5, "receptions": 6.5},
        opp_def_epa=-0.08,
    )
    assert card["position"] == "WR"
    assert card["proj_rec_yds"] is not None
    assert card["edge_rec_yds"] is not None
    assert card["air_yards_share"] == pytest.approx(0.30)


def test_te_prop_card_no_prop_lines():
    card = build_wr_te_prop_card(
        player_name="T. Kelce",
        team="KC",
        position="TE",
        season_stats={"rec_yds_pg": 72.0, "rec_pg": 5.8},
        prop_lines={},
    )
    assert card["edge_rec_yds"] is None
    assert card["edge_receptions"] is None
