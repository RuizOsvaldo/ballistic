"""Unit tests for Kelly criterion bet sizing."""

import pytest

from src.models.kelly import (
    compute_kelly,
    compute_rl_edge,
    compute_total_edge,
    moneyline_to_decimal,
    _p_home_covers_rl,
    _p_over_total,
)


# ---------------------------------------------------------------------------
# Moneyline helpers
# ---------------------------------------------------------------------------

def test_decimal_odds_positive_moneyline():
    # +150 → decimal 2.50
    assert moneyline_to_decimal(150) == pytest.approx(2.50, abs=0.001)


def test_decimal_odds_negative_moneyline():
    # -150 → decimal 1.667
    assert moneyline_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)


def test_pass_when_edge_below_threshold():
    # Model says 55%, line implies 60% → negative edge
    result = compute_kelly(model_prob=0.55, american_odds=-150)
    assert result["recommendation"] == "PASS"
    assert result["kelly_pct"] == 0.0


def test_bet_when_positive_edge():
    # Model says 65%, line is +100 (implies 50%) → strong edge
    result = compute_kelly(model_prob=0.65, american_odds=100)
    assert result["recommendation"] == "BET"
    assert result["kelly_pct"] > 0
    assert result["edge_pct"] > 3.0


def test_kelly_pct_capped_at_max():
    # Even with enormous edge, stake should not exceed MAX_STAKE_PCT * 100
    result = compute_kelly(model_prob=0.95, american_odds=300)
    assert result["kelly_pct"] <= 5.0  # MAX_STAKE_PCT = 5%


def test_edge_pct_is_correct():
    # +100 line implies 50%. Model says 60%. Edge = 10%.
    result = compute_kelly(model_prob=0.60, american_odds=100)
    assert result["edge_pct"] == pytest.approx(10.0, abs=0.1)


def test_vig_removed_implied_prob():
    # Both sides at -110: raw implied = 52.38% each, vig-free = 50.0% each.
    # Model at 53% → edge = 3.0% → BET (right at threshold).
    result = compute_kelly(model_prob=0.53, american_odds=-110, opponent_american_odds=-110)
    assert result["implied_prob"] == pytest.approx(0.50, abs=0.005)
    assert result["edge_pct"] == pytest.approx(3.0, abs=0.1)


def test_vig_removed_vs_raw_difference():
    # Without vig removal, model=0.53 vs -110 raw implied 52.38% → edge 0.62% → PASS.
    # With vig removal, same inputs → edge 3.0% → BET.
    raw = compute_kelly(model_prob=0.53, american_odds=-110)
    vig_free = compute_kelly(model_prob=0.53, american_odds=-110, opponent_american_odds=-110)
    assert raw["recommendation"] == "PASS"
    assert vig_free["recommendation"] == "BET"


def test_pass_below_threshold():
    # Edge clearly below MIN_EDGE (3%) → PASS
    # +100 implies 50%, model_prob=0.52 → edge=2% < 3%
    result = compute_kelly(model_prob=0.52, american_odds=100)
    assert result["recommendation"] == "PASS"


def test_bet_above_threshold():
    result = compute_kelly(model_prob=0.58, american_odds=100)
    assert result["recommendation"] == "BET"


# ---------------------------------------------------------------------------
# Poisson helpers
# ---------------------------------------------------------------------------

def test_rl_symmetric_teams():
    # Equal teams: P(home covers -1.5) should be roughly 0.28-0.35
    p = _p_home_covers_rl(5.0, 5.0)
    assert 0.25 < p < 0.40


def test_rl_dominant_home():
    # Home projects 8 runs, away 3: home covers with high probability
    p = _p_home_covers_rl(8.0, 3.0)
    assert p > 0.70


def test_rl_dominant_away():
    # Home projects 3 runs, away 8: away covers (home almost never covers -1.5)
    p = _p_home_covers_rl(3.0, 8.0)
    assert p < 0.10


def test_over_total_near_projection():
    # Proj total == total_line: P(over) should be near 50%
    p = _p_over_total(4.5, 4.0, 8.5)
    assert 0.35 < p < 0.65


def test_over_total_clearly_over():
    # Projected 9.5 runs vs line 7.5: P(over) should be high
    p = _p_over_total(5.0, 4.5, 7.5)
    assert p > 0.65


def test_over_total_clearly_under():
    # Projected 6.0 runs vs line 9.5: P(over) should be low
    p = _p_over_total(3.0, 3.0, 9.5)
    assert p < 0.25


# ---------------------------------------------------------------------------
# compute_rl_edge
# ---------------------------------------------------------------------------

def test_rl_edge_no_odds_returns_pass():
    result = compute_rl_edge(5.0, 5.0, None, None)
    assert result["best_rl_side"] == "PASS"
    assert result["home_rl_edge_pct"] is None
    assert result["away_rl_edge_pct"] is None


def test_rl_edge_dominant_home_has_positive_edge():
    # Home clearly better: home -1.5 at -110 should have positive edge
    result = compute_rl_edge(8.0, 3.0, home_rl_odds=-110, away_rl_odds=-110)
    assert result["home_rl_edge_pct"] > 0
    assert result["best_rl_side"] == "HOME"


def test_rl_edge_dominant_away_has_positive_edge():
    result = compute_rl_edge(3.0, 8.0, home_rl_odds=-110, away_rl_odds=-110)
    assert result["away_rl_edge_pct"] > 0
    assert result["best_rl_side"] == "AWAY"


def test_rl_edge_cover_probs_sum_to_one():
    result = compute_rl_edge(5.0, 4.5, -130, -110)
    assert result["home_rl_cover_prob"] + result["away_rl_cover_prob"] == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# compute_total_edge
# ---------------------------------------------------------------------------

def test_total_edge_no_odds_returns_none_edges():
    result = compute_total_edge(4.5, 4.0, 8.5, None, None)
    assert result["over_edge_pct"] is None
    assert result["under_edge_pct"] is None


def test_total_edge_over_probs_sum_to_one():
    result = compute_total_edge(4.5, 4.0, 8.5, -110, -110)
    assert result["total_over_prob"] + result["total_under_prob"] == pytest.approx(1.0, abs=0.001)


def test_total_edge_clear_over():
    # Proj 9.5 runs, line 7.5 → over has strong positive edge
    result = compute_total_edge(5.0, 4.5, 7.5, -110, -110)
    assert result["over_edge_pct"] > 5.0
    assert result["best_total_direction"] == "Over"


def test_total_edge_clear_under():
    # Proj 6.0 runs, line 9.5 → under has strong positive edge
    result = compute_total_edge(3.0, 3.0, 9.5, -110, -110)
    assert result["under_edge_pct"] > 5.0
    assert result["best_total_direction"] == "Under"
