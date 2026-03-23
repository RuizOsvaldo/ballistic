"""Unit tests for Kelly criterion bet sizing."""

import pytest

from src.models.kelly import compute_kelly, moneyline_to_decimal


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


def test_pass_below_threshold():
    # Edge clearly below MIN_EDGE (3%) → PASS
    # +100 implies 50%, model_prob=0.52 → edge=2% < 3%
    result = compute_kelly(model_prob=0.52, american_odds=100)
    assert result["recommendation"] == "PASS"


def test_bet_above_threshold():
    result = compute_kelly(model_prob=0.58, american_odds=100)
    assert result["recommendation"] == "BET"
