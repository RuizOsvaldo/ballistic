"""Kelly criterion bet sizing for MLB moneyline bets."""

from __future__ import annotations

import pandas as pd

MIN_EDGE = 0.03          # Minimum edge % to recommend a bet
KELLY_FRACTION = 0.5     # Half-Kelly to reduce variance
MAX_STAKE_PCT = 0.05     # Never bet more than 5% of bankroll on one game


def moneyline_to_decimal(american: int) -> float:
    """Convert American moneyline to decimal odds (European format)."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def compute_kelly(model_prob: float, american_odds: int) -> dict:
    """
    Compute edge, Kelly stake fraction, and bet recommendation for one side.

    Parameters
    ----------
    model_prob : float
        Model-derived win probability (0–1)
    american_odds : int
        Best available American moneyline for this side

    Returns
    -------
    dict with keys:
      - edge_pct       : model_prob - implied_prob (as a percentage)
      - kelly_pct      : recommended stake as % of bankroll (half-Kelly, capped)
      - decimal_odds   : decimal representation of the line
      - implied_prob   : vig-free implied probability from the line
      - recommendation : "BET" | "PASS"
    """
    decimal = moneyline_to_decimal(american_odds)
    # Implied prob from decimal odds (no vig applied here — caller should use vig-removed prob)
    implied_prob = 1 / decimal

    edge = model_prob - implied_prob

    if edge <= MIN_EDGE:
        return {
            "edge_pct": round(edge * 100, 2),
            "kelly_pct": 0.0,
            "decimal_odds": round(decimal, 3),
            "implied_prob": round(implied_prob, 4),
            "recommendation": "PASS",
        }

    # Kelly formula for decimal odds: f = (b*p - q) / b  where b = decimal - 1
    b = decimal - 1
    q = 1 - model_prob
    kelly_full = (b * model_prob - q) / b
    kelly_stake = min(kelly_full * KELLY_FRACTION, MAX_STAKE_PCT)
    kelly_stake = max(kelly_stake, 0.0)

    return {
        "edge_pct": round(edge * 100, 2),
        "kelly_pct": round(kelly_stake * 100, 2),
        "decimal_odds": round(decimal, 3),
        "implied_prob": round(implied_prob, 4),
        "recommendation": "BET",
    }


def compute_kelly_for_games(games_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    """
    Apply Kelly calculation to a DataFrame of games with model and implied probabilities.

    Required columns:
      - home_model_prob, away_model_prob
      - home_implied_prob, away_implied_prob
      - home_odds, away_odds (American)

    Added columns:
      - home_edge_pct, away_edge_pct
      - home_kelly_pct, away_kelly_pct
      - home_dollar_stake, away_dollar_stake
      - home_recommendation, away_recommendation
      - best_bet_side : "HOME" | "AWAY" | "PASS"
      - best_bet_edge : edge % of the recommended side
    """
    df = games_df.copy()

    home_results = df.apply(
        lambda r: pd.Series(compute_kelly(r["home_model_prob"], r["home_odds"])),
        axis=1,
    )
    away_results = df.apply(
        lambda r: pd.Series(compute_kelly(r["away_model_prob"], r["away_odds"])),
        axis=1,
    )

    df["home_edge_pct"] = home_results["edge_pct"]
    df["home_kelly_pct"] = home_results["kelly_pct"]
    df["home_recommendation"] = home_results["recommendation"]
    df["home_dollar_stake"] = (home_results["kelly_pct"] / 100 * bankroll).round(2)

    df["away_edge_pct"] = away_results["edge_pct"]
    df["away_kelly_pct"] = away_results["kelly_pct"]
    df["away_recommendation"] = away_results["recommendation"]
    df["away_dollar_stake"] = (away_results["kelly_pct"] / 100 * bankroll).round(2)

    def _best_bet(row):
        if row["home_recommendation"] == "BET" and row["away_recommendation"] == "BET":
            if row["home_edge_pct"] >= row["away_edge_pct"]:
                return pd.Series({"best_bet_side": "HOME", "best_bet_edge": row["home_edge_pct"]})
            return pd.Series({"best_bet_side": "AWAY", "best_bet_edge": row["away_edge_pct"]})
        if row["home_recommendation"] == "BET":
            return pd.Series({"best_bet_side": "HOME", "best_bet_edge": row["home_edge_pct"]})
        if row["away_recommendation"] == "BET":
            return pd.Series({"best_bet_side": "AWAY", "best_bet_edge": row["away_edge_pct"]})
        return pd.Series({"best_bet_side": "PASS", "best_bet_edge": max(row["home_edge_pct"], row["away_edge_pct"])})

    best = df.apply(_best_bet, axis=1)
    df["best_bet_side"] = best["best_bet_side"]
    df["best_bet_edge"] = best["best_bet_edge"]
    return df
