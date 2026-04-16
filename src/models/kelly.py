"""Kelly criterion bet sizing for MLB moneyline, run line, and game total bets."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_EDGE = 0.03          # Minimum edge % to recommend a bet
KELLY_FRACTION = 0.5     # Half-Kelly to reduce variance
MAX_STAKE_PCT = 0.05     # Never bet more than 5% of bankroll on one game
_RL_MAX_K = 35           # Max runs per team for Poisson RL computation (covers >99.9% mass)


def moneyline_to_decimal(american: int) -> float:
    """Convert American moneyline to decimal odds (European format)."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


# ---------------------------------------------------------------------------
# Poisson helpers (run line and total edge)
# ---------------------------------------------------------------------------

def _poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """Return PMF array p[k] = P(X=k) for k=0..max_k, X ~ Poisson(lam)."""
    lam = max(lam, 1e-9)
    k = np.arange(max_k + 1, dtype=np.float64)
    log_gammas = np.array([math.lgamma(ki + 1) for ki in range(max_k + 1)])
    log_pmf = k * math.log(lam) - lam - log_gammas
    return np.exp(log_pmf)


def _p_home_covers_rl(lam_home: float, lam_away: float) -> float:
    """P(home_score − away_score > 1.5) — probability home team covers −1.5."""
    ph = _poisson_pmf(lam_home, _RL_MAX_K)
    pa = _poisson_pmf(lam_away, _RL_MAX_K)
    joint = np.outer(ph, pa)                    # joint[h, a] = P(home=h, away=a)
    h_mat, a_mat = np.meshgrid(
        np.arange(_RL_MAX_K + 1), np.arange(_RL_MAX_K + 1), indexing="ij"
    )
    return float(np.sum(joint[h_mat - a_mat > 1.5]))


def _p_over_total(lam_home: float, lam_away: float, total_line: float) -> float:
    """P(home + away > total_line), using X+Y ~ Poisson(lam_home + lam_away)."""
    lam_total = lam_home + lam_away
    max_k = max(int(lam_total * 3), 60)
    pt = _poisson_pmf(lam_total, max_k)
    return float(np.sum(pt[np.arange(max_k + 1) > total_line]))


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------

def compute_kelly(
    model_prob: float,
    american_odds: int,
    opponent_american_odds: int | None = None,
) -> dict:
    """
    Compute edge, Kelly stake fraction, and bet recommendation for one side.

    Parameters
    ----------
    model_prob : float
        Model-derived win probability (0–1)
    american_odds : int
        Best available American moneyline for this side
    opponent_american_odds : int | None
        Moneyline for the other side. When provided, implied_prob is computed
        vig-free by normalising both sides' raw implied probs.

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
    if opponent_american_odds is not None:
        opp_decimal = moneyline_to_decimal(opponent_american_odds)
        raw = 1.0 / decimal
        opp_raw = 1.0 / opp_decimal
        implied_prob = raw / (raw + opp_raw)
    else:
        implied_prob = 1.0 / decimal

    edge = model_prob - implied_prob

    if edge <= MIN_EDGE:
        return {
            "edge_pct": round(edge * 100, 2),
            "kelly_pct": 0.0,
            "decimal_odds": round(decimal, 3),
            "implied_prob": round(implied_prob, 4),
            "recommendation": "PASS",
        }

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


def compute_rl_edge(
    proj_home_runs: float,
    proj_away_runs: float,
    home_rl_odds: int | None,
    away_rl_odds: int | None,
) -> dict:
    """
    Compute edge on the ±1.5 run line using a Poisson joint run distribution.

    home_rl_odds: American odds for the home team at −1.5.
    away_rl_odds: American odds for the away team at +1.5.
    """
    p_home = _p_home_covers_rl(proj_home_runs, proj_away_runs)
    p_away = 1.0 - p_home

    home_edge: float | None = None
    away_edge: float | None = None

    if home_rl_odds is not None:
        home_implied = 1.0 / moneyline_to_decimal(home_rl_odds)
        home_edge = round((p_home - home_implied) * 100, 2)

    if away_rl_odds is not None:
        away_implied = 1.0 / moneyline_to_decimal(away_rl_odds)
        away_edge = round((p_away - away_implied) * 100, 2)

    he = home_edge or 0.0
    ae = away_edge or 0.0
    min_e = MIN_EDGE * 100

    if he >= min_e and he >= ae:
        best_side, best_edge = "HOME", he
    elif ae >= min_e:
        best_side, best_edge = "AWAY", ae
    else:
        best_side, best_edge = "PASS", max(he, ae)

    return {
        "home_rl_cover_prob": round(p_home, 4),
        "away_rl_cover_prob": round(p_away, 4),
        "home_rl_edge_pct":   home_edge,
        "away_rl_edge_pct":   away_edge,
        "best_rl_side":       best_side,
        "best_rl_edge_pct":   best_edge,
    }


def compute_total_edge(
    proj_home_runs: float,
    proj_away_runs: float,
    total_line: float,
    over_odds: int | None,
    under_odds: int | None,
) -> dict:
    """
    Compute edge on the game total using Poisson run distribution.

    Returns over/under cover probabilities and edge vs. market for each direction.
    """
    p_over = _p_over_total(proj_home_runs, proj_away_runs, total_line)
    p_under = 1.0 - p_over

    over_edge: float | None = None
    under_edge: float | None = None

    if over_odds is not None:
        over_implied = 1.0 / moneyline_to_decimal(over_odds)
        over_edge = round((p_over - over_implied) * 100, 2)

    if under_odds is not None:
        under_implied = 1.0 / moneyline_to_decimal(under_odds)
        under_edge = round((p_under - under_implied) * 100, 2)

    oe = over_edge or 0.0
    ue = under_edge or 0.0
    min_e = MIN_EDGE * 100

    if oe >= min_e and oe >= ue:
        best_dir, best_edge = "Over", oe
    elif ue >= min_e:
        best_dir, best_edge = "Under", ue
    else:
        best_dir = "Over" if p_over >= p_under else "Under"
        best_edge = max(oe, ue)

    return {
        "total_over_prob":      round(p_over, 4),
        "total_under_prob":     round(p_under, 4),
        "over_edge_pct":        over_edge,
        "under_edge_pct":       under_edge,
        "best_total_direction": best_dir,
        "best_total_edge_pct":  best_edge,
    }


# ---------------------------------------------------------------------------
# Per-game Kelly application
# ---------------------------------------------------------------------------

def compute_kelly_for_games(games_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    """
    Apply Kelly calculation to a DataFrame of games with model and implied probabilities.

    Required columns:
      - home_model_prob, away_model_prob
      - home_implied_prob, away_implied_prob
      - home_odds, away_odds (American)

    Optional columns (for RL / total edge):
      - proj_home_runs, proj_away_runs
      - home_rl_odds, away_rl_odds
      - total_line, over_odds, under_odds

    Added columns:
      - home_edge_pct, away_edge_pct
      - home_kelly_pct, away_kelly_pct
      - home_dollar_stake, away_dollar_stake
      - home_recommendation, away_recommendation
      - best_bet_side : "HOME" | "AWAY" | "PASS"
      - best_bet_edge : ML edge % of the recommended side
      - home_rl_edge_pct, away_rl_edge_pct, best_rl_side, best_rl_edge_pct
      - over_edge_pct, under_edge_pct, best_total_direction, best_total_edge_pct
    """
    df = games_df.copy()

    # ── Moneyline Kelly ───────────────────────────────────────────────────────
    home_results = df.apply(
        lambda r: pd.Series(compute_kelly(r["home_model_prob"], r["home_odds"], r["away_odds"])),
        axis=1,
    )
    away_results = df.apply(
        lambda r: pd.Series(compute_kelly(r["away_model_prob"], r["away_odds"], r["home_odds"])),
        axis=1,
    )

    df["home_edge_pct"]       = home_results["edge_pct"]
    df["home_kelly_pct"]      = home_results["kelly_pct"]
    df["home_recommendation"] = home_results["recommendation"]
    df["home_dollar_stake"]   = (home_results["kelly_pct"] / 100 * bankroll).round(2)

    df["away_edge_pct"]       = away_results["edge_pct"]
    df["away_kelly_pct"]      = away_results["kelly_pct"]
    df["away_recommendation"] = away_results["recommendation"]
    df["away_dollar_stake"]   = (away_results["kelly_pct"] / 100 * bankroll).round(2)

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

    # ── Run line edge ─────────────────────────────────────────────────────────
    _rl_null = {
        "home_rl_cover_prob": None, "away_rl_cover_prob": None,
        "home_rl_edge_pct": None,   "away_rl_edge_pct": None,
        "best_rl_side": "PASS",     "best_rl_edge_pct": 0.0,
    }

    def _rl_row(row) -> pd.Series:
        phr = row.get("proj_home_runs")
        par = row.get("proj_away_runs")
        if phr is None or par is None or not np.isfinite(float(phr)) or not np.isfinite(float(par)):
            return pd.Series(_rl_null)
        hro = row.get("home_rl_odds")
        aro = row.get("away_rl_odds")
        hro_i = int(hro) if hro is not None and np.isfinite(float(hro)) else None
        aro_i = int(aro) if aro is not None and np.isfinite(float(aro)) else None
        return pd.Series(compute_rl_edge(float(phr), float(par), hro_i, aro_i))

    # ── Total edge ────────────────────────────────────────────────────────────
    _tot_null = {
        "total_over_prob": None,  "total_under_prob": None,
        "over_edge_pct": None,    "under_edge_pct": None,
        "best_total_direction": None, "best_total_edge_pct": 0.0,
    }

    def _total_row(row) -> pd.Series:
        phr = row.get("proj_home_runs")
        par = row.get("proj_away_runs")
        tl  = row.get("total_line")
        if any(v is None or not np.isfinite(float(v)) for v in (phr, par, tl)):
            return pd.Series(_tot_null)
        oo = row.get("over_odds")
        uo = row.get("under_odds")
        oo_i = int(oo) if oo is not None and np.isfinite(float(oo)) else None
        uo_i = int(uo) if uo is not None and np.isfinite(float(uo)) else None
        return pd.Series(compute_total_edge(float(phr), float(par), float(tl), oo_i, uo_i))

    rl_results    = df.apply(_rl_row, axis=1)
    total_results = df.apply(_total_row, axis=1)

    for col in rl_results.columns:
        df[col] = rl_results[col]
    for col in total_results.columns:
        df[col] = total_results[col]

    return df
