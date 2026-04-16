"""Analysis page — model performance, prediction accuracy, and ML process review."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import numpy as np
from src.data.predictions_db import load_predictions
from src.data.game_results import verify_predictions
from src.data.bet_log_db import load_bets, get_best_bet_type
from src.data.baseball_stats import get_historical_team_stats, get_team_stats
from src.models.calibration import (
    compute_calibration_table,
    compute_edge_vs_outcome,
    compute_signal_roi,
    recommend_threshold_adjustments,
)


def _accuracy_summary(df: pd.DataFrame) -> dict:
    verified = df[df["correct"].notna()].copy()
    if verified.empty:
        return {}
    n = len(verified)
    correct = int(verified["correct"].sum())
    acc = correct / n
    # Accuracy on bets only (where model had a bet recommendation)
    bets = verified[verified["bet_side"] != "PASS"]
    bet_acc = (bets["correct"].sum() / len(bets)) if not bets.empty else None
    return {
        "total_games": n,
        "correct": correct,
        "accuracy": acc,
        "bet_games": len(bets),
        "bet_accuracy": bet_acc,
    }


def _render_model_explainer() -> None:
    st.subheader("How the Model Works")
    st.caption(
        "Plain-language breakdown of every step — what the math is, why it matters, "
        "and what real-world edge it's trying to capture."
    )

    with st.expander("1. Pythagorean Win Expectation (Team Quality Baseline)", expanded=False):
        st.markdown("""
**Formula:** `W% = RS² / (RS² + RA²)`

**What it is:** Baseball's version of a team power-rating. A team's actual win-loss record contains
significant noise from close games, bullpen luck, and sequencing. Runs scored and runs allowed are
a more stable and predictive measure of true team quality — especially early in the season.

**Why it matters:** Teams that win more games than their Pythagorean W% predicts are "overperforming."
They tend to regress because they've been winning more 1-run games than expected — a streak that
doesn't persist. The model fades these teams. Teams 5%+ below Pythagorean W% are undervalued by the
market and get backed.

**Real-world edge:** The betting market overreacts to recent wins/losses (recency bias). Pythagorean
W% anchors predictions to a team's true run-scoring quality, cutting through noise.

**Data source:** MLB Stats API — season runs scored (batting) and runs allowed (pitching), merged on
numeric team ID to avoid name-format mismatches.
        """)

    with st.expander("2. Starting Pitcher FIP Adjustment", expanded=False):
        st.markdown("""
**Formula:** `pitcher_adj = (league_avg_FIP − starter_FIP) × 0.03`

**What it is:** ERA is a noisy stat — it includes unearned runs, errors, and sequencing luck.
FIP (Fielding Independent Pitching) strips ERA down to only what a pitcher directly controls:
home runs allowed, walks, hit batters, and strikeouts.

```
FIP = (13×HR + 3×(BB+HBP) − 2×K) / IP + 3.15
```

A pitcher with FIP 3.0 vs. league average 4.0 adds `+3%` win probability to his team.

**Why it matters:** Starters account for roughly 60–70% of a game's pitching. Two aces facing each
other tightens the game significantly. Two replacement-level starters in a bad park inflates total.

**Real-world edge:** The market often prices starters based on ERA (the stat casual bettors see).
FIP exposes pitchers whose ERA is about to mean-revert — either crashing back to earth or recovering.

**Computed from:** MLB Stats API pitcher stats (IP, HR, BB, HBP, K) — FIP calculated manually using
constant 3.15 (2024–2026 league calibrated).
        """)

    with st.expander("3. BABIP Regression Signal", expanded=False):
        st.markdown("""
**BABIP** = Batting Average on Balls In Play = `(H − HR) / (AB − K − HR + SF)`

**What it is:** Measures how often batted balls (that aren't home runs or strikeouts) fall in for hits.
League average is around .300. Pitchers have very little control over this — it's largely determined
by defense, luck, and sequencing.

**Why it matters:** A pitcher with BABIP .340 hasn't suddenly become bad — fielders have been
dropping balls. His ERA is inflated. Expect it to drop. Conversely, a pitcher holding a .255 BABIP
has been running hot — ERA will rise.

- **BABIP > .320** → pitcher's ERA is inflated → future performance better than it looks → back
- **BABIP < .275** → pitcher's ERA is deflated → future performance worse than it looks → fade

**Real-world edge:** The market prices today's ERA. BABIP tells you what tomorrow's ERA will look like.

**Used for:** Signal confidence rating (Low / Medium / High) and as context for AI game analysis.
        """)

    with st.expander("4. Bullpen FIP Adjustment", expanded=False):
        st.markdown("""
**Formula:** `bullpen_adj = (bullpen_FIP − league_avg_bp_FIP) × 0.33 × 0.5`

**What it is:** Starters don't finish games anymore. The bullpen handles roughly 35–40% of innings.
A terrible bullpen can blow a lead that a great starter built. This adjustment accounts for reliever
quality on both sides.

**Why it matters:** Betting markets often overlook bullpen matchups when the starter narrative
dominates. A team with a great starter but a bottom-5 bullpen is significantly riskier than it appears.

**Adjustment range:** ±1–3% per game depending on bullpen quality spread.

**Data source:** MLB Stats API reliever stats (IP, HR, BB, K) — same FIP formula applied to
non-starters with ≥5 IP on the season.
        """)

    with st.expander("5. Edge vs. Vegas Implied Probability", expanded=False):
        st.markdown("""
**What it is:** Once the model produces a win probability, it's compared to the market's vig-removed
implied probability from the best available moneyline.

```
raw_home_prob = |american_odds| / (|american_odds| + 100)   # if negative
raw_home_prob = 100 / (american_odds + 100)                  # if positive
vig_removed_prob = raw_home_prob / (raw_home_prob + raw_away_prob)
edge = model_prob − vig_removed_prob
```

**Bet threshold:** Edge ≥ 3% triggers a recommendation.

**Why it matters:** The book's vig (the overround built into both sides) means you need a real edge
just to break even. Finding a 3%+ edge after vig removal means the model sees genuine mispricing.

**Run line (±1.5):** The home team always gives 1.5 runs if favored (-1.5), receives 1.5 if underdog
(+1.5). The away team is always the mirror opposite. A team with `-110` ML at `-1.5` is a modest
favorite. A team at `+155` with `+1.5` is a significant underdog getting protection.
        """)

    with st.expander("6. Half-Kelly Bet Sizing", expanded=False):
        st.markdown("""
```
kelly_full = (b × p − q) / b      # b = decimal odds − 1, p = model prob, q = 1 − p
kelly_stake = kelly_full × 0.5    # half-Kelly: reduce variance
stake = min(kelly_stake, 0.05)    # hard cap at 5% of bankroll
```

**What it is:** Kelly criterion is the mathematically optimal bet size to maximize long-run bankroll
growth given an edge. Half-Kelly is used to reduce variance while still growing at ~75% the optimal rate.

**Why it matters:** Even with a genuine edge, full Kelly bets create dangerous drawdowns. Over a
162-game season with a 55% win rate, full Kelly can produce 30–40% swings in bankroll. Half-Kelly
cuts that in half. The 5% hard cap prevents a single mis-modeled game from being catastrophic.

**Practical rule of thumb:** A 3% edge at -110 odds → ~1.5% of bankroll stake. A 8% edge at +150
odds → ~3.5% of bankroll stake.
        """)


def _render_prediction_accuracy(preds: pd.DataFrame) -> None:
    st.subheader("Prediction Accuracy — Verified Games")
    st.caption(
        "Every game the model predicts is stored and automatically verified once the final score "
        "is posted. Overall accuracy tells you if the model picks winners better than a coin flip. "
        "Bet accuracy (games where edge ≥ 3%) is the number that actually matters for your bankroll."
    )

    if preds.empty:
        st.info(
            "No past predictions yet. Predictions are automatically saved each time you load "
            "the Games page. Come back after the day's games complete to see accuracy."
        )
        return

    verified = preds[preds["correct"].notna()].copy()
    pending = preds[preds["correct"].isna()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games Predicted", len(preds))
    col2.metric("Verified", len(verified))
    col3.metric("Pending", len(pending))

    if verified.empty:
        st.info("Results haven't been verified yet. Verification runs automatically for past games.")
        return

    stats = _accuracy_summary(preds)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy", f"{stats['accuracy']:.1%}", help="% of games where predicted winner was correct")
    c2.metric("Bet Accuracy", f"{stats['bet_accuracy']:.1%}" if stats.get("bet_accuracy") is not None else "N/A",
              help="Accuracy on games where model had an edge recommendation")
    c3.metric("Games w/ Bet Signal", stats.get("bet_games", 0))

    # Daily accuracy chart
    verified["date"] = pd.to_datetime(verified["prediction_date"]).dt.date
    daily = verified.groupby("date").agg(
        games=("correct", "count"),
        correct_count=("correct", "sum"),
    ).reset_index()
    daily["accuracy"] = daily["correct_count"] / daily["games"]

    if len(daily) > 1:
        fig = px.bar(
            daily,
            x="date",
            y="accuracy",
            text=daily["accuracy"].apply(lambda x: f"{x:.0%}"),
            color="accuracy",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            color_continuous_midpoint=0.5,
            range_color=[0.3, 0.7],
            title="Daily Prediction Accuracy",
            labels={"accuracy": "Accuracy", "date": "Date"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                      annotation_text="Coin flip (50%)")
        fig.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Accuracy by edge tier
    if "edge_pct" in verified.columns and verified["edge_pct"].notna().any():
        st.markdown("#### Does Higher Edge = More Wins?")
        st.caption(
            "If the model is calibrated correctly, games with higher edge should win at a higher "
            "rate. A flat line across all edge tiers means the edge signal isn't meaningful. "
            "A rising line means the model is finding real price discrepancies."
        )
        bins = pd.cut(verified["edge_pct"], bins=[0, 3, 5, 8, 20],
                      labels=["0-3%", "3-5%", "5-8%", "8%+"])
        edge_acc = verified.groupby(bins, observed=True)["correct"].agg(["mean", "count"])
        edge_acc.columns = ["Accuracy", "Games"]
        edge_acc.index.name = "Edge Tier"
        edge_acc["Accuracy"] = edge_acc["Accuracy"].apply(lambda x: f"{x:.1%}")
        st.dataframe(edge_acc, width="stretch")

    # Full prediction history
    st.markdown("#### Prediction History")
    st.caption(
        "Compact view of all predictions: ML (moneyline), RL (run line/spread), and O/U (over/under). "
        "Each prediction type is highlighted green if correct, red if incorrect. Full row is green if all three correct, red if all three incorrect."
    )

    history_cols = [
        "prediction_date", "away_team", "home_team",
        "predicted_winner", "home_model_prob", "edge_pct",
        "best_rl_side", "home_rl", "away_rl",
        "best_total_direction", "proj_total", "total_line",
        "correct", "rl_correct", "total_correct",
    ]
    present = [c for c in history_cols if c in verified.columns]
    display = verified[present].copy().reset_index(drop=True)

    # Save raw correct values (0/1) before emoji conversion — used for row coloring
    correct_raw = display["correct"].copy()
    rl_correct_raw = display.get("rl_correct", pd.Series([None] * len(display))).copy()
    total_correct_raw = display.get("total_correct", pd.Series([None] * len(display))).copy()

    # Build new compact columns
    new_display = pd.DataFrame()
    new_display["date"] = display["prediction_date"]
    new_display["matchup"] = display["away_team"] + " @ " + display["home_team"]

    # ML: Predicted winner with emoji (✅/❌)
    ml_emojis = correct_raw.map({1: "✅", 0: "❌", None: "?"})
    new_display["ML"] = display["predicted_winner"] + " " + ml_emojis
    new_display["ML edge"] = display["edge_pct"].round(1).astype(str) + "%"

    # RL: Predicted side with spread
    def _fmt_rl(row):
        side = row.get("best_rl_side")
        if side is None or (isinstance(side, float) and pd.isna(side)):
            return "—"
        if side.upper() == "HOME":
            spread = row.get("home_rl")
        else:
            spread = row.get("away_rl")
        if spread is None or (isinstance(spread, float) and pd.isna(spread)):
            return "—"
        team = row["home_team"] if side.upper() == "HOME" else row["away_team"]
        spread_str = f"+{spread:.1f}" if float(spread) > 0 else f"{spread:.1f}"
        return f"{team} {spread_str}"

    rl_emojis = rl_correct_raw.map({1: "✅", 0: "❌", None: "?"})
    new_display["RL"] = display.apply(_fmt_rl, axis=1) + " " + rl_emojis

    # RL edge — if available in the data
    if "rl_edge_pct" in verified.columns:
        rl_edge = verified["rl_edge_pct"].round(1).astype(str) + "%"
    else:
        rl_edge = "—"
    new_display["RL edge"] = rl_edge

    # O/U: Direction + projected total
    def _fmt_ou(row):
        direction = row.get("best_total_direction")
        total = row.get("proj_total")
        if direction is None or (isinstance(direction, float) and pd.isna(direction)):
            return "—"
        if total is None or (isinstance(total, float) and pd.isna(total)):
            return direction
        return f"{direction} {total:.1f}"

    ou_emojis = total_correct_raw.map({1: "✅", 0: "❌", None: "?"})
    new_display["O/U"] = display.apply(_fmt_ou, axis=1) + " " + ou_emojis

    # O/U edge — if available in the data
    if "total_edge_pct" in verified.columns:
        ou_edge = verified["total_edge_pct"].round(1).astype(str) + "%"
    else:
        ou_edge = "—"
    new_display["O/U edge"] = ou_edge

    # Add Home Prob as additional prediction info
    new_display["Home Prob"] = (display["home_model_prob"] * 100).round(1).astype(str) + "%"

    display = new_display.reset_index(drop=True)

    def _row_highlight(row):
        """
        Apply cell-level or row-level highlighting:
        - Green row if all three predictions (ML, RL, O/U) are correct
        - Red row if all three predictions are incorrect
        - Otherwise, green/red individual cells based on their prediction type
        """
        row_idx = row.name
        ml = correct_raw.iloc[row_idx] if row_idx < len(correct_raw) else None
        rl = rl_correct_raw.iloc[row_idx] if row_idx < len(rl_correct_raw) else None
        ou = total_correct_raw.iloc[row_idx] if row_idx < len(total_correct_raw) else None

        # Check if all three are either correct (1) or incorrect (0)
        all_correct = ml == 1 and rl == 1 and ou == 1
        all_incorrect = ml == 0 and rl == 0 and ou == 0

        if all_correct:
            return ["background-color: #0d2b0d"] * len(row)
        elif all_incorrect:
            return ["background-color: #2b0d0d"] * len(row)

        # Otherwise, highlight cells by prediction type
        styles = []
        for col_name in row.index:
            bg = ""
            if col_name in ["ML", "ML edge"]:  # ML prediction columns
                if ml == 1:
                    bg = "background-color: #0d2b0d"
                elif ml == 0:
                    bg = "background-color: #2b0d0d"
            elif col_name in ["RL", "RL edge"]:  # RL prediction columns
                if rl == 1:
                    bg = "background-color: #0d2b0d"
                elif rl == 0:
                    bg = "background-color: #2b0d0d"
            elif col_name in ["O/U", "O/U edge"]:  # O/U prediction columns
                if ou == 1:
                    bg = "background-color: #0d2b0d"
                elif ou == 0:
                    bg = "background-color: #2b0d0d"
            styles.append(bg)
        return styles

    styled = display.style.apply(_row_highlight, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)


def _render_signal_analysis(preds: pd.DataFrame) -> None:
    st.subheader("Signal Performance — Why Was the Model Right or Wrong?")
    st.caption(
        "Measures whether the model's confidence is actually predictive. A 60% model probability "
        "should win roughly 60% of the time — that's calibration. If 60% confidence games only "
        "win 48% of the time, the model is overconfident and the FIP or Pythagorean weights need tuning."
    )

    verified = preds[preds["correct"].notna()].copy()
    if verified.empty or len(verified) < 5:
        st.info("Need at least 5 verified games to analyze signal performance.")
        return

    # Pythagorean model accuracy by model confidence (model_prob buckets)
    st.markdown("#### Model Confidence vs Actual Accuracy")
    st.caption(
        "When the model assigns a high win probability, does the team actually win more often? "
        "Ideally 65% confidence → 65% win rate. Bars above the dashed line = model is "
        "underconfident (being conservative). Bars below = overconfident (too aggressive). "
        "Use this to decide whether to raise or lower the 3% edge threshold."
    )

    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
    verified["prob_bucket"] = pd.cut(
        verified["home_model_prob"], bins=bins, labels=labels, right=False
    )
    # For each bucket, is the predicted winner home or away?
    verified["picked_home"] = verified["predicted_winner"] == verified["home_team"]
    verified["home_won"] = verified["actual_winner"] == verified["home_team"]
    verified["prediction_matched_home"] = verified["picked_home"] == verified["home_won"]

    cal = verified.groupby("prob_bucket", observed=True).agg(
        games=("correct", "count"),
        model_accuracy=("correct", "mean"),
    ).reset_index()

    if not cal.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cal["prob_bucket"].astype(str),
            y=cal["model_accuracy"],
            name="Actual win rate",
            marker_color="#3498db",
            text=cal["model_accuracy"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
        ))
        # Ideal calibration line
        midpoints = [0.525, 0.575, 0.625, 0.675, 0.75]
        fig.add_trace(go.Scatter(
            x=cal["prob_bucket"].astype(str),
            y=[midpoints[i] for i in range(len(cal))],
            name="Expected (perfect calibration)",
            mode="lines+markers",
            line=dict(dash="dash", color="white"),
        ))
        fig.update_layout(
            height=300,
            title="Model Calibration: Predicted vs Actual Win Rate",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, width="stretch")


def _render_prediction_type_breakdown(preds: pd.DataFrame) -> None:
    """Win rate broken down by edge tier, bet direction, and model confidence."""
    st.subheader("Prediction Win % by Type")
    st.caption(
        "Three lenses on where the model adds value. Edge Tier shows if bigger edges actually "
        "win more — if not, the edge formula needs work. Bet Direction shows if the model is "
        "better at calling HOME wins vs AWAY wins. Model Confidence shows if high-probability "
        "picks outperform low-probability ones."
    )

    verified = preds[preds["correct"].notna()].copy()
    if verified.empty or len(verified) < 3:
        st.info("Need more verified predictions to show breakdown. Check back after more games complete.")
        return

    col_a, col_b, col_c = st.columns(3)

    # ── Edge tier ──────────────────────────────────────────────────────────
    with col_a:
        st.markdown("**By Edge Tier**")
        st.caption("Does a larger model edge actually predict more wins? 0–3% games are below the bet threshold — they're a baseline. You want 5%+ games to win noticeably more often than 3–5% games.")
        bins = pd.cut(verified["edge_pct"], bins=[-999, 3, 5, 8, 999],
                      labels=["0–3% (no bet)", "3–5%", "5–8%", "8%+"])
        tbl = (verified.groupby(bins, observed=True)["correct"]
               .agg(Win=lambda x: int(x.sum()), Games=("count"))
               .reset_index(names="Tier"))
        tbl["Win %"] = (tbl["Win"] / tbl["Games"] * 100).round(1).astype(str) + "%"
        st.dataframe(tbl, width="stretch", hide_index=True)

        if len(tbl) > 1:
            fig = px.bar(tbl, x="Tier", y=tbl["Win"] / tbl["Games"],
                         text=(tbl["Win"] / tbl["Games"]).apply(lambda x: f"{x:.0%}"),
                         color=tbl["Win"] / tbl["Games"],
                         color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                         range_color=[0.3, 0.7], labels={"y": "Win %"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                          annotation_text="50%")
            fig.update_layout(height=250, showlegend=False,
                               coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")

    # ── Bet direction ──────────────────────────────────────────────────────
    with col_b:
        st.markdown("**By Bet Direction**")
        st.caption("HOME vs AWAY win rate on model-recommended bets. If HOME bets consistently outperform AWAY, the home-field adjustment (currently +4%) may be undersized for away plays, or the model is better at reading home-field situations.")
        bets_only = verified[verified["bet_side"] != "PASS"].copy()
        if bets_only.empty:
            st.info("No directional bets yet.")
        else:
            dir_tbl = (bets_only.groupby("bet_side")["correct"]
                       .agg(Win=lambda x: int(x.sum()), Games=("count"))
                       .reset_index(names="Direction"))
            dir_tbl["Win %"] = (dir_tbl["Win"] / dir_tbl["Games"] * 100).round(1).astype(str) + "%"
            st.dataframe(dir_tbl, width="stretch", hide_index=True)

            fig = px.bar(dir_tbl, x="Direction", y=dir_tbl["Win"] / dir_tbl["Games"],
                         text=(dir_tbl["Win"] / dir_tbl["Games"]).apply(lambda x: f"{x:.0%}"),
                         color=dir_tbl["Win"] / dir_tbl["Games"],
                         color_continuous_scale=["#e74c3c", "#2ecc71"],
                         range_color=[0.3, 0.7], labels={"y": "Win %"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                          annotation_text="50%")
            fig.update_layout(height=250, showlegend=False,
                               coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")

    # ── Model confidence bucket ────────────────────────────────────────────
    with col_c:
        st.markdown("**By Model Confidence**")
        st.caption("Win rate bucketed by how strongly the model favors one team. Games near 50/50 are toss-ups regardless of model output. Games above 60% should show a meaningful win rate advantage — if they don't, the Pythagorean or FIP weights are off.")
        if "home_model_prob" not in verified.columns:
            st.info("No probability data.")
        else:
            bins2 = pd.cut(verified["home_model_prob"],
                           bins=[0, 0.52, 0.55, 0.60, 0.65, 1.0],
                           labels=["<52%", "52–55%", "55–60%", "60–65%", "65%+"])
            conf_tbl = (verified.groupby(bins2, observed=True)["correct"]
                        .agg(Win=lambda x: int(x.sum()), Games=("count"))
                        .reset_index(names="Confidence"))
            conf_tbl["Win %"] = (conf_tbl["Win"] / conf_tbl["Games"] * 100).round(1).astype(str) + "%"
            st.dataframe(conf_tbl, width="stretch", hide_index=True)

            fig = px.bar(conf_tbl, x="Confidence",
                         y=conf_tbl["Win"] / conf_tbl["Games"],
                         text=(conf_tbl["Win"] / conf_tbl["Games"]).apply(lambda x: f"{x:.0%}"),
                         color=conf_tbl["Win"] / conf_tbl["Games"],
                         color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                         range_color=[0.3, 0.7], labels={"y": "Win %"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                          annotation_text="50%")
            fig.update_layout(height=250, showlegend=False,
                               coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")


def _render_underdog_analysis(preds: pd.DataFrame) -> None:
    """Win % for underdog teams (positive ML / implied prob < 50%)."""
    st.subheader("Underdog Win Rate")
    st.caption(
        "How often does the team with worse odds (positive moneyline, implied prob < 50%) "
        "actually win? Useful for calibrating how much to trust the model against public consensus."
    )

    verified = preds[preds["correct"].notna()].copy()
    if verified.empty or len(verified) < 3:
        st.info("Need more verified games to compute underdog stats. Check back after more results come in.")
        return

    # Determine underdog from market implied probability stored in predictions
    # home_implied_prob is vig-removed; < 0.5 means home team is the underdog
    if "home_implied_prob" not in verified.columns:
        st.info("Implied probability data not stored for these predictions.")
        return

    # Home underdog games
    home_dog = verified[verified["home_implied_prob"] < 0.50].copy()
    away_dog = verified[verified["home_implied_prob"] >= 0.50].copy()

    # For home_dog: did the home team (the underdog) win?
    if "actual_winner" in verified.columns and "home_team" in verified.columns:
        home_dog["dog_won"] = home_dog["actual_winner"] == home_dog["home_team"]
        away_dog["dog_won"] = away_dog["actual_winner"] == away_dog["away_team"]
        all_dog = pd.concat([home_dog, away_dog])
    else:
        st.info("Winner data not available for underdog analysis.")
        return

    total_dog_games = len(all_dog)
    if total_dog_games == 0:
        st.info("No underdog games found in verified predictions.")
        return

    dog_wins = int(all_dog["dog_won"].sum())
    dog_win_pct = dog_wins / total_dog_games

    # Model-recommended bets on underdogs
    if "bet_side" in all_dog.columns:
        model_backed_dog = all_dog[
            ((all_dog["home_implied_prob"] < 0.50) & (all_dog["bet_side"] == "HOME")) |
            ((all_dog["home_implied_prob"] >= 0.50) & (all_dog["bet_side"] == "AWAY"))
        ].copy()
        model_dog_win_pct = (
            model_backed_dog["dog_won"].mean() if not model_backed_dog.empty else None
        )
    else:
        model_backed_dog = pd.DataFrame()
        model_dog_win_pct = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Underdog Games", total_dog_games)
    c2.metric("Underdog Actual Win %", f"{dog_win_pct:.1%}",
              help="How often the + ML team wins regardless of model recommendation")
    c3.metric("Model-Backed Underdogs", len(model_backed_dog))
    c4.metric(
        "Model Dog Win %",
        f"{model_dog_win_pct:.1%}" if model_dog_win_pct is not None else "N/A",
        help="Win % when model specifically recommended an underdog",
    )

    # Underdog win rate by implied probability bucket
    all_dog["dog_implied"] = np.where(
        all_dog["home_implied_prob"] < 0.50,
        all_dog["home_implied_prob"],
        1 - all_dog["home_implied_prob"],
    )
    all_dog["prob_bucket"] = pd.cut(
        all_dog["dog_implied"],
        bins=[0, 0.35, 0.40, 0.45, 0.50],
        labels=["< 35% (big dog)", "35–40%", "40–45%", "45–50% (slight dog)"],
    )
    bucket_tbl = (all_dog.groupby("prob_bucket", observed=True)["dog_won"]
                  .agg(Wins=lambda x: int(x.sum()), Games="count")
                  .reset_index(names="Market Prob"))
    bucket_tbl["Win %"] = (bucket_tbl["Wins"] / bucket_tbl["Games"] * 100).round(1).astype(str) + "%"

    if len(bucket_tbl) > 1:
        col_tbl, col_chart = st.columns([1, 2])
        with col_tbl:
            st.dataframe(bucket_tbl, width="stretch", hide_index=True)
        with col_chart:
            fig = px.bar(
                bucket_tbl, x="Market Prob",
                y=bucket_tbl["Wins"] / bucket_tbl["Games"],
                text=(bucket_tbl["Wins"] / bucket_tbl["Games"]).apply(lambda x: f"{x:.0%}"),
                color=bucket_tbl["Wins"] / bucket_tbl["Games"],
                color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                range_color=[0.3, 0.7],
                labels={"y": "Underdog Win %"},
                title="Underdog Win % by Market Implied Prob",
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="50%")
            fig.update_layout(height=250, showlegend=False, coloraxis_showscale=False,
                              margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch")
    else:
        st.dataframe(bucket_tbl, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Quarter definitions for season filtering
# ---------------------------------------------------------------------------

_QUARTERS = {
    "Full Season": (None, None),
    "Q1 — April":         ("{y}-04-01", "{y}-04-30"),
    "Q2 — May/June":      ("{y}-05-01", "{y}-06-30"),
    "Q3 — July/August":   ("{y}-07-01", "{y}-08-31"),
    "Q4 — Sept/Oct":      ("{y}-09-01", "{y}-10-15"),
}

_STATS_META = {
    "Hits / Game":          ("hits_pg",         "Hits per Game"),
    "HR / Game":            ("hr_pg",            "Home Runs per Game"),
    "Strikeouts / Game":    ("k_pg",             "Batter Strikeouts per Game"),
    "Runs Scored / Game":   ("runs_pg",          "Runs Scored per Game"),
    "Run Differential / G": ("run_diff_pg",      "Run Differential per Game"),
    "Win %":                ("win_pct",          "Win Percentage"),
}


def _render_historical_team_stats() -> None:
    """Interactive bar chart — team stats vs league median/max/min, 2022–current."""
    st.subheader("Historical Team Stats — League Comparison")
    st.caption(
        "Compare any team's offensive and pitching production against the rest of the league. "
        "Select a stat, season, and time period. The highlighted team (orange bar) lets you track "
        "one team across different stats without losing context. League median and average lines "
        "show where the cutoff between above-average and below-average teams sits."
    )

    current_year = datetime.date.today().year
    available_years = list(range(2022, current_year + 1))

    # ── Controls ──────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1:
        stat_label = st.selectbox("Stat", list(_STATS_META.keys()), index=0)
    with fc2:
        year_options = ["Avg Last 2 Years"] + [str(y) for y in reversed(available_years)]
        year_choice = st.selectbox("Season", year_options, index=0)
    with fc3:
        quarter_choice = st.selectbox("Period", list(_QUARTERS.keys()), index=0,
                                      disabled=(year_choice == "Avg Last 2 Years"))
    with fc4:
        highlight_team = st.selectbox("Highlight Team", ["None"] + sorted([
            "Arizona Diamondbacks", "Athletics", "Atlanta Braves",
            "Baltimore Orioles", "Boston Red Sox", "Chicago Cubs",
            "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians",
            "Colorado Rockies", "Detroit Tigers", "Houston Astros",
            "Kansas City Royals", "Los Angeles Angels", "Los Angeles Dodgers",
            "Miami Marlins", "Milwaukee Brewers", "Minnesota Twins",
            "New York Mets", "New York Yankees", "Philadelphia Phillies",
            "Pittsburgh Pirates", "San Diego Padres", "Seattle Mariners",
            "San Francisco Giants", "St. Louis Cardinals", "Tampa Bay Rays",
            "Texas Rangers", "Toronto Blue Jays", "Washington Nationals",
        ]))

    stat_col, stat_title = _STATS_META[stat_label]

    # ── Fetch data ────────────────────────────────────────────────────────
    with st.spinner("Loading historical stats..."):
        try:
            if year_choice == "Avg Last 2 Years":
                y1 = current_year - 1
                y2 = current_year - 2
                df1 = get_historical_team_stats(y1)
                df2 = get_historical_team_stats(y2)
                if df1.empty or df2.empty:
                    st.warning("Not enough historical data available.")
                    return
                merged = df1.merge(df2, on="team", suffixes=("_y1", "_y2"))
                df = df1[["team"]].copy()
                for col in [stat_col, "wins", "losses", "win_pct", "hits_pg",
                             "hr_pg", "k_pg", "runs_pg", "run_diff_pg"]:
                    if f"{col}_y1" in merged.columns and f"{col}_y2" in merged.columns:
                        df[col] = ((merged[f"{col}_y1"] + merged[f"{col}_y2"]) / 2).round(3)
                quarter_label = f"Avg {y2}–{y1}"
            else:
                y = int(year_choice)
                q_start_tmpl, q_end_tmpl = _QUARTERS[quarter_choice]
                q_start = q_start_tmpl.format(y=y) if q_start_tmpl else None
                q_end   = q_end_tmpl.format(y=y)   if q_end_tmpl   else None
                df = get_historical_team_stats(y, q_start, q_end)
                quarter_label = f"{year_choice} · {quarter_choice}"

            if df.empty or stat_col not in df.columns:
                st.warning("No data available for the selected period.")
                return
        except Exception as e:
            st.error(f"Failed to load historical stats: {e}")
            return

    df = df.sort_values(stat_col, ascending=False).reset_index(drop=True)

    # ── Compute league benchmarks ─────────────────────────────────────────
    vals = df[stat_col].dropna()
    league_median = float(vals.median())
    league_max    = float(vals.max())
    league_min    = float(vals.min())
    league_avg    = float(vals.mean())

    # ── Build chart ───────────────────────────────────────────────────────
    colors = []
    for team in df["team"]:
        if highlight_team != "None" and team == highlight_team:
            colors.append("#f39c12")
        else:
            colors.append("#3498db")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["team"],
        y=df[stat_col],
        marker_color=colors,
        name=stat_label,
        text=df[stat_col].round(2),
        textposition="outside",
        hovertemplate="%{x}<br>" + stat_label + ": %{y:.3f}<extra></extra>",
    ))

    # League reference lines
    fig.add_hline(y=league_median, line_dash="dash", line_color="#2ecc71", line_width=1.5,
                  annotation_text=f"Median {league_median:.3f}", annotation_position="top right",
                  annotation_font_color="#2ecc71")
    fig.add_hline(y=league_avg, line_dash="dot", line_color="#3498db", line_width=1,
                  annotation_text=f"Avg {league_avg:.3f}", annotation_position="bottom right",
                  annotation_font_color="#3498db")
    fig.add_hline(y=league_max, line_dash="dash", line_color="#aaa", line_width=1,
                  annotation_text=f"Max {league_max:.3f}", annotation_position="top left",
                  annotation_font_color="#aaa")
    fig.add_hline(y=league_min, line_dash="dash", line_color="#aaa", line_width=1,
                  annotation_text=f"Min {league_min:.3f}", annotation_position="bottom left",
                  annotation_font_color="#aaa")

    fig.update_layout(
        title=f"{stat_title} — {quarter_label}",
        xaxis_tickangle=-45,
        xaxis_title="",
        yaxis_title=stat_label,
        height=500,
        showlegend=False,
        margin=dict(b=120),
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f",
        font=dict(color="#e0e0e0"),
    )

    st.plotly_chart(fig, width="stretch")

    # ── League summary metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("League Median", f"{league_median:.3f}")
    m2.metric("League Avg",    f"{league_avg:.3f}")
    m3.metric("League Max",    f"{league_max:.3f}  ({df.iloc[0]['team'].split()[-1]})")
    m4.metric("League Min",    f"{league_min:.3f}  ({df.iloc[-1]['team'].split()[-1]})")

    # ── Full data table ───────────────────────────────────────────────────
    with st.expander("Full data table"):
        display = df[["team", "wins", "losses", "win_pct", "hits_pg",
                       "hr_pg", "k_pg", "runs_pg", "run_diff_pg"]].copy()
        display.columns = ["Team", "W", "L", "Win%", "H/G", "HR/G", "K/G", "R/G", "RDiff/G"]
        display["Win%"] = display["Win%"].apply(lambda x: f"{x:.3f}")
        st.dataframe(display, width="stretch", hide_index=True)

    # ── Current-season run differential leaderboard ───────────────────────
    st.divider()
    st.subheader(f"Current Season Run Differential — {current_year}")
    st.caption(
        "Run differential per game is one of the strongest predictors of future win rate "
        "(more reliable than current W-L). Teams above the league average line tend to "
        "outperform their record going forward."
    )
    with st.spinner("Loading current season run differential..."):
        try:
            rd_df = get_historical_team_stats(current_year)
        except Exception:
            rd_df = pd.DataFrame()

    if rd_df.empty or "run_diff_pg" not in rd_df.columns:
        st.info("Current season run differential not available yet.")
    else:
        rd_df = rd_df.sort_values("run_diff_pg", ascending=False).reset_index(drop=True)
        league_rd_avg = float(rd_df["run_diff_pg"].mean())
        league_rd_med = float(rd_df["run_diff_pg"].median())

        rd_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in rd_df["run_diff_pg"]]

        fig_rd = go.Figure()
        fig_rd.add_trace(go.Bar(
            x=rd_df["team"],
            y=rd_df["run_diff_pg"],
            marker_color=rd_colors,
            name="Run Diff / G",
            text=rd_df["run_diff_pg"].round(2),
            textposition="outside",
            hovertemplate="%{x}<br>Run Diff/G: %{y:.2f}<extra></extra>",
        ))
        fig_rd.add_hline(y=league_rd_avg, line_dash="dot", line_color="#3498db", line_width=1.5,
                         annotation_text=f"League Avg {league_rd_avg:.2f}",
                         annotation_position="top right", annotation_font_color="#3498db")
        fig_rd.add_hline(y=0, line_dash="solid", line_color="#555", line_width=1)
        fig_rd.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="",
            yaxis_title="Run Differential per Game",
            height=450,
            showlegend=False,
            margin=dict(b=120),
            plot_bgcolor="#0f0f0f",
            paper_bgcolor="#0f0f0f",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig_rd, width="stretch")

        rd_m1, rd_m2, rd_m3, rd_m4 = st.columns(4)
        rd_m1.metric("League Avg RDiff/G", f"{league_rd_avg:+.2f}")
        rd_m2.metric("League Median RDiff/G", f"{league_rd_med:+.2f}")
        best_row = rd_df.iloc[0]
        worst_row = rd_df.iloc[-1]
        rd_m3.metric(f"Best ({best_row['team'].split()[-1]})", f"{best_row['run_diff_pg']:+.2f}")
        rd_m4.metric(f"Worst ({worst_row['team'].split()[-1]})", f"{worst_row['run_diff_pg']:+.2f}")


def _build_type_stats(log: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    Build a win-rate summary table by bet type.

    ML row:  verified predictions where bet_side != PASS, supplemented by bet log ML entries.
    RL row:  verified predictions where rl_side is set + rl_correct is not null,
             supplemented by bet log RL entries.
    O/U row: verified predictions where total_direction is set + total_correct is not null,
             supplemented by bet log O/U entries.

    Returns columns: bet_type, Bets, Wins, Win %, Source, Staked, PnL, ROI %
    """
    # Accumulate {bet_type: {Bets, Wins, Staked, PnL, sources: set}}
    acc: dict[str, dict] = {}

    def _add(bt: str, n: int, wins: int, source: str, staked: float = 0.0, pnl: float = 0.0) -> None:
        if bt not in acc:
            acc[bt] = {"Bets": 0, "Wins": 0, "Staked": 0.0, "PnL": 0.0, "sources": set()}
        acc[bt]["Bets"]   += n
        acc[bt]["Wins"]   += wins
        acc[bt]["Staked"] += staked
        acc[bt]["PnL"]    += pnl
        acc[bt]["sources"].add(source)

    verified = preds[preds["correct"].notna()].copy() if not preds.empty else pd.DataFrame()

    # ── ML from predictions ───────────────────────────────────────────────────
    if not verified.empty:
        ml_preds = verified[
            verified["bet_side"].notna() & (verified["bet_side"] != "PASS")
        ]
        if not ml_preds.empty:
            _add("ML", len(ml_preds), int(ml_preds["correct"].sum()), "Predictions")

    # ── RL from predictions ───────────────────────────────────────────────────
    if not verified.empty and "rl_side" in verified.columns and "rl_correct" in verified.columns:
        rl_preds = verified[
            verified["rl_side"].notna() &
            verified["rl_correct"].notna() &
            (verified["rl_side"] != "PASS")
        ]
        if not rl_preds.empty:
            _add("RL", len(rl_preds), int(rl_preds["rl_correct"].sum()), "Predictions")

    # ── O/U from predictions ──────────────────────────────────────────────────
    if not verified.empty and "total_direction" in verified.columns and "total_correct" in verified.columns:
        ou_preds = verified[
            verified["total_direction"].notna() &
            verified["total_correct"].notna()
        ]
        if not ou_preds.empty:
            _add("O/U", len(ou_preds), int(ou_preds["total_correct"].sum()), "Predictions")

    # ── ML / RL / O/U from bet log (supplement predictions) ──────────────────
    settled = (
        log[log["outcome"].isin(["Win", "Loss"])].copy()
        if not log.empty else pd.DataFrame()
    )
    line_settled = (
        settled[settled["bet_type"].isin(["ML", "RL", "O/U"])].copy()
        if not settled.empty else pd.DataFrame()
    )
    if not line_settled.empty:
        for bt, grp in line_settled.groupby("bet_type"):
            n = len(grp)
            wins = int((grp["outcome"] == "Win").sum())
            _add(bt, n, wins, "Bet Log",
                 staked=float(grp["stake"].sum()),
                 pnl=float(grp["pnl"].sum()))

    if not acc:
        return pd.DataFrame()

    rows = []
    for bt, d in acc.items():
        source_str = " + ".join(sorted(d["sources"]))
        rows.append({
            "bet_type": bt,
            "Bets":     d["Bets"],
            "Wins":     d["Wins"],
            "Source":   source_str,
            "Staked":   d["Staked"],
            "PnL":      d["PnL"],
        })

    df = pd.DataFrame(rows)
    # Enforce display order
    order = {"ML": 0, "RL": 1, "O/U": 2}
    df["_order"] = df["bet_type"].map(order).fillna(99)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    df["Win %"] = df["Wins"] / df["Bets"]
    df["ROI %"] = (df["PnL"] / df["Staked"] * 100).where(df["Staked"] > 0, other=0.0)
    return df


def _render_bet_type_analysis(log: pd.DataFrame, preds: pd.DataFrame) -> None:
    """Win rate and ROI broken down by bet type: ML, Run Line, Over/Under."""
    st.subheader("Bet Type Success — ML vs Run Line vs Over/Under")
    st.caption(
        "Compares historical win rate across the three main bet types. "
        "ML, RL, and O/U accuracy are derived from verified model predictions (automatically tracked). "
        "Bet log entries supplement or override prediction data when available. "
        "The top-performing type (min 3 bets) is used to sort and highlight the Games page edge table."
    )

    type_stats = _build_type_stats(log, preds)

    if type_stats.empty:
        st.info(
            "No verified predictions or settled bets yet. "
            "Predictions (including RL and O/U) are saved automatically when you load the Games page. "
            "Games are verified once final scores are available."
        )
        return

    # Identify best type (min 3 bets) using get_best_bet_type for bet log,
    # but fall back to prediction-based ML if that has the most data
    best_type = get_best_bet_type(min_bets=3)
    if best_type is None:
        # No bet log data — pick the type with the highest win rate from combined stats
        eligible = type_stats[type_stats["Bets"] >= 3]
        if not eligible.empty:
            best_type = str(eligible.loc[eligible["Win %"].idxmax(), "bet_type"])

    if best_type:
        best_row = type_stats[type_stats["bet_type"] == best_type]
        if not best_row.empty:
            r = best_row.iloc[0]
            st.success(
                f"Top-performing bet type: **{best_type}** — "
                f"{r['Win %']:.1%} win rate over {int(r['Bets'])} bets "
                f"({r['Source']}). "
                f"The Games page edge table is sorted by **{best_type} Edge%** and that column is highlighted teal."
            )
    else:
        st.info("Need at least 3 bets per type to identify the top-performing category.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    cols = st.columns(len(type_stats))
    for i, (_, row) in enumerate(type_stats.iterrows()):
        is_best = row["bet_type"] == best_type
        label = f"{'🏆 ' if is_best else ''}{row['bet_type']}"
        delta_str = f"{row['ROI %']:+.1f}% ROI  ·  {int(row['Bets'])} bets" if row["Staked"] > 0 else f"{int(row['Bets'])} bets"
        cols[i].metric(label, f"{row['Win %']:.1%}", delta=delta_str)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig = go.Figure()
    colors = [
        "#1abc9c" if r["bet_type"] == best_type
        else "#2ecc71" if r["Win %"] >= 0.5
        else "#e74c3c"
        for _, r in type_stats.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=type_stats["bet_type"],
        y=type_stats["Win %"],
        marker_color=colors,
        text=type_stats["Win %"].apply(lambda x: f"{x:.1%}"),
        textposition="outside",
        name="Win Rate",
        customdata=type_stats["Source"],
        hovertemplate="%{x}<br>Win Rate: %{y:.1%}<br>Source: %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="50%")
    fig.update_layout(
        height=280,
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Win Rate"),
        xaxis_title="Bet Type",
        title="Win Rate by Bet Type",
        showlegend=False,
        margin=dict(t=30, b=10),
    )
    c_chart, c_tbl = st.columns([2, 1])
    with c_chart:
        st.plotly_chart(fig, width="stretch")
    with c_tbl:
        display = type_stats[["bet_type", "Bets", "Wins", "Win %", "Source"]].copy()
        display["Win %"] = display["Win %"].apply(lambda x: f"{x:.1%}")
        has_roi = type_stats["Staked"].gt(0).any()
        if has_roi:
            display["ROI %"] = type_stats["ROI %"].apply(lambda x: f"{x:+.1f}%")
        display = display.rename(columns={"bet_type": "Type"})
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption("ML, RL, and O/U win rates use verified model predictions. Logged bets supplement where available.")


def _render_bet_log_analysis() -> None:
    st.subheader("Bet Log Analysis")
    st.caption(
        "Tracks actual bets you've logged with outcomes. Win rate alone doesn't tell the whole "
        "story — a 52% win rate at -110 juice breaks even, while a 48% win rate at +150 is "
        "profitable. ROI is the number that counts. Log bets from the Games page bet slip."
    )

    log = load_bets()
    settled = log[log["outcome"].isin(["Win", "Loss"])].copy() if not log.empty else pd.DataFrame()

    if settled.empty:
        st.info("No settled bets yet. Log bets with outcomes in the Bet Log tab to see ROI analysis.")
        return

    total_staked = settled["stake"].sum()
    total_pnl = settled["pnl"].sum()
    win_rate = (settled["outcome"] == "Win").mean()
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Settled Bets", len(settled))
    c2.metric("Win Rate", f"{win_rate:.1%}")
    c3.metric("Total P&L", f"${total_pnl:+,.2f}")
    c4.metric("ROI", f"{roi:+.1f}%")

    # ROI by signal type
    if "signal_type" in settled.columns and settled["signal_type"].notna().any():
        st.markdown("#### ROI by Signal Type")
        st.caption("Which model signals actually produce profit? A signal with high win rate but negative ROI means you're winning on the wrong side of the juice. A signal with lower win rate but positive ROI means you're getting paid well when you win.")
        sig_df = compute_signal_roi(settled.dropna(subset=["signal_type"]))
        if not sig_df.empty:
            st.dataframe(
                sig_df.rename(columns={
                    "signal_type": "Signal", "bets": "Bets", "win_rate": "Win Rate",
                    "total_staked": "Staked ($)", "total_pnl": "P&L ($)", "roi_pct": "ROI%",
                }),
                width="stretch", hide_index=True,
            )

    # Calibration
    with_prob = settled.dropna(subset=["model_prob"])
    if not with_prob.empty:
        st.markdown("#### Model Calibration (Bet Log)")
        st.caption("Compares the model's stated win probability against what actually happened on bets you logged. A well-calibrated model shows actual win rate ≈ expected win rate in each bucket. Large gaps mean the model probabilities need adjustment.")
        cal_df = compute_calibration_table(with_prob)
        if not cal_df.empty:
            st.dataframe(
                cal_df.rename(columns={
                    "bucket": "Prob Bucket", "bets": "Bets",
                    "expected_win_rate": "Expected Win%", "actual_win_rate": "Actual Win%",
                    "calibration_error": "Error", "total_staked": "Staked ($)",
                    "total_pnl": "P&L ($)", "roi_pct": "ROI%",
                }),
                width="stretch", hide_index=True,
            )
            st.markdown("#### Tuning Suggestions")
            st.caption("Automated recommendations based on calibration error. These are starting points — apply judgment before changing any model constants.")
            for s in recommend_threshold_adjustments(cal_df):
                st.write(f"- {s}")


def render() -> None:
    st.header("📊 Analysis — Model Performance & Prediction Audit")
    today = datetime.date.today()

    # Load and auto-verify predictions
    with st.spinner("Loading predictions and verifying results..."):
        preds = load_predictions(days=60)
        if not preds.empty:
            preds = verify_predictions(preds)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Prediction Accuracy", "Prediction Type Win%", "Bet Type Comparison",
        "Historical Team Stats", "Signal Analysis", "Bet Log ROI",
    ])

    with tab1:
        st.caption(
            "Tracks how often the model picks the correct game winner. "
            "Verified automatically for completed games. Use this to understand whether the "
            "overall model is adding value — and whether bet-only games outperform all games."
        )
        _render_prediction_accuracy(preds)
        st.divider()
        _render_underdog_analysis(preds)

    with tab2:
        st.caption(
            "Breaks down win rate by edge tier, bet direction (HOME vs AWAY), and model confidence. "
            "Reveals where the model is adding the most value — and where it may need recalibration. "
            "A well-calibrated model should show higher win % at higher edge tiers."
        )
        _render_prediction_type_breakdown(preds)

    with tab3:
        st.caption(
            "Compares win rate and ROI across ML, Run Line, and Over/Under bets. "
            "The top-performing type is automatically used to sort the Games page edge table "
            "and highlight that column teal."
        )
        log = load_bets()
        _render_bet_type_analysis(log, preds)

    with tab4:
        st.caption(
            "Team-level batting, pitching, and run production stats across seasons and date ranges. "
            "Run differential per game is the strongest single predictor of future team win rate — "
            "more reliable than current W-L record, especially in the first half of the season."
        )
        _render_historical_team_stats()

    with tab5:
        st.caption(
            "Step-by-step breakdown of every formula in the model — what it calculates, why the "
            "stat matters, and what real-world market inefficiency it targets. Read this to "
            "understand why a game was recommended or passed."
        )
        _render_model_explainer()
        st.divider()
        _render_signal_analysis(preds)

    with tab6:
        st.caption(
            "Actual P&L tracking on logged bets. Win rate, ROI by signal type, and model "
            "calibration versus real outcomes. This is the ground truth — did the edge translate "
            "to profit? Use ROI by signal type to decide which signals to trust more or less."
        )
        _render_bet_log_analysis()
