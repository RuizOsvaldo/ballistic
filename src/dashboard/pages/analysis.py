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

from src.data.predictions_db import load_predictions
from src.data.game_results import verify_predictions
from src.data.bet_log_db import load_bets
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
    st.caption("A plain-language overview of the ML process behind every prediction.")

    with st.expander("1. Pythagorean Win Expectation (Team Quality Baseline)", expanded=False):
        st.markdown("""
**Formula:** `W% = RS² / (RS² + RA²)`

This is the foundation. A team's actual win-loss record contains significant noise from
close games, bullpen luck, and sequencing. Runs scored and runs allowed are a more stable
measure of true team quality over a full season.

**How it's used:** Each team's Pythagorean W% becomes the base win probability going into
every game. Teams whose actual W% is 5%+ above their Pythagorean W% are "overperforming"
— the model fades them. Teams 5%+ below are "underperforming" — the model backs them.

**Data source:** FanGraphs team batting (runs scored) and team pitching (runs allowed) via pybaseball.
        """)

    with st.expander("2. Starting Pitcher FIP Adjustment", expanded=False):
        st.markdown("""
**Formula:** `pitcher_adj = (league_avg_FIP − starter_FIP) × 0.03`

ERA is polluted by defense and luck. FIP (Fielding Independent Pitching) strips ERA down
to only what a pitcher controls: home runs, walks, and strikeouts. A pitcher with FIP 3.0
vs. league average 4.0 adds `+3%` win probability to his team.

**Adjustment range:** Typically ±3–6% depending on starter quality.

**Data source:** FanGraphs pitcher leaderboard via pybaseball.
        """)

    with st.expander("3. BABIP Regression Signal", expanded=False):
        st.markdown("""
**BABIP** (Batting Average on Balls in Play) is largely random at the pitcher level.
League average is .300. Extreme BABIP values always normalize:

- Pitcher BABIP > .320 → has been allowing too many hits to fall in → ERA will rise → fade
- Pitcher BABIP < .275 → has been artificially suppressing hits → ERA will drop → back

**Used for:** Escalating signal confidence (Low / Medium / High) and AI reasoning context.
        """)

    with st.expander("4. Edge vs. Vegas Implied Probability", expanded=False):
        st.markdown("""
Once the model computes a win probability, it's compared to the vig-removed implied
probability from the best available moneyline across all tracked sportsbooks.

```
vig_removal: home_prob / (home_prob + away_prob)
edge = model_prob − market_implied_prob
```

**Bet threshold:** Edge ≥ 3% triggers a recommendation. Kelly criterion (half-Kelly,
capped at 5% of bankroll) sizes the stake.
        """)

    with st.expander("5. Half-Kelly Bet Sizing", expanded=False):
        st.markdown("""
```
kelly_full = (b × p − q) / b      # b = decimal odds − 1, p = model prob
kelly_stake = kelly_full × 0.5    # half-Kelly to reduce variance
stake = min(kelly_stake, 0.05)    # cap at 5% of bankroll
```

Half-Kelly is used instead of full Kelly to reduce bankroll swings while still
growing at roughly 75% of the optimal long-run rate. The 5% cap prevents
any single game from being catastrophic.
        """)


def _render_prediction_accuracy(preds: pd.DataFrame) -> None:
    st.subheader("Prediction Accuracy — Verified Games")

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
        st.plotly_chart(fig, use_container_width=True)

    # Accuracy by edge tier
    if "edge_pct" in verified.columns and verified["edge_pct"].notna().any():
        st.markdown("#### Does Higher Edge = More Wins?")
        bins = pd.cut(verified["edge_pct"], bins=[0, 3, 5, 8, 20],
                      labels=["0-3%", "3-5%", "5-8%", "8%+"])
        edge_acc = verified.groupby(bins, observed=True)["correct"].agg(["mean", "count"])
        edge_acc.columns = ["Accuracy", "Games"]
        edge_acc.index.name = "Edge Tier"
        edge_acc["Accuracy"] = edge_acc["Accuracy"].apply(lambda x: f"{x:.1%}")
        st.dataframe(edge_acc, use_container_width=True)

    # Full prediction history
    st.markdown("#### Prediction History")
    display = verified[["prediction_date", "away_team", "home_team",
                         "predicted_winner", "actual_winner", "correct",
                         "home_model_prob", "edge_pct", "bet_side"]].copy()
    display["correct"] = display["correct"].map({1: "✅", 0: "❌"})
    display["home_model_prob"] = (display["home_model_prob"] * 100).round(1).astype(str) + "%"
    display["edge_pct"] = display["edge_pct"].round(1).astype(str) + "%"
    display = display.rename(columns={
        "prediction_date": "Date", "away_team": "Away", "home_team": "Home",
        "predicted_winner": "Predicted", "actual_winner": "Actual",
        "correct": "Result", "home_model_prob": "Home Prob",
        "edge_pct": "Edge", "bet_side": "Bet Side",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_signal_analysis(preds: pd.DataFrame) -> None:
    st.subheader("Signal Performance — Why Was the Model Right or Wrong?")

    verified = preds[preds["correct"].notna()].copy()
    if verified.empty or len(verified) < 5:
        st.info("Need at least 5 verified games to analyze signal performance.")
        return

    # Pythagorean model accuracy by model confidence (model_prob buckets)
    st.markdown("#### Model Confidence vs Actual Accuracy")
    st.caption(
        "When the model is very confident (high home_model_prob), does it actually win more? "
        "Ideally, 65% model confidence games should win ~65% of the time."
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
        st.plotly_chart(fig, use_container_width=True)


def _render_bet_log_analysis() -> None:
    st.subheader("Bet Log Analysis")

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
        sig_df = compute_signal_roi(settled.dropna(subset=["signal_type"]))
        if not sig_df.empty:
            st.dataframe(
                sig_df.rename(columns={
                    "signal_type": "Signal", "bets": "Bets", "win_rate": "Win Rate",
                    "total_staked": "Staked ($)", "total_pnl": "P&L ($)", "roi_pct": "ROI%",
                }),
                use_container_width=True, hide_index=True,
            )

    # Calibration
    with_prob = settled.dropna(subset=["model_prob"])
    if not with_prob.empty:
        st.markdown("#### Model Calibration (Bet Log)")
        cal_df = compute_calibration_table(with_prob)
        if not cal_df.empty:
            st.dataframe(
                cal_df.rename(columns={
                    "bucket": "Prob Bucket", "bets": "Bets",
                    "expected_win_rate": "Expected Win%", "actual_win_rate": "Actual Win%",
                    "calibration_error": "Error", "total_staked": "Staked ($)",
                    "total_pnl": "P&L ($)", "roi_pct": "ROI%",
                }),
                use_container_width=True, hide_index=True,
            )
            st.markdown("#### Tuning Suggestions")
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

    tab1, tab2, tab3 = st.tabs(["Prediction Accuracy", "Signal Analysis", "Bet Log ROI"])

    with tab1:
        _render_prediction_accuracy(preds)

    with tab2:
        _render_model_explainer()
        st.divider()
        _render_signal_analysis(preds)

    with tab3:
        _render_bet_log_analysis()
