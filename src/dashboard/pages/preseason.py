"""Preseason win total projections page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.baseball_stats import get_team_stats
from src.models.preseason import compute_preseason_projections, MIN_EDGE_WINS
from src.shared.groq_agent import analyze_preseason_projection


CURRENT_SEASON = datetime.datetime.now().year
PRIOR_SEASON = CURRENT_SEASON - 1


@st.cache_data(ttl=3600 * 24, show_spinner="Loading prior season stats...")
def _load_prior_stats(season: int) -> pd.DataFrame:
    try:
        return get_team_stats(season)
    except Exception as e:
        st.error(f"Could not load prior season data: {e}")
        return pd.DataFrame()


def render() -> None:
    st.header("Preseason Win Total Projections")
    st.caption(
        f"Uses {PRIOR_SEASON} run differential and Pythagorean W% to project {CURRENT_SEASON} win totals. "
        "Compare to Vegas O/U lines to find season-long bet value."
    )

    prior_stats = _load_prior_stats(PRIOR_SEASON)

    if prior_stats.empty:
        st.warning("Could not load prior season data.")
        return

    # ---- Vegas lines input ----
    st.subheader("Vegas Win Total Lines")
    st.caption("Enter each team's Vegas over/under win total. Leave blank to see projections only.")

    with st.expander("Enter Vegas win total lines", expanded=True):
        teams = sorted(prior_stats["team"].dropna().unique().tolist())

        col1, col2, col3 = st.columns(3)
        vegas_entries = {}

        for i, team in enumerate(teams):
            col = [col1, col2, col3][i % 3]
            with col:
                val = st.number_input(
                    team,
                    min_value=40.0,
                    max_value=120.0,
                    value=81.0,
                    step=0.5,
                    key=f"vegas_{team}",
                )
                vegas_entries[team] = val

    # Build vegas_lines DataFrame
    vegas_df = pd.DataFrame([
        {"team": team, "vegas_total": total}
        for team, total in vegas_entries.items()
    ])

    # ---- Compute projections ----
    projections = compute_preseason_projections(prior_stats, vegas_df)

    # ---- Summary metrics ----
    bets = projections[projections["bet_direction"] != "PASS"]
    over_bets = bets[bets["bet_direction"] == "OVER"]
    under_bets = bets[bets["bet_direction"] == "UNDER"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Teams", len(projections))
    c2.metric("OVER Bets", len(over_bets))
    c3.metric("UNDER Bets", len(under_bets))

    # ---- Projection vs Vegas chart ----
    if "vegas_total" in projections.columns:
        st.subheader("Projected Wins vs Vegas Lines")
        chart_df = projections.dropna(subset=["vegas_total"]).copy()
        chart_df["gap"] = chart_df["projected_wins"] - chart_df["vegas_total"]

        fig = px.bar(
            chart_df.sort_values("gap"),
            x="team",
            y="gap",
            color="gap",
            color_continuous_scale=["#c0392b", "#95a5a6", "#2ecc71"],
            color_continuous_midpoint=0,
            labels={"gap": "Projection vs Vegas (wins)", "team": "Team"},
            title=f"Model Projection vs Vegas Line (positive = project more wins than Vegas)",
        )
        fig.add_hline(y=MIN_EDGE_WINS, line_dash="dash", line_color="green",
                      annotation_text=f"+{MIN_EDGE_WINS} OVER threshold")
        fig.add_hline(y=-MIN_EDGE_WINS, line_dash="dash", line_color="red",
                      annotation_text=f"-{MIN_EDGE_WINS} UNDER threshold")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # ---- Projections table ----
    st.subheader("Full Projection Table")

    signal_filter = st.selectbox(
        "Filter by bet signal",
        ["All", "OVER bets only", "UNDER bets only", "All bets (exclude PASS)"],
    )

    display = projections.copy()
    if signal_filter == "OVER bets only":
        display = display[display["bet_direction"] == "OVER"]
    elif signal_filter == "UNDER bets only":
        display = display[display["bet_direction"] == "UNDER"]
    elif signal_filter == "All bets (exclude PASS)":
        display = display[display["bet_direction"] != "PASS"]

    display_cols = {
        "team": "Team",
        "prior_wins": f"{PRIOR_SEASON} Wins",
        "prior_pyth_wins": "Pyth Wins",
        "prior_run_diff": "Run Diff",
        "projected_wins": "Projected",
        "vegas_total": "Vegas O/U",
        "edge_wins": "Edge (wins)",
        "bet_direction": "Bet",
        "signal_strength": "Confidence",
    }
    available = {k: v for k, v in display_cols.items() if k in display.columns}
    display_df = display[list(available.keys())].rename(columns=available)

    def _color_bet(val):
        if val == "OVER":
            return "background-color: #2ecc71; color: black"
        if val == "UNDER":
            return "background-color: #c0392b; color: white"
        return ""

    def _color_edge(val):
        try:
            v = float(val)
            if v >= MIN_EDGE_WINS:
                return "color: #2ecc71; font-weight: bold"
            if v <= -MIN_EDGE_WINS:
                return "color: #c0392b; font-weight: bold"
        except (TypeError, ValueError):
            pass
        return ""

    styled = display_df.style
    if "Bet" in display_df.columns:
        styled = styled.applymap(_color_bet, subset=["Bet"])
    if "Edge (wins)" in display_df.columns:
        styled = styled.applymap(_color_edge, subset=["Edge (wins)"])

    st.dataframe(styled, width="stretch", hide_index=True)

    # ---- AI Reasoning for top bets ----
    st.subheader("AI Reasoning — Top Win Total Bets")
    top_bets = projections[projections["bet_direction"] != "PASS"].nlargest(5, "edge_wins", keep="all")
    top_bets = pd.concat([
        projections[projections["bet_direction"] == "OVER"].nlargest(3, "edge_wins"),
        projections[projections["bet_direction"] == "UNDER"].nsmallest(3, "edge_wins"),
    ]).drop_duplicates()

    if top_bets.empty:
        st.info("Enter Vegas lines to generate bet recommendations.")
        return

    for _, row in top_bets.iterrows():
        if row["bet_direction"] == "PASS":
            continue
        with st.expander(
            f"{row['team']} — {row['bet_direction']} {row.get('vegas_total', '?')} wins "
            f"(projection: {row['projected_wins']})"
        ):
            signals = {
                "prior_pyth_win_pct": row.get("prior_pyth_pct"),
                "prior_run_diff": row.get("prior_run_diff"),
            }

            if st.button(f"Generate Analysis — {row['team']}", key=f"preseason_ai_{row['team']}"):
                with st.spinner("Analyzing..."):
                    result = analyze_preseason_projection(
                        team=row["team"],
                        projected_wins=row["projected_wins"],
                        vegas_line=row.get("vegas_total", row["projected_wins"]),
                        bet_direction=row["bet_direction"],
                        edge_wins=row.get("edge_wins", 0),
                        signals=signals,
                    )
                st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Projected Wins", row["projected_wins"])
                col2.metric("Prior Run Diff", f"{row.get('prior_run_diff', 0):+d}")
                col3.metric("Edge", f"{row.get('edge_wins', 0):+.1f} wins")
