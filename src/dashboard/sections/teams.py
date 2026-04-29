"""Teams page — Pythagorean records, run differentials, and regression signals."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components.signal_cards import render_signal_summary, severity_badge


def render(team_stats: pd.DataFrame) -> None:
    st.header("Team Analysis — Pythagorean & Regression Signals")

    if team_stats.empty:
        st.warning("No team stats loaded. Check your data connection.")
        return

    df = team_stats.copy()

    # ---- Filters ----
    col1, col2 = st.columns(2)
    with col1:
        signal_filter = st.selectbox(
            "Filter by signal",
            ["All", "Overperforming", "Underperforming", "On Track"],
        )
    with col2:
        sort_col = st.selectbox(
            "Sort by",
            ["pyth_deviation", "win_pct", "run_diff", "runs_scored"],
            format_func=lambda x: {
                "pyth_deviation": "Pythagorean Deviation",
                "win_pct": "Win %",
                "run_diff": "Run Differential",
                "runs_scored": "Runs Scored",
            }.get(x, x),
        )

    if signal_filter != "All" and "pyth_signal" in df.columns:
        df = df[df["pyth_signal"] == signal_filter]

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)

    # ---- Pythagorean deviation chart ----
    if "pyth_deviation" in df.columns and not df.empty:
        st.subheader("Pythagorean Deviation by Team")
        st.caption("Positive = overperforming their run differential (fade). Negative = underperforming (back).")

        chart_df = df[["team", "pyth_deviation"]].dropna()
        fig = px.bar(
            chart_df,
            x="team",
            y="pyth_deviation",
            color="pyth_deviation",
            color_continuous_scale=["#c0392b", "#f39c12", "#2ecc71"],
            color_continuous_midpoint=0,
            labels={"pyth_deviation": "Deviation", "team": "Team"},
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="+5% (overperforming)")
        fig.add_hline(y=-0.05, line_dash="dash", line_color="green", annotation_text="-5% (underperforming)")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Team stats table ----
    st.subheader("Team Stats Table")
    display_cols = {
        "team": "Team",
        "wins": "W",
        "losses": "L",
        "win_pct": "W%",
        "pyth_win_pct": "Pyth W%",
        "pyth_deviation": "Deviation",
        "pyth_signal": "Signal",
        "runs_scored": "RS",
        "runs_allowed": "RA",
        "run_diff": "RD",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    display_df = df[list(available.keys())].rename(columns=available)

    if "W%" in display_df.columns:
        display_df["W%"] = (display_df["W%"] * 100).round(1).astype(str) + "%"
    if "Pyth W%" in display_df.columns:
        display_df["Pyth W%"] = (display_df["Pyth W%"] * 100).round(1).astype(str) + "%"
    if "Deviation" in display_df.columns:
        display_df["Deviation"] = display_df["Deviation"].apply(
            lambda x: f"+{x*100:.1f}%" if x > 0 else f"{x*100:.1f}%"
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- Regression signal cards ----
    if "team_signal_severity" in df.columns:
        st.subheader("Team Regression Signals")
        render_signal_summary(
            df,
            name_col="team",
            severity_col="team_signal_severity",
            direction_col="team_signal_direction",
        )
