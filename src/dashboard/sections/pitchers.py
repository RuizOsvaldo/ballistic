"""Pitchers page — FIP/ERA/BABIP breakdown with regression signals."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components.signal_cards import render_signal_summary


def render(pitcher_stats: pd.DataFrame) -> None:
    st.header("Pitcher Analysis — FIP, ERA & Regression Signals")

    if pitcher_stats.empty:
        st.warning("No pitcher stats loaded. Check your data connection.")
        return

    df = pitcher_stats.copy()

    # ---- Filters ----
    col1, col2, col3 = st.columns(3)
    with col1:
        teams = ["All"] + sorted(df["team"].dropna().unique().tolist()) if "team" in df.columns else ["All"]
        selected_team = st.selectbox("Filter by team", teams)
    with col2:
        severity_opts = ["All", "High", "Medium", "Low", "None"]
        selected_severity = st.selectbox("Signal severity", severity_opts)
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["fip_era_gap", "fip", "era", "babip"],
            format_func=lambda x: {
                "fip_era_gap": "FIP-ERA Gap (regression risk)",
                "fip": "FIP",
                "era": "ERA",
                "babip": "BABIP",
            }.get(x, x),
        )

    if selected_team != "All" and "team" in df.columns:
        df = df[df["team"] == selected_team]

    if selected_severity != "All" and "signal_severity" in df.columns:
        df = df[df["signal_severity"] == selected_severity]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_by not in ["fip_era_gap"]))

    st.caption(f"Showing {len(df)} qualified pitcher(s)")

    # ---- FIP-ERA scatter ----
    if "fip" in df.columns and "era" in df.columns and not df.empty:
        st.subheader("FIP vs ERA")
        st.caption("Points above the diagonal: ERA < FIP → expect ERA to rise (regression risk)")

        fig = px.scatter(
            df,
            x="era",
            y="fip",
            text="name" if "name" in df.columns else None,
            color="fip_era_gap" if "fip_era_gap" in df.columns else None,
            color_continuous_scale=["#2ecc71", "#f39c12", "#c0392b"],
            hover_data=["name", "team", "era", "fip", "babip"] if all(c in df.columns for c in ["name", "team", "babip"]) else None,
            labels={"era": "ERA", "fip": "FIP"},
        )
        # Diagonal = ERA equals FIP
        min_val = min(df["era"].min(), df["fip"].min()) - 0.2
        max_val = max(df["era"].max(), df["fip"].max()) + 0.2
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="grey"))
        fig.update_traces(textposition="top center", marker=dict(size=8))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Stats table ----
    st.subheader("Pitcher Stats Table")
    display_cols = {
        "name": "Pitcher",
        "team": "Team",
        "ip": "IP",
        "era": "ERA",
        "fip": "FIP",
        "fip_era_gap": "FIP-ERA Gap",
        "babip": "BABIP",
        "k_pct": "K%",
        "bb_pct": "BB%",
        "signal_severity": "Signal",
        "signal_direction": "Direction",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    display_df = df[list(available.keys())].rename(columns=available)

    for col in ["ERA", "FIP", "FIP-ERA Gap", "BABIP"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- Signal cards ----
    if "signal_severity" in df.columns:
        st.subheader("Active Regression Signals")
        render_signal_summary(
            df,
            name_col="name",
            severity_col="signal_severity",
            direction_col="signal_direction",
            notes_col="signal_notes",
        )
