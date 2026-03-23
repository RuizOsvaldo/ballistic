"""Reusable edge % table with conditional formatting."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_edge_table(df: pd.DataFrame, edge_col: str = "best_bet_edge") -> None:
    """
    Render a styled DataFrame with green/red highlighting on edge %.
    Positive edge = green, negative = red, near-zero = neutral.
    """
    if df.empty:
        st.info("No games match the current filters.")
        return

    def _color_edge(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 5.0:
            return "background-color: #1a7a1a; color: white"
        if v >= 3.0:
            return "background-color: #2d9e2d; color: white"
        if v > 0:
            return "background-color: #5cb85c; color: white"
        if v < 0:
            return "background-color: #c0392b; color: white"
        return ""

    pct_cols = [c for c in df.columns if "edge_pct" in c or c == edge_col]

    styled = df.style
    for col in pct_cols:
        if col in df.columns:
            styled = styled.applymap(_color_edge, subset=[col])

    styled = styled.format(
        {col: "{:.1f}%" for col in pct_cols if col in df.columns},
        na_rep="—",
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)
