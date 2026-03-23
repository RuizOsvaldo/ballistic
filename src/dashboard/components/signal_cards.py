"""Visual regression signal indicator cards."""

from __future__ import annotations

import streamlit as st

SEVERITY_COLORS = {
    "High": "#c0392b",
    "Medium": "#e67e22",
    "Low": "#f1c40f",
    "None": "#95a5a6",
}

DIRECTION_ICONS = {
    "ERA likely UP": "↑",
    "ERA likely DOWN": "↓",
    "Likely to decline": "↓",
    "Likely to improve": "↑",
    "Stable": "→",
}


def severity_badge(severity: str, label: str = "") -> str:
    """Return an HTML badge string for inline display."""
    color = SEVERITY_COLORS.get(severity, "#95a5a6")
    text = label or severity
    return (
        f'<span style="background-color:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:bold;">{text}</span>'
    )


def render_signal_card(
    name: str,
    severity: str,
    direction: str,
    notes: str = "",
) -> None:
    """Render a single regression signal card using Streamlit."""
    color = SEVERITY_COLORS.get(severity, "#95a5a6")
    icon = DIRECTION_ICONS.get(direction, "→")

    border_style = f"border-left: 4px solid {color}; padding: 8px 12px; margin-bottom: 8px; background-color: #1e1e1e;"

    card_html = f"""
    <div style="{border_style}">
        <strong>{name}</strong>
        &nbsp;&nbsp;{severity_badge(severity)}
        &nbsp;<span style="font-size:1.1em;">{icon}</span>
        &nbsp;<span style="color:#aaa;font-size:0.85em;">{direction}</span>
        {"<br><small style='color:#888;'>" + notes + "</small>" if notes else ""}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_signal_summary(signals_df, name_col: str, severity_col: str, direction_col: str, notes_col: str = "") -> None:
    """Render a list of signal cards from a DataFrame."""
    active = signals_df[signals_df[severity_col] != "None"]
    if active.empty:
        st.success("No active regression signals.")
        return

    for _, row in active.iterrows():
        render_signal_card(
            name=row[name_col],
            severity=row[severity_col],
            direction=row[direction_col],
            notes=row[notes_col] if notes_col and notes_col in row else "",
        )
