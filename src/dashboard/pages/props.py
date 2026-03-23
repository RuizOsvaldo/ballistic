"""Player props page — pitcher strikeouts and batter hits with edge analysis."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.baseball_stats import get_batter_stats, get_pitcher_stats
from src.data.odds import get_mlb_odds, get_mlb_player_props, get_best_prop_lines
from src.models.player_props import (
    project_pitcher_strikeouts,
    project_batter_hits,
    compute_prop_edge,
    MIN_PROP_EDGE_PCT,
)
from src.shared.groq_agent import analyze_mlb_prop

CURRENT_SEASON = datetime.datetime.now().year


@st.cache_data(ttl=3600 * 6, show_spinner="Loading pitcher stats...")
def _load_pitchers(season: int) -> pd.DataFrame:
    try:
        return get_pitcher_stats(season)
    except Exception as e:
        st.error(f"Could not load pitcher stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading batter stats...")
def _load_batters(season: int) -> pd.DataFrame:
    try:
        return get_batter_stats(season)
    except Exception as e:
        st.error(f"Could not load batter stats: {e}")
        return pd.DataFrame()


def _render_pitcher_props(pitcher_df: pd.DataFrame) -> None:
    st.subheader("Pitcher Strikeout Props")
    st.caption(
        "Model projects K total using pitcher K%, IP/start, and opponent K%. "
        "Enter sportsbook lines to find over/under value."
    )

    if pitcher_df.empty:
        st.warning("No pitcher data available.")
        return

    teams = ["All"] + sorted(pitcher_df["team"].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.selectbox("Filter by team", teams, key="props_pitcher_team")
    with col2:
        min_edge = st.slider("Min edge %", 0.0, 30.0, MIN_PROP_EDGE_PCT, 1.0, key="props_pitcher_edge")

    filtered = pitcher_df.copy()
    if team_filter != "All":
        filtered = filtered[filtered["team"] == team_filter]

    results = []
    for _, row in filtered.iterrows():
        if pd.isna(row.get("k_pct")) or pd.isna(row.get("ip")):
            continue
        ip_per_start = max(row["ip"] / 30, 3.0)
        proj_k = project_pitcher_strikeouts(
            pitcher_k_pct=row["k_pct"],
            pitcher_ip_per_start=ip_per_start,
        )
        results.append({
            "name": row["name"],
            "team": row["team"],
            "k_pct": row.get("k_pct"),
            "xfip": row.get("xfip"),
            "babip": row.get("babip"),
            "ip_per_start": round(ip_per_start, 1),
            "projected_k": proj_k,
        })

    if not results:
        st.info("No pitcher data to display.")
        return

    proj_df = pd.DataFrame(results).sort_values("projected_k", ascending=False)

    # Allow manual prop line input
    st.caption("Enter sportsbook K line for each pitcher to compute edge:")
    line_inputs = {}
    cols = st.columns(4)
    for i, (_, row) in enumerate(proj_df.head(20).iterrows()):
        c = cols[i % 4]
        with c:
            line_inputs[row["name"]] = st.number_input(
                f"{row['name']} ({row['team']})",
                min_value=0.5,
                max_value=15.0,
                value=float(row["projected_k"]),
                step=0.5,
                key=f"kline_{row['name']}",
            )

    # Compute edges
    edge_rows = []
    for _, row in proj_df.iterrows():
        if row["name"] not in line_inputs:
            continue
        line = line_inputs[row["name"]]
        proj = row["projected_k"]
        direction = "OVER" if proj > line else "UNDER"
        edge = compute_prop_edge(proj, line, direction)
        edge_rows.append({
            "Pitcher": row["name"],
            "Team": row["team"],
            "K%": f"{row['k_pct']:.1%}" if row.get("k_pct") else "—",
            "xFIP": round(row["xfip"], 2) if row.get("xfip") else "—",
            "Proj K": proj,
            "Line": line,
            "Bet": direction,
            "Edge %": edge["edge_pct"],
            "Signal": "BET" if abs(edge["edge_pct"]) >= min_edge else "PASS",
        })

    if edge_rows:
        edge_df = pd.DataFrame(edge_rows)
        edge_df = edge_df[edge_df["Signal"] == "BET"].sort_values("Edge %", ascending=False) \
            if min_edge > 0 else pd.DataFrame(edge_rows)

        def _color(val):
            if val == "BET":
                return "background-color: #2ecc71; color: black"
            if val == "PASS":
                return "background-color: #555; color: white"
            return ""

        st.dataframe(
            edge_df.style.applymap(_color, subset=["Signal"]),
            use_container_width=True,
            hide_index=True,
        )

        # AI analysis for top props
        top = edge_df[edge_df["Signal"] == "BET"].head(3)
        if not top.empty:
            st.subheader("AI Analysis — Top Strikeout Props")
            for _, r in top.iterrows():
                with st.expander(f"{r['Pitcher']} — {r['Bet']} {r['Line']} K (Edge: {r['Edge %']:+.1f}%)"):
                    if st.button(f"Analyze {r['Pitcher']}", key=f"ai_k_{r['Pitcher']}"):
                        with st.spinner("Analyzing..."):
                            pitcher_row = pitcher_df[pitcher_df["name"] == r["Pitcher"]].iloc[0]
                            result = analyze_mlb_prop(
                                player_name=r["Pitcher"],
                                team=r["Team"],
                                prop_type="Strikeouts",
                                line=r["Line"],
                                model_projection=r["Proj K"],
                                edge_pct=r["Edge %"],
                                bet_direction=r["Bet"],
                                signals={
                                    "k_pct": pitcher_row.get("k_pct"),
                                    "whiff_rate": pitcher_row.get("whiff_pct"),
                                    "babip": pitcher_row.get("babip"),
                                },
                            )
                        st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                        st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                        st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")


def _render_batter_props(batter_df: pd.DataFrame) -> None:
    st.subheader("Batter Hit Props")
    st.caption(
        "Model uses BABIP regression to project hits. "
        "Batters with extreme BABIP are regression candidates — prime prop value."
    )

    if batter_df.empty:
        st.warning("No batter data available.")
        return

    teams = ["All"] + sorted(batter_df["team"].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.selectbox("Filter by team", teams, key="props_batter_team")
    with col2:
        babip_filter = st.selectbox(
            "BABIP signal filter",
            ["All", "High BABIP (>0.320) — fade", "Low BABIP (<0.275) — back"],
            key="props_babip_filter",
        )

    filtered = batter_df.copy()
    if team_filter != "All":
        filtered = filtered[filtered["team"] == team_filter]
    if babip_filter == "High BABIP (>0.320) — fade" and "babip" in filtered.columns:
        filtered = filtered[filtered["babip"] > 0.320]
    elif babip_filter == "Low BABIP (<0.275) — back" and "babip" in filtered.columns:
        filtered = filtered[filtered["babip"] < 0.275]

    if filtered.empty:
        st.info("No batters match the current filters.")
        return

    # Build projection table
    results = []
    for _, row in filtered.iterrows():
        if pd.isna(row.get("babip")) or pd.isna(row.get("k_pct")):
            continue
        ab_per_game = row.get("ab", 450) / 140
        proj_hits = project_batter_hits(
            batter_babip=row["babip"],
            at_bats_projected=ab_per_game,
            batter_k_pct=row["k_pct"],
        )
        babip_dev = row["babip"] - 0.300
        results.append({
            "Batter": row["name"],
            "Team": row["team"],
            "BABIP": row["babip"],
            "BABIP Dev": f"{babip_dev:+.3f}",
            "wRC+": row.get("wrc_plus"),
            "K%": f"{row['k_pct']:.1%}",
            "Proj Hits/G": proj_hits,
            "Signal": "High BABIP" if row["babip"] > 0.320 else ("Low BABIP" if row["babip"] < 0.275 else "Normal"),
        })

    if not results:
        st.info("No qualifying batters.")
        return

    result_df = pd.DataFrame(results).sort_values("BABIP Dev")

    def _color_babip(val):
        try:
            v = float(val.replace("+", ""))
            if v > 0.020:
                return "color: #c0392b; font-weight: bold"
            if v < -0.020:
                return "color: #2ecc71; font-weight: bold"
        except (ValueError, AttributeError):
            pass
        return ""

    st.dataframe(
        result_df.style.applymap(_color_babip, subset=["BABIP Dev"]),
        use_container_width=True,
        hide_index=True,
    )

    # BABIP distribution chart
    if "BABIP" in result_df.columns and len(result_df) > 5:
        fig = px.histogram(
            result_df,
            x="BABIP",
            nbins=20,
            title="BABIP Distribution — Batters",
            labels={"BABIP": "BABIP"},
            color_discrete_sequence=["#3498db"],
        )
        fig.add_vline(x=0.300, line_dash="dash", line_color="white", annotation_text="League avg .300")
        fig.add_vline(x=0.320, line_dash="dash", line_color="red", annotation_text=".320 threshold")
        fig.add_vline(x=0.275, line_dash="dash", line_color="green", annotation_text=".275 threshold")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.header("Player Props — MLB")
    st.caption(
        "Stat-model projections for pitcher and batter props. "
        "Find value where sportsbook lines diverge from underlying performance metrics."
    )

    tab1, tab2 = st.tabs(["Pitcher Props (K)", "Batter Props (Hits)"])

    pitcher_df = _load_pitchers(CURRENT_SEASON)
    batter_df = _load_batters(CURRENT_SEASON)

    with tab1:
        _render_pitcher_props(pitcher_df)

    with tab2:
        _render_batter_props(batter_df)
