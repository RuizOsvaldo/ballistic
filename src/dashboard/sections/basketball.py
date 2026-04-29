"""Basketball (NBA) dashboard — Games, Teams, Players, Props, Season Totals."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.sports.basketball.data.nba_stats import (
    get_nba_team_stats,
    get_nba_player_stats,
    get_todays_nba_games,
    get_team_game_log,
)
from src.sports.basketball.models.net_rating import (
    compute_net_rating_signals,
    compute_nba_win_probabilities,
)
from src.sports.basketball.models.four_factors import project_game_total, compute_total_edge
from src.sports.basketball.models.rest_schedule import (
    compute_rest_adjustments_df,
    get_team_last_game_dates as build_last_game_map,
)
from src.sports.basketball.models.player_props import (
    compute_prop_edge,
    project_points,
    project_rebounds,
    project_assists,
    project_three_pm,
    project_pra,
)
from src.sports.basketball.models.win_probability import build_nba_games_with_edge
from src.sports.basketball.agent.groq_prompts import analyze_nba_game, analyze_nba_prop
from src.data.odds import get_mlb_odds   # reuse odds client pattern — NBA uses same client

CURRENT_SEASON = f"{datetime.datetime.now().year - (0 if datetime.datetime.now().month >= 10 else 1)}"
NBA_SEASON = f"{CURRENT_SEASON}-{str(int(CURRENT_SEASON) + 1)[-2:]}"

SEVERITY_COLORS = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#f1c40f", "None": "#2ecc71"}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600 * 6, show_spinner="Loading NBA team stats...")
def load_nba_teams(season: str) -> pd.DataFrame:
    try:
        df = get_nba_team_stats(season)
        return compute_net_rating_signals(df)
    except Exception as e:
        st.error(f"Could not load NBA team stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading NBA player stats...")
def load_nba_players(season: str) -> pd.DataFrame:
    try:
        return get_nba_player_stats(season)
    except Exception as e:
        st.error(f"Could not load NBA player stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner="Loading today's NBA games...")
def load_nba_games() -> pd.DataFrame:
    try:
        return get_todays_nba_games()
    except Exception as e:
        st.error(f"Could not load NBA schedule: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(bankroll: float = 1000.0) -> None:
    st.header("Basketball (NBA)")
    st.caption(f"Season: {NBA_SEASON} | Net Rating methodology + Four Factors + Rest analysis")

    tab_games, tab_teams, tab_players, tab_props, tab_totals = st.tabs([
        "Today's Games", "Teams", "Players", "Player Props", "Season Totals"
    ])

    team_df = load_nba_teams(NBA_SEASON)
    player_df = load_nba_players(NBA_SEASON)
    games_df = load_nba_games()

    with tab_games:
        _render_games(games_df, team_df, bankroll)

    with tab_teams:
        _render_teams(team_df)

    with tab_players:
        _render_players(player_df, team_df)

    with tab_props:
        _render_props(player_df, team_df)

    with tab_totals:
        _render_season_totals(team_df)


# ---------------------------------------------------------------------------
# Games tab
# ---------------------------------------------------------------------------

def _render_games(games_df: pd.DataFrame, team_df: pd.DataFrame, bankroll: float) -> None:
    st.subheader("Today's NBA Games")

    if games_df.empty or team_df.empty:
        st.info("No games today or stats unavailable. Check back during the NBA season.")
        return

    # Build win probabilities
    try:
        probs_df = compute_nba_win_probabilities(games_df, team_df)
    except Exception as e:
        st.error(f"Could not compute win probabilities: {e}")
        return

    if probs_df.empty:
        st.warning("Could not match game teams to team stats.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_edge = st.slider("Min edge %", 0.0, 15.0, 3.0, 0.5, key="nba_min_edge")
    with col2:
        bet_only = st.checkbox("BET recommendations only", key="nba_bet_only")

    st.divider()

    for _, row in probs_df.iterrows():
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        hp = row.get("home_model_prob", float("nan"))
        ap = row.get("away_model_prob", float("nan"))
        diff = row.get("net_rtg_diff", 0)

        if pd.isna(hp):
            continue

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"**{away} @ {home}**")
                st.caption(row.get("game_time", ""))
                if abs(diff) >= 5:
                    direction = home if diff > 0 else away
                    st.caption(f"Net Rtg edge: **{direction}** (+{abs(diff):.1f} pts)")

            with c2:
                st.metric(f"{home} win prob", f"{hp:.1%}")
                st.metric(f"{away} win prob", f"{ap:.1%}")

            with c3:
                if st.button("AI Analysis", key=f"nba_ai_{home}_{away}"):
                    team_lookup = team_df.set_index("team_name") if "team_name" in team_df.columns else pd.DataFrame()
                    signals = {
                        "home_team": home, "away_team": away,
                        "home_net_rtg": team_lookup.loc[home, "net_rtg"] if home in team_lookup.index else "N/A",
                        "away_net_rtg": team_lookup.loc[away, "net_rtg"] if away in team_lookup.index else "N/A",
                        "net_rtg_diff": diff,
                        "home_rest_type": row.get("home_rest_type", "normal"),
                        "away_rest_type": row.get("away_rest_type", "normal"),
                        "rest_mismatch": row.get("rest_mismatch", False),
                        "home_efg_pct": team_lookup.loc[home, "efg_pct"] if home in team_lookup.index else "N/A",
                        "away_efg_pct": team_lookup.loc[away, "efg_pct"] if away in team_lookup.index else "N/A",
                        "home_drtg": team_lookup.loc[home, "drtg"] if home in team_lookup.index else "N/A",
                        "away_drtg": team_lookup.loc[away, "drtg"] if away in team_lookup.index else "N/A",
                        "home_model_prob": hp, "away_model_prob": ap,
                        "home_implied_prob": row.get("home_implied_prob", "N/A"),
                        "away_implied_prob": row.get("away_implied_prob", "N/A"),
                        "home_edge_pct": row.get("home_edge_pct", "N/A"),
                        "away_edge_pct": row.get("away_edge_pct", "N/A"),
                        "best_bet_side": row.get("best_bet_side", "N/A"),
                        "best_bet_edge": row.get("best_bet_edge", "N/A"),
                    }
                    with st.spinner("Analyzing..."):
                        result = analyze_nba_game(signals)
                    st.info(result.get("reasoning", ""))
                    st.caption(f"Confidence: {result.get('confidence')} | Risk: {result.get('key_risk')}")


# ---------------------------------------------------------------------------
# Teams tab
# ---------------------------------------------------------------------------

def _render_teams(team_df: pd.DataFrame) -> None:
    st.subheader("NBA Teams — Net Rating & Four Factors")

    if team_df.empty:
        st.info("Team stats unavailable.")
        return

    # Filter by signal
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect(
            "Signal severity", ["High", "Medium", "Low", "None"],
            default=["High", "Medium"],
            key="nba_team_severity"
        )
    with col2:
        sort_col = st.selectbox("Sort by", ["net_rtg", "win_pct", "ortg", "drtg", "net_rtg_deviation"], key="nba_team_sort")

    display_df = team_df.copy()
    if severity_filter and "net_rtg_severity" in display_df.columns:
        display_df = display_df[display_df["net_rtg_severity"].isin(severity_filter)]

    if sort_col in display_df.columns:
        asc = sort_col == "drtg"
        display_df = display_df.sort_values(sort_col, ascending=asc)

    # Net rating bar chart
    if "net_rtg" in team_df.columns and "team_name" in team_df.columns:
        chart_df = team_df.sort_values("net_rtg", ascending=False)
        fig = px.bar(
            chart_df,
            x="team_name",
            y="net_rtg",
            color="net_rtg",
            color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],
            color_continuous_midpoint=0,
            title="NBA Net Rating — All Teams",
            labels={"net_rtg": "Net Rating", "team_name": "Team"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Regression signal cards
    flagged = team_df[team_df.get("net_rtg_severity", pd.Series()) != "None"] if "net_rtg_severity" in team_df.columns else pd.DataFrame()
    if not flagged.empty:
        st.subheader("Regression Signals")
        cols = st.columns(min(len(flagged), 4))
        for i, (_, row) in enumerate(flagged.iterrows()):
            sev = row.get("net_rtg_severity", "None")
            color = SEVERITY_COLORS.get(sev, "#95a5a6")
            with cols[i % 4]:
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:8px;margin:4px 0">
                    <b>{row.get('team_name','')}</b><br>
                    Net Rtg: {row.get('net_rtg',''):.1f} | W%: {row.get('win_pct',''):.3f}<br>
                    <span style="color:{color}">{row.get('net_rtg_signal','')} — {sev}</span><br>
                    <small>{row.get('net_rtg_direction','')}</small>
                    </div>""",
                    unsafe_allow_html=True
                )

    # Stats table
    st.subheader("Full Team Stats")
    show_cols = [c for c in ["team_name", "w", "l", "win_pct", "net_rtg", "ortg", "drtg", "pace",
                              "efg_pct", "opp_efg_pct", "net_rtg_signal", "net_rtg_severity"] if c in display_df.columns]
    st.dataframe(display_df[show_cols].reset_index(drop=True), use_container_width=True)


# ---------------------------------------------------------------------------
# Players tab
# ---------------------------------------------------------------------------

def _render_players(player_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
    st.subheader("NBA Players — Stats & Prop Context")

    if player_df.empty:
        st.info("Player stats unavailable.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        team_filter = st.selectbox(
            "Team",
            ["All"] + sorted(player_df["team_abbrev"].dropna().unique().tolist()),
            key="nba_player_team"
        )
    with col2:
        stat_sort = st.selectbox("Sort by", ["pts", "reb", "ast", "usg_pct", "three_pm"], key="nba_player_sort")
    with col3:
        min_min = st.slider("Min minutes", 10, 36, 20, key="nba_player_min")

    display_df = player_df.copy()
    if team_filter != "All":
        display_df = display_df[display_df["team_abbrev"] == team_filter]
    display_df = display_df[display_df["min"] >= min_min]
    if stat_sort in display_df.columns:
        display_df = display_df.sort_values(stat_sort, ascending=False)

    show_cols = [c for c in ["player_name", "team_abbrev", "min", "pts", "reb", "ast",
                              "usg_pct", "ts_pct", "three_pm", "three_pct"] if c in display_df.columns]
    st.dataframe(display_df[show_cols].head(50).reset_index(drop=True), use_container_width=True)

    # Usage vs. scoring scatter
    if all(c in player_df.columns for c in ["usg_pct", "pts", "player_name"]):
        fig = px.scatter(
            player_df[player_df["min"] >= min_min],
            x="usg_pct", y="pts",
            hover_name="player_name",
            color="team_abbrev",
            title="Usage Rate vs. Points Per Game",
            labels={"usg_pct": "Usage %", "pts": "Points/game"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Props tab
# ---------------------------------------------------------------------------

def _render_props(player_df: pd.DataFrame, team_df: pd.DataFrame) -> None:
    st.subheader("NBA Player Props")
    st.caption("Select a player and enter the sportsbook line to compute model edge.")

    if player_df.empty or team_df.empty:
        st.info("Player or team stats unavailable.")
        return

    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("Player name search", key="nba_prop_search")
        player_list = player_df["player_name"].dropna().tolist()
        if search:
            player_list = [p for p in player_list if search.lower() in p.lower()]
        selected_player = st.selectbox("Select player", player_list[:50], key="nba_prop_player")

    with col2:
        opp_team = st.selectbox(
            "Opponent team",
            sorted(team_df["team_name"].dropna().unique().tolist()) if "team_name" in team_df.columns else [],
            key="nba_prop_opp"
        )

    if not selected_player:
        return

    player_row = player_df[player_df["player_name"] == selected_player]
    if player_row.empty:
        st.warning("Player not found.")
        return

    p = player_row.iloc[0].to_dict()

    opp_stats = {}
    if opp_team and "team_name" in team_df.columns:
        opp_row = team_df[team_df["team_name"] == opp_team]
        if not opp_row.empty:
            opp_stats = opp_row.iloc[0].to_dict()

    home_team_stats = {}
    if "team_abbrev" in p and "team_abbrev" in team_df.columns:
        home_row = team_df[team_df["team_abbrev"] == p.get("team_abbrev", "")]
        if not home_row.empty:
            home_team_stats = home_row.iloc[0].to_dict()

    avg_pace = (home_team_stats.get("pace", 99.0) + opp_stats.get("pace", 99.0)) / 2
    opp_drtg = opp_stats.get("drtg", 113.0)

    # Projections
    proj_pts = project_points(p.get("pts", 0), p.get("usg_pct", 0.2), opp_drtg, avg_pace, home_team_stats.get("pace", 99.0))
    proj_reb = project_rebounds(p.get("reb", 0), avg_pace, home_team_stats.get("pace", 99.0))
    proj_ast = project_assists(p.get("ast", 0), avg_pace, home_team_stats.get("pace", 99.0))
    proj_3pm = project_three_pm(p.get("three_pa", 0), p.get("three_pct", 0.36))
    proj_pra = project_pra(proj_pts, proj_reb, proj_ast)

    st.divider()
    st.markdown(f"**{selected_player}** projections vs. {opp_team or 'selected opponent'}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Points", proj_pts, delta=round(proj_pts - p.get("pts", proj_pts), 1))
    c2.metric("Rebounds", proj_reb, delta=round(proj_reb - p.get("reb", proj_reb), 1))
    c3.metric("Assists", proj_ast, delta=round(proj_ast - p.get("ast", proj_ast), 1))
    c4.metric("3PM", proj_3pm)
    c5.metric("PRA", proj_pra)

    # Edge calculator
    st.divider()
    st.markdown("**Prop Line Edge Calculator**")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        stat_type = st.selectbox("Stat", ["pts", "reb", "ast", "pra", "3pm"], key="nba_prop_stat")
    with ec2:
        prop_line = st.number_input("Prop line", min_value=0.0, value=float(p.get("pts", 20.0)), step=0.5, key="nba_prop_line")
    with ec3:
        direction = st.radio("Direction", ["OVER", "UNDER"], horizontal=True, key="nba_prop_dir")

    proj_map = {"pts": proj_pts, "reb": proj_reb, "ast": proj_ast, "pra": proj_pra, "3pm": proj_3pm}
    projection = proj_map.get(stat_type, proj_pts)

    edge_result = compute_prop_edge(projection, prop_line, direction)

    color = "#2ecc71" if edge_result["recommendation"] == "BET" else "#e74c3c"
    st.markdown(
        f"""<div style="border-left:4px solid {color};padding:10px;margin:8px 0">
        <b>{edge_result['recommendation']}</b> — Edge: {edge_result['edge_pct']:.1f}% |
        Projection: {projection} vs Line: {prop_line}
        </div>""",
        unsafe_allow_html=True
    )

    if edge_result["recommendation"] == "BET":
        if st.button("AI Prop Analysis", key="nba_prop_ai"):
            signals = {
                "player_name": selected_player,
                "team": p.get("team_abbrev", ""),
                "stat_type": stat_type,
                "prop_line": prop_line,
                "model_projection": projection,
                "bet_direction": direction,
                "edge_pct": edge_result["edge_pct"],
                "season_avg": p.get(stat_type, "N/A"),
                "usg_pct": p.get("usg_pct", "N/A"),
                "opp_drtg": opp_drtg,
                "game_pace": avg_pace,
                "rest_type": "normal",
            }
            with st.spinner("Analyzing prop..."):
                result = analyze_nba_prop(signals)
            st.info(result.get("reasoning", ""))
            st.caption(f"Confidence: {result.get('confidence')} | Risk: {result.get('key_risk')}")


# ---------------------------------------------------------------------------
# Season Totals tab
# ---------------------------------------------------------------------------

def _render_season_totals(team_df: pd.DataFrame) -> None:
    st.subheader("NBA Season Win Totals")
    st.caption("Compare model projected wins (based on net rating) to Vegas over/under lines.")

    if team_df.empty:
        st.info("Team stats unavailable.")
        return

    if "net_rtg" not in team_df.columns or "win_pct" not in team_df.columns:
        st.warning("Net rating data required for projections.")
        return

    games_played = 82

    # Project wins: net rating → logistic win prob → wins
    import math
    from src.sports.basketball.models.net_rating import net_rtg_to_win_prob

    proj_rows = []
    for _, row in team_df.iterrows():
        net_rtg = row.get("net_rtg", 0.0)
        current_win_pct = row.get("win_pct", 0.5)
        # Regress 20% toward .500 — sample size adjustment
        regressed_pct = current_win_pct * 0.80 + 0.500 * 0.20
        proj_wins = round(regressed_pct * games_played, 1)
        proj_rows.append({
            "team_name": row.get("team_name", ""),
            "current_wins": row.get("w", 0),
            "current_losses": row.get("l", 0),
            "current_win_pct": round(current_win_pct, 3),
            "net_rtg": net_rtg,
            "projected_wins": proj_wins,
            "vegas_total": float("nan"),
            "edge_wins": float("nan"),
            "bet_direction": "N/A",
        })

    proj_df = pd.DataFrame(proj_rows).sort_values("projected_wins", ascending=False)

    # Vegas line inputs
    st.markdown("**Enter Vegas win total lines (optional):**")
    st.caption("Leave blank to view projections without edge calculation.")

    with st.expander("Vegas Line Input"):
        vegas_lines = {}
        cols = st.columns(3)
        for i, row in proj_df.iterrows():
            col = cols[i % 3]
            val = col.number_input(
                row["team_name"],
                min_value=0.0, max_value=82.0,
                value=0.0, step=0.5,
                key=f"nba_vegas_{row['team_name']}"
            )
            if val > 0:
                vegas_lines[row["team_name"]] = val

    if vegas_lines:
        for i, row in proj_df.iterrows():
            team = row["team_name"]
            if team in vegas_lines:
                line = vegas_lines[team]
                edge = row["projected_wins"] - line
                proj_df.at[i, "vegas_total"] = line
                proj_df.at[i, "edge_wins"] = round(edge, 1)
                if abs(edge) >= 3:
                    proj_df.at[i, "bet_direction"] = "OVER" if edge > 0 else "UNDER"
                else:
                    proj_df.at[i, "bet_direction"] = "PASS"

    # Chart
    chart_df = proj_df.dropna(subset=["projected_wins"])
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x="team_name",
            y="projected_wins",
            color="net_rtg",
            color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],
            color_continuous_midpoint=0,
            title="NBA Projected Season Wins",
            labels={"projected_wins": "Projected Wins", "team_name": "Team"},
        )
        fig.add_hline(y=41, line_dash="dash", line_color="white", opacity=0.4, annotation_text=".500")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Table
    show_cols = [c for c in ["team_name", "current_wins", "current_losses", "net_rtg",
                              "projected_wins", "vegas_total", "edge_wins", "bet_direction"] if c in proj_df.columns]
    st.dataframe(proj_df[show_cols].reset_index(drop=True), use_container_width=True)

    # Bet recommendations
    bets = proj_df[proj_df["bet_direction"].isin(["OVER", "UNDER"])]
    if not bets.empty:
        st.subheader("Flagged Season Win Total Bets")
        for _, row in bets.iterrows():
            color = "#2ecc71" if row["bet_direction"] == "OVER" else "#e74c3c"
            st.markdown(
                f"""<div style="border-left:4px solid {color};padding:8px;margin:4px 0">
                <b>{row['team_name']}</b> — {row['bet_direction']} {row['vegas_total']} wins |
                Model: {row['projected_wins']} | Edge: {row['edge_wins']:+.1f} wins
                </div>""",
                unsafe_allow_html=True
            )
