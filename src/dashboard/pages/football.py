"""Football (NFL) dashboard — Games, Teams, Players, Props, Season Totals."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.sports.football.data.nfl_stats import (
    get_current_week_games,
    get_nfl_player_stats,
    get_nfl_schedule,
    get_nfl_snap_counts,
    get_nfl_team_epa,
    get_nfl_win_totals,
)
from src.sports.football.models.epa import (
    compute_epa_signals,
    compute_nfl_win_probabilities,
)
from src.sports.football.models.weather import add_weather_adjustments
from src.sports.football.models.rest_schedule import compute_rest_adjustments
from src.sports.football.models.player_props import (
    build_qb_prop_card,
    build_rb_prop_card,
    build_wr_te_prop_card,
)
from src.sports.football.agent.groq_prompts import (
    analyze_nfl_game,
    analyze_nfl_prop,
    analyze_nfl_season_total,
)

NFL_SEASON = datetime.datetime.now().year if datetime.datetime.now().month >= 9 else datetime.datetime.now().year - 1

SEVERITY_COLORS = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#f1c40f", "None": "#2ecc71"}
POSITION_ICONS = {"QB": "🏈", "RB": "🏃", "WR": "📡", "TE": "🏹"}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600 * 6, show_spinner="Loading NFL team EPA stats...")
def load_nfl_teams(season: int) -> pd.DataFrame:
    try:
        df = get_nfl_team_epa(season)
        return compute_epa_signals(df)
    except Exception as e:
        st.error(f"Could not load NFL team stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading NFL player stats...")
def load_nfl_players(season: int) -> pd.DataFrame:
    try:
        return get_nfl_player_stats(season)
    except Exception as e:
        st.error(f"Could not load NFL player stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 2, show_spinner="Loading NFL schedule...")
def load_nfl_games(season: int) -> pd.DataFrame:
    try:
        games = get_current_week_games(season)
        if games.empty:
            return games
        return add_weather_adjustments(games)
    except Exception as e:
        st.error(f"Could not load NFL schedule: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 24, show_spinner="Loading NFL win totals...")
def load_nfl_win_totals(season: int) -> pd.DataFrame:
    try:
        return get_nfl_win_totals(season)
    except Exception as e:
        return pd.DataFrame()


def render(bankroll: float = 1000.0) -> None:
    st.header("Football (NFL)")

    team_df = load_nfl_teams(NFL_SEASON)
    player_df = load_nfl_players(NFL_SEASON)
    games_df = load_nfl_games(NFL_SEASON)
    win_totals_df = load_nfl_win_totals(NFL_SEASON)

    tab_games, tab_teams, tab_players, tab_props, tab_totals = st.tabs(
        ["Games", "Teams", "Players", "Player Props", "Season Totals"]
    )

    # =========================================================================
    # TAB 1 — Games
    # =========================================================================
    with tab_games:
        st.subheader("This Week's Games")

        if games_df.empty:
            st.info("No upcoming NFL games found. NFL season runs September–February.")
            st.caption(f"Using season: {NFL_SEASON}")
        else:
            # Compute win probabilities if team data is available
            games_with_prob = games_df.copy()
            if not team_df.empty:
                games_with_prob = compute_nfl_win_probabilities(games_with_prob, team_df)
                games_with_prob = compute_rest_adjustments(games_with_prob)

            # Filter to games with model probs
            display_cols = ["gameday", "week", "home_team", "away_team"]
            optional_cols = [
                "spread_line", "total_line", "adjusted_total",
                "home_model_prob", "away_model_prob", "spread_equivalent",
                "weather_flag", "weather_summary",
                "home_rest_type", "away_rest_type", "rest_mismatch",
            ]
            display_cols += [c for c in optional_cols if c in games_with_prob.columns]

            st.dataframe(
                games_with_prob[display_cols].rename(columns={
                    "gameday": "Date", "week": "Week",
                    "home_team": "Home", "away_team": "Away",
                    "spread_line": "Spread", "total_line": "Total",
                    "adjusted_total": "Adj. Total",
                    "home_model_prob": "Home Win%", "away_model_prob": "Away Win%",
                    "spread_equivalent": "Model Spread",
                    "weather_flag": "Weather⚠", "weather_summary": "Weather",
                    "home_rest_type": "Home Rest", "away_rest_type": "Away Rest",
                    "rest_mismatch": "Rest Mismatch",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # ---- AI reasoning per game ----
            if not team_df.empty and "home_model_prob" in games_with_prob.columns:
                st.subheader("AI Game Analysis")
                st.caption("Select a game to generate Groq/Llama 3.3 reasoning.")

                settled = games_with_prob.dropna(subset=["home_model_prob"])
                if not settled.empty:
                    game_labels = [
                        f"{row['away_team']} @ {row['home_team']} (Wk {row.get('week', '?')})"
                        for _, row in settled.iterrows()
                    ]
                    selected_idx = st.selectbox("Game", range(len(game_labels)), format_func=lambda i: game_labels[i])
                    row = settled.iloc[selected_idx]

                    if st.button("Analyze Game", key="nfl_game_ai"):
                        home_epa = team_df[team_df["team"] == row["home_team"]]["epa_composite"].values
                        away_epa = team_df[team_df["team"] == row["away_team"]]["epa_composite"].values
                        signals = {
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "home_epa_composite": float(home_epa[0]) if len(home_epa) > 0 else None,
                            "away_epa_composite": float(away_epa[0]) if len(away_epa) > 0 else None,
                            "epa_diff": round(float(home_epa[0]) - float(away_epa[0]), 3) if len(home_epa) > 0 and len(away_epa) > 0 else None,
                            "spread_equivalent": row.get("spread_equivalent"),
                            "posted_spread": row.get("spread_line"),
                            "home_model_prob": f"{row['home_model_prob']:.1%}",
                            "away_model_prob": f"{row['away_model_prob']:.1%}",
                            "home_rest_type": row.get("home_rest_type", "Normal"),
                            "away_rest_type": row.get("away_rest_type", "Normal"),
                            "rest_mismatch": row.get("rest_mismatch", False),
                            "weather_summary": row.get("weather_summary", ""),
                            "weather_flag": row.get("weather_flag", False),
                        }
                        with st.spinner("Asking Groq..."):
                            result = analyze_nfl_game(signals)
                        conf = result.get("confidence", "Low")
                        conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
                        st.markdown(f"**{conf_color} Confidence: {conf}**")
                        st.markdown(f"**Reasoning:** {result.get('reasoning', '')}")
                        st.markdown(f"**Key Risk:** {result.get('key_risk', '')}")

            # ---- Weather alerts ----
            if "weather_flag" in games_with_prob.columns:
                flagged = games_with_prob[games_with_prob["weather_flag"] == True]
                if not flagged.empty:
                    st.subheader("Weather Alerts")
                    for _, row in flagged.iterrows():
                        st.warning(
                            f"**{row['away_team']} @ {row['home_team']}** — "
                            f"{row.get('weather_summary', 'Adverse conditions')} | "
                            f"Posted Total: {row.get('total_line', 'N/A')} → "
                            f"Adjusted: {row.get('adjusted_total', 'N/A')}"
                        )

    # =========================================================================
    # TAB 2 — Teams
    # =========================================================================
    with tab_teams:
        st.subheader("Team EPA Efficiency Rankings")
        st.caption("EPA/play measures play-by-play efficiency vs. expected points in situation. Higher = better.")

        if team_df.empty:
            st.info("Team EPA data unavailable. Check that nfl_data_py is installed and the season is active.")
        else:
            # Summary table
            display_cols = ["team", "off_epa", "def_epa", "epa_composite"]
            opt_cols = ["off_success_rate", "def_success_rate", "games", "win_pct",
                        "epa_signal", "epa_severity", "epa_direction"]
            display_cols += [c for c in opt_cols if c in team_df.columns]

            st.dataframe(
                team_df[display_cols].rename(columns={
                    "team": "Team", "off_epa": "Off EPA/play", "def_epa": "Def EPA/play",
                    "epa_composite": "Composite", "off_success_rate": "Off Success%",
                    "def_success_rate": "Def Success%", "games": "Games",
                    "win_pct": "Win%", "epa_signal": "Signal",
                    "epa_severity": "Severity", "epa_direction": "Outlook",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Regression signal cards
            if "epa_signal" in team_df.columns:
                flagged = team_df[team_df["epa_signal"] != "On Track"]
                if not flagged.empty:
                    st.subheader("Regression Signal Cards")
                    cols = st.columns(min(3, len(flagged)))
                    for i, (_, row) in enumerate(flagged.iterrows()):
                        with cols[i % len(cols)]:
                            sev = row.get("epa_severity", "None")
                            color = SEVERITY_COLORS.get(sev, "#888")
                            signal = row.get("epa_signal", "")
                            direction = row.get("epa_direction", "")
                            st.markdown(
                                f"""<div style="border-left:4px solid {color};padding:8px;margin:4px 0">
                                <b>{row['team']}</b> — <span style="color:{color}">{signal}</span><br/>
                                <small>{direction}</small><br/>
                                <small>Composite: {row.get('epa_composite', 'N/A'):.3f}</small>
                                </div>""",
                                unsafe_allow_html=True,
                            )

    # =========================================================================
    # TAB 3 — Players
    # =========================================================================
    with tab_players:
        st.subheader("Player Stats")

        if player_df.empty:
            st.info("Player data unavailable.")
        else:
            positions = ["All", "QB", "RB", "WR", "TE"]
            if "position" in player_df.columns:
                pos_filter = st.selectbox("Position", positions, key="nfl_pos_filter")
            else:
                pos_filter = "All"

            df = player_df.copy()
            if pos_filter != "All" and "position" in df.columns:
                df = df[df["position"] == pos_filter]

            display_cols = [c for c in [
                "player_name", "position", "recent_team", "games",
                "passing_yards_pg", "rushing_yards_pg", "receiving_yards_pg",
                "passing_tds", "rushing_tds", "receiving_tds",
                "targets", "target_share", "air_yards",
                "receptions", "attempts",
            ] if c in df.columns]

            st.dataframe(
                df[display_cols].rename(columns={
                    "player_name": "Player", "position": "Pos", "recent_team": "Team",
                    "games": "G", "passing_yards_pg": "Pass Yds/G",
                    "rushing_yards_pg": "Rush Yds/G", "receiving_yards_pg": "Rec Yds/G",
                    "passing_tds": "Pass TDs", "rushing_tds": "Rush TDs",
                    "receiving_tds": "Rec TDs", "targets": "Targets",
                    "target_share": "Tgt Share", "air_yards": "Air Yds",
                    "receptions": "Rec", "attempts": "Att",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================================
    # TAB 4 — Player Props
    # =========================================================================
    with tab_props:
        st.subheader("Player Prop Analysis")
        st.caption("Enter a player's season stats and sportsbook prop line to compute model edge.")

        position = st.selectbox("Position", ["QB", "RB", "WR", "TE"], key="nfl_prop_pos")
        player_name = st.text_input("Player Name", placeholder="e.g. P. Mahomes")
        team = st.text_input("Team", placeholder="e.g. KC")
        opponent = st.text_input("Opponent", placeholder="e.g. BUF")

        # Get opponent's def EPA if available
        opp_def_epa = None
        if not team_df.empty and opponent:
            opp_row = team_df[team_df["team"] == opponent.upper()]
            if not opp_row.empty:
                opp_def_epa = float(opp_row.iloc[0]["def_epa"])
                st.caption(f"{opponent.upper()} defensive EPA/play: {opp_def_epa:.3f}")

        if position == "QB":
            col1, col2 = st.columns(2)
            with col1:
                pass_yds_avg = st.number_input("Season Avg Passing Yards/G", value=270.0, step=5.0)
                comp_avg = st.number_input("Season Avg Completions/G", value=22.0, step=1.0)
            with col2:
                pass_yds_line = st.number_input("Sportsbook Pass Yards Line", value=254.5, step=0.5)
                comp_line = st.number_input("Sportsbook Completions Line", value=21.5, step=0.5)

            if st.button("Project QB Props"):
                card = build_qb_prop_card(
                    player_name=player_name or "QB",
                    team=team or "TM",
                    season_stats={"pass_yds_pg": pass_yds_avg, "comp_pg": comp_avg,
                                  "pass_tds_pg": 1.8, "int_pg": 0.5},
                    prop_lines={"pass_yds": pass_yds_line, "completions": comp_line},
                    opp_def_epa=opp_def_epa,
                )
                col1, col2 = st.columns(2)
                with col1:
                    edge = card.get("edge_pass_yds", 0) or 0
                    delta_color = "normal" if abs(edge) < 5 else ("off" if edge < 0 else "normal")
                    st.metric("Pass Yards Projection", f"{card['proj_pass_yds']:.1f}",
                              delta=f"{edge:+.1f}% edge", delta_color=delta_color)
                with col2:
                    edge2 = card.get("edge_completions", 0) or 0
                    st.metric("Completions Projection", f"{card['proj_completions']:.1f}",
                              delta=f"{edge2:+.1f}% edge")

                if opp_def_epa is not None and abs(card.get("edge_pass_yds", 0) or 0) >= 5.0:
                    if st.button("Get AI Analysis", key="nfl_qb_ai"):
                        with st.spinner("Asking Groq..."):
                            result = analyze_nfl_prop({
                                "player_name": player_name, "team": team,
                                "position": "QB", "opponent": opponent,
                                "prop_type": "Passing Yards",
                                "line": pass_yds_line,
                                "projection": card["proj_pass_yds"],
                                "edge_pct": card.get("edge_pass_yds"),
                                "bet_direction": "OVER" if (card.get("edge_pass_yds") or 0) > 0 else "UNDER",
                                "opp_def_epa": opp_def_epa,
                                "season_avg": pass_yds_avg,
                            })
                        st.markdown(f"**Reasoning:** {result.get('reasoning', '')}")
                        st.caption(f"Confidence: {result.get('confidence', 'N/A')} | Risk: {result.get('key_risk', '')}")

        elif position == "RB":
            col1, col2 = st.columns(2)
            with col1:
                rush_avg = st.number_input("Season Avg Rush Yards/G", value=75.0, step=5.0)
                rush_line = st.number_input("Sportsbook Rush Yards Line", value=72.5, step=0.5)
            with col2:
                rec_avg = st.number_input("Season Avg Rec Yards/G", value=25.0, step=2.0)
                rec_line = st.number_input("Sportsbook Rec Yards Line", value=22.5, step=0.5)

            if st.button("Project RB Props"):
                card = build_rb_prop_card(
                    player_name=player_name or "RB",
                    team=team or "TM",
                    season_stats={"rush_yds_pg": rush_avg, "rec_yds_pg": rec_avg},
                    prop_lines={"rush_yds": rush_line, "rec_yds": rec_line},
                    opp_def_epa=opp_def_epa,
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rush Yards Projection", f"{card['proj_rush_yds']:.1f}",
                              delta=f"{card.get('edge_rush_yds', 0) or 0:+.1f}% edge")
                with col2:
                    st.metric("Rec Yards Projection", f"{card['proj_rec_yds']:.1f}",
                              delta=f"{card.get('edge_rec_yds', 0) or 0:+.1f}% edge")

        else:  # WR or TE
            col1, col2 = st.columns(2)
            with col1:
                rec_yds_avg = st.number_input("Season Avg Rec Yards/G", value=65.0, step=5.0)
                rec_avg = st.number_input("Season Avg Receptions/G", value=5.5, step=0.5)
                air_yds_share = st.number_input("Air Yards Share (0-1)", min_value=0.0,
                                                 max_value=1.0, value=0.22, step=0.01)
            with col2:
                rec_yds_line = st.number_input("Sportsbook Rec Yards Line", value=62.5, step=0.5)
                rec_line = st.number_input("Sportsbook Receptions Line", value=5.5, step=0.5)
                tgt_share = st.number_input("Target Share (0-1)", min_value=0.0,
                                             max_value=1.0, value=0.22, step=0.01)

            if st.button(f"Project {position} Props"):
                card = build_wr_te_prop_card(
                    player_name=player_name or position,
                    team=team or "TM",
                    position=position,
                    season_stats={
                        "rec_yds_pg": rec_yds_avg, "rec_pg": rec_avg,
                        "air_yards_share": air_yds_share, "target_share": tgt_share,
                    },
                    prop_lines={"rec_yds": rec_yds_line, "receptions": rec_line},
                    opp_def_epa=opp_def_epa,
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rec Yards Projection", f"{card['proj_rec_yds']:.1f}",
                              delta=f"{card.get('edge_rec_yds', 0) or 0:+.1f}% edge")
                with col2:
                    st.metric("Receptions Projection", f"{card['proj_receptions']:.1f}",
                              delta=f"{card.get('edge_receptions', 0) or 0:+.1f}% edge")

    # =========================================================================
    # TAB 5 — Season Totals
    # =========================================================================
    with tab_totals:
        st.subheader("Preseason Win Total Projections")
        st.caption(
            "Model projects wins from prior-season EPA. Flag when projection diverges from Vegas line by 1.5+ wins."
        )

        if team_df.empty:
            st.info("Team EPA data needed for win total projections.")
        else:
            # Build projected wins from EPA composite using logistic model
            # Map EPA composite to win% → multiply by 17 games
            from src.sports.football.models.epa import spread_to_win_prob, epa_composite_to_spread

            projections = []
            for _, row in team_df.iterrows():
                spread_eq = epa_composite_to_spread(row.get("epa_composite", 0))
                win_pct = spread_to_win_prob(spread_eq)
                proj_wins = round(win_pct * 17, 1)
                projections.append({"team": row["team"], "projected_wins": proj_wins})

            proj_df = pd.DataFrame(projections)

            # Merge with Vegas win totals if available
            if not win_totals_df.empty and "team" in win_totals_df.columns and "wins" in win_totals_df.columns:
                proj_df = proj_df.merge(win_totals_df[["team", "wins"]].rename(
                    columns={"wins": "vegas_line"}), on="team", how="left")
                proj_df["edge_wins"] = (proj_df["projected_wins"] - proj_df["vegas_line"]).round(1)
                proj_df["bet"] = proj_df["edge_wins"].apply(
                    lambda e: "OVER" if e >= 1.5 else ("UNDER" if e <= -1.5 else "PASS")
                )
            else:
                st.caption("Vegas win total lines not available for this season.")

            st.dataframe(
                proj_df.rename(columns={
                    "team": "Team", "projected_wins": "Projected Wins",
                    "vegas_line": "Vegas O/U", "edge_wins": "Edge (wins)",
                    "bet": "Signal",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # AI analysis for flagged teams
            if "bet" in proj_df.columns:
                flagged = proj_df[proj_df["bet"] != "PASS"]
                if not flagged.empty:
                    st.subheader("Best Value Win Total Bets")
                    for _, row in flagged.sort_values("edge_wins", key=abs, ascending=False).iterrows():
                        label = f"{'📈' if row['bet'] == 'OVER' else '📉'} **{row['team']}** — {row['bet']} {row.get('vegas_line', '?')} wins (Model: {row['projected_wins']}, Edge: {row.get('edge_wins', '?'):+} wins)"
                        with st.expander(label):
                            prior_epa = team_df[team_df["team"] == row["team"]]["epa_composite"].values
                            if st.button(f"AI Analysis — {row['team']}", key=f"nfl_tot_{row['team']}"):
                                with st.spinner("Asking Groq..."):
                                    result = analyze_nfl_season_total({
                                        "team": row["team"],
                                        "projected_wins": row["projected_wins"],
                                        "vegas_line": row.get("vegas_line", 0),
                                        "bet_direction": row["bet"],
                                        "edge_wins": row.get("edge_wins", 0),
                                        "prior_epa_composite": float(prior_epa[0]) if len(prior_epa) > 0 else None,
                                    })
                                st.markdown(f"**Reasoning:** {result.get('reasoning', '')}")
                                st.caption(f"Confidence: {result.get('confidence', 'N/A')} | Risk: {result.get('key_risk', '')}")
