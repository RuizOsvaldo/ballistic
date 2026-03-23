"""Ballistic Dashboard — multi-sport betting analytics."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.baseball_stats import get_pitcher_stats, get_team_stats
from src.data.odds import get_mlb_odds
from src.models.kelly import compute_kelly_for_games
from src.models.pythagorean import compute_pythagorean
from src.models.regression_signals import compute_pitcher_signals, compute_team_signals
from src.models.win_probability import compute_win_probabilities
from src.dashboard.pages import games as games_page
from src.dashboard.pages import teams as teams_page
from src.dashboard.pages import pitchers as pitchers_page
from src.dashboard.pages import bet_log as bet_log_page
from src.dashboard.pages import props as props_page
from src.dashboard.pages import preseason as preseason_page
from src.dashboard.pages import basketball as basketball_page
from src.dashboard.pages import football as football_page

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ballistic",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENT_SEASON = datetime.datetime.now().year

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎯 Ballistic")
    st.caption("Quantitative bet analysis")
    st.divider()

    sport = st.radio(
        "Sport",
        ["⚾ Baseball", "🏀 Basketball", "🏈 Football"],
        index=0,
    )

    st.divider()

    if sport == "⚾ Baseball":
        page = st.radio(
            "Section",
            ["Games", "Teams", "Pitchers", "Player Props", "Preseason Projections", "Bet Log"],
            index=0,
        )
    else:
        page = "overview"

    st.divider()

    # Kelly / bankroll calculator (baseball only)
    if sport == "⚾ Baseball":
        st.subheader("Kelly Calculator")
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=100,
            max_value=1_000_000,
            value=1_000,
            step=100,
        )
        min_edge_sidebar = st.slider(
            "Min edge to show",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.5,
            key="sidebar_min_edge",
        )
        st.divider()
    else:
        bankroll = 1000

    st.caption(f"Season: {CURRENT_SEASON}")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Data loading — baseball
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600 * 6, show_spinner="Loading team stats...")
def load_team_stats(season: int) -> pd.DataFrame:
    try:
        raw = get_team_stats(season)
        df = compute_pythagorean(raw)
        return compute_team_signals(df)
    except Exception as e:
        st.error(f"Could not load team stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading pitcher stats...")
def load_pitcher_stats(season: int) -> pd.DataFrame:
    try:
        raw = get_pitcher_stats(season)
        return compute_pitcher_signals(raw)
    except Exception as e:
        st.error(f"Could not load pitcher stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 2, show_spinner="Loading odds...")
def load_games(team_stats: pd.DataFrame, pitcher_stats: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    try:
        odds_df = get_mlb_odds()
        if odds_df.empty:
            return pd.DataFrame()
        games_with_prob = compute_win_probabilities(odds_df, team_stats, pitcher_stats)
        games_with_prob = games_with_prob.dropna(subset=["home_model_prob", "away_model_prob"])
        return compute_kelly_for_games(games_with_prob, bankroll=bankroll)
    except Exception as e:
        st.error(f"Could not load games/odds: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Route to page
# ---------------------------------------------------------------------------

if sport == "⚾ Baseball":
    team_stats_df = load_team_stats(CURRENT_SEASON)
    pitcher_stats_df = load_pitcher_stats(CURRENT_SEASON)

    if page in ("Games",):
        games_df = load_games(team_stats_df, pitcher_stats_df, bankroll)

    if page == "Games":
        games_page.render(games_df, bankroll, team_stats=team_stats_df, pitcher_stats=pitcher_stats_df)
    elif page == "Teams":
        teams_page.render(team_stats_df)
    elif page == "Pitchers":
        pitchers_page.render(pitcher_stats_df)
    elif page == "Player Props":
        props_page.render()
    elif page == "Preseason Projections":
        preseason_page.render()
    elif page == "Bet Log":
        bet_log_page.render()

elif sport == "🏀 Basketball":
    basketball_page.render(bankroll=bankroll)

elif sport == "🏈 Football":
    football_page.render(bankroll=bankroll)
