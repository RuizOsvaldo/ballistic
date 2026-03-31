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
from src.data.game_results import get_probable_starters
from src.models.kelly import compute_kelly_for_games
from src.models.pythagorean import compute_pythagorean
from src.models.regression_signals import compute_pitcher_signals, compute_team_signals
from src.models.win_probability import compute_win_probabilities

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ballistic",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENT_SEASON = datetime.datetime.now().year

# ---------------------------------------------------------------------------
# Data loaders (shared across pages)
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


def load_games_data(team_stats: pd.DataFrame, pitcher_stats: pd.DataFrame, bankroll: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (games_df, raw_odds).
    games_df includes model probs, Kelly stakes, probable starters, and a
    'lineup_status' column: 'Full' | 'Partial' | 'None'
    """
    try:
        raw_odds = get_mlb_odds()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    if raw_odds.empty or team_stats.empty:
        return pd.DataFrame(), raw_odds

    try:
        # Merge probable starters into odds DataFrame
        starters = get_probable_starters(datetime.date.today())
        odds_with_starters = raw_odds.copy()
        if not starters.empty:
            odds_with_starters = odds_with_starters.merge(
                starters[["home_team", "away_team", "home_starter", "away_starter",
                           "home_starter_announced", "away_starter_announced"]],
                on=["home_team", "away_team"],
                how="left",
            )
        else:
            odds_with_starters["home_starter"] = None
            odds_with_starters["away_starter"] = None
            odds_with_starters["home_starter_announced"] = False
            odds_with_starters["away_starter_announced"] = False

        # Tag lineup status
        def _lineup_status(row) -> str:
            h = bool(row.get("home_starter_announced"))
            a = bool(row.get("away_starter_announced"))
            if h and a:
                return "Full"
            if h or a:
                return "Partial"
            return "None"

        odds_with_starters["lineup_status"] = odds_with_starters.apply(_lineup_status, axis=1)

        # Compute win probabilities (FIP adjustment applies only where starters are known)
        games_with_prob = compute_win_probabilities(odds_with_starters, team_stats, pitcher_stats)

        ready = games_with_prob.dropna(subset=["home_model_prob", "away_model_prob"])
        games_df = compute_kelly_for_games(ready, bankroll=bankroll)

        # Re-attach any games where team stats couldn't be matched
        pending = games_with_prob[
            games_with_prob["home_model_prob"].isna() | games_with_prob["away_model_prob"].isna()
        ]
        if not pending.empty:
            games_df = pd.concat([games_df, pending], ignore_index=True)

        return games_df, raw_odds

    except Exception as e:
        st.error(f"Could not compute game predictions: {e}")
        return pd.DataFrame(), raw_odds


# ---------------------------------------------------------------------------
# Page render functions
# ---------------------------------------------------------------------------

def _render_home() -> None:
    from src.dashboard.pages import games as games_page
    from src.dashboard.pages import teams as teams_page
    from src.dashboard.pages import pitchers as pitchers_page
    from src.dashboard.pages import props as props_page
    from src.dashboard.pages import preseason as preseason_page
    from src.dashboard.pages import basketball as basketball_page
    from src.dashboard.pages import football as football_page

    with st.sidebar:
        st.title("🎯 Ballistic")
        st.caption("Quantitative bet analysis")
        st.divider()

        sport = st.selectbox(
            "Sport",
            ["⚾ Baseball", "🏀 Basketball", "🏈 Football"],
        )

        if sport == "⚾ Baseball":
            st.divider()
            section = st.selectbox(
                "Section",
                ["Games", "Teams", "Pitchers", "Player Props", "Preseason Projections"],
            )
            st.divider()
            st.subheader("Kelly Calculator")
            bankroll = st.number_input(
                "Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100,
            )
        else:
            section = "overview"
            bankroll = 1000

        st.divider()
        st.caption(f"Season: {CURRENT_SEASON}")
        if st.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            st.rerun()

    if sport == "⚾ Baseball":
        team_stats_df = load_team_stats(CURRENT_SEASON)
        pitcher_stats_df = load_pitcher_stats(CURRENT_SEASON)

        if section == "Games":
            games_df, _ = load_games_data(team_stats_df, pitcher_stats_df, bankroll)
            games_page.render(games_df, bankroll, team_stats=team_stats_df, pitcher_stats=pitcher_stats_df)
        elif section == "Teams":
            teams_page.render(team_stats_df)
        elif section == "Pitchers":
            pitchers_page.render(pitcher_stats_df)
        elif section == "Player Props":
            props_page.render()
        elif section == "Preseason Projections":
            preseason_page.render()

    elif sport == "🏀 Basketball":
        basketball_page.render(bankroll=bankroll)
    elif sport == "🏈 Football":
        football_page.render(bankroll=bankroll)


def _render_bet_log() -> None:
    from src.dashboard.pages import bet_log as bet_log_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
    bet_log_page.render()


def _render_analysis() -> None:
    from src.dashboard.pages import analysis as analysis_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
    analysis_page.render()


# ---------------------------------------------------------------------------
# Navigation — replaces Streamlit's auto-discovered pages/ tabs
# ---------------------------------------------------------------------------

pg = st.navigation(
    [
        st.Page(_render_home, title="Home", icon="🏠", default=True),
        st.Page(_render_bet_log, title="Bet Log", icon="📒"),
        st.Page(_render_analysis, title="Analysis", icon="📊"),
    ],
    position="sidebar",
)
pg.run()
