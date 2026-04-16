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
from src.data.ballpark import get_bullpen_stats
from src.models.kelly import compute_kelly_for_games
from src.models.pythagorean import compute_pythagorean
from src.models.regression_signals import compute_pitcher_signals, compute_team_signals
from src.models.win_probability import compute_win_probabilities
from src.data.bet_log_db import insert_bet, insert_parlay

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
def load_team_stats(season: int, _v: int = 2) -> pd.DataFrame:
    """_v is a cache-bust version; increment when data source changes."""
    try:
        raw = get_team_stats(season)
        if raw.empty:
            st.error("Could not load team stats: no data returned from MLB Stats API.")
            return pd.DataFrame()
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


@st.cache_data(ttl=3600 * 6, show_spinner="Loading bullpen stats...")
def load_bullpen_stats(season: int) -> pd.DataFrame:
    try:
        return get_bullpen_stats(season)
    except Exception as e:
        st.error(f"Could not load bullpen stats: {e}")
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
        from src.data.ballpark import get_bullpen_stats as _get_bp
        try:
            bullpen = _get_bp(datetime.date.today().year)
        except Exception:
            bullpen = pd.DataFrame()
        games_with_prob = compute_win_probabilities(odds_with_starters, team_stats, pitcher_stats, bullpen_df=bullpen)

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

def _to_decimal(american: float) -> float:
    """Convert American odds to decimal multiplier."""
    if american is None or american == 0:
        return 1.909  # -110 default
    return (american / 100 + 1) if american > 0 else (100 / abs(american) + 1)


def _fmt_american(val) -> str:
    """Format American odds with sign."""
    if val is None or (isinstance(val, float) and (val != val)):
        return "N/A"
    v = int(val)
    return f"+{v}" if v > 0 else str(v)


@st.dialog("Review & Submit Bets")
def _bet_slip_submit_dialog(slip: list, today_str: str) -> None:
    """Modal showing each bet with editable odds/stake before saving."""
    st.markdown(f"**{len(slip)} bet(s) — adjust odds to match your book before saving**")
    st.divider()

    # Build a mutable copy so edits in the form persist within this dialog
    edited = []
    for i, bet in enumerate(slip):
        st.markdown(f"**{bet['matchup']}**")
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.caption(bet["description"])
        with c2:
            new_odds = st.number_input(
                "Odds",
                value=int(bet.get("line") or -110),
                step=5,
                key=f"dlg_odds_{i}",
                help="Edit to match your sportsbook's actual line",
            )
        with c3:
            new_stake = st.number_input(
                "Stake $",
                min_value=1.0,
                value=float(bet.get("stake") or 50.0),
                step=5.0,
                key=f"dlg_stake_{i}",
            )
        edited.append({**bet, "line": float(new_odds), "stake": float(new_stake)})
        if i < len(slip) - 1:
            st.divider()

    # Parlay preview (uses edited odds)
    if len(edited) >= 2:
        st.divider()
        combined_decimal = 1.0
        for bet in edited:
            combined_decimal *= _to_decimal(bet.get("line", -110))
        total_stake = edited[0].get("stake", 50.0)
        payout = total_stake * combined_decimal
        profit = payout - total_stake
        parlay_odds = int((combined_decimal - 1) * 100)
        st.markdown(
            f"**Parlay odds:** +{parlay_odds}  "
            f"| **Payout:** ${payout:,.2f}  |  **Profit:** ${profit:,.2f}"
        )
        st.caption(f"Based on ${total_stake:.0f} stake on first leg.")

    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("✅ Save Singles", type="primary", width="stretch"):
            for bet in edited:
                insert_bet({
                    "date":        today_str,
                    "matchup":     bet["matchup"],
                    "bet_side":    bet["description"],
                    "line":        bet["line"],
                    "stake":       bet["stake"],
                    "edge_pct":    bet.get("edge_pct", 0),
                    "model_prob":  bet.get("model_prob"),
                    "signal_type": "Multiple",
                    "outcome":     "Pending",
                    "pnl":         0.0,
                    "bet_type":    "Single",
                })
            st.session_state.bet_slip = []
            st.success(f"Saved {len(edited)} single bet(s)!")
            st.rerun()

    with btn_col2:
        if st.button("🔗 Save Parlay", type="primary", width="stretch"):
            legs = [
                {
                    "date":        today_str,
                    "matchup":     bet["matchup"],
                    "bet_side":    bet["description"],
                    "line":        bet["line"],
                    "stake":       bet["stake"],
                    "edge_pct":    bet.get("edge_pct", 0),
                    "model_prob":  bet.get("model_prob"),
                    "signal_type": "Multiple",
                    "outcome":     "Pending",
                    "pnl":         0.0,
                }
                for bet in edited
            ]
            insert_parlay(legs)
            st.session_state.bet_slip = []
            st.success(f"Saved {len(legs)}-leg parlay!")
            st.rerun()

    with btn_col3:
        if st.button("Cancel", width="stretch"):
            st.rerun()


def _render_bet_slip_sidebar(bankroll: float) -> None:
    """Render the bet slip in the sidebar."""
    if "bet_slip" not in st.session_state:
        st.session_state.bet_slip = []

    slip = st.session_state.bet_slip
    st.subheader("📋 Bet Slip")

    if not slip:
        st.caption("No bets added yet. Click **+ ML**, **+ RL**, or **+ O/U** on a game card.")
        return

    st.caption(f"{len(slip)} bet(s) selected")

    to_remove = []
    for i, bet in enumerate(slip):
        col_desc, col_del = st.columns([5, 1])
        with col_desc:
            stake_val = st.number_input(
                bet["description"],
                min_value=1.0,
                value=float(bet.get("stake", 50.0)),
                step=5.0,
                key=f"slip_stake_{bet['key']}",
                label_visibility="visible",
            )
            slip[i]["stake"] = stake_val
        with col_del:
            st.write("")
            if st.button("✕", key=f"slip_del_{bet['key']}"):
                to_remove.append(i)

    for idx in reversed(to_remove):
        st.session_state.bet_slip.pop(idx)
    if to_remove:
        st.rerun()

    # ── Parlay payout preview ──────────────────────────────────────
    if len(slip) >= 2:
        combined_decimal = 1.0
        total_stake = slip[0].get("stake", 50.0)
        for bet in slip:
            combined_decimal *= _to_decimal(bet.get("line", -110))
        payout = total_stake * combined_decimal
        profit = payout - total_stake
        st.markdown(
            f"**Parlay odds:** +{int((combined_decimal - 1) * 100)}  "
            f"| **Payout:** ${payout:,.2f}  |  **Profit:** ${profit:,.2f}",
        )
        st.caption(f"Based on ${total_stake:.0f} stake on first leg.")

    st.divider()

    import datetime as _dt
    today_str = str(_dt.date.today())

    sub_col, clr_col = st.columns(2)
    with sub_col:
        if st.button("📤 Submit", width="stretch", type="primary"):
            _bet_slip_submit_dialog(list(slip), today_str)
    with clr_col:
        if st.button("🗑 Clear", width="stretch"):
            st.session_state.bet_slip = []
            st.rerun()


def _render_game_analysis() -> None:
    from src.dashboard.pages import game_analysis as ga_page
    from src.data.ballpark import get_bullpen_stats as _get_bp
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
        bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100)
        st.divider()
        _render_bet_slip_sidebar(bankroll)

    team_stats_df   = load_team_stats(CURRENT_SEASON)
    pitcher_stats_df = load_pitcher_stats(CURRENT_SEASON)
    try:
        bullpen_df = _get_bp(CURRENT_SEASON)
    except Exception:
        bullpen_df = pd.DataFrame()
    games_df, _ = load_games_data(team_stats_df, pitcher_stats_df, bankroll)
    ga_page.render(games_df, team_stats=team_stats_df, pitcher_stats=pitcher_stats_df, bullpen_df=bullpen_df)


def _render_player_analysis() -> None:
    from src.dashboard.pages import player_analysis as pa_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
        bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100)
        st.divider()
        _render_bet_slip_sidebar(bankroll)
    pa_page.render()


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
                ["Games", "Teams", "Pitchers", "Player Props", "Preseason Projections", "Game Analysis", "Player Analysis"],
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

        st.divider()
        _render_bet_slip_sidebar(bankroll)

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
        elif section == "Game Analysis":
            from src.dashboard.pages import game_analysis as ga_page
            from src.data.ballpark import get_bullpen_stats as _get_bp
            try:
                bullpen_df = _get_bp(CURRENT_SEASON)
            except Exception:
                bullpen_df = pd.DataFrame()
            games_df, _ = load_games_data(team_stats_df, pitcher_stats_df, bankroll)
            ga_page.render(games_df, team_stats=team_stats_df, pitcher_stats=pitcher_stats_df, bullpen_df=bullpen_df)
        elif section == "Player Analysis":
            from src.dashboard.pages import player_analysis as pa_page
            pa_page.render()

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


def _render_about() -> None:
    from src.dashboard.pages import about as about_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
    about_page.render()


# ---------------------------------------------------------------------------
# Navigation — replaces Streamlit's auto-discovered pages/ tabs
# ---------------------------------------------------------------------------

pg = st.navigation(
    [
        st.Page(_render_home, title="Home", icon="🏠", default=True),
        st.Page(_render_bet_log, title="Bet Log", icon="📒"),
        st.Page(_render_analysis, title="Analysis", icon="📊"),
        st.Page(_render_about, title="About", icon="📖"),
    ],
    position="sidebar",
)
pg.run()
