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

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #4a0000;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small {
    color: #f0d0d0 !important;
}
[data-testid="stSidebarNav"] a span {
    color: #f0d0d0 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: #ffffff !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

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

        # Tag lineup status — use actual starter name as ground truth (NaN announced flag = True in Python)
        def _lineup_status(row) -> str:
            h = isinstance(row.get("home_starter"), str) and bool(row.get("home_starter"))
            a = isinstance(row.get("away_starter"), str) and bool(row.get("away_starter"))
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
    """Modal — choose side, edit odds/stake, then save as singles, parlay, or both."""
    st.markdown(f"**{len(slip)} bet(s) — choose side, adjust odds to match your book**")
    st.divider()

    edited = []
    for i, bet in enumerate(slip):
        st.markdown(f"**{bet['matchup']}**")
        bet_type = bet.get("bet_type", "ML")

        if bet_type == "RL":
            home_t = bet.get("home_team", "")
            away_t = bet.get("away_team", "")
            has_side = bool(
                home_t and away_t
                and bet.get("home_rl") is not None
                and bet.get("away_rl") is not None
            )
            cols = st.columns([2, 1, 1, 1])
            if has_side:
                with cols[0]:
                    curr_rl_side = bet.get("rl_side", "HOME")
                    chosen_rl = st.selectbox(
                        "Side", [home_t, away_t],
                        index=0 if curr_rl_side == "HOME" else 1,
                        key=f"dlg_rl_side_{i}", label_visibility="collapsed",
                    )
                    is_home_rl = chosen_rl == home_t
                    new_rl_side = "HOME" if is_home_rl else "AWAY"
                    rl_pt_raw = bet.get("home_rl") if is_home_rl else bet.get("away_rl")
                    rl_pt_val = float(rl_pt_raw) if rl_pt_raw is not None else -1.5
                    rl_odds_raw = bet.get("home_rl_odds") if is_home_rl else bet.get("away_rl_odds")
                    rl_odds_def = int(rl_odds_raw) if isinstance(rl_odds_raw, (int, float)) else -110
            else:
                chosen_rl = bet.get("rl_team", "")
                new_rl_side = bet.get("rl_side", "HOME")
                rl_pt_val = float(bet.get("rl_spread") or -1.5)
                rl_odds_def = int(bet.get("line") or -110)
                with cols[0]:
                    st.caption(bet["description"])
            with cols[1]:
                new_spread = st.number_input(
                    "Spread", value=rl_pt_val, step=0.5,
                    key=f"dlg_spread_{i}_{new_rl_side}",
                    help="Adjust run line spread to match your sportsbook",
                )
            with cols[2]:
                new_odds = st.number_input(
                    "Odds", value=rl_odds_def, step=5,
                    key=f"dlg_odds_{i}_{new_rl_side}",
                )
            with cols[3]:
                new_stake = st.number_input(
                    "Stake $", min_value=1.0, value=float(bet.get("stake") or 50.0),
                    step=5.0, key=f"dlg_stake_{i}",
                )
            sign = "+" if new_spread > 0 else ""
            odds_sign = "+" if int(new_odds) > 0 else ""
            new_desc = f"RL: {chosen_rl} {sign}{new_spread:.1f} ({odds_sign}{int(new_odds)})"
            edited.append({
                **bet,
                "line": float(new_odds), "stake": float(new_stake),
                "rl_spread": new_spread, "rl_side": new_rl_side,
                "rl_team": chosen_rl, "description": new_desc,
            })

        elif bet_type == "O/U":
            over_o = bet.get("over_odds")
            under_o = bet.get("under_odds")
            has_side = over_o is not None and under_o is not None
            cols = st.columns([2, 1, 1, 1])
            if has_side:
                with cols[0]:
                    curr_dir = bet.get("ou_direction", "Over")
                    chosen_dir = st.selectbox(
                        "Direction", ["Over", "Under"],
                        index=0 if curr_dir == "Over" else 1,
                        key=f"dlg_ou_dir_{i}", label_visibility="collapsed",
                    )
                    ou_odds_def = int(over_o if chosen_dir == "Over" else under_o)
            else:
                chosen_dir = bet.get("ou_direction", "Over")
                ou_odds_def = int(bet.get("line") or -110)
                with cols[0]:
                    st.caption(bet["description"])
            with cols[1]:
                new_total = st.number_input(
                    "Total", value=float(bet.get("ou_total") or 8.5), step=0.5,
                    key=f"dlg_total_{i}",
                    help="Adjust O/U line to match your sportsbook",
                )
            with cols[2]:
                new_odds = st.number_input(
                    "Odds", value=ou_odds_def, step=5,
                    key=f"dlg_odds_{i}_{chosen_dir}",
                )
            with cols[3]:
                new_stake = st.number_input(
                    "Stake $", min_value=1.0, value=float(bet.get("stake") or 50.0),
                    step=5.0, key=f"dlg_stake_{i}",
                )
            odds_sign = "+" if int(new_odds) > 0 else ""
            new_desc = f"{chosen_dir} {new_total:.1f} ({odds_sign}{int(new_odds)})"
            edited.append({
                **bet,
                "line": float(new_odds), "stake": float(new_stake),
                "ou_total": new_total, "ou_direction": chosen_dir, "description": new_desc,
            })

        else:  # ML
            home_t = bet.get("home_team", "")
            away_t = bet.get("away_team", "")
            has_side = bool(
                home_t and away_t
                and bet.get("home_odds") is not None
                and bet.get("away_odds") is not None
            )
            cols = st.columns([2, 1, 1]) if has_side else st.columns([3, 1, 1])
            chosen_ml = None
            new_side = bet.get("bet_side", "HOME")
            ml_default = int(bet.get("line") or -110)
            if has_side:
                with cols[0]:
                    curr_side = bet.get("bet_side", "HOME")
                    chosen_ml = st.selectbox(
                        "Side", [home_t, away_t],
                        index=0 if curr_side == "HOME" else 1,
                        key=f"dlg_ml_side_{i}", label_visibility="collapsed",
                    )
                    is_home = chosen_ml == home_t
                    new_side = "HOME" if is_home else "AWAY"
                    ml_default = int(bet.get("home_odds") if is_home else bet.get("away_odds") or -110)
            else:
                with cols[0]:
                    st.caption(bet["description"])
            with cols[1]:
                new_odds = st.number_input(
                    "Odds", value=ml_default, step=5,
                    key=f"dlg_odds_{i}_{new_side}",
                    help="Edit to match your sportsbook's actual line",
                )
            with cols[2]:
                new_stake = st.number_input(
                    "Stake $", min_value=1.0, value=float(bet.get("stake") or 50.0),
                    step=5.0, key=f"dlg_stake_{i}",
                )
            if has_side and chosen_ml:
                odds_sign = "+" if int(new_odds) > 0 else ""
                new_desc = f"ML: {chosen_ml} {odds_sign}{int(new_odds)}"
            else:
                new_desc = bet["description"]
            edited.append({
                **bet,
                "line": float(new_odds), "stake": float(new_stake),
                "bet_side": new_side, "description": new_desc,
            })

        if i < len(slip) - 1:
            st.divider()

    # Parlay preview
    parlay_total_stake = None
    if len(edited) >= 2:
        st.divider()
        combined_decimal = 1.0
        for bet in edited:
            combined_decimal *= _to_decimal(bet.get("line", -110))
        parlay_total_stake = st.number_input(
            "Parlay Total Wagered ($)",
            min_value=1.0,
            value=float(edited[0].get("stake") or 50.0),
            step=5.0,
            key="dlg_parlay_total_stake",
            help="Single amount wagered on the entire parlay",
        )
        payout = parlay_total_stake * combined_decimal
        profit = payout - parlay_total_stake
        parlay_odds = int((combined_decimal - 1) * 100)
        st.markdown(
            f"**Parlay odds:** +{parlay_odds}  "
            f"| **Payout:** ${payout:,.2f}  |  **Profit:** ${profit:,.2f}"
        )
        st.caption(f"Win: +${profit:,.2f}  ·  Loss: -${parlay_total_stake:,.2f}")

    # Build parlay legs once — shared by Parlay and Both buttons
    legs = [
        {
            "date":        today_str,
            "matchup":     bet["matchup"],
            "bet_side":    bet["description"],
            "line":        bet["line"],
            "stake":       bet.get("stake", 50.0),
            "edge_pct":    bet.get("edge_pct", 0),
            "model_prob":  bet.get("model_prob"),
            "signal_type": "Multiple",
            "outcome":     "Pending",
            "pnl":         0.0,
        }
        for bet in edited
    ]
    parlay_ok = len(edited) >= 2 and parlay_total_stake is not None

    st.divider()
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

    with btn_c1:
        if st.button("✅ Singles", type="primary", width="stretch"):
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

    with btn_c2:
        if st.button("🔗 Parlay", type="primary", width="stretch", disabled=not parlay_ok):
            insert_parlay(legs, total_stake=parlay_total_stake)
            st.session_state.bet_slip = []
            st.success(f"Saved {len(legs)}-leg parlay! Total wagered: ${parlay_total_stake:,.2f}")
            st.rerun()

    with btn_c3:
        if st.button("💾 Both", type="primary", width="stretch", disabled=not parlay_ok):
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
            insert_parlay(legs, total_stake=parlay_total_stake)
            st.session_state.bet_slip = []
            st.success(f"Saved {len(edited)} singles + {len(legs)}-leg parlay!")
            st.rerun()

    with btn_c4:
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
    from src.dashboard.sections import game_analysis as ga_page
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
    from src.dashboard.sections import player_analysis as pa_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
        bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100)
        st.divider()
        _render_bet_slip_sidebar(bankroll)
    pa_page.render()


def _render_home() -> None:
    from src.dashboard.sections import games as games_page
    from src.dashboard.sections import teams as teams_page
    from src.dashboard.sections import pitchers as pitchers_page
    from src.dashboard.sections import props as props_page
    from src.dashboard.sections import preseason as preseason_page
    from src.dashboard.sections import basketball as basketball_page
    from src.dashboard.sections import football as football_page

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
            from src.dashboard.sections import game_analysis as ga_page
            from src.data.ballpark import get_bullpen_stats as _get_bp
            try:
                bullpen_df = _get_bp(CURRENT_SEASON)
            except Exception:
                bullpen_df = pd.DataFrame()
            games_df, _ = load_games_data(team_stats_df, pitcher_stats_df, bankroll)
            ga_page.render(games_df, team_stats=team_stats_df, pitcher_stats=pitcher_stats_df, bullpen_df=bullpen_df)
        elif section == "Player Analysis":
            from src.dashboard.sections import player_analysis as pa_page
            pa_page.render()

    elif sport == "🏀 Basketball":
        basketball_page.render(bankroll=bankroll)
    elif sport == "🏈 Football":
        football_page.render(bankroll=bankroll)


def _render_analysis() -> None:
    from src.dashboard.sections import analysis as analysis_page
    with st.sidebar:
        st.title("🎯 Ballistic")
        st.divider()
    analysis_page.render()


def _render_about() -> None:
    from src.dashboard.sections import about as about_page
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
        st.Page(_render_analysis, title="Analysis", icon="📊"),
        st.Page(_render_about, title="About", icon="📖"),
    ],
    position="sidebar",
)
pg.run()
