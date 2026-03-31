"""Games page — today's matchups with edge table, filters, and AI reasoning."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.dashboard.components.edge_table import render_edge_table
from src.dashboard.components.signal_cards import severity_badge
from src.shared.groq_agent import analyze_mlb_game
from src.data.predictions_db import save_predictions


# Status badge HTML
_STATUS_HTML = {
    "Full":    '<span style="background:#2ecc71;color:#000;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">✅ Starters Set</span>',
    "Partial": '<span style="background:#e67e22;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">⚠️ Partial Lineup</span>',
    "None":    '<span style="background:#555;color:#ccc;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">⏳ No Lineup Yet</span>',
}


def _format_matchup(row) -> str:
    return f"{row.get('away_team', '?')} @ {row.get('home_team', '?')}"


def _fmt_odds(val) -> str:
    """Format American odds as string, e.g. +150 or -110."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    v = int(val)
    return f"+{v}" if v > 0 else str(v)


def _fmt_line(point, odds) -> str:
    """Format a spread/run-line entry like '-1.5 (-115)'."""
    if point is None or (isinstance(point, float) and pd.isna(point)):
        return "N/A"
    sign = "+" if float(point) > 0 else ""
    odds_str = f" ({_fmt_odds(odds)})" if odds is not None and not (isinstance(odds, float) and pd.isna(odds)) else ""
    return f"{sign}{point}{odds_str}"


def _clear_odds_cache() -> None:
    from src.data.cache import _cache_path
    for key in ["mlb_odds", f"mlb_starters_{datetime.date.today().isoformat()}"]:
        p = _cache_path(key)
        if p.exists():
            p.unlink()


def _render_bet_cards(bets: pd.DataFrame, bankroll: float,
                      team_stats: pd.DataFrame | None,
                      pitcher_stats: pd.DataFrame | None) -> None:
    if bets.empty:
        st.info("No bets meet the minimum edge threshold today.")
        return

    for _, row in bets.iterrows():
        side = row["best_bet_side"]
        team = row["home_team"] if side == "HOME" else row["away_team"]
        opp = row["away_team"] if side == "HOME" else row["home_team"]
        odds_key = "home_odds" if side == "HOME" else "away_odds"
        kelly_key = "home_kelly_pct" if side == "HOME" else "away_kelly_pct"
        edge_key = "home_edge_pct" if side == "HOME" else "away_edge_pct"
        model_key = "home_model_prob" if side == "HOME" else "away_model_prob"
        implied_key = "home_implied_prob" if side == "HOME" else "away_implied_prob"

        edge = row[edge_key]
        kelly_pct = row[kelly_key]
        dollar_stake = bankroll * kelly_pct / 100
        line = row.get(odds_key, "N/A")
        line_str = f"+{line}" if isinstance(line, (int, float)) and line > 0 else str(line)
        model_prob = row.get(model_key, 0)
        implied_prob = row.get(implied_key, 0)
        status = row.get("lineup_status", "None")

        severity = "High" if edge >= 7 else "Medium" if edge >= 4 else "Low"
        badge = severity_badge(severity, f"{edge:.1f}% edge")

        with st.expander(
            f"{team} vs {opp}  |  {edge:.1f}% edge  |  Line: {line_str}  |  Stake: ${dollar_stake:,.0f}",
            expanded=edge >= 6,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Prob", f"{model_prob:.1%}")
            c2.metric("Market Prob", f"{implied_prob:.1%}")
            c3.metric("Edge", f"{edge:+.1f}%")
            c4.metric("Kelly Stake", f"${dollar_stake:,.0f} ({kelly_pct:.1f}%)")

            # Starter info
            home_starter = row.get("home_starter")
            away_starter = row.get("away_starter")
            s_col1, s_col2, s_col3 = st.columns([1, 2, 2])
            with s_col1:
                st.markdown(_STATUS_HTML.get(status, ""), unsafe_allow_html=True)
            with s_col2:
                st.caption(f"🏠 {row.get('home_team','?')}: **{home_starter or 'TBD'}**")
            with s_col3:
                st.caption(f"✈️ {row.get('away_team','?')}: **{away_starter or 'TBD'}**")

            if status == "None":
                st.caption("⚠️ No starters confirmed — prediction uses season Pythagorean only. Pitcher FIP adjustment not applied.")
            elif status == "Partial":
                st.caption("⚠️ One starter TBD — FIP adjustment applied for confirmed starter only.")

            st.markdown(
                f"**Bet:** {team} &nbsp; {badge} &nbsp; Line: `{line_str}`",
                unsafe_allow_html=True,
            )

            # Build signals dict
            signals = {}
            if team_stats is not None and not team_stats.empty:
                for t, prefix in [(row["home_team"], "home"), (row["away_team"], "away")]:
                    t_row = team_stats[team_stats["team"] == t]
                    if not t_row.empty and "pyth_deviation" in t_row.columns:
                        signals[f"{prefix}_pyth_deviation"] = float(t_row.iloc[0]["pyth_deviation"])

            if pitcher_stats is not None and not pitcher_stats.empty:
                for starter_col, prefix in [("home_starter", "home"), ("away_starter", "away")]:
                    name = row.get(starter_col)
                    if name:
                        p = pitcher_stats[pitcher_stats["name"] == name]
                        if not p.empty:
                            p = p.iloc[0]
                            signals[f"{prefix}_starter"] = name
                            signals[f"{prefix}_starter_fip"] = p.get("fip")
                            signals[f"{prefix}_starter_era"] = p.get("era")
                            if p.get("fip") and p.get("era"):
                                signals[f"{prefix}_fip_era_gap"] = round(float(p["fip"]) - float(p["era"]), 2)
                            signals[f"{prefix}_babip"] = p.get("babip")

            # ── Additional Markets ──────────────────────────────────────
            st.markdown("**Additional Markets**")
            m1, m2, m3, m4 = st.columns(4)

            # Game total
            total = row.get("total_line")
            over_o = row.get("over_odds")
            under_o = row.get("under_odds")
            if total is not None and not (isinstance(total, float) and pd.isna(total)):
                m1.metric("Game Total (O/U)", str(total))
                m1.caption(f"Over {_fmt_odds(over_o)} / Under {_fmt_odds(under_o)}")
            else:
                m1.metric("Game Total", "N/A")

            # Run line (home)
            home_rl = row.get("home_rl")
            away_rl = row.get("away_rl")
            if home_rl is not None and not (isinstance(home_rl, float) and pd.isna(home_rl)):
                m2.metric(
                    "Run Line",
                    f"{row.get('home_team','Home')} {_fmt_line(home_rl, row.get('home_rl_odds'))}",
                )
                m2.caption(f"{row.get('away_team','Away')} {_fmt_line(away_rl, row.get('away_rl_odds'))}")
            else:
                m2.metric("Run Line", "N/A")

            # F5 and F1 — require API plan upgrade
            m3.metric("F5 ML / F5 Total", "—")
            m3.caption("Requires The Odds API 'Pro' plan")
            m4.metric("1st Inning ML", "—")
            m4.caption("Requires The Odds API 'Pro' plan")

            st.divider()

            if st.button("Generate AI Analysis", key=f"ai_{row.get('game_id', team)}"):
                with st.spinner("Analyzing with Llama 3.3 70B..."):
                    result = analyze_mlb_game(
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        model_prob=model_prob,
                        implied_prob=implied_prob,
                        edge_pct=edge,
                        bet_side=f"{team} ({side})",
                        signals=signals,
                    )
                conf = result.get("confidence", "Low")
                conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
                st.markdown(f"{conf_icon} **Confidence: {conf}**")
                st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")


def render(
    games_with_kelly: pd.DataFrame,
    bankroll: float,
    team_stats: pd.DataFrame | None = None,
    pitcher_stats: pd.DataFrame | None = None,
) -> None:
    today = datetime.date.today()
    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.header(f"Today's Games — {today.strftime('%A, %B %-d, %Y')}")
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", help="Fetches fresh odds and lineup announcements"):
            _clear_odds_cache()
            st.cache_data.clear()
            st.rerun()

    if games_with_kelly.empty:
        st.warning("No games loaded. Check your ODDS_API_KEY in .env and ensure there are upcoming MLB games.")
        return

    # Auto-save predictions
    df = games_with_kelly.copy()
    if df["home_model_prob"].notna().any():
        try:
            save_predictions(df, today.isoformat())
        except Exception:
            pass

    # Split: no team stats match vs has model probs
    has_prob = df["home_model_prob"].notna() & df["away_model_prob"].notna()
    ready_df = df[has_prob].copy()
    pending_df = df[~has_prob].copy()

    ready_df["matchup"] = ready_df.apply(_format_matchup, axis=1)

    # ---- Lineup status legend ----
    status_counts = ready_df["lineup_status"].value_counts() if "lineup_status" in ready_df.columns else {}
    full_n = status_counts.get("Full", 0)
    partial_n = status_counts.get("Partial", 0)
    none_n = status_counts.get("None", 0) + len(pending_df)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f'{_STATUS_HTML["Full"]} &nbsp; **{full_n}** game(s)', unsafe_allow_html=True)
    lc2.markdown(f'{_STATUS_HTML["Partial"]} &nbsp; **{partial_n}** game(s)', unsafe_allow_html=True)
    lc3.markdown(f'{_STATUS_HTML["None"]} &nbsp; **{none_n}** game(s)', unsafe_allow_html=True)
    st.divider()

    # ---- Games with no team stats (cannot model at all) ----
    if not pending_df.empty:
        st.subheader(f"⏳ No Team Data — Cannot Predict ({len(pending_df)})")
        st.caption("Team stats could not be matched. These games are shown for reference only.")
        st.dataframe(
            pending_df[[c for c in ["away_team", "home_team", "away_odds", "home_odds"] if c in pending_df.columns]]
            .rename(columns={"away_team": "Away", "home_team": "Home",
                             "away_odds": "Away Line", "home_odds": "Home Line"}),
            use_container_width=True, hide_index=True,
        )
        st.divider()

    if ready_df.empty:
        return

    # ---- Filters ----
    col1, col2, col3 = st.columns(3)
    with col1:
        all_teams = sorted(set(ready_df["home_team"].tolist() + ready_df["away_team"].tolist()))
        selected_team = st.selectbox("Filter by team", ["All"] + all_teams)
    with col2:
        min_edge = st.slider("Min edge %", 0.0, 20.0, 3.0, 0.5)
    with col3:
        show_only_bets = st.checkbox("BET recommendations only", value=False)

    filtered = ready_df.copy()
    if selected_team != "All":
        filtered = filtered[(filtered["home_team"] == selected_team) | (filtered["away_team"] == selected_team)]
    if show_only_bets:
        filtered = filtered[filtered["best_bet_side"] != "PASS"]
    filtered = filtered[filtered["best_bet_edge"] >= min_edge].sort_values("best_bet_edge", ascending=False)

    st.caption(f"{len(filtered)} game(s) shown | Bankroll: ${bankroll:,.0f}")

    # ---- Section 1: Full lineups confirmed ----
    full_games = filtered[filtered.get("lineup_status", pd.Series(dtype=str)) == "Full"] \
        if "lineup_status" in filtered.columns else pd.DataFrame()
    partial_games = filtered[filtered.get("lineup_status", pd.Series(dtype=str)) == "Partial"] \
        if "lineup_status" in filtered.columns else pd.DataFrame()
    no_lineup_games = filtered[filtered.get("lineup_status", pd.Series(dtype=str)) == "None"] \
        if "lineup_status" in filtered.columns else filtered

    # ---- Edge summary table (all filtered games) ----
    display_cols = {
        "matchup": "Matchup",
        "lineup_status": "Lineup",
        "home_starter": "Home SP",
        "away_starter": "Away SP",
        "home_model_prob": "Home Model%",
        "away_model_prob": "Away Model%",
        "home_implied_prob": "Home Mkt%",
        "away_implied_prob": "Away Mkt%",
        "best_bet_side": "Bet Side",
        "best_bet_edge": "Best Edge%",
        "home_odds": "Home ML",
        "away_odds": "Away ML",
        "total_line": "Total",
        "home_rl": "Home RL",
        "away_rl": "Away RL",
    }
    available = {k: v for k, v in display_cols.items() if k in filtered.columns}
    display_df = filtered[list(available.keys())].rename(columns=available)
    for col in ["Home Model%", "Away Model%", "Home Mkt%", "Away Mkt%"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    render_edge_table(display_df, edge_col="Best Edge%")

    # ---- Bet recommendations — grouped by lineup status ----
    st.subheader("Recommended Bets")

    bets = filtered[filtered["best_bet_side"] != "PASS"].sort_values("best_bet_edge", ascending=False)
    if bets.empty:
        st.info("No bets meet the minimum edge threshold today.")
        return

    full_bets = bets[bets.get("lineup_status", pd.Series(dtype=str)) == "Full"] \
        if "lineup_status" in bets.columns else pd.DataFrame()
    other_bets = bets[bets.get("lineup_status", pd.Series(dtype=str)) != "Full"] \
        if "lineup_status" in bets.columns else bets

    if not full_bets.empty:
        st.markdown(f'### {_STATUS_HTML["Full"]} &nbsp; Full Confidence — Both Starters Confirmed', unsafe_allow_html=True)
        st.caption("FIP, xFIP, and BABIP adjustments applied. Highest-confidence predictions.")
        _render_bet_cards(full_bets, bankroll, team_stats, pitcher_stats)

    if not other_bets.empty:
        partial = other_bets[other_bets.get("lineup_status", pd.Series(dtype=str)) == "Partial"] \
            if "lineup_status" in other_bets.columns else pd.DataFrame()
        none_bets = other_bets[other_bets.get("lineup_status", pd.Series(dtype=str)) == "None"] \
            if "lineup_status" in other_bets.columns else other_bets

        if not partial.empty:
            st.markdown(f'### {_STATUS_HTML["Partial"]} &nbsp; Partial Lineup — One Starter Confirmed', unsafe_allow_html=True)
            st.caption("FIP adjustment applied for confirmed starter only. Refresh when second starter is announced.")
            _render_bet_cards(partial, bankroll, team_stats, pitcher_stats)

        if not none_bets.empty:
            st.markdown(f'### {_STATUS_HTML["None"]} &nbsp; No Lineup — Pythagorean Model Only', unsafe_allow_html=True)
            st.caption("No starters announced yet. Prediction based on season run differential only. Use with caution.")
            _render_bet_cards(none_bets, bankroll, team_stats, pitcher_stats)
