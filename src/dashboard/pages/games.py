"""Games page — today's matchups with edge table, filters, and AI reasoning."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.dashboard.components.edge_table import render_edge_table
from src.dashboard.components.signal_cards import severity_badge
from src.shared.groq_agent import analyze_mlb_game


def _format_matchup(row) -> str:
    return f"{row.get('away_team', '?')} @ {row.get('home_team', '?')}"


def render(games_with_kelly: pd.DataFrame, bankroll: float, team_stats: pd.DataFrame = None, pitcher_stats: pd.DataFrame = None) -> None:
    st.header("Today's Games — Bet Edge Finder")

    if games_with_kelly.empty:
        st.warning(
            "No games loaded. Check your ODDS_API_KEY in .env and ensure there are upcoming MLB games."
        )
        return

    df = games_with_kelly.copy()
    df["matchup"] = df.apply(_format_matchup, axis=1)

    # ---- Filters ----
    col1, col2, col3 = st.columns(3)
    with col1:
        all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        selected_team = st.selectbox("Filter by team", ["All"] + all_teams)
    with col2:
        min_edge = st.slider("Min edge %", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    with col3:
        show_only_bets = st.checkbox("Show only BET recommendations", value=False)

    if selected_team != "All":
        df = df[(df["home_team"] == selected_team) | (df["away_team"] == selected_team)]
    if show_only_bets:
        df = df[df["best_bet_side"] != "PASS"]
    df = df[df["best_bet_edge"] >= min_edge].sort_values("best_bet_edge", ascending=False)

    st.caption(f"Showing {len(df)} game(s) | Bankroll: ${bankroll:,.0f}")

    # ---- Edge table ----
    display_cols = {
        "matchup": "Matchup",
        "home_model_prob": "Home Model%",
        "away_model_prob": "Away Model%",
        "home_implied_prob": "Home Mkt%",
        "away_implied_prob": "Away Mkt%",
        "home_edge_pct": "Home Edge%",
        "away_edge_pct": "Away Edge%",
        "best_bet_side": "Bet Side",
        "best_bet_edge": "Best Edge%",
        "home_odds": "Home Line",
        "away_odds": "Away Line",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    display_df = df[list(available.keys())].rename(columns=available)

    for col in ["Home Model%", "Away Model%", "Home Mkt%", "Away Mkt%"]:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    render_edge_table(display_df, edge_col="Best Edge%")

    # ---- Bet recommendations with AI reasoning ----
    st.subheader("Recommended Bets")
    bets = games_with_kelly[games_with_kelly["best_bet_side"] != "PASS"].sort_values(
        "best_bet_edge", ascending=False
    )

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

            st.markdown(
                f"**Bet:** {team} &nbsp; {badge} &nbsp; Line: `{line_str}`",
                unsafe_allow_html=True,
            )

            # Build signals dict from available data
            signals = {}
            if team_stats is not None and not team_stats.empty:
                home_row = team_stats[team_stats["team"] == row["home_team"]]
                away_row = team_stats[team_stats["team"] == row["away_team"]]
                if not home_row.empty and "pyth_deviation" in home_row.columns:
                    signals["home_pyth_deviation"] = float(home_row.iloc[0]["pyth_deviation"])
                if not away_row.empty and "pyth_deviation" in away_row.columns:
                    signals["away_pyth_deviation"] = float(away_row.iloc[0]["pyth_deviation"])

            if pitcher_stats is not None and not pitcher_stats.empty:
                for starter_col, prefix in [("home_starter", "home"), ("away_starter", "away")]:
                    starter_name = row.get(starter_col)
                    if starter_name:
                        p = pitcher_stats[pitcher_stats["name"] == starter_name]
                        if not p.empty:
                            p = p.iloc[0]
                            signals[f"{prefix}_starter"] = starter_name
                            signals[f"{prefix}_starter_fip"] = p.get("fip")
                            signals[f"{prefix}_starter_era"] = p.get("era")
                            if p.get("fip") and p.get("era"):
                                signals[f"{prefix}_fip_era_gap"] = round(p["fip"] - p["era"], 2)
                            signals[f"{prefix}_babip"] = p.get("babip")

            if st.button(f"Generate AI Analysis", key=f"ai_{row.get('game_id', team)}"):
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
                confidence_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                conf = result.get("confidence", "Low")
                st.markdown(f"{confidence_colors.get(conf, '⚪')} **Confidence: {conf}**")
                st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")
