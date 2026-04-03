"""
Game Analysis page — deep-dive breakdown of every today's game prediction.

Shows the full model logic: Pythagorean baseline, starter FIP, bullpen,
park factor, win probability, O/U projection, and AI reasoning.
Auto-generates analysis for every game — no button required.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.pages.games import _fmt_odds, _fmt_line, _build_runs_per_game, _proj_total_for
from src.data.ballpark import PARK_FACTORS, get_park_factor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal_bar(label: str, value: float, league_avg: float, low: float, high: float,
                higher_is_better: bool = True) -> None:
    """Render a labeled metric with a colored delta vs league average."""
    delta = value - league_avg
    delta_str = f"{delta:+.3f} vs lg avg"
    if higher_is_better:
        color = "normal" if delta > 0 else "inverse"
    else:
        color = "inverse" if delta > 0 else "normal"
    st.metric(label, f"{value:.3f}", delta=delta_str, delta_color=color)


def _confidence_badge(confidence: str) -> str:
    colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
    icons  = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    c = colors.get(confidence, "#888")
    i = icons.get(confidence, "⚪")
    return (
        f'<span style="background:{c};color:#000;padding:3px 10px;'
        f'border-radius:8px;font-weight:bold;font-size:13px">{i} {confidence} Confidence</span>'
    )


def _win_prob_chart(home: str, away: str,
                   home_model: float, away_model: float,
                   home_implied: float, away_implied: float) -> go.Figure:
    fig = go.Figure()
    categories = [away, home]
    fig.add_trace(go.Bar(
        name="Model Prob",
        x=categories,
        y=[away_model * 100, home_model * 100],
        marker_color=["#3498db", "#2ecc71"],
        text=[f"{away_model:.1%}", f"{home_model:.1%}"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Market Implied",
        x=categories,
        y=[away_implied * 100, home_implied * 100],
        marker_color=["rgba(52,152,219,0.3)", "rgba(46,204,113,0.3)"],
        text=[f"{away_implied:.1%}", f"{home_implied:.1%}"],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=260,
        yaxis=dict(title="Win Probability %", range=[0, 85]),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=10, b=10),
    )
    return fig


def _render_starter_card(name: str | None, team: str,
                          pitcher_stats: pd.DataFrame | None) -> None:
    if not name or pitcher_stats is None or pitcher_stats.empty:
        st.caption(f"**{team}:** {name or 'TBD'} — no stats available")
        return

    row = pitcher_stats[pitcher_stats["name"] == name]
    if row.empty:
        st.caption(f"**{team}:** {name} — not in FanGraphs leaderboard yet")
        return
    r = row.iloc[0]

    era  = r.get("era")
    fip  = r.get("fip")
    xfip = r.get("xfip")
    babip = r.get("babip")
    k_pct = r.get("k_pct")
    ip    = r.get("ip")
    fip_era_gap = round(float(fip) - float(era), 2) if fip and era else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ERA",   f"{era:.2f}"  if era   is not None else "—")
    c2.metric("FIP",   f"{fip:.2f}"  if fip   is not None else "—")
    c3.metric("xFIP",  f"{xfip:.2f}" if xfip  is not None else "—")
    c4.metric("K%",    f"{k_pct:.1%}" if k_pct is not None else "—")
    c5.metric("BABIP", f"{babip:.3f}" if babip is not None else "—")

    if fip_era_gap is not None:
        if fip_era_gap > 0.75:
            st.warning(
                f"⚠️ FIP-ERA gap: **+{fip_era_gap:.2f}** — ERA is artificially low, "
                f"expect regression upward. Model fades {team}.",
                icon=None,
            )
        elif fip_era_gap < -0.75:
            st.success(
                f"✅ FIP-ERA gap: **{fip_era_gap:.2f}** — ERA overstates true performance, "
                f"pitcher is better than numbers look. Model backs {team}.",
                icon=None,
            )
        else:
            st.caption(f"FIP-ERA gap: {fip_era_gap:+.2f} — ERA tracking FIP closely (stable signal)")

    if babip is not None:
        if babip > 0.320:
            st.caption(f"🔴 BABIP {babip:.3f} → pitcher has been lucky, hits will normalize upward")
        elif babip < 0.275:
            st.caption(f"🟢 BABIP {babip:.3f} → pitcher has been unlucky, expect ERA to improve")


def _render_bullpen_card(team: str, bullpen_df: pd.DataFrame | None,
                          league_avg_bp_fip: float) -> None:
    if bullpen_df is None or bullpen_df.empty:
        st.caption(f"No bullpen data available for {team}")
        return
    row = bullpen_df[bullpen_df["team"] == team]
    if row.empty:
        st.caption(f"{team} bullpen: not in dataset")
        return
    r = row.iloc[0]
    bp_era = r.get("bullpen_era")
    bp_fip = r.get("bullpen_fip")
    bp_ip  = r.get("bullpen_ip")
    bp_k   = r.get("bullpen_k_pct")

    c1, c2, c3 = st.columns(3)
    c1.metric("Bullpen ERA", f"{bp_era:.2f}" if bp_era else "—")
    c2.metric("Bullpen FIP", f"{bp_fip:.2f}" if bp_fip else "—")
    c3.metric("K%",          f"{bp_k:.1%}"   if bp_k  else "—")

    if bp_fip is not None:
        diff = bp_fip - league_avg_bp_fip
        if diff > 0.5:
            st.caption(f"🔴 Bullpen FIP {bp_fip:.2f} (+{diff:.2f} vs lg avg) — weak pen, adds runs to total")
        elif diff < -0.5:
            st.caption(f"🟢 Bullpen FIP {bp_fip:.2f} ({diff:.2f} vs lg avg) — strong pen, suppresses total")
        else:
            st.caption(f"Bullpen FIP {bp_fip:.2f} — near league average ({league_avg_bp_fip:.2f})")


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_ai_analysis(home: str, away: str, model_prob: float, implied_prob: float,
                         edge: float, bet_side: str, signals_json: str) -> dict:
    """Cache AI reasoning per game for 30 min — avoids re-calling Groq on every rerender."""
    import json as _json
    from src.shared.groq_agent import analyze_mlb_game
    signals = _json.loads(signals_json)
    return analyze_mlb_game(
        home_team=home, away_team=away,
        model_prob=model_prob, implied_prob=implied_prob,
        edge_pct=edge, bet_side=bet_side, signals=signals,
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render(
    games_df: pd.DataFrame,
    team_stats: pd.DataFrame | None = None,
    pitcher_stats: pd.DataFrame | None = None,
    bullpen_df: pd.DataFrame | None = None,
) -> None:
    today = datetime.date.today()
    st.header(f"Game Analysis — {today.strftime('%A, %B %-d, %Y')}")
    st.caption(
        "Full model breakdown for every game today: Pythagorean baseline, "
        "starter FIP signals, bullpen strength, park factor, and AI reasoning."
    )

    if games_df.empty:
        st.warning("No games loaded. Check ODDS_API_KEY in .env.")
        return

    has_prob = games_df["home_model_prob"].notna() & games_df["away_model_prob"].notna()
    df = games_df[has_prob].copy()

    if df.empty:
        st.warning("Team stats could not be matched — cannot run model analysis today.")
        return

    # Sort by edge descending so top picks appear first
    if "best_bet_edge" in df.columns:
        df = df.sort_values("best_bet_edge", ascending=False)

    rpg = _build_runs_per_game(team_stats)

    # League avg bullpen FIP for comparisons
    league_avg_bp_fip = 4.20
    if bullpen_df is not None and not bullpen_df.empty and "bullpen_fip" in bullpen_df.columns:
        v = bullpen_df["bullpen_fip"].dropna()
        if not v.empty:
            league_avg_bp_fip = float(v.mean())

    # ── Game selector ─────────────────────────────────────────────────────────
    matchups = [f"{r['away_team']} @ {r['home_team']}" for _, r in df.iterrows()]
    selected = st.selectbox("Select a game to analyze", matchups)
    st.divider()

    row = df[df.apply(
        lambda r: f"{r['away_team']} @ {r['home_team']}" == selected, axis=1
    )].iloc[0]

    home  = row["home_team"]
    away  = row["away_team"]
    side  = row.get("best_bet_side", "PASS")
    edge  = row.get("best_bet_edge", 0.0)
    bet_team = home if side == "HOME" else away if side != "PASS" else None
    home_prob   = float(row.get("home_model_prob", 0.5))
    away_prob   = float(row.get("away_model_prob", 0.5))
    home_impl   = float(row.get("home_implied_prob", 0.5))
    away_impl   = float(row.get("away_implied_prob", 0.5))
    home_ml     = _fmt_odds(row.get("home_odds"))
    away_ml     = _fmt_odds(row.get("away_odds"))
    total       = row.get("total_line")
    over_o      = _fmt_odds(row.get("over_odds"))
    under_o     = _fmt_odds(row.get("under_odds"))
    lineup      = row.get("lineup_status", "None")
    home_sp     = row.get("home_starter") or "TBD"
    away_sp     = row.get("away_starter") or "TBD"
    park_factor = get_park_factor(home)
    proj_total  = _proj_total_for(row, rpg)

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.subheader(f"{away} @ {home}")
        st.caption(f"Starters: {away_sp} (away) vs {home_sp} (home)  |  Lineup: {lineup}")
    with h2:
        if side != "PASS" and bet_team:
            bet_odds = home_ml if side == "HOME" else away_ml
            st.markdown(
                f'<div style="background:#1a3a1a;padding:12px;border-radius:10px;text-align:center">'
                f'<div style="color:#2ecc71;font-size:11px;font-weight:bold">MODEL PICK</div>'
                f'<div style="color:#fff;font-size:18px;font-weight:bold">{bet_team}</div>'
                f'<div style="color:#aaa;font-size:13px">{bet_odds}  |  {edge:+.1f}% edge</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#2a2a2a;padding:12px;border-radius:10px;text-align:center">'
                '<div style="color:#aaa;font-size:14px">NO BET — below 3% edge threshold</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Section 1: Win Probability ────────────────────────────────────────────
    st.subheader("Win Probability Breakdown")
    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        fig = _win_prob_chart(home, away, home_prob, away_prob, home_impl, away_impl)
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric(f"{away} Model Prob", f"{away_prob:.1%}",
                  delta=f"{(away_prob - away_impl)*100:+.1f}% vs market")
        st.metric(f"{home} Model Prob", f"{home_prob:.1%}",
                  delta=f"{(home_prob - home_impl)*100:+.1f}% vs market")
        st.caption("**Edge** = Model Prob − Market Implied Prob")
        st.caption("Positive delta = model sees more value than market prices in.")

    st.divider()

    # ── Section 2: Pythagorean Signal ─────────────────────────────────────────
    st.subheader("Team Quality — Pythagorean Signal")
    st.caption(
        "Season RS/RA is a more stable measure of team quality than win-loss record. "
        "Teams significantly over or under their Pythagorean expectation tend to regress."
    )

    py_c1, py_c2 = st.columns(2)
    if team_stats is not None and not team_stats.empty:
        for col, team in [(py_c1, away), (py_c2, home)]:
            with col:
                t = team_stats[team_stats["team"] == team]
                if not t.empty:
                    r2 = t.iloc[0]
                    g = max(float(r2.get("wins", 0) or 0) + float(r2.get("losses", 0) or 0), 1)
                    rs_pg = float(r2.get("runs_scored", 0) or 0) / g
                    ra_pg = float(r2.get("runs_allowed", 0) or 0) / g
                    pyth  = r2.get("pyth_win_pct")
                    actual = r2.get("win_pct")
                    dev   = r2.get("pyth_deviation")
                    signal = r2.get("pyth_signal", "—")

                    st.markdown(f"**{team}**")
                    m1, m2 = st.columns(2)
                    m1.metric("RS/game",  f"{rs_pg:.2f}")
                    m2.metric("RA/game",  f"{ra_pg:.2f}")
                    m1.metric("Actual W%", f"{actual:.1%}" if actual else "—")
                    m2.metric("Pyth W%",  f"{pyth:.1%}"   if pyth   else "—")

                    if dev is not None:
                        if dev > 0.05:
                            st.warning(
                                f"⚠️ **Overperforming** by {dev:+.1%} — wins above Pythagorean. "
                                f"Expect regression toward {pyth:.1%} W%. Model reduces confidence.",
                                icon=None,
                            )
                        elif dev < -0.05:
                            st.success(
                                f"✅ **Underperforming** by {dev:+.1%} — runs suggest better team "
                                f"than record shows. Model increases confidence.",
                                icon=None,
                            )
                        else:
                            st.caption(f"On track — W% closely matches Pythagorean expectation ({dev:+.1%})")
                else:
                    st.caption(f"No season data for {team}")
    st.divider()

    # ── Section 3: Starting Pitchers ──────────────────────────────────────────
    st.subheader("Starting Pitcher Analysis")
    sp_c1, sp_c2 = st.columns(2)
    with sp_c1:
        st.markdown(f"**{away} — {away_sp}**")
        _render_starter_card(away_sp if away_sp != "TBD" else None, away, pitcher_stats)
    with sp_c2:
        st.markdown(f"**{home} — {home_sp}**")
        _render_starter_card(home_sp if home_sp != "TBD" else None, home, pitcher_stats)

    st.divider()

    # ── Section 4: Bullpen ────────────────────────────────────────────────────
    st.subheader("Bullpen Strength")
    st.caption("Bullpens cover ~33% of innings. Strong pens suppress totals; weak pens inflate them.")
    bp_c1, bp_c2 = st.columns(2)
    with bp_c1:
        st.markdown(f"**{away} Bullpen**")
        _render_bullpen_card(away, bullpen_df, league_avg_bp_fip)
    with bp_c2:
        st.markdown(f"**{home} Bullpen**")
        _render_bullpen_card(home, bullpen_df, league_avg_bp_fip)

    st.divider()

    # ── Section 5: Park Factor & Game Total ───────────────────────────────────
    st.subheader("Park Factor & Game Total Projection")
    pf_pct = (park_factor - 1.0) * 100
    pf_c1, pf_c2, pf_c3 = st.columns(3)
    pf_c1.metric(
        "Park Factor",
        f"{park_factor:.2f}",
        delta=f"{pf_pct:+.0f}% vs neutral",
        delta_color="inverse",
        help="1.00 = neutral. >1 = hitter-friendly (more runs). <1 = pitcher-friendly (fewer runs)."
    )
    pf_c2.metric(
        "Sportsbook Total",
        str(total) if total else "N/A",
        help="The over/under set by the sportsbook."
    )
    if proj_total is not None and total is not None:
        diff = proj_total - float(total)
        ou_label = "Over" if diff >= 0 else "Under"
        ou_odds  = over_o  if diff >= 0 else under_o
        pf_c3.metric(
            "Model Projection",
            f"{proj_total:.1f} runs",
            delta=f"{diff:+.1f} → {ou_label} {total} ({ou_odds})",
            delta_color="normal" if diff >= 0 else "inverse",
        )
    else:
        pf_c3.metric("Model Projection", "N/A")

    park_name = PARK_FACTORS.get(home)
    if park_factor >= 1.05:
        st.caption(
            f"🏟️ **Hitter-friendly park** (+{pf_pct:.0f}%) — Coors-type environment "
            f"adds runs to projected total."
        )
    elif park_factor <= 0.96:
        st.caption(
            f"🏟️ **Pitcher-friendly park** ({pf_pct:.0f}%) — Suppresses scoring. "
            f"Under bets benefit here."
        )
    else:
        st.caption(f"🏟️ Neutral park — minimal effect on run totals.")

    st.divider()

    # ── Section 6: AI Reasoning (auto-generated, cached) ─────────────────────
    st.subheader("AI Reasoning")
    st.caption("Generated by Llama 3.3 70B — synthesizes all signals above into a plain-language rationale.")

    import json as _json
    signals = {}
    if team_stats is not None and not team_stats.empty:
        for team, prefix in [(home, "home"), (away, "away")]:
            t = team_stats[team_stats["team"] == team]
            if not t.empty and "pyth_deviation" in t.columns:
                signals[f"{prefix}_pyth_deviation"] = float(t.iloc[0]["pyth_deviation"])
    if pitcher_stats is not None and not pitcher_stats.empty:
        for sp, prefix in [(home_sp, "home"), (away_sp, "away")]:
            if sp and sp != "TBD":
                p = pitcher_stats[pitcher_stats["name"] == sp]
                if not p.empty:
                    p = p.iloc[0]
                    signals[f"{prefix}_starter"]     = sp
                    signals[f"{prefix}_starter_fip"] = p.get("fip")
                    signals[f"{prefix}_starter_era"] = p.get("era")
                    if p.get("fip") and p.get("era"):
                        signals[f"{prefix}_fip_era_gap"] = round(float(p["fip"]) - float(p["era"]), 2)
                    signals[f"{prefix}_babip"] = p.get("babip")

    model_prob_for_ai = home_prob if side == "HOME" else away_prob if side != "PASS" else home_prob
    impl_prob_for_ai  = home_impl if side == "HOME" else away_impl if side != "PASS" else home_impl
    bet_side_str = f"{bet_team} ({'HOME' if side == 'HOME' else 'AWAY'})" if bet_team else "No bet (PASS)"

    with st.spinner("Generating AI analysis..."):
        ai = _cached_ai_analysis(
            home=home, away=away,
            model_prob=model_prob_for_ai,
            implied_prob=impl_prob_for_ai,
            edge=edge,
            bet_side=bet_side_str,
            signals_json=_json.dumps({k: v for k, v in signals.items() if v is not None}),
        )

    conf = ai.get("confidence", "Low")
    st.markdown(_confidence_badge(conf), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"**Reasoning:**  \n{ai.get('reasoning', 'N/A')}")
    st.markdown(f"**Key Risk:**  \n{ai.get('key_risk', 'N/A')}")
