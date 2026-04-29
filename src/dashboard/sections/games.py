"""Games page — today's matchups with edge table, filters, and AI reasoning."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import requests
import streamlit as st

from src.dashboard.components.edge_table import render_edge_table
from src.dashboard.components.signal_cards import severity_badge
from src.shared.groq_agent import analyze_mlb_game
from src.data.predictions_db import save_predictions
from src.data.game_results import get_live_game_state
from src.data.bet_log_db import load_bets, get_best_bet_type, update_bet_outcome, update_parlay, OUTCOMES as _BET_OUTCOMES


# Full team name → 3-letter abbreviation (for compact bet cell display)
_FULL_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH",
    "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}


def _abbrev_bet_desc(desc: str) -> str:
    """Replace full team names in a bet description with 3-letter abbreviations."""
    for full, abbr in _FULL_TO_ABBR.items():
        desc = desc.replace(full, abbr)
    return desc


# Status badge HTML
_STATUS_HTML = {
    "Full":    '<span style="background:#2ecc71;color:#000;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">✅ Starters Set</span>',
    "Partial": '<span style="background:#e67e22;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">⚠️ Partial Lineup</span>',
    "None":    '<span style="background:#555;color:#ccc;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">⏳ No Lineup Yet</span>',
}


def _format_matchup(row) -> str:
    return f"{row.get('away_team', '?')} @ {row.get('home_team', '?')}"


def _fmt_odds(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    v = int(val)
    return f"+{v}" if v > 0 else str(v)


def _fmt_line(point, odds) -> str:
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


def _add_to_slip(bet: dict) -> None:
    """Add a bet to the session-state bet slip, avoiding duplicate keys."""
    if "bet_slip" not in st.session_state:
        st.session_state.bet_slip = []
    existing_keys = {b["key"] for b in st.session_state.bet_slip}
    if bet["key"] not in existing_keys:
        st.session_state.bet_slip.append(bet)


def _build_runs_per_game(team_stats: pd.DataFrame | None) -> dict:
    """Return {team: (rs_per_g, ra_per_g)} from team_stats."""
    lookup: dict[str, tuple[float, float]] = {}
    if team_stats is None or team_stats.empty:
        return lookup
    for _, r in team_stats.iterrows():
        try:
            w = float(r.get("wins", 0) or 0)
            l = float(r.get("losses", 0) or 0)
            g = max(w + l, 1.0)
            rs = float(r.get("runs_scored", 0) or 0)
            ra = float(r.get("runs_allowed", 0) or 0)
            lookup[r["team"]] = (rs / g, ra / g)
        except Exception:
            continue
    return lookup


def _proj_total_for(row, rpg: dict) -> float | None:
    """Compute projected game total from runs-per-game lookup."""
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    if home not in rpg or away not in rpg:
        return None
    h_rs, h_ra = rpg[home]
    a_rs, a_ra = rpg[away]
    home_exp = (h_rs + a_ra) / 2
    away_exp = (a_rs + h_ra) / 2
    return round(home_exp + away_exp, 2)


def _render_bet_cards(
    bets: pd.DataFrame,
    bankroll: float,
    team_stats: pd.DataFrame | None,
    pitcher_stats: pd.DataFrame | None,
    rpg: dict,
) -> None:
    if bets.empty:
        st.info("No bets meet the minimum edge threshold today.")
        return

    for _, row in bets.iterrows():
        side = row["best_bet_side"]
        team = row["home_team"] if side == "HOME" else row["away_team"]
        opp  = row["away_team"] if side == "HOME" else row["home_team"]
        odds_key    = "home_odds"     if side == "HOME" else "away_odds"
        kelly_key   = "home_kelly_pct" if side == "HOME" else "away_kelly_pct"
        edge_key    = "home_edge_pct"  if side == "HOME" else "away_edge_pct"
        model_key   = "home_model_prob" if side == "HOME" else "away_model_prob"
        implied_key = "home_implied_prob" if side == "HOME" else "away_implied_prob"

        edge        = row[edge_key]
        kelly_pct   = row[kelly_key]
        dollar_stake = bankroll * kelly_pct / 100
        line        = row.get(odds_key, "N/A")
        line_str    = f"+{line}" if isinstance(line, (int, float)) and line > 0 else str(line)
        model_prob  = row.get(model_key, 0)
        implied_prob = row.get(implied_key, 0)
        status      = row.get("lineup_status", "None")
        matchup     = f"{row.get('away_team','?')} @ {row.get('home_team','?')}"
        game_key    = row.get("game_id", f"{row.get('home_team','')}_{row.get('away_team','')}")

        severity = "High" if edge >= 7 else "Medium" if edge >= 4 else "Low"
        badge = severity_badge(severity, f"{edge:.1f}% edge")

        # Compute O/U recommendation fresh from team stats
        total     = row.get("total_line")
        over_o    = row.get("over_odds")
        under_o   = row.get("under_odds")
        proj_total = _proj_total_for(row, rpg)

        with st.expander(
            f"{team} vs {opp}  |  {edge:.1f}% edge  |  Line: {line_str}  |  Stake: ${dollar_stake:,.0f}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Prob",  f"{model_prob:.1%}")
            c2.metric("Market Prob", f"{implied_prob:.1%}")
            c3.metric("Edge",        f"{edge:+.1f}%")
            c4.metric("Kelly Stake", f"${dollar_stake:,.0f} ({kelly_pct:.1f}%)")

            home_starter = row.get("home_starter")
            away_starter = row.get("away_starter")
            s_col1, s_col2, s_col3 = st.columns([1, 2, 2])
            with s_col1:
                st.markdown(_STATUS_HTML.get(status, ""), unsafe_allow_html=True)
            with s_col2:
                st.caption(f"🏠 {row.get('home_team','?')}: **{home_starter or 'TBD'}**")
            with s_col3:
                st.caption(f"✈️ {row.get('away_team','?')}: **{away_starter or 'TBD'}**")

            st.markdown(
                f"**Bet:** {team} &nbsp; {badge} &nbsp; Line: `{line_str}`",
                unsafe_allow_html=True,
            )

            # Build signals dict for AI
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

            # ── Additional Markets ─────────────────────────────────────
            st.markdown("**Additional Markets**")
            m1, m2 = st.columns(2)

            # Game Total — always show Over or Under using fresh proj_total
            if total is not None and not (isinstance(total, float) and pd.isna(total)):
                if proj_total is not None:
                    if proj_total >= float(total):
                        ou_label   = f"Over {total}"
                        ou_odds    = _fmt_odds(over_o)
                        ou_bet_key = f"ou_{game_key}"
                        ou_line    = over_o
                    else:
                        ou_label   = f"Under {total}"
                        ou_odds    = _fmt_odds(under_o)
                        ou_bet_key = f"ou_{game_key}"
                        ou_line    = under_o
                    m1.metric("Game Total", ou_label)
                    m1.caption(f"Model projects {proj_total:.1f} runs  |  Odds: {ou_odds}")
                else:
                    ou_label   = None
                    ou_bet_key = f"ou_{game_key}"
                    ou_line    = None
                    m1.metric("Game Total", str(total))
                    m1.caption(f"Over {_fmt_odds(over_o)}  |  Under {_fmt_odds(under_o)}")
            else:
                ou_label = None
                ou_bet_key = None
                ou_line = None
                m1.metric("Game Total", "N/A")

            # Run Line
            home_rl = row.get("home_rl")
            away_rl = row.get("away_rl")
            rl_label = None
            rl_line  = None
            if home_rl is not None and not (isinstance(home_rl, float) and pd.isna(home_rl)):
                rl_point    = home_rl if side == "HOME" else away_rl
                rl_odds_val = row.get("home_rl_odds") if side == "HOME" else row.get("away_rl_odds")
                sign        = "+" if float(rl_point) > 0 else ""
                rl_label    = f"{team} {sign}{rl_point}"
                rl_line     = rl_odds_val
                m2.metric("Run Line", rl_label)
                m2.caption(f"Odds: {_fmt_odds(rl_odds_val)}")
            else:
                m2.metric("Run Line", "N/A")

            # ── Bet Slip Buttons ───────────────────────────────────────
            st.divider()
            st.caption("**Add to Bet Slip:**")
            slip_c1, slip_c2, slip_c3, slip_c4 = st.columns([2, 2, 2, 4])

            with slip_c1:
                if st.button(f"+ ML: {team}", key=f"slip_ml_{game_key}"):
                    _add_to_slip({
                        "key": f"ml_{game_key}",
                        "matchup": matchup,
                        "description": f"ML: {team} {line_str}",
                        "bet_type": "ML",
                        "line": float(line) if isinstance(line, (int, float)) else 0,
                        "edge_pct": edge,
                        "model_prob": round(model_prob, 4),
                        "stake": round(dollar_stake, 2),
                        "bet_side": side,
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "home_odds": row.get("home_odds"),
                        "away_odds": row.get("away_odds"),
                    })
                    st.toast(f"Added: ML {team}", icon="📌")

            with slip_c2:
                if rl_label and st.button(f"+ RL: {rl_label}", key=f"slip_rl_{game_key}"):
                    _add_to_slip({
                        "key": f"rl_{game_key}",
                        "matchup": matchup,
                        "description": f"RL: {rl_label}",
                        "bet_type": "RL",
                        "line": float(rl_line) if isinstance(rl_line, (int, float)) else 0,
                        "edge_pct": edge,
                        "model_prob": round(model_prob, 4),
                        "stake": round(dollar_stake, 2),
                        "rl_team": team,
                        "rl_spread": float(rl_point) if rl_point is not None else -1.5,
                        "rl_side": side,
                        "home_team": row.get("home_team", ""),
                        "away_team": row.get("away_team", ""),
                        "home_rl": home_rl,
                        "away_rl": away_rl,
                        "home_rl_odds": row.get("home_rl_odds"),
                        "away_rl_odds": row.get("away_rl_odds"),
                    })
                    st.toast(f"Added: RL {rl_label}", icon="📌")
                elif not rl_label:
                    st.button("+ RL: N/A", key=f"slip_rl_{game_key}", disabled=True)

            with slip_c3:
                if ou_label and st.button(f"+ {ou_label}", key=f"slip_ou_{game_key}"):
                    _add_to_slip({
                        "key": f"ou_{game_key}",
                        "matchup": matchup,
                        "description": f"O/U: {ou_label}",
                        "bet_type": "O/U",
                        "line": float(ou_line) if isinstance(ou_line, (int, float)) else 0,
                        "edge_pct": 0.0,
                        "model_prob": None,
                        "stake": 50.0,
                        "ou_direction": "Over" if ou_label.startswith("Over") else "Under",
                        "ou_total": float(total) if total is not None else 8.5,
                        "over_odds": float(over_o) if isinstance(over_o, (int, float)) else None,
                        "under_odds": float(under_o) if isinstance(under_o, (int, float)) else None,
                    })
                    st.toast(f"Added: {ou_label}", icon="📌")
                elif not ou_label:
                    st.button("+ O/U: N/A", key=f"slip_ou_{game_key}", disabled=True)

            st.divider()

            if st.button("Generate AI Analysis", key=f"ai_{game_key}"):
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


def _render_game_detail_card(
    row: pd.Series,
    bankroll: float,
    team_stats: pd.DataFrame | None,
    pitcher_stats: pd.DataFrame | None,
    rpg: dict,
) -> None:
    """
    Render a detailed breakdown for a single game — model signals, projected runs,
    starter stats, and an AI analysis button. Works for both complete and incomplete games.
    """
    home = row.get("home_team", "?")
    away = row.get("away_team", "?")
    game_key = row.get("game_id", f"{home}_{away}")
    matchup  = f"{away} @ {home}"
    status   = row.get("lineup_status", "None")
    complete = row.get("_complete", True) and pd.notna(row.get("home_model_prob"))

    edge      = row.get("best_bet_edge", 0.0) or 0.0
    bet_side  = row.get("best_bet_side", "PASS")
    is_bet    = bet_side != "PASS" and complete

    # ── Header row ────────────────────────────────────────────────────────────
    home_prob    = row.get("home_model_prob")
    away_prob    = row.get("away_model_prob")
    home_implied = row.get("home_implied_prob")
    away_implied = row.get("away_implied_prob")

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        st.markdown(f"**{matchup}**")
        st.markdown(_STATUS_HTML.get(status, ""), unsafe_allow_html=True)
    if complete:
        c2.metric(f"{home} Model", f"{home_prob:.1%}" if home_prob else "—")
        c3.metric(f"{away} Model", f"{away_prob:.1%}" if away_prob else "—")
        c4.metric("Edge", f"{edge:+.1f}%")
        c5.metric("Bet Signal", bet_side if is_bet else "PASS")
    else:
        missing = row.get("_missing", [])
        c2.caption("Analysis incomplete")
        c3.caption(", ".join(missing) if missing else "missing data")

    # ── Starters ──────────────────────────────────────────────────────────────
    home_starter = row.get("home_starter")
    away_starter = row.get("away_starter")

    def _pitcher_line(name: str | None, label: str) -> None:
        if not name:
            st.caption(f"{label}: **TBD**")
            return
        st.caption(f"{label}: **{name}**")
        if pitcher_stats is not None and not pitcher_stats.empty:
            p = pitcher_stats[pitcher_stats["name"] == name]
            if not p.empty:
                p = p.iloc[0]
                fip  = p.get("fip");  fip_str  = f"{fip:.2f}" if fip  else "—"
                era  = p.get("era");  era_str  = f"{era:.2f}" if era  else "—"
                babip = p.get("babip"); babip_str = f"{babip:.3f}" if babip else "—"
                k_pct = p.get("k_pct"); k_str = f"{k_pct:.1%}" if k_pct else "—"
                gap_str = ""
                if fip and era:
                    gap = round(float(fip) - float(era), 2)
                    gap_str = f"  ·  FIP-ERA gap: {gap:+.2f}"
                st.caption(f"ERA {era_str}  ·  FIP {fip_str}  ·  BABIP {babip_str}  ·  K% {k_str}{gap_str}")

    p1, p2 = st.columns(2)
    with p1:
        _pitcher_line(home_starter, f"🏠 {home} SP")
    with p2:
        _pitcher_line(away_starter, f"✈️ {away} SP")

    # ── Projected runs ────────────────────────────────────────────────────────
    proj_hr = row.get("proj_home_runs")
    proj_ar = row.get("proj_away_runs")
    proj_t  = row.get("proj_total")
    total_line = row.get("total_line")

    if proj_hr is not None and proj_ar is not None:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"{home} Proj Runs", f"{proj_hr:.1f}")
        r2.metric(f"{away} Proj Runs", f"{proj_ar:.1f}")
        if proj_t is not None:
            r3.metric("Proj Total", f"{proj_t:.1f}")
        if total_line is not None and not (isinstance(total_line, float) and pd.isna(total_line)):
            over_o  = row.get("over_odds")
            under_o = row.get("under_odds")
            if proj_t is not None:
                direction = "Over" if proj_t >= float(total_line) else "Under"
                odds_val  = over_o if direction == "Over" else under_o
                r4.metric("O/U Pick", f"{direction} {total_line}", help=f"Odds: {_fmt_odds(odds_val)}")
            else:
                r4.metric("O/U Line", f"{total_line}")

    # ── Run line ──────────────────────────────────────────────────────────────
    home_rl = row.get("home_rl")
    away_rl = row.get("away_rl")
    if home_rl is not None and not (isinstance(home_rl, float) and pd.isna(home_rl)):
        home_rl_sign = "+" if float(home_rl) > 0 else ""
        away_rl_sign = "+" if float(away_rl) > 0 else ""
        st.caption(
            f"Run Line — {home}: {home_rl_sign}{home_rl} ({_fmt_odds(row.get('home_rl_odds'))})  "
            f"·  {away}: {away_rl_sign}{away_rl} ({_fmt_odds(row.get('away_rl_odds'))})"
        )

    # ── Bet Slip Buttons — all markets, both teams ────────────────────────────
    st.divider()
    if status != "Full":
        st.info("⏳ Bet slip locked — both starting pitchers must be confirmed before adding bets.")
        return
    st.caption("**Add to Bet Slip**")

    home_odds     = row.get("home_odds")
    away_odds     = row.get("away_odds")
    home_rl_odds  = row.get("home_rl_odds")
    away_rl_odds  = row.get("away_rl_odds")
    over_odds     = row.get("over_odds")
    under_odds    = row.get("under_odds")
    home_kelly    = row.get("home_kelly_pct", 0.0) or 0.0
    away_kelly    = row.get("away_kelly_pct", 0.0) or 0.0
    home_edge_pct = row.get("home_edge_pct", 0.0) or 0.0
    away_edge_pct = row.get("away_edge_pct", 0.0) or 0.0

    def _odds_label(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        v = int(val)
        return f"+{v}" if v > 0 else str(v)

    def _rl_label(team_name: str, point, odds) -> str:
        if point is None or (isinstance(point, float) and pd.isna(point)):
            return "N/A"
        sign = "+" if float(point) > 0 else ""
        return f"{team_name} {sign}{point} ({_odds_label(odds)})"

    has_rl    = home_rl is not None and not (isinstance(home_rl, float) and pd.isna(home_rl))
    has_total = total_line is not None and not (isinstance(total_line, float) and pd.isna(total_line))

    # Row 1: ML buttons
    ml_c1, ml_c2 = st.columns(2)
    with ml_c1:
        lbl = f"✈️ {away} ML  {_odds_label(away_odds)}"
        if st.button(lbl, key=f"slip_ml_away_d_{game_key}", disabled=(away_odds is None)):
            stake = round(bankroll * away_kelly / 100, 2) or 50.0
            _add_to_slip({
                "key": f"ml_away_{game_key}",
                "matchup": matchup,
                "description": f"ML: {away} {_odds_label(away_odds)}",
                "bet_type": "ML",
                "line": float(away_odds) if isinstance(away_odds, (int, float)) else 0,
                "edge_pct": away_edge_pct,
                "model_prob": round(float(away_prob), 4) if away_prob else None,
                "stake": stake,
                "bet_side": "AWAY",
                "home_team": home,
                "away_team": away,
                "home_odds": home_odds,
                "away_odds": away_odds,
            })
            st.toast(f"Added: ML {away}", icon="📌")
    with ml_c2:
        lbl = f"🏠 {home} ML  {_odds_label(home_odds)}"
        if st.button(lbl, key=f"slip_ml_home_d_{game_key}", disabled=(home_odds is None)):
            stake = round(bankroll * home_kelly / 100, 2) or 50.0
            _add_to_slip({
                "key": f"ml_home_{game_key}",
                "matchup": matchup,
                "description": f"ML: {home} {_odds_label(home_odds)}",
                "bet_type": "ML",
                "line": float(home_odds) if isinstance(home_odds, (int, float)) else 0,
                "edge_pct": home_edge_pct,
                "model_prob": round(float(home_prob), 4) if home_prob else None,
                "stake": stake,
                "bet_side": "HOME",
                "home_team": home,
                "away_team": away,
                "home_odds": home_odds,
                "away_odds": away_odds,
            })
            st.toast(f"Added: ML {home}", icon="📌")

    # ── Spread (Run Line) recommendation ──────────────────────────────────────
    best_rl_side     = row.get("best_rl_side", "PASS")
    best_rl_edge     = row.get("best_rl_edge_pct", 0.0) or 0.0
    home_rl_cov_prob = row.get("home_rl_cover_prob")
    away_rl_cov_prob = row.get("away_rl_cover_prob")
    home_rl_edge_rl  = row.get("home_rl_edge_pct", 0.0) or 0.0
    away_rl_edge_rl  = row.get("away_rl_edge_pct", 0.0) or 0.0

    # Row 2: Run Line buttons
    rl_c1, rl_c2 = st.columns(2)
    with rl_c1:
        away_rl_str = _rl_label(away, away_rl, away_rl_odds) if has_rl else "Spread N/A"
        away_rl_rec = has_rl and best_rl_side == "AWAY"
        btn_lbl = f"✈️ {away_rl_str}" + (" ⭐" if away_rl_rec else "")
        if st.button(btn_lbl, key=f"slip_rl_away_d_{game_key}", disabled=(not has_rl)):
            stake = round(bankroll * away_kelly / 100, 2) or 50.0
            away_rl_sign = "+" if float(away_rl) > 0 else ""
            _add_to_slip({
                "key": f"rl_away_{game_key}",
                "matchup": matchup,
                "description": f"RL: {away} {away_rl_sign}{away_rl} ({_odds_label(away_rl_odds)})",
                "bet_type": "RL",
                "line": float(away_rl_odds) if isinstance(away_rl_odds, (int, float)) else 0,
                "edge_pct": away_rl_edge_rl,
                "model_prob": round(float(away_rl_cov_prob), 4) if away_rl_cov_prob else None,
                "stake": stake,
                "rl_team": away,
                "rl_spread": float(away_rl),
                "rl_side": "AWAY",
                "home_team": home,
                "away_team": away,
                "home_rl": home_rl,
                "away_rl": away_rl,
                "home_rl_odds": home_rl_odds,
                "away_rl_odds": away_rl_odds,
            })
            st.toast(f"Added: RL {away}", icon="📌")
    with rl_c2:
        home_rl_str = _rl_label(home, home_rl, home_rl_odds) if has_rl else "Spread N/A"
        home_rl_rec = has_rl and best_rl_side == "HOME"
        btn_lbl = f"🏠 {home_rl_str}" + (" ⭐" if home_rl_rec else "")
        if st.button(btn_lbl, key=f"slip_rl_home_d_{game_key}", disabled=(not has_rl)):
            stake = round(bankroll * home_kelly / 100, 2) or 50.0
            home_rl_sign = "+" if float(home_rl) > 0 else ""
            _add_to_slip({
                "key": f"rl_home_{game_key}",
                "matchup": matchup,
                "description": f"RL: {home} {home_rl_sign}{home_rl} ({_odds_label(home_rl_odds)})",
                "bet_type": "RL",
                "line": float(home_rl_odds) if isinstance(home_rl_odds, (int, float)) else 0,
                "edge_pct": home_rl_edge_rl,
                "model_prob": round(float(home_rl_cov_prob), 4) if home_rl_cov_prob else None,
                "stake": stake,
                "rl_team": home,
                "rl_spread": float(home_rl),
                "rl_side": "HOME",
                "home_team": home,
                "away_team": away,
                "home_rl": home_rl,
                "away_rl": away_rl,
                "home_rl_odds": home_rl_odds,
                "away_rl_odds": away_rl_odds,
            })
            st.toast(f"Added: RL {home}", icon="📌")

    # Spread recommendation line
    if has_rl and home_rl_cov_prob is not None:
        if best_rl_side == "HOME":
            rl_sign = "+" if float(home_rl) > 0 else ""
            st.caption(
                f"⭐ Model pick: **{home} {rl_sign}{home_rl}** · "
                f"Cover prob: {home_rl_cov_prob:.1%} · Edge: {best_rl_edge:+.1f}%"
            )
        elif best_rl_side == "AWAY":
            rl_sign = "+" if float(away_rl) > 0 else ""
            st.caption(
                f"⭐ Model pick: **{away} {rl_sign}{away_rl}** · "
                f"Cover prob: {away_rl_cov_prob:.1%} · Edge: {best_rl_edge:+.1f}%"
            )
        else:
            h_str = f"{home}: {home_rl_cov_prob:.1%}" if home_rl_cov_prob else ""
            a_str = f"{away}: {away_rl_cov_prob:.1%}" if away_rl_cov_prob else ""
            st.caption(f"No spread edge — {h_str}  ·  {a_str}")

    # ── Game Total recommendation ──────────────────────────────────────────────
    best_total_dir  = row.get("best_total_direction")
    best_total_edge = row.get("best_total_edge_pct", 0.0) or 0.0
    over_prob_model = row.get("total_over_prob")
    under_prob_model = row.get("total_under_prob")
    over_edge_pct   = row.get("over_edge_pct", 0.0) or 0.0
    under_edge_pct  = row.get("under_edge_pct", 0.0) or 0.0
    has_total_model = over_prob_model is not None

    # Row 3: Game Total (Over / Under)
    ou_c1, ou_c2 = st.columns(2)
    with ou_c1:
        over_lbl = f"Over {total_line}  {_odds_label(over_odds)}" if has_total else "Over N/A"
        over_rec = has_total and best_total_dir == "Over" and has_total_model
        btn_lbl = f"⬆ {over_lbl}" + (" ⭐" if over_rec else "")
        if st.button(btn_lbl, key=f"slip_over_d_{game_key}", disabled=(not has_total)):
            _add_to_slip({
                "key": f"over_{game_key}",
                "matchup": matchup,
                "description": f"Over {total_line} ({_odds_label(over_odds)})",
                "bet_type": "O/U",
                "line": float(over_odds) if isinstance(over_odds, (int, float)) else 0,
                "edge_pct": over_edge_pct,
                "model_prob": round(float(over_prob_model), 4) if over_prob_model else None,
                "stake": 50.0,
                "ou_direction": "Over",
                "ou_total": float(total_line) if total_line is not None else 8.5,
                "over_odds": float(over_odds) if isinstance(over_odds, (int, float)) else None,
                "under_odds": float(under_odds) if isinstance(under_odds, (int, float)) else None,
            })
            st.toast(f"Added: Over {total_line}", icon="📌")
    with ou_c2:
        under_lbl = f"Under {total_line}  {_odds_label(under_odds)}" if has_total else "Under N/A"
        under_rec = has_total and best_total_dir == "Under" and has_total_model
        btn_lbl = f"⬇ {under_lbl}" + (" ⭐" if under_rec else "")
        if st.button(btn_lbl, key=f"slip_under_d_{game_key}", disabled=(not has_total)):
            _add_to_slip({
                "key": f"under_{game_key}",
                "matchup": matchup,
                "description": f"Under {total_line} ({_odds_label(under_odds)})",
                "bet_type": "O/U",
                "line": float(under_odds) if isinstance(under_odds, (int, float)) else 0,
                "edge_pct": under_edge_pct,
                "model_prob": round(float(under_prob_model), 4) if under_prob_model else None,
                "stake": 50.0,
                "ou_direction": "Under",
                "ou_total": float(total_line) if total_line is not None else 8.5,
                "over_odds": float(over_odds) if isinstance(over_odds, (int, float)) else None,
                "under_odds": float(under_odds) if isinstance(under_odds, (int, float)) else None,
            })
            st.toast(f"Added: Under {total_line}", icon="📌")

    # Total recommendation line
    if has_total and has_total_model:
        rec_prob = over_prob_model if best_total_dir == "Over" else under_prob_model
        min_edge = 3.0
        if abs(best_total_edge) >= min_edge:
            st.caption(
                f"⭐ Model pick: **{best_total_dir} {total_line}** · "
                f"Prob: {rec_prob:.1%} · Edge: {best_total_edge:+.1f}%"
            )
        else:
            st.caption(
                f"Model leans **{best_total_dir} {total_line}** · "
                f"Prob: {rec_prob:.1%} · Edge: {best_total_edge:+.1f}% (no edge)"
            )

    # ── AI Analysis ───────────────────────────────────────────────────────────
    st.divider()
    if complete:
        if st.button("Generate AI Analysis", key=f"ai_detail_{game_key}"):
            signals: dict = {
                "home_starter": home_starter,
                "away_starter": away_starter,
            }
            if team_stats is not None and not team_stats.empty:
                for t, prefix in [(home, "home"), (away, "away")]:
                    t_row = team_stats[team_stats["team"] == t]
                    if not t_row.empty and "pyth_deviation" in t_row.columns:
                        signals[f"{prefix}_pyth_deviation"] = float(t_row.iloc[0]["pyth_deviation"])
            if pitcher_stats is not None and not pitcher_stats.empty:
                for name, prefix in [(home_starter, "home"), (away_starter, "away")]:
                    if name:
                        p = pitcher_stats[pitcher_stats["name"] == name]
                        if not p.empty:
                            p = p.iloc[0]
                            signals[f"{prefix}_starter_fip"] = p.get("fip")
                            signals[f"{prefix}_starter_era"] = p.get("era")
                            if p.get("fip") and p.get("era"):
                                signals[f"{prefix}_fip_era_gap"] = round(float(p["fip"]) - float(p["era"]), 2)
                            signals[f"{prefix}_babip"] = p.get("babip")
            with st.spinner("Analyzing with Llama 3.3 70B..."):
                result = analyze_mlb_game(
                    home_team=home,
                    away_team=away,
                    model_prob=float(home_prob) if home_prob else 0.5,
                    implied_prob=float(home_implied) if home_implied else 0.5,
                    edge_pct=edge,
                    bet_side=bet_side,
                    signals=signals,
                )
            conf = result.get("confidence", "Low")
            conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
            st.markdown(f"{conf_icon} **Confidence: {conf}**")
            st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
            st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")
    else:
        st.caption("AI analysis unavailable — team stats not matched for this game.")


def _mlb_session() -> requests.Session:
    """Return a requests Session with retry logic for MLB Stats API calls."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_team_records() -> dict[int, dict]:
    """Return {team_id: {record, last5}} from MLB standings API. Returns {} on failure."""
    try:
        resp = _mlb_session().get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": datetime.date.today().year,
                    "standingsTypes": "regularSeason", "hydrate": "team,record"},
            timeout=8,
        )
        resp.raise_for_status()
    except Exception:
        return {}
    out: dict[int, dict] = {}
    for division in resp.json().get("records", []):
        for tr in division.get("teamRecords", []):
            tid = tr["team"]["id"]
            w = tr.get("wins", 0)
            l = tr.get("losses", 0)
            splits = tr.get("records", {}).get("splitRecords", [])
            last10 = next((s for s in splits if s["type"] == "lastTen"), None)
            out[tid] = {
                "record": f"{w}-{l}",
                "last10_w": last10["wins"] if last10 else None,
                "last10_l": last10["losses"] if last10 else None,
            }
    return out


def _fetch_last5_results() -> dict[int, list[str]]:
    """Return {team_id: ['W','L','W','W','L']} from the past 14 days. Returns {} on failure."""
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=14)).isoformat()
    try:
        resp = _mlb_session().get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "startDate": start, "endDate": today.isoformat(),
                    "gameType": "R"},
            timeout=8,
        )
        resp.raise_for_status()
    except Exception:
        return {}
    from collections import defaultdict
    results: dict[int, list[str]] = defaultdict(list)
    for date_entry in resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            results[home["team"]["id"]].append("W" if home.get("isWinner") else "L")
            results[away["team"]["id"]].append("W" if away.get("isWinner") else "L")
    return {tid: res[-5:] for tid, res in results.items()}


def _fetch_mlb_schedule_enrichment() -> dict[str, dict]:
    """
    Return {'{away_name}@{home_name}': {series, travel, game_pk}} using MLB Stats API.
    series: 'Game 2 of 3 (Tied 1-1)'
    travel: 'W→E' | 'E→W' | None
    game_pk: int MLB game primary key for live feed
    """
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    sess = _mlb_session()

    # Today's schedule for series info + venue coords
    try:
        today_resp = sess.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": today.isoformat(), "gameType": "R",
                    "hydrate": "linescore,venue(location,fieldInfo),weather"},
            timeout=8,
        )
        today_resp.raise_for_status()
    except Exception:
        return {}

    # Yesterday's schedule for travel — where did away team play?
    yest_venue_by_team: dict[int, float] = {}
    try:
        yest_resp = sess.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": yesterday.isoformat(), "gameType": "R",
                    "hydrate": "venue(location)"},
            timeout=8,
        )
        yest_resp.raise_for_status()
        for d in yest_resp.json().get("dates", []):
            for g in d.get("games", []):
                lon = (g.get("venue", {}).get("location", {})
                       .get("defaultCoordinates", {}).get("longitude"))
                if lon is None:
                    continue
                for side in ("home", "away"):
                    tid = g["teams"][side]["team"]["id"]
                    yest_venue_by_team[tid] = float(lon)
    except Exception:
        pass  # travel data is optional — continue without it

    enrichment: dict[str, dict] = {}
    for d in today_resp.json().get("dates", []):
        for g in d.get("games", []):
            away_name = g["teams"]["away"]["team"]["name"]
            home_name = g["teams"]["home"]["team"]["name"]
            away_id   = g["teams"]["away"]["team"]["id"]
            key = f"{away_name}@{home_name}"

            # Series
            game_num   = g.get("seriesGameNumber", "?")
            games_in   = g.get("gamesInSeries", "?")
            series_status = g.get("seriesStatus", {})
            series_result = series_status.get("result", "")
            series_str = f"Game {game_num} of {games_in}"
            if series_result:
                series_str += f" · {series_result}"

            # Travel: compare yesterday's venue longitude to today's home venue longitude
            today_home_lon = (g.get("venue", {}).get("location", {})
                              .get("defaultCoordinates", {}).get("longitude"))
            travel = None
            if today_home_lon is not None and away_id in yest_venue_by_team:
                yest_lon = yest_venue_by_team[away_id]
                delta = float(today_home_lon) - yest_lon
                if abs(delta) > 5:  # ignore same-city or nearby moves
                    travel = "W→E" if delta > 0 else "E→W"

            weather = g.get("weather", {})
            fi = g.get("venue", {}).get("fieldInfo", {})
            enrichment[key] = {
                "series":  series_str,
                "travel":  travel,
                "game_pk": g.get("gamePk"),
                "weather": {
                    "condition": weather.get("condition", ""),
                    "temp":      weather.get("temp", ""),
                    "wind":      weather.get("wind", ""),
                    "roof":      fi.get("roofType", ""),
                },
            }

    return enrichment


def _team_sub(name: str, name_to_id: dict, records: dict, last5: dict) -> str:
    """Return record + L5 subtitle HTML for a team name."""
    tid = name_to_id.get(name)
    if tid is None:
        return ""
    rec = records.get(tid, {})
    res = last5.get(tid, [])
    record_str = rec.get("record", "")
    last5_str = "".join(
        f'<span style="color:{"#2ecc71" if r == "W" else "#e74c3c"}">{r}</span>'
        for r in res
    )
    parts = []
    if record_str:
        parts.append(record_str)
    if last5_str:
        parts.append(f"L5: {last5_str}")
    return "  ·  ".join(parts)


def _fetch_schedule_for_date(date: datetime.date) -> list[dict]:
    """
    Fetch games for any date from MLB Stats API + ESPN logos.
    Returns same schema as _fetch_daily_schedule so _render_daily_schedule works unchanged.
    For future dates: state='pre', no score, odds unavailable (ESPN only covers today).
    """
    records    = _fetch_team_records()
    last5      = _fetch_last5_results()
    enrichment = _fetch_mlb_schedule_enrichment()
    sess = _mlb_session()

    # Build name → team_id from standings
    name_to_id: dict[str, int] = {}
    try:
        standings_resp = sess.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": date.year,
                    "standingsTypes": "regularSeason", "hydrate": "team"},
            timeout=8,
        )
        standings_resp.raise_for_status()
        for division in standings_resp.json().get("records", []):
            for tr in division.get("teamRecords", []):
                name_to_id[tr["team"]["name"]] = tr["team"]["id"]
    except Exception:
        pass  # team records are optional enrichment

    # Logos from ESPN — keyed by displayName
    logo_map: dict[str, str] = {}
    try:
        espn_resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            timeout=8,
        )
        if espn_resp.ok:
            for evt in espn_resp.json().get("events", []):
                for c in evt["competitions"][0].get("competitors", []):
                    logo_map[c["team"]["displayName"]] = c["team"].get("logo", "")
    except Exception:
        pass

    # MLB Stats API schedule for the target date — this is the critical call
    sched_resp = sess.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date.isoformat(), "gameType": "R",
                "hydrate": "probablePitcher,linescore,venue(location,fieldInfo),weather"},
        timeout=12,
    )
    sched_resp.raise_for_status()

    pt = ZoneInfo("America/Los_Angeles")
    games = []
    for date_entry in sched_resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            teams = g.get("teams", {})
            home_info = teams.get("home", {})
            away_info = teams.get("away", {})
            home = home_info.get("team", {}).get("name", "")
            away = away_info.get("team", {}).get("name", "")

            abstract_state = g.get("status", {}).get("abstractGameState", "Preview")
            if abstract_state == "Final":
                state = "post"
            elif abstract_state == "Live":
                state = "in"
            else:
                state = "pre"

            detail = g.get("status", {}).get("detailedState", "")
            # Skip games that are not actually being played today
            if any(kw in detail.lower() for kw in ("postponed", "cancel", "suspend")):
                continue
            home_score = str(home_info.get("score", "")) if state != "pre" else ""
            away_score = str(away_info.get("score", "")) if state != "pre" else ""

            try:
                game_dt = datetime.datetime.fromisoformat(
                    g["gameDate"].replace("Z", "+00:00")
                ).astimezone(pt)
                time_str = game_dt.strftime("%-I:%M %p PT")
            except Exception:
                time_str = "—"

            # Probable starters from hydrated schedule
            home_sp = home_info.get("probablePitcher", {}).get("fullName")
            away_sp = away_info.get("probablePitcher", {}).get("fullName")

            weather = g.get("weather", {})
            fi      = g.get("venue", {}).get("fieldInfo", {})
            weather["roof"] = fi.get("roofType", "")

            enr     = enrichment.get(f"{away}@{home}", {})
            series  = enr.get("series", "")
            travel  = enr.get("travel")
            game_pk = g.get("gamePk") or enr.get("game_pk")
            travel_label = f"✈️ Long Travel ({travel})" if travel else ""

            games.append({
                "time":      time_str,
                "away":      away,
                "home":      home,
                "away_logo": logo_map.get(away, ""),
                "home_logo": logo_map.get(home, ""),
                "away_sub":  _team_sub(away, name_to_id, records, last5),
                "home_sub":  _team_sub(home, name_to_id, records, last5),
                "away_sp":   away_sp,
                "home_sp":   home_sp,
                "series":    series,
                "travel":    travel_label,
                "weather":   weather,
                "score":     f"{away_score}–{home_score}" if (away_score or home_score) else "—",
                "detail":    detail,
                "state":     state,
                "game_pk":   game_pk,
            })
    return games


def _fetch_daily_schedule() -> list[dict]:
    """Fetch today's games. Delegates to _fetch_schedule_for_date with ESPN live state overlay."""
    today = datetime.date.today()

    # Start from MLB Stats API (same as tomorrow path) for consistency
    games = _fetch_schedule_for_date(today)

    # Overlay live state + scores from ESPN scoreboard (richer than MLB API for today)
    try:
        espn_resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            timeout=10,
        )
        if espn_resp.ok:
            pt = ZoneInfo("America/Los_Angeles")
            espn_by_matchup: dict[str, dict] = {}
            for event in espn_resp.json().get("events", []):
                comp = event["competitions"][0]
                h = a = ""
                h_score = a_score = ""
                h_sp = a_sp = None
                h_logo = a_logo = ""
                for c in comp.get("competitors", []):
                    pitcher = c.get("probables", [{}])[0] if c.get("probables") else {}
                    sp_name = (pitcher.get("athlete", {}).get("displayName")
                               or pitcher.get("displayName"))
                    logo = c["team"].get("logo", "")
                    if c["homeAway"] == "home":
                        h = c["team"]["displayName"]
                        h_score = c.get("score", "")
                        h_sp   = sp_name
                        h_logo = logo
                    else:
                        a = c["team"]["displayName"]
                        a_score = c.get("score", "")
                        a_sp   = sp_name
                        a_logo = logo
                status = comp["status"]
                state  = status["type"].get("state", "pre")
                detail = status["type"].get("shortDetail", "")
                try:
                    game_dt = datetime.datetime.fromisoformat(
                        event["date"].replace("Z", "+00:00")
                    ).astimezone(pt)
                    time_str = game_dt.strftime("%-I:%M %p PT")
                except Exception:
                    time_str = "—"
                espn_by_matchup[f"{a}@{h}"] = {
                    "state": state, "detail": detail, "time": time_str,
                    "score": f"{a_score}–{h_score}" if (a_score or h_score) else "—",
                    "away_sp": a_sp, "home_sp": h_sp,
                    "away_logo": a_logo, "home_logo": h_logo,
                }

            # Merge ESPN state into MLB-sourced games
            for g in games:
                key = f"{g['away']}@{g['home']}"
                if key in espn_by_matchup:
                    ov = espn_by_matchup[key]
                    g["state"]  = ov["state"]
                    g["detail"] = ov["detail"]
                    g["time"]   = ov["time"]
                    g["score"]  = ov["score"]
                    if ov["away_sp"]:  g["away_sp"]   = ov["away_sp"]
                    if ov["home_sp"]:  g["home_sp"]   = ov["home_sp"]
                    if ov["away_logo"]: g["away_logo"] = ov["away_logo"]
                    if ov["home_logo"]: g["home_logo"] = ov["home_logo"]
    except Exception:
        pass

    return games


# Wind direction → unicode arrow + label
_WIND_ARROWS: dict[str, tuple[str, str]] = {
    "out to cf":   ("↑", "Out to CF"),
    "out to rf":   ("↗", "Out to RF"),
    "l to r":      ("→", "L to R"),
    "in from lf":  ("↘", "In from LF"),
    "in from cf":  ("↓", "In from CF"),
    "in from rf":  ("↙", "In from RF"),
    "r to l":      ("←", "R to L"),
    "out to lf":   ("↖", "Out to LF"),
}


def _wind_label(wind_str: str) -> str:
    """Return '↗ 12 mph (Out to RF)' or empty string."""
    if not wind_str or "0 mph" in wind_str or wind_str.lower() == "none":
        return ""
    parts = wind_str.lower().split(",", 1)
    mph_part  = wind_str.split(",")[0].strip()
    direction = parts[1].strip() if len(parts) > 1 else ""
    arrow, label = _WIND_ARROWS.get(direction, ("💨", direction.title()))
    return f"{arrow} {mph_part} ({label})"


def _weather_and_park_text(weather: dict, home_team: str) -> str:
    """Return plain-text weather + park factor line (no nested HTML)."""
    from src.data.ballpark import PARK_FACTORS
    roof      = weather.get("roof", "")
    condition = weather.get("condition", "")
    temp      = weather.get("temp", "")
    wind_str  = weather.get("wind", "")
    is_dome   = roof in ("Dome", "Retractable") or condition in ("Dome", "Roof Closed")

    cond_icon = {"Sunny": "☀️", "Clear": "☀️", "Partly Cloudy": "⛅", "Cloudy": "☁️",
                 "Overcast": "☁️", "Rain": "🌧️", "Drizzle": "🌦️",
                 "Dome": "🏟️", "Roof Closed": "🏟️"}.get(condition, "🌡️")

    parts = []
    if temp:
        parts.append(f"{cond_icon} {temp}°F")
    if is_dome:
        parts.append("Dome")
    elif wind_str:
        wl = _wind_label(wind_str)
        if wl:
            parts.append(wl)

    pf = PARK_FACTORS.get(home_team)
    if pf is not None:
        if pf >= 105:
            parts.append(f"⬆ Hitter-friendly ({pf})")
        elif pf <= 95:
            parts.append(f"⬇ Pitcher-friendly ({pf})")
        else:
            parts.append(f"Neutral park ({pf})")

    return "  ·  ".join(parts)




def _render_pitcher_line(p: dict, is_pitching: bool, right_align: bool = False) -> str:
    """
    Pitcher line for live game.
    Left side (away): SP John Doe (45p) · 3.0 IP 2H 4K 0ER
    Right side (home): 3.0 IP 2H 4K 0ER · (45p) John Doe SP  (stats before name)
    SP label only shown if pitcher is the starter.
    """
    if not p:
        return "TBD"
    name      = p.get("name", "")
    pc        = p.get("pc", 0)
    is_sp     = p.get("is_starter", False)
    sp_badge  = '<span style="font-size:10px;color:#3498db;font-weight:bold">SP</span> ' if is_sp else ""
    sp_badge_r = ' <span style="font-size:10px;color:#3498db;font-weight:bold">SP</span>' if is_sp else ""
    pc_str    = f'({pc}p)' if is_pitching and pc else ""
    stats_str = f'{p["ip"]} IP  {p["h"]}H  {p["k"]}K  {p["er"]}ER'

    if right_align:
        # stats · (pc) Name SP
        parts = []
        parts.append(f'<span style="font-size:11px;color:#888">{stats_str}</span>')
        if pc_str:
            parts.append(f'<span style="font-size:11px;color:#aaa">{pc_str}</span>')
        parts.append(f'<span style="font-weight:bold">{name}</span>{sp_badge_r}')
        return " · ".join(p for p in parts if p)
    else:
        # SP Name (pc) · stats
        parts = []
        if pc_str:
            name_part = f'{sp_badge}<span style="font-weight:bold">{name}</span> <span style="font-size:11px;color:#aaa">{pc_str}</span>'
        else:
            name_part = f'{sp_badge}<span style="font-weight:bold">{name}</span>'
        parts.append(name_part)
        parts.append(f'<span style="font-size:11px;color:#888">{stats_str}</span>')
        return " · ".join(parts)


def _diamond_html(bases: dict) -> str:
    """3x3 CSS grid diamond. Gold = occupied, dark grey = empty."""
    def sq(occ: bool) -> str:
        bg = "#f1c40f" if occ else "#2a2a2a"
        return f'<div style="width:9px;height:9px;background:{bg};transform:rotate(45deg);border:1px solid #444"></div>'
    e = '<div></div>'
    f = bases.get("first", False)
    s = bases.get("second", False)
    t = bases.get("third", False)
    return (
        f'<div style="display:inline-grid;grid-template-columns:11px 11px 11px;'
        f'grid-template-rows:11px 11px 11px;gap:1px">'
        f'{e}{sq(s)}{e}'
        f'{sq(t)}{e}{sq(f)}'
        f'{e}{e}{e}'
        f'</div>'
    )


def _batter_detail_html(ba: dict, align: str = "left") -> str:
    """Compact batter line: AB/H summary, hit types, R, RBI."""
    if not ba or not ba.get("name"):
        return ""
    parts = []
    if ba.get("ab", 0) > 0:
        parts.append(f'{ba.get("hits",0)}-{ba.get("ab",0)}')
    hit_types = []
    if ba.get("doubles", 0):  hit_types.append(f'{ba["doubles"]}2B')
    if ba.get("triples", 0):  hit_types.append(f'{ba["triples"]}3B')
    if ba.get("hr", 0):       hit_types.append(f'{ba["hr"]}HR')
    if ba.get("k", 0):        hit_types.append(f'{ba["k"]}K')
    if hit_types:
        parts.append(" ".join(hit_types))
    if ba.get("runs", 0):   parts.append(f'R:{ba["runs"]}')
    if ba.get("rbi", 0):    parts.append(f'RBI:{ba["rbi"]}')
    stat_line = "  ".join(parts)
    ta = "right" if align == "right" else "left"
    return (
        f'<div style="font-size:11px;margin-top:4px;text-align:{ta}">'
        f'<span style="color:#f39c12">● {ba["name"]}</span>'
        f'<span style="color:#777"> {stat_line}</span>'
        f'</div>'
    )


def _render_daily_schedule(games: list[dict]) -> None:
    """Render schedule using st.components.v1.html to avoid Streamlit HTML sanitisation."""
    import streamlit.components.v1 as components

    your_bets = _load_today_bets()   # {matchup: [{"desc": str, "outcome": str}]}

    live_states: dict[int, dict] = {}
    for g in games:
        if g["state"] == "in" and g.get("game_pk"):
            try:
                s = get_live_game_state(g["game_pk"])
                if s:
                    live_states[g["game_pk"]] = s
            except Exception:
                pass

    rows_html = []
    for g in games:
        state = g["state"]
        pk    = g.get("game_pk")
        live  = live_states.get(pk) if pk else None

        time_cell   = ("🔴 " if state == "in" else "") + g["time"]
        row_opacity = "opacity:0.4;" if state == "post" else ""

        wx      = live["weather"] if live and live.get("weather") else g.get("weather", {})
        wx_text = _weather_and_park_text(wx, g["home"])

        # ── Pitchers ───────────────────────────────────────────────────
        if live:
            ps   = live.get("pitching_side", "home")
            hp   = live.get("home_pitcher", {})
            ap   = live.get("away_pitcher", {})
            away_p = _render_pitcher_line(ap, ps == "away", right_align=False)
            home_p = _render_pitcher_line(hp, ps == "home", right_align=True)
        else:
            def _sp(name, right=False):
                if name:
                    label = f'<span style="font-size:10px;color:#3498db;font-weight:bold">SP</span> <span style="font-size:11px;color:#888">{name}</span>'
                    return f'<div style="text-align:right">{label}</div>' if right else label
                tbd = '<span style="font-size:11px;color:#444">SP: TBD</span>'
                return f'<div style="text-align:right">{tbd}</div>' if right else tbd
            away_p = _sp(g.get("away_sp"))
            home_p = _sp(g.get("home_sp"), right=True)

        # ── Batter (live only) ─────────────────────────────────────────
        away_batter = home_batter = ""
        on_deck_html = ""
        if live:
            ba  = live.get("batter", {})
            od  = live.get("on_deck", "")
            if live.get("batting_side") == "away":
                away_batter  = _batter_detail_html(ba, "left")
                on_deck_html = f'<div style="font-size:10px;color:#555;margin-top:1px">On deck: {od or "—"}</div>'
            else:
                home_batter  = _batter_detail_html(ba, "right")
                on_deck_html = f'<div style="font-size:10px;color:#555;margin-top:1px;text-align:right">On deck: {od or "—"}</div>'

        # ── Center column ──────────────────────────────────────────────
        center_parts = ['<div style="font-size:16px;color:#fff;font-weight:bold;line-height:1">@</div>']
        if g["series"]:
            center_parts.append(f'<div style="font-size:10px;color:#777;margin-top:2px">{g["series"]}</div>')
        if g["travel"]:
            center_parts.append(f'<div style="font-size:10px;color:#555">{g["travel"]}</div>')
        if wx_text:
            center_parts.append(f'<div style="font-size:10px;color:#666;margin-top:4px">{wx_text}</div>')
        center_html = "\n".join(center_parts)

        # ── Score column ───────────────────────────────────────────────
        away_logo = g.get("away_logo", "")
        home_logo = g.get("home_logo", "")
        logo_a = f'<img src="{away_logo}" style="width:28px;height:28px;object-fit:contain;vertical-align:middle">' if away_logo else ""
        logo_h = f'<img src="{home_logo}" style="width:28px;height:28px;object-fit:contain;vertical-align:middle">' if home_logo else ""

        if live:
            inning_half = live.get("inning_half", "Top")
            arrow  = "▲" if inning_half.lower() == "top" else "▼"
            inning = live.get("inning", "?")
            b  = live.get("balls", 0)
            s2 = live.get("strikes", 0)
            o  = live.get("outs", 0)
            diamond = _diamond_html(live.get("bases", {}))
            ah = live.get("away_hits", 0)
            hh = live.get("home_hits", 0)
            away_score = live.get("away_score", 0)
            home_score = live.get("home_score", 0)
            score_html = (
                f'<div style="text-align:center">'
                f'<div style="font-size:10px;color:#888;margin-bottom:2px">{arrow}{inning} &nbsp; {b}-{s2} &nbsp; {o} out</div>'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:6px;font-size:15px;font-weight:bold">'
                f'{logo_a}{away_score} <span style="color:#555;font-size:12px">–</span> {home_score}{logo_h}'
                f'</div>'
                f'<div style="font-size:10px;color:#666;margin-top:2px">{ah}H &nbsp;·&nbsp; {hh}H</div>'
                f'<div style="margin:4px auto;display:flex;justify-content:center">{diamond}</div>'
                f'</div>'
            )
        elif state == "post":
            score_html = (
                f'<div style="text-align:center">'
                f'<div style="font-size:10px;color:#555">Final</div>'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:6px;font-weight:bold">'
                f'{logo_a}{g["score"]}{logo_h}</div>'
                f'</div>'
            )
        else:
            score_html = (
                f'<div style="text-align:center">'
                f'<div style="display:flex;align-items:center;justify-content:center;gap:6px">'
                f'{logo_a}<span style="font-size:11px;color:#555">{g["detail"] or "—"}</span>{logo_h}'
                f'</div></div>'
            )

        matchup_key = f"{g['away']} @ {g['home']}"
        bet_entries = your_bets.get(matchup_key, [])
        if bet_entries:
            _outcome_color = {"Win": "#2ecc71", "Loss": "#e74c3c", "Push": "#95a5a6", "Pending": "#f1c40f"}
            _outcome_icon  = {"Win": "✅", "Loss": "❌", "Push": "↩️", "Pending": "🎯"}
            lines = []
            for e in bet_entries:
                color = _outcome_color.get(e["outcome"], "#f1c40f")
                icon  = _outcome_icon.get(e["outcome"], "🎯")
                tag   = "🔗 Parlay" if str(e.get("bet_type", "Single")) == "Parlay" else "Single"
                tag_color = "#9b59b6" if tag == "🔗 Parlay" else "#555"
                lines.append(
                    f'<span style="color:{color};font-size:11px;font-weight:600">'
                    f'{icon} {_abbrev_bet_desc(e["desc"])}</span>'
                    f'<br><span style="color:{tag_color};font-size:10px">{tag}</span>'
                )
            bet_cell = "<br>".join(lines)
        else:
            bet_cell = '<span style="color:#444;font-size:11px">—</span>'

        row = f"""
        <tr style="border-bottom:1px solid #1c1c1c;{row_opacity}">
          <td style="padding:8px 12px;vertical-align:top;white-space:nowrap;font-size:12px;color:#ccc;text-align:center">{time_cell}</td>
          <td style="padding:8px 12px;vertical-align:top">
            <div style="display:flex;align-items:flex-start;gap:12px">
              <div style="flex:1;text-align:left">
                <div style="font-weight:600;font-size:13px">{g["away"]}</div>
                <div style="font-size:11px;color:#888;margin:1px 0">{g["away_sub"]}</div>
                <div style="margin-top:2px">{away_p}</div>
                {away_batter}
                {on_deck_html if live and live.get("batting_side") == "away" else ""}
              </div>
              <div style="text-align:center;flex-shrink:0;padding-top:2px;min-width:90px">{center_html}</div>
              <div style="flex:1;text-align:right">
                <div style="font-weight:600;font-size:13px">{g["home"]}</div>
                <div style="font-size:11px;color:#888;margin:1px 0">{g["home_sub"]}</div>
                <div style="margin-top:2px">{home_p}</div>
                {home_batter}
                {on_deck_html if live and live.get("batting_side") == "home" else ""}
              </div>
            </div>
          </td>
          <td style="padding:8px 12px;vertical-align:top;min-width:140px;text-align:center">{score_html}</td>
          <td style="padding:8px 6px;vertical-align:top;width:90px;max-width:90px;text-align:center">{bet_cell}</td>
        </tr>"""
        rows_html.append(row)

    html = f"""<!DOCTYPE html>
    <html><head><style>
      body {{ margin:0; background:#0f0f0f; color:#e0e0e0;
              font-family:Arial,sans-serif; font-size:13px; }}
      table {{ border-collapse:collapse; width:100%; }}
      th {{ background:#1a1a2e; color:#3498db; padding:8px 12px;
            text-align:center; border-bottom:2px solid #3498db; font-size:12px; }}
    </style></head><body>
    <table>
      <thead><tr>
        <th style="width:90px">Time</th>
        <th>Matchup</th>
        <th style="width:150px">Score</th>
        <th style="width:90px">Bet</th>
      </tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table></body></html>"""

    # Live rows are taller (batter + on-deck lines); pre-game rows ~90px
    live_count = sum(1 for g in games if g["state"] == "in")
    pre_count  = len(games) - live_count
    height = max(live_count * 140 + pre_count * 95, 300)
    components.html(html, height=height, scrolling=True)


def _build_tomorrow_predictions(
    team_stats: pd.DataFrame | None,
    pitcher_stats: pd.DataFrame | None,
    bankroll: float,
) -> pd.DataFrame:
    """
    Build a predictions DataFrame for tomorrow using probable starters + team stats.
    No odds available, so edge/kelly columns are absent. Returns same schema as
    games_with_kelly minus odds-derived columns.
    """
    from src.data.game_results import get_probable_starters
    from src.models.win_probability import compute_win_probabilities
    from src.data.ballpark import get_bullpen_stats

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    if team_stats is None or team_stats.empty:
        return pd.DataFrame()

    # Get tomorrow's probable starters
    try:
        starters = get_probable_starters(tomorrow)
    except Exception:
        starters = pd.DataFrame()

    if starters.empty:
        return pd.DataFrame()

    # Build a stub games DataFrame from starters (no odds)
    games_stub = starters[["home_team", "away_team", "home_starter", "away_starter",
                            "home_starter_announced", "away_starter_announced"]].copy()

    # Placeholder odds columns — NaN so model skips Kelly but still computes win prob
    for col in ["home_odds", "away_odds", "home_implied_prob", "away_implied_prob",
                "home_rl", "away_rl", "home_rl_odds", "away_rl_odds",
                "total_line", "over_odds", "under_odds"]:
        games_stub[col] = float("nan")

    def _lineup_status(row) -> str:
        h = isinstance(row.get("home_starter"), str) and bool(row.get("home_starter"))
        a = isinstance(row.get("away_starter"), str) and bool(row.get("away_starter"))
        if h and a:   return "Full"
        if h or a:    return "Partial"
        return "None"

    games_stub["lineup_status"] = games_stub.apply(_lineup_status, axis=1)

    try:
        bullpen = get_bullpen_stats(tomorrow.year)
    except Exception:
        bullpen = pd.DataFrame()

    ps = pitcher_stats if pitcher_stats is not None else pd.DataFrame()
    games_with_prob = compute_win_probabilities(games_stub, team_stats, ps, bullpen_df=bullpen)

    # No Kelly (no odds) — add stub columns so render() doesn't crash
    for col in ["best_bet_side", "best_bet_edge", "home_edge_pct", "away_edge_pct",
                "home_kelly_pct", "away_kelly_pct", "home_dollar_stake", "away_dollar_stake",
                "home_recommendation", "away_recommendation"]:
        if col not in games_with_prob.columns:
            games_with_prob[col] = "PASS" if "side" in col or "recommendation" in col else 0.0

    games_with_prob["best_bet_side"] = "PASS"
    games_with_prob["best_bet_edge"] = 0.0
    return games_with_prob


def _load_today_bets() -> dict[str, list[dict]]:
    """
    Return {matchup: [{"desc", "outcome", "bet_type", "parlay_id"}, ...]} for today's bets.
    """
    import datetime as _dt
    try:
        log = load_bets()
        if log.empty:
            return {}
        today_str = str(_dt.date.today())
        today_bets = log[log["date"] == today_str]
        lookup: dict[str, list[dict]] = {}
        for _, row in today_bets.iterrows():
            key = str(row.get("matchup", ""))
            if not key:
                continue
            lookup.setdefault(key, []).append({
                "desc":      str(row.get("bet_side", "")),
                "outcome":   str(row.get("outcome", "Pending")),
                "bet_type":  str(row.get("bet_type", "Single")),
                "parlay_id": row.get("parlay_id"),
            })
        return lookup
    except Exception:
        return {}


def _today_bets_summary() -> list[dict] | None:
    """Return all of today's bets as a flat list for the summary banner, or None if none."""
    import datetime as _dt
    try:
        log = load_bets()
        if log.empty:
            return None
        today_str = str(_dt.date.today())
        today_bets = log[log["date"] == today_str]
        if today_bets.empty:
            return None
        return today_bets.to_dict("records")
    except Exception:
        return None


def _render_games_analysis(
    df: pd.DataFrame,
    bankroll: float,
    team_stats: pd.DataFrame | None,
    pitcher_stats: pd.DataFrame | None,
    is_tomorrow: bool = False,
) -> None:
    """Render the full analysis section — bet table, game details, recommended bets."""
    import streamlit.components.v1 as components

    today = datetime.date.today()

    if df is None or df.empty:
        if is_tomorrow:
            st.info("No probable starters found for tomorrow yet. Check back later.")
        else:
            st.warning("No games loaded. Check your ODDS_API_KEY in .env and ensure there are upcoming MLB games.")
        return

    rpg = _build_runs_per_game(team_stats)

    # Load today's pending bets for the "Your Bet" column
    your_bets: dict[str, list[str]] = {} if is_tomorrow else _load_today_bets()

    # Auto-save predictions — today only, full lineups only (both starters confirmed)
    if not is_tomorrow and df["home_model_prob"].notna().any():
        try:
            full_df = df[df["lineup_status"] == "Full"] if "lineup_status" in df.columns else df
            if not full_df.empty:
                save_predictions(full_df, today.isoformat())
        except Exception:
            pass

    has_prob = df["home_model_prob"].notna() & df["away_model_prob"].notna()
    ready_df   = df[has_prob].copy()
    pending_df = df[~has_prob].copy()

    ready_df["matchup"]   = ready_df.apply(_format_matchup, axis=1)
    pending_df["matchup"] = pending_df.apply(_format_matchup, axis=1)

    # ---- Lineup status legend ----
    status_counts = ready_df["lineup_status"].value_counts() if "lineup_status" in ready_df.columns else {}
    full_n    = status_counts.get("Full", 0)
    partial_n = status_counts.get("Partial", 0)
    none_n    = status_counts.get("None", 0) + len(pending_df)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f'{_STATUS_HTML["Full"]} &nbsp; **{full_n}** game(s)', unsafe_allow_html=True)
    lc2.markdown(f'{_STATUS_HTML["Partial"]} &nbsp; **{partial_n}** game(s)', unsafe_allow_html=True)
    lc3.markdown(f'{_STATUS_HTML["None"]} &nbsp; **{none_n}** game(s)', unsafe_allow_html=True)
    st.divider()

    if ready_df.empty and pending_df.empty:
        return

    # ---- Filters ----
    all_df = pd.concat([ready_df, pending_df], ignore_index=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        all_teams     = sorted(set(all_df["home_team"].tolist() + all_df["away_team"].tolist()))
        tab_key       = "tmr" if is_tomorrow else "today"
        selected_team = st.selectbox("Filter by team", ["All"] + all_teams, key=f"team_filter_{tab_key}")
    with col2:
        min_edge = st.slider("Min edge %", 0.0, 20.0, 3.0, 0.5, key=f"min_edge_{tab_key}")
    with col3:
        show_only_bets = st.checkbox("BET recommendations only", value=False, key=f"bets_only_{tab_key}")

    # Table view: team filter only — all games appear
    table_ready = ready_df.copy()
    if selected_team != "All":
        table_ready = table_ready[
            (table_ready["home_team"] == selected_team) | (table_ready["away_team"] == selected_team)
        ]
    # Determine sort column from best-performing bet type in logged history
    _BEST_TYPE_SORT = {
        "ML":  "best_bet_edge",
        "RL":  "best_rl_edge_pct",
        "O/U": "best_total_edge_pct",
    }
    _best_bet_type = get_best_bet_type(min_bets=3)
    _sort_col = _BEST_TYPE_SORT.get(_best_bet_type or "", "best_bet_edge")
    if _sort_col not in table_ready.columns:
        _sort_col = "best_bet_edge"
    _lineup_order = {"Full": 0, "Partial": 1, "None": 2}
    table_ready["_lineup_sort"] = table_ready.get("lineup_status", pd.Series(dtype=str)).map(_lineup_order).fillna(3)
    table_ready = table_ready.sort_values(["_lineup_sort", _sort_col], ascending=[True, False])
    table_ready = table_ready.drop(columns=["_lineup_sort"])

    table_pending = pending_df.copy()
    if selected_team != "All":
        table_pending = table_pending[
            (table_pending["home_team"] == selected_team) | (table_pending["away_team"] == selected_team)
        ]

    # Recommended bets: all three filters applied
    filtered_ready = table_ready.copy()
    if show_only_bets:
        filtered_ready = filtered_ready[filtered_ready["best_bet_side"] != "PASS"]
    filtered_ready = filtered_ready[filtered_ready["best_bet_edge"] >= min_edge]

    filtered_pending = table_pending.copy()
    if show_only_bets:
        filtered_pending = pd.DataFrame(columns=pending_df.columns)

    st.caption(f"{len(table_ready) + len(table_pending)} game(s) shown | Bankroll: ${bankroll:,.0f}")

    # ---- Pick summary columns ----
    def _ml_pick(row) -> str:
        side = row.get("best_bet_side", "PASS")
        if side == "PASS":
            return "PASS"
        t = row.get("home_team") if side == "HOME" else row.get("away_team")
        odds_key = "home_odds" if side == "HOME" else "away_odds"
        return f"{t} {_fmt_odds(row.get(odds_key))}"

    def _rl_pick(row) -> str:
        side = row.get("best_rl_side")
        if not side or pd.isna(side) or str(side).upper() == "PASS":
            return "—"
        side_upper = str(side).upper()
        t = row.get("home_team") if side_upper == "HOME" else row.get("away_team")
        rl_key      = "home_rl"      if side_upper == "HOME" else "away_rl"
        rl_odds_key = "home_rl_odds" if side_upper == "HOME" else "away_rl_odds"
        rl = row.get(rl_key)
        if rl is None or (isinstance(rl, float) and pd.isna(rl)):
            return "—"
        sign = "+" if float(rl) > 0 else ""
        return f"{t} {sign}{rl} ({_fmt_odds(row.get(rl_odds_key))})"

    def _total_pick(row) -> str:
        total = row.get("total_line")
        if total is None or (isinstance(total, float) and pd.isna(total)):
            return "N/A"
        over_o  = _fmt_odds(row.get("over_odds"))
        under_o = _fmt_odds(row.get("under_odds"))
        proj = _proj_total_for(row, rpg)
        if proj is not None:
            return f"Over {total} ({over_o})" if proj >= float(total) else f"Under {total} ({under_o})"
        return f"O/U {total}  ({over_o} / {under_o})"

    def _rl_edge_str(row) -> str:
        he = row.get("home_rl_edge_pct")
        ae = row.get("away_rl_edge_pct")
        if he is not None or ae is not None:
            # Odds available — show edge of the recommended side (or best available)
            side = row.get("best_rl_side", "PASS")
            if side == "HOME":
                return f"{he:+.1f}%"
            elif side == "AWAY":
                return f"{ae:+.1f}%"
            # Both below threshold — show the less-negative edge
            best = max(he or -99.0, ae or -99.0)
            return f"{best:+.1f}%"
        # No market odds — fall back to model cover probability as directional signal
        p = row.get("home_rl_cover_prob")
        if p is not None:
            p = float(p)
            if p >= 0.5:
                return f"Home {p:.0%}"
            else:
                return f"Away {1 - p:.0%}"
        return "—"

    def _total_edge_str(row) -> str:
        oe = row.get("over_edge_pct")
        ue = row.get("under_edge_pct")
        if oe is not None or ue is not None:
            # Odds available — show edge of the recommended direction
            d = row.get("best_total_direction")
            e = row.get("best_total_edge_pct", 0.0)
            if d is None:
                return "—"
            return f"{e:+.1f}%"
        # No market odds — fall back to model over probability as directional signal
        p = row.get("total_over_prob")
        if p is not None:
            p = float(p)
            d = "Over" if p >= 0.5 else "Under"
            prob = p if p >= 0.5 else 1 - p
            return f"{d} {prob:.0%}"
        return "—"

    def _your_bet_cell(row) -> str:
        matchup = row.get("matchup", "")
        entries = your_bets.get(matchup, [])
        return " · ".join(e["desc"] for e in entries) if entries else "—"

    if not table_ready.empty:
        table_ready = table_ready.copy()
        table_ready["_ml_pick"]       = table_ready.apply(_ml_pick, axis=1)
        table_ready["_rl_pick"]       = table_ready.apply(_rl_pick, axis=1)
        table_ready["_total_pick"]    = table_ready.apply(_total_pick, axis=1)
        table_ready["_rl_edge_str"]   = table_ready.apply(_rl_edge_str, axis=1)
        table_ready["_total_edge_str"] = table_ready.apply(_total_edge_str, axis=1)
        table_ready["_complete"]      = True
        table_ready["_your_bet"]      = table_ready.apply(_your_bet_cell, axis=1)
        # Blank out bet predictions for games without both starters confirmed
        if "lineup_status" in table_ready.columns:
            no_full = table_ready["lineup_status"] != "Full"
            for _col in ["_ml_pick", "_rl_pick", "_total_pick", "_rl_edge_str", "_total_edge_str"]:
                table_ready.loc[no_full, _col] = "—"
    if not table_pending.empty:
        table_pending = table_pending.copy()
        table_pending["_ml_pick"]        = "—"
        table_pending["_rl_pick"]        = "—"
        table_pending["_total_pick"]     = "—"
        table_pending["_rl_edge_str"]    = "—"
        table_pending["_total_edge_str"] = "—"
        table_pending["_complete"]       = False
        table_pending["_your_bet"]       = table_pending.apply(_your_bet_cell, axis=1)

    # Propagate pick columns to filtered_ready for recommended bets section
    if not filtered_ready.empty and "_ml_pick" not in filtered_ready.columns:
        filtered_ready = filtered_ready.merge(
            table_ready[["home_team", "away_team", "_ml_pick", "_rl_pick", "_total_pick", "_complete"]],
            on=["home_team", "away_team"],
            how="left",
        )

    combined = pd.concat([table_ready, table_pending], ignore_index=True)

    # ---- Build "what's missing" per game ----
    def _missing_reasons(row) -> list[str]:
        reasons = []
        if not row.get("_complete", True):
            reasons.append("no team stats matched")
        away_sp = row.get("away_starter") or ""
        home_sp = row.get("home_starter") or ""
        if not away_sp and not home_sp:
            reasons.append("no starters announced")
        elif not away_sp:
            reasons.append(f"no {row.get('away_team','away')} starter")
        elif not home_sp:
            reasons.append(f"no {row.get('home_team','home')} starter")
        return reasons

    combined["_missing"] = combined.apply(lambda r: _missing_reasons(r), axis=1)
    combined["_incomplete"] = combined["_missing"].apply(lambda m: len(m) > 0)

    # ---- Main Line Game Bets table ----
    st.subheader("Main Line Game Bets")
    if _best_bet_type:
        _col_label = {"ML": "ML Edge%", "RL": "RL Edge%", "O/U": "Total Edge%"}.get(_best_bet_type, _best_bet_type)
        st.caption(
            f"Sorted by **{_col_label}** — historically your best-performing bet type "
            f"(**{_best_bet_type}**). That column is highlighted teal. "
            "Track more bets in the Bet Log to update this automatically."
        )

    # ---- Render unified styled table via components.html ----
    display_cols_order = [
        ("matchup",            "Matchup"),
        ("lineup_status",      "Lineup"),
        ("away_starter",       "Away SP"),
        ("home_starter",       "Home SP"),
        ("_ml_pick",           "ML Pick"),
        ("best_bet_edge",      "ML Edge%"),
        ("_rl_pick",           "Run Line"),
        ("_rl_edge_str",       "RL Edge%"),
        ("_total_pick",        "Total"),
        ("_total_edge_str",    "Total Edge%"),
    ]
    available_cols = [(k, v) for k, v in display_cols_order if k in combined.columns]

    # Map bet type to its edge column key for teal highlight
    _BEST_TYPE_COL = {"ML": "best_bet_edge", "RL": "_rl_edge_str", "O/U": "_total_edge_str"}
    _teal_col_key = _BEST_TYPE_COL.get(_best_bet_type or "")

    header_html = "".join(
        f'<th style="background:#0d3030;color:#1abc9c;border-bottom:2px solid #1abc9c">{label}</th>'
        if col_key == _teal_col_key else
        f"<th>{label}</th>"
        for col_key, label in available_cols
    )
    row_html_parts = []
    for _, row in combined.iterrows():
        incomplete = row.get("_incomplete", False)
        style = "opacity:0.4;" if incomplete else ""
        strike_open  = "<s>" if incomplete else ""
        strike_close = "</s>" if incomplete else ""
        cells = []
        for col_key, _ in available_cols:
            val = row.get(col_key, "")
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = "—"
            if col_key in ("home_model_prob", "away_model_prob"):
                try:
                    val = f"{float(val)*100:.1f}%"
                except (TypeError, ValueError):
                    val = "—"
            elif col_key == "best_bet_edge":
                if str(row.get("lineup_status", "None")) != "Full":
                    val = "—"
                else:
                    try:
                        fv = float(val)
                        if col_key == _teal_col_key:
                            colour = "#1abc9c" if fv >= 3.0 else "#e74c3c" if fv < 0 else "#888"
                        else:
                            colour = "#2ecc71" if fv >= 3.0 else "#e74c3c" if fv < 0 else "#888"
                        val = f'<span style="color:{colour};font-weight:600">{fv:.1f}%</span>'
                    except (TypeError, ValueError):
                        val = "—"
            elif col_key in ("_rl_edge_str", "_total_edge_str"):
                s = str(val)
                if s not in ("—", ""):
                    try:
                        fv = float(s.replace("%", ""))
                        if col_key == _teal_col_key:
                            colour = "#1abc9c" if fv >= 3.0 else "#e74c3c" if fv < 0 else "#888"
                        else:
                            colour = "#2ecc71" if fv >= 3.0 else "#e74c3c" if fv < 0 else "#888"
                        val = f'<span style="color:{colour};font-weight:600">{s}</span>'
                    except ValueError:
                        pass
            elif col_key == "lineup_status":
                colours = {"Full": "#2ecc71", "Partial": "#e67e22", "None": "#888"}
                c = colours.get(str(val), "#888")
                val = f'<span style="color:{c};font-weight:600">{val}</span>'
            elif col_key == "_your_bet":
                if str(val) not in ("—", ""):
                    val = f'<span style="color:#f1c40f;font-weight:600">{val}</span>'
            elif col_key in ("_ml_pick", "_rl_pick"):
                if str(val) not in ("PASS", "—", ""):
                    val = f'<span style="color:#3498db;font-weight:600">{val}</span>'
            cells.append(f"<td>{strike_open}{val}{strike_close}</td>")
        row_html_parts.append(f'<tr style="border-bottom:1px solid #1c1c1c;{style}">{"".join(cells)}</tr>')

    table_html = f"""<!DOCTYPE html><html><head><style>
      body{{margin:0;background:#0f0f0f;color:#e0e0e0;font-family:Arial,sans-serif;font-size:12px}}
      table{{border-collapse:collapse;width:100%}}
      th{{background:#1a1a2e;color:#3498db;padding:6px 10px;text-align:left;
          border-bottom:2px solid #3498db;font-size:11px;white-space:nowrap}}
      td{{padding:6px 10px;vertical-align:middle;white-space:nowrap}}
      tr:hover td{{background:#1a1a1a}}
    </style></head><body>
    <table><thead><tr>{header_html}</tr></thead>
    <tbody>{"".join(row_html_parts)}</tbody></table>
    </body></html>"""

    table_height = len(combined) * 29 + 32  # 29px/row + 32px header
    components.html(table_html, height=max(table_height, 64), scrolling=True)

    # ---- Missing data summary ----
    missing_games = combined[combined["_incomplete"]].copy()
    if not missing_games.empty:
        st.markdown("**Missing data (analysis incomplete):**")
        lines = []
        for _, row in missing_games.iterrows():
            matchup = row.get("matchup", "?")
            reasons = row.get("_missing", [])
            lines.append(f"- **{matchup}** — {', '.join(reasons)}")
        st.markdown("\n".join(lines))

    st.divider()

    # ---- Game Details — all games ----------------------------------------
    detail_label = "Tomorrow's Game Details" if is_tomorrow else "Game Details"
    detail_caption = (
        "Model win probabilities and projected scores for tomorrow's games. Odds not yet available."
        if is_tomorrow
        else "Model signals, projected scores, and AI analysis for every game today."
    )
    st.subheader(detail_label)
    st.caption(detail_caption)

    all_detail = pd.concat([table_ready, table_pending], ignore_index=True)
    if show_only_bets and "best_bet_side" in all_detail.columns:
        all_detail = all_detail[all_detail["best_bet_side"] != "PASS"]
    if "best_bet_edge" in all_detail.columns:
        _lo = {"Full": 0, "Partial": 1, "None": 2}
        all_detail["_lineup_sort"] = all_detail.get("lineup_status", pd.Series(dtype=str)).map(_lo).fillna(3)
        all_detail = all_detail.sort_values(
            ["_lineup_sort", "_complete", "best_bet_edge"], ascending=[True, False, False]
        ).drop(columns=["_lineup_sort"]).reset_index(drop=True)

    for idx, row in all_detail.iterrows():
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        matchup_key = f"{away} @ {home}"
        edge = row.get("best_bet_edge", 0.0) or 0.0
        bet_side = row.get("best_bet_side", "PASS")
        is_bet = bet_side != "PASS" and row.get("_complete", True)
        your_bet_entries = your_bets.get(matchup_key, [])
        _oc = {"Win": "✅", "Loss": "❌", "Push": "↩️", "Pending": "🎯"}
        bet_badge   = f"  🔵 BET {bet_side}  {edge:+.1f}%" if is_bet else ""
        your_badge  = ("  " + "  ".join(
            f"{_oc.get(e['outcome'], '🎯')} {e['desc']}" for e in your_bet_entries
        )) if your_bet_entries else ""
        label = f"{matchup_key}{bet_badge}{your_badge}"
        card_key_suffix = f"{'tmr' if is_tomorrow else 'today'}_{idx}"
        with st.expander(label, expanded=bool(your_bet_entries)):
            _render_game_detail_card(row, bankroll, team_stats, pitcher_stats, rpg)

    if is_tomorrow:
        return  # No recommended bets section for tomorrow (no odds/edge yet)

    st.divider()

    # ---- Recommended Bets grouped by lineup status (today only) ----
    if filtered_ready.empty:
        return

    st.subheader("Recommended Bets")

    # Only show bets where both starting lineups are confirmed
    if "lineup_status" in filtered_ready.columns:
        bets = filtered_ready[
            (filtered_ready["best_bet_side"] != "PASS") &
            (filtered_ready["lineup_status"] == "Full")
        ].sort_values("best_bet_edge", ascending=False)
    else:
        bets = filtered_ready[filtered_ready["best_bet_side"] != "PASS"].sort_values("best_bet_edge", ascending=False)

    if bets.empty:
        st.info("No bets with confirmed starting lineups meet the edge threshold today. Check back once starters are announced.")
        return

    st.markdown(f'### {_STATUS_HTML["Full"]} &nbsp; Both Starters Confirmed', unsafe_allow_html=True)
    st.caption("FIP, BABIP, and bullpen adjustments applied. Predictions only available with full starting lineups.")
    _render_bet_cards(bets, bankroll, team_stats, pitcher_stats, rpg)


def _render_formula_banner(team_stats: pd.DataFrame | None) -> None:
    """Show a seasonal formula state banner explaining which model mode is active."""
    from src.models.win_probability import get_formula_state
    if team_stats is None or team_stats.empty:
        return
    state = get_formula_state(team_stats)
    if state["state"] == "EARLY_SEASON":
        st.info(state["message"])
    else:
        st.success(state["message"])


def render(
    games_with_kelly: pd.DataFrame,
    bankroll: float,
    team_stats: pd.DataFrame | None = None,
    pitcher_stats: pd.DataFrame | None = None,
) -> None:
    today = datetime.date.today()

    # Settle any pending bets from completed games (runs silently)
    try:
        from src.data.bet_log_db import settle_pending_bets as _settle
        n = _settle()
        if n > 0:
            st.toast(f"Auto-settled {n} bet(s) from completed games.", icon="✅")
    except Exception:
        pass

    _render_formula_banner(team_stats)

    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.header(f"Today's Games — {today.strftime('%A, %B %-d, %Y')}")
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", help="Fetches fresh odds and lineup announcements"):
            _clear_odds_cache()
            st.cache_data.clear()
            st.rerun()

    # ── Today's Logged Bets Summary ───────────────────────────────────────────
    summary_bets = _today_bets_summary()
    if summary_bets:
        _oc = {"Win": "#2ecc71", "Loss": "#e74c3c", "Push": "#95a5a6", "Pending": "#f1c40f"}
        _oi = {"Win": "✅", "Loss": "❌", "Push": "↩️", "Pending": "🎯"}

        singles = [b for b in summary_bets if str(b.get("bet_type", "Single")) != "Parlay"]
        parlay_groups: dict[int, list[dict]] = {}
        for b in summary_bets:
            if str(b.get("bet_type", "Single")) == "Parlay":
                pid = int(b.get("parlay_id") or 0)
                parlay_groups.setdefault(pid, []).append(b)

        all_cards = (
            [("single", b) for b in singles]
            + [("parlay", (pid, legs)) for pid, legs in parlay_groups.items()]
        )

        for chunk_start in range(0, len(all_cards), 4):
            chunk = all_cards[chunk_start : chunk_start + 4]
            cols = st.columns(len(chunk))
            for col, (ctype, card_data) in zip(cols, chunk):
                with col:
                    if ctype == "single":
                        b = card_data
                        outcome = str(b.get("outcome", "Pending"))
                        color   = _oc.get(outcome, "#f1c40f")
                        icon    = _oi.get(outcome, "🎯")
                        stake   = float(b.get("stake") or 0)
                        pnl     = float(b.get("pnl") or 0)
                        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                        bet_id  = int(b.get("id", 0))
                        st.markdown(
                            f'<div style="border:1px solid {color};border-radius:8px;'
                            f'padding:8px 12px;margin-bottom:4px">'
                            f'<div style="font-size:10px;color:#888">Single</div>'
                            f'<div style="font-size:12px;font-weight:600;color:{color}">'
                            f'{icon} {_abbrev_bet_desc(str(b.get("bet_side",""))[:40])}</div>'
                            f'<div style="font-size:11px;color:#aaa">{b.get("matchup","")}</div>'
                            f'<div style="font-size:11px;color:#888">Stake: ${stake:.2f} · '
                            f'P&L: <span style="color:{color}">{pnl_str}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.popover("✏️ Edit", use_container_width=True):
                            st.markdown(f"**{b.get('matchup', '')}**")
                            st.caption(
                                f"Bet: {b.get('bet_side', '')}  ·  "
                                f"Stake: ${stake:.2f}  ·  Line: {b.get('line', 'N/A')}"
                            )
                            new_outcome = st.selectbox(
                                "Outcome",
                                _BET_OUTCOMES,
                                index=_BET_OUTCOMES.index(outcome) if outcome in _BET_OUTCOMES else 0,
                                key=f"banner_outcome_{bet_id}",
                            )
                            line_val = float(b.get("line") or -110)
                            if new_outcome == "Win":
                                new_pnl = (
                                    round(stake * line_val / 100, 2) if line_val > 0
                                    else round(stake * 100 / abs(line_val), 2)
                                )
                            elif new_outcome == "Loss":
                                new_pnl = -abs(stake)
                            else:
                                new_pnl = 0.0
                            st.caption(f"Computed P&L: **${new_pnl:+.2f}**")
                            if st.button("Save", key=f"banner_save_{bet_id}"):
                                update_bet_outcome(bet_id, new_outcome, new_pnl)
                                st.rerun()

                    else:
                        pid, legs = card_data
                        total_stake = float(
                            next((l.get("stake") or 0 for l in legs if float(l.get("stake") or 0) > 0), 0)
                        )
                        total_pnl = sum(float(l.get("pnl") or 0) for l in legs)
                        outcome   = str(legs[0].get("outcome", "Pending"))
                        color     = _oc.get(outcome, "#f1c40f")
                        icon      = _oi.get(outcome, "🎯")
                        pnl_str   = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
                        leg_lines = "".join(
                            f'<div style="font-size:11px;color:#ccc">'
                            f'· {_abbrev_bet_desc(str(l.get("bet_side",""))[:35])}</div>'
                            for l in legs
                        )
                        st.markdown(
                            f'<div style="border:1px solid {color};border-radius:8px;'
                            f'padding:8px 12px;margin-bottom:4px">'
                            f'<div style="font-size:10px;color:#9b59b6">🔗 {len(legs)}-Leg Parlay</div>'
                            f'<div style="font-size:12px;font-weight:600;color:{color}">{icon} {outcome}</div>'
                            f'{leg_lines}'
                            f'<div style="font-size:11px;color:#888;margin-top:4px">'
                            f'Stake: ${total_stake:.2f} · P&L: <span style="color:{color}">{pnl_str}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.popover("✏️ Edit", use_container_width=True):
                            st.markdown(f"**{len(legs)}-Leg Parlay #{pid}**")
                            for l in legs:
                                st.caption(f"• {l.get('matchup', '')}  —  {l.get('bet_side', '')}")
                            new_outcome = st.selectbox(
                                "Outcome",
                                _BET_OUTCOMES,
                                index=_BET_OUTCOMES.index(outcome) if outcome in _BET_OUTCOMES else 0,
                                key=f"banner_parlay_outcome_{pid}",
                            )
                            new_stake = st.number_input(
                                "Total Stake ($)",
                                min_value=0.0,
                                value=max(total_stake, 1.0),
                                step=5.0,
                                key=f"banner_parlay_stake_{pid}",
                            )
                            if st.button("Save", key=f"banner_parlay_save_{pid}"):
                                update_parlay(pid, new_outcome, new_stake)
                                st.rerun()

    # ── Full Day Schedule ─────────────────────────────────────────────────────
    try:
        schedule = _fetch_daily_schedule()
    except Exception as _sched_err:
        st.warning(f"Schedule table failed to load: {_sched_err}")
        schedule = []

    # Check if ≥80% of today's games are final — unlock Tomorrow tab if so
    tomorrow = today + datetime.timedelta(days=1)
    finished  = sum(1 for g in schedule if g.get("state") == "post")
    total_g   = len(schedule)
    show_tomorrow = total_g > 0 and (finished / total_g) >= 0.80

    if show_tomorrow:
        tab1, tab2 = st.tabs([
            f"Today · {today.strftime('%a %b %-d')}",
            f"Tomorrow · {tomorrow.strftime('%a %b %-d')}",
        ])
        with tab1:
            if schedule:
                _render_daily_schedule(schedule)
            else:
                st.info("No games scheduled today.")
            st.divider()
            _render_games_analysis(games_with_kelly, bankroll, team_stats, pitcher_stats, is_tomorrow=False)
        with tab2:
            try:
                tmr_schedule = _fetch_schedule_for_date(tomorrow)
                if tmr_schedule:
                    st.caption(
                        f"Showing {len(tmr_schedule)} game(s) for tomorrow. "
                        "Starters listed if announced. Odds unavailable until ESPN posts lines (usually by late evening)."
                    )
                    _render_daily_schedule(tmr_schedule)
                else:
                    st.info(f"No games scheduled for {tomorrow.strftime('%A, %B %-d')}.")
            except Exception as _tmr_err:
                st.warning(f"Could not load tomorrow's schedule: {_tmr_err}")
            st.divider()
            tmr_df = _build_tomorrow_predictions(team_stats, pitcher_stats, bankroll)
            _render_games_analysis(tmr_df, bankroll, team_stats, pitcher_stats, is_tomorrow=True)
    else:
        if schedule:
            _render_daily_schedule(schedule)
        else:
            st.info("No games scheduled today.")
        st.divider()
        _render_games_analysis(games_with_kelly, bankroll, team_stats, pitcher_stats, is_tomorrow=False)
