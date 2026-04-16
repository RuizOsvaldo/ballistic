"""
Player Analysis page — deep-dive breakdown of every batter and pitcher prop today.

Shows the full model logic per player: BABIP regression, K%, wRC+, opposing pitcher
FIP, park factor, and AI reasoning. Auto-generates analysis — no button required.
"""

from __future__ import annotations

import datetime
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.baseball_stats import get_batter_stats, get_pitcher_stats
from src.data.game_results import get_today_lineups
from src.data.odds import get_mlb_odds, get_all_batter_hits_props
from src.data.ballpark import get_park_factor
from src.models.player_props import (
    project_batter_hits, compute_prop_edge, MIN_PROP_EDGE_PCT, LEAGUE_AVG_BABIP
)

CURRENT_SEASON = datetime.datetime.now().year

_FG_TEAM_MAP: dict[str, str] = {
    "ARI": "Arizona Diamondbacks", "ATH": "Athletics", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",    "BOS": "Boston Red Sox",  "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",     "DET": "Detroit Tigers",  "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",   "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",        "MIL": "Milwaukee Brewers",  "MIN": "Minnesota Twins",
    "NYM": "New York Mets",        "NYY": "New York Yankees",   "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",   "SDP": "San Diego Padres",   "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals","TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",        "TOR": "Toronto Blue Jays",  "WSN": "Washington Nationals",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower().strip()


def _babip_signal_color(babip: float) -> tuple[str, str]:
    """Return (label, hex_color) for a BABIP value."""
    if babip > 0.320:
        return "High — hitting lucky", "#e74c3c"
    if babip < 0.275:
        return "Low — underperforming", "#2ecc71"
    return "Normal", "#aaa"


def _babip_gauge(babip: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=babip,
        number={"valueformat": ".3f"},
        gauge={
            "axis": {"range": [0.200, 0.420], "tickformat": ".3f"},
            "bar":  {"color": "#3498db"},
            "steps": [
                {"range": [0.200, 0.275], "color": "#1a5c1a"},
                {"range": [0.275, 0.320], "color": "#2a2a2a"},
                {"range": [0.320, 0.420], "color": "#5c1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 0.300,
            },
        },
        title={"text": "BABIP (white line = .300 avg)"},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20))
    return fig


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_prop_ai(player: str, team: str, prop_type: str, line: float,
                    proj: float, edge: float, direction: str, signals_json: str) -> dict:
    import json as _json
    from src.shared.groq_agent import analyze_mlb_prop
    return analyze_mlb_prop(
        player_name=player, team=team, prop_type=prop_type,
        line=line, model_projection=proj, edge_pct=edge,
        bet_direction=direction,
        signals=_json.loads(signals_json),
    )


@st.cache_data(ttl=3600 * 3, show_spinner="Loading batter stats...")
def _load_batters(season: int) -> pd.DataFrame:
    try:
        return get_batter_stats(season, min_pa=10)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 3, show_spinner="Loading pitcher stats...")
def _load_pitchers(season: int) -> pd.DataFrame:
    try:
        return get_pitcher_stats(season)
    except Exception:
        return pd.DataFrame()


def _build_batter_rows(
    batter_df: pd.DataFrame,
    games_df: pd.DataFrame,
    props_df: pd.DataFrame,
    lineups_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
) -> list[dict]:
    """Combine batter stats, prop lines, lineup, and opposing pitcher into one row per player."""
    if batter_df.empty or games_df.empty:
        return []

    today_teams = set(games_df["home_team"].tolist() + games_df["away_team"].tolist())
    today_teams_norm = {_norm(t) for t in today_teams}

    # Build per-team lineup sets so we only filter teams that have posted lineups
    teams_with_lineups: dict[str, set[str]] = {}
    if not lineups_df.empty:
        for _, lr in lineups_df.iterrows():
            t_norm = _norm(lr["team"])
            teams_with_lineups.setdefault(t_norm, set()).add(_norm(lr["player_name"]))

    # Build opposing pitcher lookup: {batter_team: (pitcher_name, pitcher_row)}
    opp_pitcher: dict[str, tuple[str | None, pd.Series | None]] = {}
    for _, g in games_df.iterrows():
        home_t = g.get("home_team", "")
        away_t = g.get("away_team", "")
        home_sp = g.get("home_starter")
        away_sp = g.get("away_starter")
        # Away batters face home pitcher and vice-versa
        if home_sp and not pitcher_df.empty:
            p = pitcher_df[pitcher_df["name"] == home_sp]
            opp_pitcher[away_t] = (home_sp, p.iloc[0] if not p.empty else None)
        if away_sp and not pitcher_df.empty:
            p = pitcher_df[pitcher_df["name"] == away_sp]
            opp_pitcher[home_t] = (away_sp, p.iloc[0] if not p.empty else None)

    # Build prop line lookup by normalized name
    prop_lookup: dict[str, dict] = {}
    if not props_df.empty:
        props_df = props_df.drop_duplicates(subset="player_name")
        for _, pr in props_df.iterrows():
            prop_lookup[_norm(pr["player_name"])] = pr.to_dict()

    rows = []
    for _, b in batter_df.iterrows():
        if pd.isna(b.get("babip")) or pd.isna(b.get("k_pct")):
            continue
        fg_abbr = b.get("team", "")
        full_team = _FG_TEAM_MAP.get(fg_abbr, fg_abbr)
        if _norm(full_team) not in today_teams_norm:
            continue
        name = b.get("name", "")
        team_norm = _norm(full_team)
        # If this team has posted a lineup, only show confirmed players; otherwise show all
        if team_norm in teams_with_lineups and _norm(name) not in teams_with_lineups[team_norm]:
            continue

        pa = b.get("pa", 0) or 0
        ab = b.get("ab", 0) or 0
        est_games = max(pa / 4.5, 1)
        ab_per_game = ab / est_games if pa > 0 else 3.5

        proj_hits = project_batter_hits(b["babip"], ab_per_game, b["k_pct"])

        # Opposing pitcher
        opp_name, opp_row = opp_pitcher.get(full_team, (None, None))
        opp_fip  = float(opp_row["fip"])  if opp_row is not None and pd.notna(opp_row.get("fip"))  else None
        opp_era  = float(opp_row["era"])  if opp_row is not None and pd.notna(opp_row.get("era"))  else None
        opp_babip = float(opp_row["babip"]) if opp_row is not None and pd.notna(opp_row.get("babip")) else None

        # Find opponent team for matchup label
        opp_team = None
        for _, g in games_df.iterrows():
            if g["home_team"] == full_team:
                opp_team = g["away_team"]
            elif g["away_team"] == full_team:
                opp_team = g["home_team"]

        # Park factor for the home team
        home_team_for_park = None
        for _, g in games_df.iterrows():
            if g["home_team"] == full_team or g["away_team"] == full_team:
                home_team_for_park = g["home_team"]
                break
        park_factor = get_park_factor(home_team_for_park) if home_team_for_park else 1.0

        # Prop line
        prop = prop_lookup.get(_norm(name), {})
        prop_line = prop.get("prop_line")
        over_odds  = prop.get("over_odds")
        under_odds = prop.get("under_odds")

        if prop_line and not pd.isna(prop_line):
            direction = "OVER" if proj_hits >= float(prop_line) else "UNDER"
            edge_data = compute_prop_edge(proj_hits, float(prop_line), direction)
            edge_pct  = edge_data["edge_pct"]
            rec       = "BET" if edge_pct >= MIN_PROP_EDGE_PCT else "PASS"
        else:
            direction = "OVER" if proj_hits >= 0.5 else "UNDER"
            edge_pct  = 0.0
            rec       = "No line"

        rows.append({
            "name":        name,
            "team":        full_team,
            "opp_team":    opp_team or "?",
            "proj_hits":   proj_hits,
            "babip":       float(b["babip"]),
            "k_pct":       float(b["k_pct"]),
            "wrc_plus":    b.get("wrc_plus"),
            "ab_per_game": round(ab_per_game, 1),
            "prop_line":   prop_line,
            "over_odds":   over_odds,
            "under_odds":  under_odds,
            "direction":   direction,
            "edge_pct":    edge_pct,
            "rec":         rec,
            "opp_pitcher": opp_name,
            "opp_fip":     opp_fip,
            "opp_era":     opp_era,
            "opp_babip_allowed": opp_babip,
            "park_factor": park_factor,
            "home_team_for_park": home_team_for_park,
        })

    return sorted(rows, key=lambda r: r["edge_pct"], reverse=True)


def _render_batter_card(row: dict) -> None:
    name       = row["name"]
    team       = row["team"]
    opp        = row["opp_team"]
    proj       = row["proj_hits"]
    babip      = row["babip"]
    k_pct      = row["k_pct"]
    wrc        = row["wrc_plus"]
    ab_pg      = row["ab_per_game"]
    prop_line  = row["prop_line"]
    direction  = row["direction"]
    edge_pct   = row["edge_pct"]
    rec        = row["rec"]
    opp_pitcher = row["opp_pitcher"]
    opp_fip    = row["opp_fip"]
    opp_era    = row["opp_era"]
    opp_babip  = row["opp_babip_allowed"]
    park_factor = row["park_factor"]
    over_odds  = row.get("over_odds")
    under_odds = row.get("under_odds")
    home_park  = row.get("home_team_for_park", "")

    rec_color = "#2ecc71" if rec == "BET" else "#aaa" if rec == "PASS" else "#888"

    with st.expander(
        f"{'🟢' if rec == 'BET' else '⚪'} {name} ({team} vs {opp})  "
        f"| Proj: {proj:.2f} hits  "
        + (f"| Line: {direction} {prop_line}  | Edge: {edge_pct:+.1f}%" if prop_line else "| No line yet"),
        expanded=(rec == "BET"),
    ):
        # ── Row 1: key metrics ────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Proj Hits/G", f"{proj:.2f}")
        c2.metric("BABIP",       f"{babip:.3f}",
                  delta=f"{babip - 0.300:+.3f} vs .300",
                  delta_color="inverse")
        c3.metric("K%",          f"{k_pct:.1%}")
        c4.metric("wRC+",        str(int(wrc)) if wrc and not pd.isna(wrc) else "—",
                  help="100 = league average. 130 = 30% above avg.")
        c5.metric("AB/Game",     f"{ab_pg}")

        # ── Row 2: BABIP explanation ──────────────────────────────────────────
        sig_label, sig_color = _babip_signal_color(babip)
        st.markdown(
            f'<span style="background:{sig_color};color:#000;padding:2px 8px;'
            f'border-radius:6px;font-size:12px;font-weight:bold">'
            f'BABIP Signal: {sig_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption("")

        if babip > 0.320:
            st.caption(
                f"BABIP of {babip:.3f} is **{babip - 0.300:.0%}** above league avg (.300). "
                f"Hits on balls in play are unsustainably high — expect regression. "
                f"Model projects decline toward {(babip * 0.6 + 0.300 * 0.4):.3f} regressed BABIP."
            )
        elif babip < 0.275:
            st.caption(
                f"BABIP of {babip:.3f} is **{0.300 - babip:.0%}** below league avg (.300). "
                f"Player is making good contact but not getting results — expect improvement. "
                f"Model projects regressed BABIP of {(babip * 0.6 + 0.300 * 0.4):.3f}."
            )
        else:
            st.caption(f"BABIP of {babip:.3f} is near league average — neutral signal, projection uses raw BABIP.")

        col_gauge, col_right = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(_babip_gauge(babip), width="stretch")

        with col_right:
            # ── Opposing pitcher breakdown ────────────────────────────────────
            if opp_pitcher:
                st.markdown(f"**Opposing Pitcher: {opp_pitcher}**")
                p1, p2, p3 = st.columns(3)
                p1.metric("ERA",   f"{opp_era:.2f}"  if opp_era  is not None else "—")
                p2.metric("FIP",   f"{opp_fip:.2f}"  if opp_fip  is not None else "—")
                p3.metric("BABIP Allowed", f"{opp_babip:.3f}" if opp_babip is not None else "—")

                if opp_fip is not None and opp_era is not None:
                    gap = opp_fip - opp_era
                    if gap > 0.75:
                        st.caption(
                            f"🔴 Pitcher FIP-ERA gap: +{gap:.2f} — ERA is artificially low, "
                            f"this pitcher is easier to hit than ERA suggests. **Favors Over.**"
                        )
                    elif gap < -0.75:
                        st.caption(
                            f"🟢 Pitcher FIP-ERA gap: {gap:.2f} — ERA overstates difficulty, "
                            f"pitcher is better than numbers look. **Favors Under.**"
                        )

                if opp_babip is not None:
                    if opp_babip > 0.320:
                        st.caption(f"Pitcher BABIP allowed: {opp_babip:.3f} → high, has been allowing lots of hits — favors Over.")
                    elif opp_babip < 0.275:
                        st.caption(f"Pitcher BABIP allowed: {opp_babip:.3f} → low, suppressing hits — favors Under.")
            else:
                st.caption("Opposing pitcher not confirmed yet.")

            # ── Park factor ───────────────────────────────────────────────────
            pf_pct = (park_factor - 1.0) * 100
            if park_factor >= 1.05:
                st.caption(f"🏟️ Hitter-friendly park (+{pf_pct:.0f}%) — favors Over on hits props.")
            elif park_factor <= 0.96:
                st.caption(f"🏟️ Pitcher-friendly park ({pf_pct:.0f}%) — suppresses hits, favors Under.")

        # ── Prop line & recommendation ────────────────────────────────────────
        if prop_line and not pd.isna(float(prop_line)):
            st.divider()
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Sportsbook Line", f"{direction} {prop_line}")
            rc2.metric("Model Projection", f"{proj:.2f}")
            rc3.metric("Edge", f"{edge_pct:+.1f}%")
            bet_odds = over_odds if direction == "OVER" else under_odds
            rc4.metric("Odds", f"+{int(bet_odds)}" if bet_odds and bet_odds > 0 else str(int(bet_odds)) if bet_odds else "—")

            if rec == "BET":
                st.success(f"✅ **BET: {direction} {prop_line}** — Model projects {proj:.2f}, edge of {edge_pct:+.1f}%")
            else:
                st.info(f"Edge {edge_pct:+.1f}% is below the {MIN_PROP_EDGE_PCT}% threshold — PASS")
        else:
            st.caption("No sportsbook line available yet for this player.")

        # ── AI Reasoning ─────────────────────────────────────────────────────
        st.divider()
        st.caption("**AI Reasoning** (Llama 3.3 70B):")

        import json as _json
        signals = {
            "babip": babip,
            "babip_deviation": round(babip - 0.300, 3),
            "k_pct": k_pct,
            "wrc_plus": wrc,
        }
        if opp_pitcher:
            signals["opponent_pitcher"] = opp_pitcher
            if opp_fip:
                signals["opponent_fip"] = opp_fip

        with st.spinner(""):
            ai = _cached_prop_ai(
                player=name, team=team,
                prop_type="Hits",
                line=float(prop_line) if prop_line and not pd.isna(float(prop_line)) else 0.5,
                proj=proj,
                edge=edge_pct,
                direction=direction,
                signals_json=_json.dumps({k: v for k, v in signals.items() if v is not None}),
            )

        conf = ai.get("confidence", "Low")
        conf_color = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}.get(conf, "#888")
        st.markdown(
            f'<span style="background:{conf_color};color:#000;padding:2px 8px;'
            f'border-radius:6px;font-size:12px;font-weight:bold">'
            f'{conf} Confidence</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Reasoning:** {ai.get('reasoning', 'N/A')}")
        st.markdown(f"**Key Risk:** {ai.get('key_risk', 'N/A')}")


def render() -> None:
    today = datetime.date.today()
    st.header(f"Player Analysis — {today.strftime('%A, %B %-d, %Y')}")
    st.caption(
        "Full model breakdown for every batter in today's confirmed lineups: "
        "BABIP regression, opposing pitcher FIP, park factor, and AI reasoning."
    )

    with st.spinner("Loading player and game data..."):
        batter_df  = _load_batters(CURRENT_SEASON)
        pitcher_df = _load_pitchers(CURRENT_SEASON)
        try:
            games_df = get_mlb_odds()
        except Exception:
            games_df = pd.DataFrame()
        lineups_df = get_today_lineups(today)
        props_df   = pd.DataFrame()
        if not games_df.empty:
            try:
                props_df = get_all_batter_hits_props(games_df)
            except Exception:
                pass

    if batter_df.empty:
        st.warning("No batter stats available.")
        return

    if games_df.empty:
        st.warning("No games found today — check ODDS_API_KEY.")
        return

    # Merge probable starters into games_df so we know who's pitching against each batter
    from src.data.game_results import get_probable_starters
    starters = get_probable_starters(today)
    if not starters.empty and not games_df.empty:
        games_df = games_df.merge(
            starters[["home_team", "away_team", "home_starter", "away_starter"]],
            on=["home_team", "away_team"], how="left",
        )

    if not lineups_df.empty:
        confirmed_teams = lineups_df["team"].nunique()
        lineup_status = f"✅ Confirmed lineups for {confirmed_teams} team(s) — other teams show all rostered batters"
    else:
        lineup_status = "⏳ Lineups not yet posted — showing all rostered batters for today's games"
    st.info(lineup_status)

    rows = _build_batter_rows(batter_df, games_df, props_df, lineups_df, pitcher_df)

    if not rows:
        st.info("No qualifying batters found for today's games.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        all_teams = sorted({r["team"] for r in rows})
        sel_team  = st.selectbox("Filter by team", ["All"] + all_teams)
    with f2:
        show_bets_only = st.checkbox("BET recommendations only", value=False)
    with f3:
        sort_by = st.selectbox("Sort by", ["Edge %", "Projected Hits", "BABIP"])

    filtered = rows
    if sel_team != "All":
        filtered = [r for r in filtered if r["team"] == sel_team]
    if show_bets_only:
        filtered = [r for r in filtered if r["rec"] == "BET"]

    sort_key = {"Edge %": "edge_pct", "Projected Hits": "proj_hits", "BABIP": "babip"}[sort_by]
    filtered = sorted(filtered, key=lambda r: r[sort_key], reverse=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader(f"{len(filtered)} player(s)")
    summary_rows = []
    for r in filtered:
        bet_odds = r.get("over_odds") if r["direction"] == "OVER" else r.get("under_odds")
        odds_str = (f"+{int(bet_odds)}" if bet_odds and bet_odds > 0 else str(int(bet_odds))) if bet_odds else "—"
        summary_rows.append({
            "Player":     r["name"],
            "Team":       r["team"],
            "vs":         r["opp_team"],
            "Opp SP":     r["opp_pitcher"] or "TBD",
            "Proj Hits":  f"{r['proj_hits']:.2f}",
            "BABIP":      f"{r['babip']:.3f}",
            "Line":       f"{r['direction']} {r['prop_line']}" if r["prop_line"] else "No line",
            "Edge %":     f"{r['edge_pct']:+.1f}%",
            "Odds":       odds_str,
            "Rec":        r["rec"],
        })
    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)

        def _color_rec(val):
            if val == "BET":
                return "background-color:#1a3a1a;color:#2ecc71;font-weight:bold"
            if val == "PASS":
                return "background-color:#2a2a2a;color:#aaa"
            return ""

        st.dataframe(
            sum_df.style.applymap(_color_rec, subset=["Rec"]),
            width="stretch", hide_index=True,
        )

    st.divider()
    st.subheader("Detailed Breakdown")

    for r in filtered:
        _render_batter_card(r)
