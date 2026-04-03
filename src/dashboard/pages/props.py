"""Player props page — pitcher strikeouts and batter hits with edge analysis."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import datetime
import unicodedata

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.baseball_stats import get_batter_stats, get_pitcher_stats
from src.data.game_results import get_today_lineups
from src.data.odds import get_mlb_odds, get_mlb_player_props, get_best_prop_lines, get_all_batter_hits_props
from src.models.player_props import (
    project_pitcher_strikeouts,
    project_batter_hits,
    compute_prop_edge,
    MIN_PROP_EDGE_PCT,
)
from src.shared.groq_agent import analyze_mlb_prop
from src.dashboard.pages.games import _add_to_slip
from src.data.game_results import get_live_games

CURRENT_SEASON = datetime.datetime.now().year

# FanGraphs abbreviation → Odds API full team name
_FG_TEAM_MAP: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATH": "Athletics",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}


def _normalize(name: str) -> str:
    """Strip accents and lowercase for fuzzy name matching."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii").lower().strip()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading pitcher stats...")
def _load_pitchers(season: int) -> pd.DataFrame:
    try:
        return get_pitcher_stats(season)
    except Exception as e:
        st.error(f"Could not load pitcher stats: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 6, show_spinner="Loading batter stats...")
def _load_batters(season: int) -> pd.DataFrame:
    try:
        return get_batter_stats(season, min_pa=10)
    except Exception as e:
        st.error(f"Could not load batter stats: {e}")
        return pd.DataFrame()


def _render_pitcher_props(pitcher_df: pd.DataFrame) -> None:
    st.subheader("Pitcher Strikeout Props")
    st.caption(
        "Model projects K total using pitcher K%, IP/start, and opponent K%. "
        "Enter sportsbook lines to find over/under value."
    )

    if pitcher_df.empty:
        st.warning("No pitcher data available.")
        return

    teams = ["All"] + sorted(pitcher_df["team"].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.selectbox("Filter by team", teams, key="props_pitcher_team")
    with col2:
        min_edge = st.slider("Min edge %", 0.0, 30.0, MIN_PROP_EDGE_PCT, 1.0, key="props_pitcher_edge")

    filtered = pitcher_df.copy()
    if team_filter != "All":
        filtered = filtered[filtered["team"] == team_filter]

    results = []
    for _, row in filtered.iterrows():
        if pd.isna(row.get("k_pct")) or pd.isna(row.get("ip")):
            continue
        ip_per_start = max(row["ip"] / 30, 3.0)
        proj_k = project_pitcher_strikeouts(
            pitcher_k_pct=row["k_pct"],
            pitcher_ip_per_start=ip_per_start,
        )
        results.append({
            "name": row["name"],
            "team": row["team"],
            "k_pct": row.get("k_pct"),
            "xfip": row.get("xfip"),
            "babip": row.get("babip"),
            "ip_per_start": round(ip_per_start, 1),
            "projected_k": proj_k,
        })

    if not results:
        st.info("No pitcher data to display.")
        return

    proj_df = pd.DataFrame(results).sort_values("projected_k", ascending=False)

    # Allow manual prop line input
    st.caption("Enter sportsbook K line for each pitcher to compute edge:")
    line_inputs = {}
    cols = st.columns(4)
    for i, (_, row) in enumerate(proj_df.head(20).iterrows()):
        c = cols[i % 4]
        with c:
            line_inputs[row["name"]] = st.number_input(
                f"{row['name']} ({row['team']})",
                min_value=0.5,
                max_value=15.0,
                value=float(row["projected_k"]),
                step=0.5,
                key=f"kline_{row['name']}",
            )

    # Compute edges
    edge_rows = []
    for _, row in proj_df.iterrows():
        if row["name"] not in line_inputs:
            continue
        line = line_inputs[row["name"]]
        proj = row["projected_k"]
        direction = "OVER" if proj > line else "UNDER"
        edge = compute_prop_edge(proj, line, direction)
        edge_rows.append({
            "Pitcher": row["name"],
            "Team": row["team"],
            "K%": f"{row['k_pct']:.1%}" if row.get("k_pct") else "—",
            "xFIP": round(row["xfip"], 2) if row.get("xfip") else "—",
            "Proj K": proj,
            "Line": line,
            "Bet": direction,
            "Edge %": edge["edge_pct"],
            "Signal": "BET" if abs(edge["edge_pct"]) >= min_edge else "PASS",
        })

    if edge_rows:
        edge_df = pd.DataFrame(edge_rows)
        edge_df = edge_df[edge_df["Signal"] == "BET"].sort_values("Edge %", ascending=False) \
            if min_edge > 0 else pd.DataFrame(edge_rows)

        def _color(val):
            if val == "BET":
                return "background-color: #2ecc71; color: black"
            if val == "PASS":
                return "background-color: #555; color: white"
            return ""

        st.dataframe(
            edge_df.style.applymap(_color, subset=["Signal"]),
            use_container_width=True,
            hide_index=True,
        )

        # AI analysis for top props
        top = edge_df[edge_df["Signal"] == "BET"].head(3)
        if not top.empty:
            st.subheader("AI Analysis — Top Strikeout Props")
            for _, r in top.iterrows():
                with st.expander(f"{r['Pitcher']} — {r['Bet']} {r['Line']} K (Edge: {r['Edge %']:+.1f}%)"):
                    if st.button(f"Analyze {r['Pitcher']}", key=f"ai_k_{r['Pitcher']}"):
                        with st.spinner("Analyzing..."):
                            pitcher_row = pitcher_df[pitcher_df["name"] == r["Pitcher"]].iloc[0]
                            result = analyze_mlb_prop(
                                player_name=r["Pitcher"],
                                team=r["Team"],
                                prop_type="Strikeouts",
                                line=r["Line"],
                                model_projection=r["Proj K"],
                                edge_pct=r["Edge %"],
                                bet_direction=r["Bet"],
                                signals={
                                    "k_pct": pitcher_row.get("k_pct"),
                                    "whiff_rate": pitcher_row.get("whiff_pct"),
                                    "babip": pitcher_row.get("babip"),
                                },
                            )
                        st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                        st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                        st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")


def _format_odds(odds: int | float | None) -> str:
    if odds is None or pd.isna(odds):
        return "N/A"
    odds = int(odds)
    return f"+{odds}" if odds > 0 else str(odds)


def _odds_eligible(odds: int | float | None) -> bool:
    """Return True if odds meet the minimum threshold to consider betting (-200 or better)."""
    if odds is None or pd.isna(odds):
        return False
    return int(odds) >= -200


def _build_batter_projections(
    batter_df: pd.DataFrame,
    today_teams: set[str],
    lineup_names: set[str],
    lineup_available: bool,
) -> pd.DataFrame:
    """Project hits/game for all eligible batters and return sorted projection DataFrame."""
    # Build normalized lookup sets for matching
    today_teams_norm = {_normalize(t) for t in today_teams}
    lineup_norm = {_normalize(n) for n in lineup_names}

    results = []
    for _, row in batter_df.iterrows():
        if pd.isna(row.get("babip")) or pd.isna(row.get("k_pct")):
            continue

        fg_abbr = row.get("team", "")
        full_team = _FG_TEAM_MAP.get(fg_abbr, fg_abbr)
        if _normalize(full_team) not in today_teams_norm:
            continue

        name = row.get("name", "")

        # Lineup filter: if lineups are posted, only include confirmed starters
        if lineup_available and _normalize(name) not in lineup_norm:
            continue

        pa = row.get("pa", 0) or 0
        ab = row.get("ab", 0) or 0
        estimated_games = max(pa / 4.5, 1)
        ab_per_game = ab / estimated_games if pa > 0 else 3.5

        proj_hits = project_batter_hits(
            batter_babip=row["babip"],
            at_bats_projected=ab_per_game,
            batter_k_pct=row["k_pct"],
        )
        babip_dev = row["babip"] - 0.300
        signal = "🔴 High BABIP" if row["babip"] > 0.320 else ("🟢 Low BABIP" if row["babip"] < 0.275 else "⚪ Normal")

        results.append({
            "name": name,
            "Team": full_team,
            "babip": round(row["babip"], 3),
            "babip_dev": babip_dev,
            "wrc_plus": row.get("wrc_plus"),
            "k_pct": row.get("k_pct"),
            "proj_hits": proj_hits,
            "ab_per_game": round(ab_per_game, 1),
            "signal": signal,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("proj_hits", ascending=False)

    # If no lineup posted, fall back to top 2 per team by wRC+ (likely starters)
    if not lineup_available:
        df = df.sort_values(["Team", "wrc_plus"], ascending=[True, False])
        df = df.groupby("Team").head(2).sort_values("proj_hits", ascending=False)

    return df.reset_index(drop=True)


def _render_batter_props(batter_df: pd.DataFrame) -> None:
    today = datetime.date.today()

    if batter_df.empty:
        st.warning("No batter data available.")
        return

    # Load today's games (for team filtering and props fetching)
    with st.spinner("Loading today's lineups and prop lines..."):
        try:
            games_df = get_mlb_odds()
        except Exception:
            games_df = pd.DataFrame()

        lineups_df = get_today_lineups(today)
        lineup_available = not lineups_df.empty

        props_df = pd.DataFrame()
        if not games_df.empty:
            try:
                props_df = get_all_batter_hits_props(games_df)
            except Exception:
                pass

    # Build set of today's teams and confirmed starters
    today_teams: set[str] = set()
    if not games_df.empty:
        today_teams = set(games_df["home_team"].tolist() + games_df["away_team"].tolist())

    lineup_names: set[str] = set()
    if lineup_available:
        lineup_names = set(lineups_df["player_name"].tolist())

    # Lineup status notice
    if lineup_available:
        n = len(lineup_names)
        st.success(f"✅ Confirmed lineups loaded — {n} starters across {len(lineups_df['team'].unique())} teams")
    else:
        st.info(
            "⏳ Lineups not yet posted (typically 1–2 hrs before first pitch). "
            "Showing estimated regulars based on season stats."
        )

    if not today_teams:
        st.warning("No games found for today. Check your ODDS_API_KEY.")
        return

    # Build projections
    proj_df = _build_batter_projections(batter_df, today_teams, lineup_names, lineup_available)

    if proj_df.empty:
        st.info("No qualifying batters for today's games.")
        return

    # ── Section 1: Top 10 Hitter Projections ──────────────────────────────────
    st.subheader("Top 10 Hitter Projections Today")
    st.caption(
        "Ranked by projected hits/game using BABIP regression (regressed 60% batter / 40% league avg). "
        "🔴 High BABIP = hitting lucky, expect regression. 🟢 Low BABIP = underperforming, expect improvement."
    )

    top10 = proj_df.head(10).copy()
    top10_display = top10.rename(columns={
        "name": "Batter",
        "babip": "BABIP",
        "wrc_plus": "wRC+",
        "proj_hits": "Proj Hits/G",
        "signal": "BABIP Signal",
        "ab_per_game": "AB/G",
    })[["Batter", "Team", "BABIP", "wRC+", "AB/G", "Proj Hits/G", "BABIP Signal"]]

    def _style_signal(val: str) -> str:
        if "High" in val:
            return "color: #e74c3c; font-weight: bold"
        if "Low" in val:
            return "color: #2ecc71; font-weight: bold"
        return ""

    st.dataframe(
        top10_display.style.applymap(_style_signal, subset=["BABIP Signal"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Section 2: Bet Recommendations ────────────────────────────────────────
    st.subheader("Bet Recommendations")
    st.caption(
        "Only shown where sportsbook lines are available. "
        "**Minimum odds: -200** (anything more negative is skipped). "
        "**Positive odds are prioritized** — better value per dollar risked."
    )

    if props_df.empty:
        st.info(
            "No sportsbook prop lines available yet. Lines typically appear 2–3 hours before first pitch. "
            "Check back closer to game time."
        )
        return

    # Merge projections with prop lines using normalized names (handles accents)
    proj_df["_norm"] = proj_df["name"].apply(_normalize)
    props_df["_norm"] = props_df["player_name"].apply(_normalize)

    merged = proj_df.merge(props_df, on="_norm", how="inner")
    merged = merged.drop(columns=["_norm"], errors="ignore")

    if merged.empty:
        st.info("Could not match player names between model and prop lines. Name format may differ.")
        return

    # Determine best side (over vs under) based on model projection
    bet_rows = []
    for _, row in merged.iterrows():
        line = row.get("prop_line")
        if pd.isna(line):
            continue

        proj = row["proj_hits"]
        if proj > line:
            direction = "OVER"
            bet_odds = row.get("over_odds")
        else:
            direction = "UNDER"
            bet_odds = row.get("under_odds")

        if not _odds_eligible(bet_odds):
            continue

        edge = compute_prop_edge(proj, line, direction)
        is_positive = isinstance(bet_odds, (int, float)) and bet_odds > 0

        bet_rows.append({
            "name": row["name"],
            "Team": row.get("Team", ""),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "proj_hits": proj,
            "prop_line": line,
            "direction": direction,
            "odds": int(bet_odds),
            "odds_str": _format_odds(bet_odds),
            "edge_pct": edge["edge_pct"],
            "babip": row.get("babip"),
            "babip_dev": row.get("babip_dev", 0),
            "wrc_plus": row.get("wrc_plus"),
            "signal": row.get("signal", ""),
            "is_positive_odds": is_positive,
            "rec": "BET" if edge["edge_pct"] >= MIN_PROP_EDGE_PCT else "PASS",
        })

    if not bet_rows:
        st.info("No prop lines with odds ≥ -200 found for today's projected starters.")
        return

    bets_df = pd.DataFrame(bet_rows).sort_values(
        ["is_positive_odds", "edge_pct"], ascending=[False, False]
    )

    bet_count = (bets_df["rec"] == "BET").sum()
    pass_count = (bets_df["rec"] == "PASS").sum()
    st.caption(f"**{bet_count} BET** recommendations | **{pass_count} PASS** | Sorted: positive odds first, then edge")

    # Build live pitcher lookup {team: current_pitcher} for mid-game context
    live_pitcher: dict[str, str] = {}
    try:
        live_df = get_live_games()
        if not live_df.empty:
            for _, lg in live_df.iterrows():
                if lg.get("home_current_pitcher"):
                    live_pitcher[lg["home_team"]] = lg["home_current_pitcher"]
                if lg.get("away_current_pitcher"):
                    live_pitcher[lg["away_team"]] = lg["away_current_pitcher"]
    except Exception:
        pass

    for _, row in bets_df.iterrows():
        is_bet = row["rec"] == "BET"
        plus_tag = " ⭐ Positive Odds" if row["is_positive_odds"] else ""

        # Determine who's currently pitching against this batter
        batter_team = row.get("Team", "")
        opp_team    = row.get("home_team") if batter_team == row.get("away_team") else row.get("away_team")
        current_pitcher = live_pitcher.get(opp_team or "")

        label = (
            f"{'🟢 BET' if is_bet else '⚪ PASS'} — {row['name']}  |  "
            f"{row['direction']} {row['prop_line']}  |  "
            f"Odds: {row['odds_str']}{plus_tag}  |  "
            f"Edge: {row['edge_pct']:+.1f}%"
        )

        with st.expander(label, expanded=is_bet and row["is_positive_odds"]):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Proj", f"{row['proj_hits']:.2f} hits")
            c2.metric("Sportsbook Line", f"{row['direction']} {row['prop_line']}")
            c3.metric("Odds", row["odds_str"])
            c4.metric("Edge", f"{row['edge_pct']:+.1f}%")

            matchup = f"{row['away_team']} @ {row['home_team']}"
            st.caption(f"**{row['name']}** ({row['Team']}) — {matchup}")
            st.caption(f"BABIP: {row['babip']:.3f}  |  wRC+: {row.get('wrc_plus', 'N/A')}  |  Signal: {row['signal']}")

            if current_pitcher:
                st.info(
                    f"🔴 **Live:** {opp_team} currently pitching **{current_pitcher}**.  "
                    f"The pre-game prop line was set against the starter — "
                    f"hit Refresh to recalculate if you want updated projection against {current_pitcher}.",
                    icon=None,
                )

            if row["is_positive_odds"]:
                st.markdown(
                    f'<span style="background:#f39c12;color:#000;padding:3px 10px;border-radius:8px;font-weight:bold;font-size:13px">⭐ +ODDS VALUE — Win more than you risk</span>',
                    unsafe_allow_html=True,
                )

            btn_col1, btn_col2 = st.columns([1, 3])
            with btn_col1:
                slip_label = f"+ {row['direction']} {row['prop_line']} Hits — {row['name']}"
                if st.button("📌 Add to Slip", key=f"slip_prop_{row['name']}"):
                    _add_to_slip({
                        "key": f"prop_hit_{row['name']}",
                        "matchup": f"{row['away_team']} @ {row['home_team']}",
                        "description": slip_label,
                        "bet_type": "Prop",
                        "line": float(row["odds"]),
                        "edge_pct": row["edge_pct"],
                        "model_prob": None,
                        "stake": 50.0,
                    })
                    st.toast(f"Added: {slip_label}", icon="📌")
            with btn_col2:
                if is_bet and st.button("AI Analysis", key=f"ai_hit_{row['name']}"):
                    with st.spinner("Analyzing..."):
                        result = analyze_mlb_prop(
                            player_name=row["name"],
                            team=row["Team"],
                            prop_type="Hits",
                            line=row["prop_line"],
                            model_projection=row["proj_hits"],
                            edge_pct=row["edge_pct"],
                            bet_direction=row["direction"],
                            signals={
                                "babip": row["babip"],
                                "babip_deviation": round(row.get("babip_dev", 0), 3),
                                "wrc_plus": row.get("wrc_plus"),
                            },
                        )
                    st.markdown(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                    st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                    st.markdown(f"**Key Risk:** {result.get('key_risk', 'N/A')}")


def render() -> None:
    st.header("Player Props — MLB")
    st.caption(
        "Stat-model projections for pitcher and batter props. "
        "Find value where sportsbook lines diverge from underlying performance metrics."
    )

    tab1, tab2 = st.tabs(["Pitcher Props (K)", "Batter Props (Hits)"])

    pitcher_df = _load_pitchers(CURRENT_SEASON)
    batter_df = _load_batters(CURRENT_SEASON)

    with tab1:
        _render_pitcher_props(pitcher_df)

    with tab2:
        _render_batter_props(batter_df)
