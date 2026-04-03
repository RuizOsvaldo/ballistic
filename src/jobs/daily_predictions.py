"""
Daily predictions job — runs at 8am, emails top game picks and player props.

Usage:
    venv/bin/python3 -m src.jobs.daily_predictions

Requires in .env:
    EMAIL_SENDER       Gmail address to send from
    EMAIL_APP_PASSWORD Gmail App Password (not your regular password)
    EMAIL_RECIPIENT    Address to deliver the report to
"""

from __future__ import annotations

import datetime
import os
import smtplib
import sys
import traceback
import unicodedata
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

# Load .env relative to project root — works regardless of CWD
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Email config
# ---------------------------------------------------------------------------

EMAIL_SENDER       = os.getenv("EMAIL_SENDER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT    = os.getenv("EMAIL_RECIPIENT", "")


# ---------------------------------------------------------------------------
# Data helpers (same pipeline as the dashboard)
# ---------------------------------------------------------------------------

def _load_games() -> pd.DataFrame:
    from src.data.odds import get_mlb_odds
    from src.data.game_results import get_probable_starters
    from src.data.baseball_stats import get_team_stats, get_pitcher_stats
    from src.models.pythagorean import compute_pythagorean
    from src.models.regression_signals import compute_pitcher_signals, compute_team_signals
    from src.models.win_probability import compute_win_probabilities
    from src.models.kelly import compute_kelly_for_games

    season = datetime.date.today().year
    today = datetime.date.today()

    raw_odds = get_mlb_odds()
    if raw_odds.empty:
        return pd.DataFrame()

    raw_team = get_team_stats(season)
    team_stats = compute_pythagorean(raw_team)
    team_stats = compute_team_signals(team_stats)

    raw_pitchers = get_pitcher_stats(season)
    pitcher_stats = compute_pitcher_signals(raw_pitchers)

    starters = get_probable_starters(today)
    odds = raw_odds.copy()
    if not starters.empty:
        odds = odds.merge(
            starters[["home_team", "away_team", "home_starter", "away_starter",
                       "home_starter_announced", "away_starter_announced"]],
            on=["home_team", "away_team"], how="left",
        )
    else:
        for col in ["home_starter", "away_starter", "home_starter_announced", "away_starter_announced"]:
            odds[col] = None if "starter" in col else False

    def _ls(row) -> str:
        h = bool(row.get("home_starter_announced"))
        a = bool(row.get("away_starter_announced"))
        return "Full" if h and a else "Partial" if h or a else "None"

    odds["lineup_status"] = odds.apply(_ls, axis=1)

    from src.data.ballpark import get_bullpen_stats
    try:
        bullpen_stats = get_bullpen_stats(season)
    except Exception:
        bullpen_stats = pd.DataFrame()
    games_with_prob = compute_win_probabilities(odds, team_stats, pitcher_stats, bullpen_df=bullpen_stats)
    ready = games_with_prob.dropna(subset=["home_model_prob", "away_model_prob"])
    if ready.empty:
        return pd.DataFrame()

    return compute_kelly_for_games(ready, bankroll=1000)


def _load_batter_props() -> pd.DataFrame:
    from src.data.baseball_stats import get_batter_stats
    from src.data.game_results import get_today_lineups
    from src.data.odds import get_mlb_odds, get_all_batter_hits_props
    from src.models.player_props import project_batter_hits, compute_prop_edge, MIN_PROP_EDGE_PCT

    _FG_TEAM_MAP = {
        "ARI": "Arizona Diamondbacks", "ATH": "Athletics", "ATL": "Atlanta Braves",
        "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs",
        "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
        "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros",
        "KCR": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
        "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins",
        "NYM": "New York Mets", "NYY": "New York Yankees", "PHI": "Philadelphia Phillies",
        "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
        "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
        "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    }

    def norm(s: str) -> str:
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower().strip()

    today = datetime.date.today()
    season = today.year

    batters = get_batter_stats(season, min_pa=10)
    lineups = get_today_lineups(today)
    games = get_mlb_odds()

    today_teams = set(games["home_team"].tolist() + games["away_team"].tolist())
    today_teams_norm = {norm(t) for t in today_teams}
    lineup_norm = {norm(n) for n in lineups["player_name"].tolist()} if not lineups.empty else set()
    lineup_available = bool(lineup_norm)

    projections = []
    for _, row in batters.iterrows():
        if pd.isna(row.get("babip")) or pd.isna(row.get("k_pct")):
            continue
        full_team = _FG_TEAM_MAP.get(row.get("team", ""), row.get("team", ""))
        if norm(full_team) not in today_teams_norm:
            continue
        name = row.get("name", "")
        if lineup_available and norm(name) not in lineup_norm:
            continue
        pa = row.get("pa", 0) or 0
        ab = row.get("ab", 0) or 0
        est_games = max(pa / 4.5, 1)
        ab_per_game = ab / est_games if pa > 0 else 3.5
        proj = project_batter_hits(row["babip"], ab_per_game, row["k_pct"])
        projections.append({
            "name": name,
            "team": full_team,
            "proj_hits": proj,
            "babip": round(row["babip"], 3),
            "wrc_plus": row.get("wrc_plus"),
            "k_pct": row.get("k_pct"),
        })

    if not projections:
        return pd.DataFrame()

    proj_df = (
        pd.DataFrame(projections)
        .drop_duplicates(subset="name")
        .sort_values("proj_hits", ascending=False)
        .head(10)
    )

    # Merge with live prop lines
    props_df = pd.DataFrame()
    if not games.empty:
        try:
            props_df = get_all_batter_hits_props(games)
        except Exception:
            pass

    if props_df.empty:
        proj_df["prop_line"] = None
        proj_df["over_odds"] = None
        proj_df["under_odds"] = None
        proj_df["rec"] = "No line"
        return proj_df

    props_df["_norm"] = props_df["player_name"].apply(norm)
    props_df = props_df.drop_duplicates(subset="_norm")
    proj_df["_norm"] = proj_df["name"].apply(norm)
    merged = proj_df.merge(props_df, on="_norm", how="left").drop(columns=["_norm"], errors="ignore")
    merged = merged.drop_duplicates(subset="name")

    results = []
    for _, row in merged.iterrows():
        line = row.get("prop_line")
        proj = row["proj_hits"]
        if pd.isna(line) if isinstance(line, float) else line is None:
            row = row.copy()
            row["rec"] = "No line yet"
            row["direction"] = "—"
            results.append(row)
            continue
        direction = "OVER" if proj > line else "UNDER"
        edge = compute_prop_edge(proj, float(line), direction)
        row = row.copy()
        row["direction"] = direction
        row["rec"] = "BET ✅" if edge["edge_pct"] >= MIN_PROP_EDGE_PCT else "PASS"
        results.append(row)

    return pd.DataFrame(results) if results else proj_df


# ---------------------------------------------------------------------------
# HTML email builder
# ---------------------------------------------------------------------------

def _fmt_odds(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    v = int(val)
    return f"+{v}" if v > 0 else str(v)


def _fmt_rl(point, odds) -> str:
    if point is None or (isinstance(point, float) and pd.isna(point)):
        return "N/A"
    sign = "+" if float(point) > 0 else ""
    return f"{sign}{point} ({_fmt_odds(odds)})"


_CSS = """
body { font-family: Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 20px; }
h1 { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
h2 { color: #2ecc71; margin-top: 32px; }
h3 { color: #e67e22; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }
th { background: #1a1a2e; color: #3498db; padding: 8px 12px; text-align: left; border-bottom: 2px solid #3498db; }
td { padding: 7px 12px; border-bottom: 1px solid #2a2a2a; }
tr:nth-child(even) { background: #1a1a1a; }
.bet  { background: #1a3a1a; color: #2ecc71; font-weight: bold; padding: 3px 8px; border-radius: 6px; }
.pass { background: #2a2a2a; color: #aaa; padding: 3px 8px; border-radius: 6px; }
.high { color: #2ecc71; font-weight: bold; }
.med  { color: #f39c12; font-weight: bold; }
.low  { color: #e74c3c; }
.note { color: #aaa; font-size: 12px; margin-top: 4px; }
.footer { margin-top: 40px; color: #555; font-size: 11px; border-top: 1px solid #222; padding-top: 12px; }
"""


def _edge_class(edge: float) -> str:
    if edge >= 7:
        return "high"
    if edge >= 4:
        return "med"
    return "low"


def _build_html(games_df: pd.DataFrame, props_df: pd.DataFrame, today: datetime.date) -> str:
    date_str = today.strftime("%A, %B %-d, %Y")
    sections: list[str] = []

    # ── Game Predictions ──────────────────────────────────────────────────
    sections.append(f"<h2>⚾ Top Game Predictions</h2>")

    if games_df.empty:
        sections.append("<p>No games loaded today — check ODDS_API_KEY.</p>")
    else:
        bets = games_df[games_df["best_bet_side"] != "PASS"].sort_values("best_bet_edge", ascending=False).head(10)
        all_games = games_df.sort_values("best_bet_edge", ascending=False).head(10)
        display = bets if not bets.empty else all_games

        rows_html = []
        for rank, (_, row) in enumerate(display.iterrows(), 1):
            side = row.get("best_bet_side", "PASS")
            edge = row.get("best_bet_edge", 0)
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            matchup = f"{away} @ {home}"
            bet_team = home if side == "HOME" else away if side in ("HOME", "AWAY") else "—"
            home_prob = row.get("home_model_prob", 0)
            away_prob = row.get("away_model_prob", 0)
            home_ml = _fmt_odds(row.get("home_odds"))
            away_ml = _fmt_odds(row.get("away_odds"))
            total = row.get("total_line")
            over_o = _fmt_odds(row.get("over_odds"))
            under_o = _fmt_odds(row.get("under_odds"))
            home_rl = _fmt_rl(row.get("home_rl"), row.get("home_rl_odds"))
            away_rl = _fmt_rl(row.get("away_rl"), row.get("away_rl_odds"))
            lineup = row.get("lineup_status", "—")
            home_sp = row.get("home_starter") or "TBD"
            away_sp = row.get("away_starter") or "TBD"
            commence = row.get("commence_time", "")
            try:
                game_dt = datetime.datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                game_time = game_dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%-I:%M %p PT")
            except Exception:
                game_time = "—"

            ec = _edge_class(edge)

            # O/U recommendation from projected total
            proj_total = row.get("proj_total")
            if total is not None and proj_total is not None and not (isinstance(proj_total, float) and pd.isna(proj_total)):
                if float(proj_total) >= float(total):
                    ou_rec = f'<span class="bet">Over {total} ({over_o})</span>'
                else:
                    ou_rec = f'<span class="bet">Under {total} ({under_o})</span>'
            else:
                ou_rec = f'<span class="note">O/U {total if total else "N/A"} &nbsp; O {over_o} / U {under_o}</span>'

            if side != "PASS":
                bet_ml_odds = home_ml if side == "HOME" else away_ml
                bet_rl_point = row.get("home_rl") if side == "HOME" else row.get("away_rl")
                bet_rl_odds_val = row.get("home_rl_odds") if side == "HOME" else row.get("away_rl_odds")
                rl_str = _fmt_rl(bet_rl_point, bet_rl_odds_val)

                rec_html = (
                    f'<span class="bet">ML: {bet_team} @ {bet_ml_odds} ({edge:+.1f}%)</span><br>'
                    f'<span class="note">RL: {bet_team} {rl_str}</span><br>'
                    f'{ou_rec}'
                )
            else:
                rec_html = (
                    f'<span class="pass">PASS</span><br>'
                    f'{ou_rec}'
                )

            rows_html.append(f"""
            <tr>
              <td>{rank}</td>
              <td><strong>{matchup}</strong><br>
                  <span class="note">{away_sp} vs {home_sp} | Lineup: {lineup} | {game_time}</span></td>
              <td>{away_ml} / {home_ml}</td>
              <td class="{ec}">{home_prob:.1%} / {away_prob:.1%}</td>
              <td>{rec_html}</td>
            </tr>""")

        sections.append(f"""
        <table>
          <thead>
            <tr>
              <th>#</th><th>Matchup</th><th>Moneyline (Away/Home)</th>
              <th>Model % (H/A)</th><th>Recommendation (ML · RL · O/U)</th>
            </tr>
          </thead>
          <tbody>{"".join(rows_html)}</tbody>
        </table>""")

        if bets.empty:
            sections.append('<p class="note">No games cleared the 3% edge threshold today — all shown for reference.</p>')

    # ── Player Props ──────────────────────────────────────────────────────
    sections.append("<h2>🏏 Top 10 Batter Hit Projections</h2>")

    if props_df.empty:
        sections.append("<p>No batter data available.</p>")
    else:
        prop_rows = []
        for rank, (_, row) in enumerate(props_df.head(10).iterrows(), 1):
            name = row.get("name", "?")
            team = row.get("team", "?")
            proj = row.get("proj_hits", 0)
            babip = row.get("babip", 0)
            wrc = row.get("wrc_plus", "N/A")
            line = row.get("prop_line")
            direction = row.get("direction", "—")
            rec = row.get("rec", "No line yet")
            over_o = _fmt_odds(row.get("over_odds"))
            under_o = _fmt_odds(row.get("under_odds"))

            if rec == "BET ✅":
                rec_html = f'<span class="bet">{rec}</span>'
            elif rec == "PASS":
                rec_html = f'<span class="pass">PASS</span>'
            else:
                rec_html = f'<span class="note">{rec}</span>'

            line_str = f"{direction} {line}" if line and not (isinstance(line, float) and pd.isna(line)) else "No line"
            odds_str = f"O {over_o} / U {under_o}" if line else "—"

            prop_rows.append(f"""
            <tr>
              <td>{rank}</td>
              <td><strong>{name}</strong><br><span class="note">{team}</span></td>
              <td>{proj:.2f}</td>
              <td>{babip:.3f}</td>
              <td>{wrc}</td>
              <td>{line_str}<br><span class="note">{odds_str}</span></td>
              <td>{rec_html}</td>
            </tr>""")

        sections.append(f"""
        <table>
          <thead>
            <tr>
              <th>#</th><th>Batter</th><th>Proj Hits/G</th>
              <th>BABIP</th><th>wRC+</th><th>Sportsbook Line</th><th>Rec</th>
            </tr>
          </thead>
          <tbody>{"".join(prop_rows)}</tbody>
        </table>""")

    sections.append("""
    <div class="footer">
      Generated by Ballistic — quantitative baseball analytics.<br>
      Model: Pythagorean win expectation + FIP pitcher adjustment + BABIP regression.<br>
      Bet responsibly. This is for informational purposes only.
    </div>""")

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>
  <h1>🎯 Ballistic Daily Report — {date_str}</h1>
  {body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> None:
    if not EMAIL_SENDER:
        raise EnvironmentError("EMAIL_SENDER not set in .env")
    if not EMAIL_APP_PASSWORD:
        raise EnvironmentError("EMAIL_APP_PASSWORD not set in .env")
    if not EMAIL_RECIPIENT:
        raise EnvironmentError("EMAIL_RECIPIENT not set in .env")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECIPIENT
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    today = datetime.date.today()
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] Starting daily predictions job...")

    print("  Loading game predictions...")
    try:
        games_df = _load_games()
        print(f"  Games loaded: {len(games_df)}")
    except Exception:
        print(f"  ERROR loading games:\n{traceback.format_exc()}")
        games_df = pd.DataFrame()

    print("  Loading batter props...")
    try:
        props_df = _load_batter_props()
        print(f"  Props loaded: {len(props_df)}")
    except Exception:
        print(f"  ERROR loading props:\n{traceback.format_exc()}")
        props_df = pd.DataFrame()

    print("  Building email...")
    html = _build_html(games_df, props_df, today)
    subject = f"🎯 Ballistic — MLB Picks {today.strftime('%a %b %-d')}"

    print(f"  Sending to {EMAIL_RECIPIENT}...")
    try:
        send_email(subject, html)
        print("  Email sent successfully.")
    except Exception:
        print(f"  ERROR sending email:\n{traceback.format_exc()}")
        raise

    print("Done.")


if __name__ == "__main__":
    run()
