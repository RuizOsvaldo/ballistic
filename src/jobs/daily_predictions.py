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


def _build_html(games_df: pd.DataFrame, today: datetime.date) -> str:
    date_str = today.strftime("%A, %B %-d, %Y")
    sections: list[str] = []

    # ── Top 10 Bets by Edge ───────────────────────────────────────────────
    sections.append("<h2>🎯 Top 10 Bets to Place</h2>")

    all_bets = []

    # Collect ML bets
    ml_bets = games_df[games_df["best_bet_side"] != "PASS"].copy()
    for _, row in ml_bets.iterrows():
        team = row["home_team"] if row["best_bet_side"] == "HOME" else row["away_team"]
        all_bets.append({
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "bet": f"{team} (ML)",
            "edge": row.get("best_bet_edge", 0),
            "type": "ML",
        })

    # Collect RL bets
    rl_bets = games_df[games_df["best_rl_side"] != "PASS"].copy() if "best_rl_side" in games_df.columns else pd.DataFrame()
    for _, row in rl_bets.iterrows():
        team = row["home_team"] if row["best_rl_side"] == "HOME" else row["away_team"]
        if row["best_rl_side"] == "HOME":
            rl_val = float(row['home_rl'])
        else:
            rl_val = float(row['away_rl'])
        side_str = f"+{rl_val:.1f}" if rl_val > 0 else f"{rl_val:.1f}"
        all_bets.append({
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "bet": f"{team} (RL {side_str})",
            "edge": row.get("best_rl_edge_pct", 0) or 0,
            "type": "RL",
        })

    # Collect O/U bets
    ou_bets = games_df[games_df["best_total_direction"].notna()].copy() if "best_total_direction" in games_df.columns else pd.DataFrame()
    for _, row in ou_bets.iterrows():
        direction = row.get("best_total_direction", "?")
        all_bets.append({
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "bet": f"{direction} (O/U {row.get('total_line', '?')})",
            "edge": row.get("best_total_edge_pct", 0) or 0,
            "type": "O/U",
        })

    if all_bets:
        bets_df = pd.DataFrame(all_bets).sort_values("edge", ascending=False).head(10)
        bet_rows = []
        for rank, (_, row) in enumerate(bets_df.iterrows(), 1):
            edge = row["edge"]
            ec = _edge_class(edge)
            bet_rows.append(f"""
            <tr>
              <td>{rank}</td>
              <td><strong>{row['matchup']}</strong></td>
              <td>{row['bet']}</td>
              <td class="{ec}">{edge:+.1f}%</td>
            </tr>""")

        sections.append(f"""
        <table>
          <thead>
            <tr>
              <th>#</th><th>Matchup</th><th>Bet</th><th>Edge</th>
            </tr>
          </thead>
          <tbody>{"".join(bet_rows)}</tbody>
        </table>""")
    else:
        sections.append("<p>No bets available today.</p>")

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

    print("  Building email...")
    html = _build_html(games_df, today)
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
