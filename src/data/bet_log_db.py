"""SQLite-backed bet log — replaces CSV storage. Uses Python's built-in sqlite3."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/bet_log.db")

COLUMNS = [
    "id", "date", "matchup", "bet_side", "line", "stake", "edge_pct",
    "model_prob", "signal_type", "outcome", "pnl", "notes", "bet_type", "parlay_id",
]

OUTCOMES     = ["Pending", "Win", "Loss", "Push"]
SIGNAL_TYPES = ["None", "Pythagorean", "FIP-ERA", "BABIP", "Multiple"]
BET_TYPES    = ["Single", "Parlay"]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def _init() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                matchup     TEXT,
                bet_side    TEXT,
                line        REAL,
                stake       REAL,
                edge_pct    REAL,
                model_prob  REAL,
                signal_type TEXT,
                outcome     TEXT DEFAULT 'Pending',
                pnl         REAL DEFAULT 0.0,
                notes       TEXT,
                bet_type    TEXT DEFAULT 'Single',
                parlay_id   INTEGER
            )
        """)
        # Migrate existing databases
        for col, definition in [
            ("bet_type",  "TEXT DEFAULT 'Single'"),
            ("parlay_id", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE bets ADD COLUMN {col} {definition}")
            except Exception:
                pass


def load_bets() -> pd.DataFrame:
    _init()
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM bets ORDER BY date DESC, id DESC", conn)
    return df if not df.empty else pd.DataFrame(columns=COLUMNS)


def get_best_bet_type(min_bets: int = 3) -> str | None:
    """
    Return 'ML', 'RL', or 'O/U' — whichever has the highest win rate in the bet log.
    Returns None when no type has at least min_bets settled bets.
    """
    log = load_bets()
    if log.empty:
        return None
    settled = log[log["outcome"].isin(["Win", "Loss"])]
    line_bets = settled[settled["bet_type"].isin(["ML", "RL", "O/U"])]
    if line_bets.empty:
        return None
    counts = line_bets.groupby("bet_type").size()
    win_rate = line_bets.groupby("bet_type")["outcome"].apply(
        lambda x: (x == "Win").sum() / len(x)
    )
    valid = win_rate[counts[win_rate.index] >= min_bets]
    if valid.empty:
        return None
    return str(valid.idxmax())


def insert_bet(row: dict) -> None:
    _init()
    cols = [c for c in COLUMNS if c != "id" and c in row]
    placeholders = ", ".join(["?" for _ in cols])
    sql = f"INSERT INTO bets ({', '.join(cols)}) VALUES ({placeholders})"
    values = [row[c] for c in cols]
    with _connect() as conn:
        conn.execute(sql, values)


def insert_parlay(legs: list[dict]) -> None:
    """Log multiple bets sharing the same auto-generated parlay_id."""
    if not legs:
        return
    import time
    parlay_id = int(time.time())
    _init()
    with _connect() as conn:
        for leg in legs:
            row = {**leg, "bet_type": "Parlay", "parlay_id": parlay_id}
            cols = [c for c in COLUMNS if c != "id" and c in row]
            placeholders = ", ".join(["?" for _ in cols])
            conn.execute(
                f"INSERT INTO bets ({', '.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )


def update_bet_outcome(bet_id: int, outcome: str, pnl: float) -> None:
    """Update outcome and PnL for a single bet row by primary key."""
    _init()
    with _connect() as conn:
        conn.execute(
            "UPDATE bets SET outcome = ?, pnl = ? WHERE id = ?",
            (outcome, round(pnl, 2), bet_id),
        )


def _resolve_outcome(bet_side: str, matchup: str, home_score: int, away_score: int) -> str | None:
    """
    Return 'Win', 'Loss', or 'Push' for a settled game, or None if unparseable.

    bet_side formats:
      ML  — "ML: {team} {odds}"
      RL  — "RL: {team} +/-1.5 ({odds})"
      O/U — "Over {line} ({odds})" | "Under {line} ({odds})"
    """
    import re

    # Parse home/away from "{away} @ {home}"
    try:
        away_team, home_team = matchup.split(" @ ", 1)
    except ValueError:
        return None

    # ── Moneyline ─────────────────────────────────────────────────────────────
    if bet_side.startswith("ML:"):
        # "ML: New York Yankees -140"  → team = everything before the trailing odds token
        rest = bet_side[4:].strip()
        parts = rest.rsplit(" ", 1)
        team = parts[0] if len(parts) > 1 else rest
        winner = home_team if home_score > away_score else away_team
        if team.lower() in winner.lower() or winner.lower() in team.lower():
            return "Win"
        return "Loss"

    # ── Run Line ──────────────────────────────────────────────────────────────
    if bet_side.startswith("RL:"):
        rest = bet_side[4:].strip()
        m = re.match(r"(.+?)\s+([\+\-]\d+\.?\d*)", rest)
        if not m:
            return None
        team, spread_str = m.group(1).strip(), float(m.group(2))
        is_home = team.lower() in home_team.lower() or home_team.lower() in team.lower()
        margin = (home_score - away_score) if is_home else (away_score - home_score)
        if spread_str < 0:      # -1.5 (favourite must win by 2+)
            if margin > abs(spread_str):
                return "Win"
            elif margin == abs(spread_str):
                return "Push"
            return "Loss"
        else:                   # +1.5 (underdog can lose by 1 or win outright)
            if margin > -abs(spread_str):
                return "Win"
            elif margin == -abs(spread_str):
                return "Push"
            return "Loss"

    # ── Over / Under ──────────────────────────────────────────────────────────
    m = re.match(r"(Over|Under)\s+([\d.]+)", bet_side, re.IGNORECASE)
    if m:
        direction = m.group(1).capitalize()
        line = float(m.group(2))
        total = home_score + away_score
        if total == line:
            return "Push"
        over_wins = total > line
        if direction == "Over":
            return "Win" if over_wins else "Loss"
        return "Win" if not over_wins else "Loss"

    return None


def settle_pending_bets() -> int:
    """
    Settle all pending bets whose game date has passed.

    Fetches completed game results from the MLB Stats API for each relevant date,
    evaluates ML / RL / O/U outcomes, and writes Win / Loss / Push + PnL to the DB.
    Returns the number of bets settled.
    """
    import datetime as _dt
    from src.data.cache import _cache_path
    from src.data.game_results import get_games_for_date

    _init()
    with _connect() as conn:
        pending_df = pd.read_sql_query(
            "SELECT * FROM bets WHERE outcome = 'Pending'", conn
        )

    if pending_df.empty:
        return 0

    today = _dt.date.today()
    settled = 0

    for date_str, group in pending_df.groupby("date"):
        try:
            game_date = _dt.date.fromisoformat(str(date_str))
        except (ValueError, TypeError):
            continue
        if game_date > today:
            continue  # game hasn't happened yet

        # Bust the results cache for today so we always get the latest scores
        if game_date == today:
            try:
                _cache_path(f"mlb_results_{today.isoformat()}").unlink(missing_ok=True)
            except Exception:
                pass

        try:
            results = get_games_for_date(game_date)
        except Exception:
            continue
        if results is None or results.empty:
            continue

        # Build lookup: "{away} @ {home}" → result row
        results_lookup: dict[str, dict] = {}
        for _, r in results.iterrows():
            key = f"{r['away_team']} @ {r['home_team']}"
            results_lookup[key] = r.to_dict()

        for _, bet in group.iterrows():
            matchup = str(bet.get("matchup", ""))
            result  = results_lookup.get(matchup)
            if result is None:
                continue

            outcome = _resolve_outcome(
                str(bet.get("bet_side", "")),
                matchup,
                int(result["home_score"]),
                int(result["away_score"]),
            )
            if outcome is None:
                continue

            stake = float(bet.get("stake", 0) or 0)
            line  = float(bet.get("line", -110) or -110)
            if outcome == "Win":
                pnl = (stake * line / 100) if line > 0 else (stake * 100 / abs(line))
                pnl = round(pnl, 2)
            elif outcome == "Loss":
                pnl = -abs(stake)
            else:
                pnl = 0.0

            update_bet_outcome(int(bet["id"]), outcome, pnl)
            settled += 1

    return settled


def save_all(df: pd.DataFrame) -> None:
    """Overwrite the entire bet log with the given DataFrame (used after data_editor edits)."""
    _init()
    write_df = df.copy()
    with _connect() as conn:
        conn.execute("DELETE FROM bets")
        write_cols = [c for c in COLUMNS if c != "id" and c in write_df.columns]
        for _, row in write_df.iterrows():
            values = [row.get(c) for c in write_cols]
            placeholders = ", ".join(["?" for _ in write_cols])
            conn.execute(
                f"INSERT INTO bets ({', '.join(write_cols)}) VALUES ({placeholders})",
                values,
            )
