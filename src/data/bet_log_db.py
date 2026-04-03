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
