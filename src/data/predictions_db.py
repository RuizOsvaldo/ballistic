"""SQLite table for storing daily model predictions — enables automated accuracy tracking."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/bet_log.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def _init() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date  TEXT NOT NULL,
                home_team        TEXT NOT NULL,
                away_team        TEXT NOT NULL,
                predicted_winner TEXT,
                home_model_prob  REAL,
                away_model_prob  REAL,
                home_implied_prob REAL,
                away_implied_prob REAL,
                edge_pct         REAL,
                bet_side         TEXT,
                actual_winner    TEXT,
                correct          INTEGER,
                verified_at      TEXT,
                UNIQUE(prediction_date, home_team, away_team)
            )
        """)


def save_predictions(games_df: pd.DataFrame, prediction_date: str) -> None:
    """
    Persist today's model predictions. Skips games already stored for this date.
    games_df must have: home_team, away_team, home_model_prob, away_model_prob,
                        home_implied_prob, away_implied_prob, best_bet_side, best_bet_edge
    """
    _init()
    if games_df.empty:
        return

    with _connect() as conn:
        for _, row in games_df.iterrows():
            if pd.isna(row.get("home_model_prob")):
                continue
            home_prob = row.get("home_model_prob", 0)
            away_prob = row.get("away_model_prob", 0)
            predicted_winner = (
                row["home_team"] if home_prob >= away_prob else row["away_team"]
            )
            bet_side = row.get("best_bet_side", "PASS")
            edge = row.get("best_bet_edge", 0)
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (prediction_date, home_team, away_team, predicted_winner,
                     home_model_prob, away_model_prob, home_implied_prob, away_implied_prob,
                     edge_pct, bet_side)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_date,
                    row["home_team"], row["away_team"], predicted_winner,
                    round(float(home_prob), 4), round(float(away_prob), 4),
                    round(float(row.get("home_implied_prob", 0)), 4),
                    round(float(row.get("away_implied_prob", 0)), 4),
                    round(float(edge), 2),
                    bet_side,
                ))
            except Exception:
                pass


def load_predictions(days: int = 30) -> pd.DataFrame:
    _init()
    with _connect() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT * FROM predictions
            WHERE prediction_date >= date('now', '-{days} days')
            ORDER BY prediction_date DESC, id DESC
            """,
            conn,
        )
    return df


def update_result(prediction_date: str, home_team: str, away_team: str,
                  actual_winner: str, correct: bool, verified_at: str) -> None:
    _init()
    with _connect() as conn:
        conn.execute("""
            UPDATE predictions
            SET actual_winner = ?, correct = ?, verified_at = ?
            WHERE prediction_date = ? AND home_team = ? AND away_team = ?
        """, (actual_winner, int(correct), verified_at, prediction_date, home_team, away_team))
