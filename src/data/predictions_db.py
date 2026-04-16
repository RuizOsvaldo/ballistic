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
                proj_home_runs   REAL,
                proj_away_runs   REAL,
                proj_total       REAL,
                home_rl          REAL,
                away_rl          REAL,
                total_line       REAL,
                rl_side          TEXT,
                rl_edge_pct      REAL,
                rl_correct       INTEGER,
                total_direction  TEXT,
                total_edge_pct   REAL,
                total_correct    INTEGER,
                actual_winner    TEXT,
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                actual_total      REAL,
                correct          INTEGER,
                verified_at      TEXT,
                UNIQUE(prediction_date, home_team, away_team)
            )
        """)
        # Migrate existing databases that are missing the new columns
        existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        for col, coltype in [
            ("proj_home_runs",    "REAL"),
            ("proj_away_runs",    "REAL"),
            ("proj_total",        "REAL"),
            ("home_rl",           "REAL"),
            ("away_rl",           "REAL"),
            ("total_line",        "REAL"),
            ("rl_side",           "TEXT"),
            ("rl_edge_pct",       "REAL"),
            ("rl_correct",        "INTEGER"),
            ("total_direction",   "TEXT"),
            ("total_edge_pct",    "REAL"),
            ("total_correct",     "INTEGER"),
            ("actual_home_score", "INTEGER"),
            ("actual_away_score", "INTEGER"),
            ("actual_total",      "REAL"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")


def save_predictions(games_df: pd.DataFrame, prediction_date: str) -> None:
    """
    Persist today's model predictions. Skips games already stored for this date.
    games_df must have: home_team, away_team, home_model_prob, away_model_prob,
                        home_implied_prob, away_implied_prob, best_bet_side, best_bet_edge
    Optional: proj_home_runs, proj_away_runs, proj_total, home_rl, away_rl, total_line,
              best_rl_side, best_rl_edge_pct, best_total_direction, best_total_edge_pct
    """
    _init()
    if games_df.empty:
        return

    def _float_or_none(val) -> float | None:
        if val is None:
            return None
        try:
            f = float(val)
            return None if pd.isna(f) else round(f, 2)
        except (TypeError, ValueError):
            return None

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
            rl_side = row.get("best_rl_side") or None
            rl_edge = _float_or_none(row.get("best_rl_edge_pct"))
            total_direction = row.get("best_total_direction") or None
            total_edge = _float_or_none(row.get("best_total_edge_pct"))
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (prediction_date, home_team, away_team, predicted_winner,
                     home_model_prob, away_model_prob, home_implied_prob, away_implied_prob,
                     edge_pct, bet_side,
                     proj_home_runs, proj_away_runs, proj_total,
                     home_rl, away_rl, total_line,
                     rl_side, rl_edge_pct, total_direction, total_edge_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_date,
                    row["home_team"], row["away_team"], predicted_winner,
                    round(float(home_prob), 4), round(float(away_prob), 4),
                    round(float(row.get("home_implied_prob", 0)), 4),
                    round(float(row.get("away_implied_prob", 0)), 4),
                    round(float(edge), 2),
                    bet_side,
                    _float_or_none(row.get("proj_home_runs")),
                    _float_or_none(row.get("proj_away_runs")),
                    _float_or_none(row.get("proj_total")),
                    _float_or_none(row.get("home_rl")),
                    _float_or_none(row.get("away_rl")),
                    _float_or_none(row.get("total_line")),
                    rl_side,
                    rl_edge,
                    total_direction,
                    total_edge,
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


def update_result(
    prediction_date: str,
    home_team: str,
    away_team: str,
    actual_winner: str,
    correct: bool,
    verified_at: str,
    actual_home_score: int | None = None,
    actual_away_score: int | None = None,
    rl_correct: int | None = None,
    total_correct: int | None = None,
) -> None:
    """Persist ML result plus RL and total outcome columns."""
    _init()
    actual_total = (
        float(actual_home_score + actual_away_score)
        if actual_home_score is not None and actual_away_score is not None
        else None
    )
    with _connect() as conn:
        conn.execute("""
            UPDATE predictions
            SET actual_winner = ?, correct = ?, verified_at = ?,
                actual_home_score = ?, actual_away_score = ?, actual_total = ?,
                rl_correct = ?, total_correct = ?
            WHERE prediction_date = ? AND home_team = ? AND away_team = ?
        """, (
            actual_winner, int(correct), verified_at,
            actual_home_score, actual_away_score, actual_total,
            rl_correct, total_correct,
            prediction_date, home_team, away_team,
        ))
