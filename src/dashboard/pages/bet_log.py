"""Bet log page — CSV-backed bet tracking with model calibration reporting."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.models.calibration import (
    compute_calibration_table,
    compute_edge_vs_outcome,
    compute_signal_roi,
    recommend_threshold_adjustments,
)

BET_LOG_PATH = Path("data/bet_log.csv")

COLUMNS = [
    "date", "matchup", "bet_side", "line", "stake", "edge_pct",
    "model_prob", "signal_type", "outcome", "pnl", "notes",
]

OUTCOMES = ["Pending", "Win", "Loss", "Push"]

SIGNAL_TYPES = ["None", "Pythagorean", "FIP-ERA", "BABIP", "Multiple"]


def _load_log() -> pd.DataFrame:
    if BET_LOG_PATH.exists():
        df = pd.read_csv(BET_LOG_PATH)
        # Back-fill columns added in this sprint for older rows
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _save_log(df: pd.DataFrame) -> None:
    BET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BET_LOG_PATH, index=False)


def _compute_pnl(outcome: str, stake: float, line: int | float) -> float:
    """Compute profit/loss from outcome, stake, and American line."""
    if outcome == "Win":
        if line > 0:
            return round(stake * line / 100, 2)
        else:
            return round(stake * 100 / abs(line), 2)
    elif outcome == "Loss":
        return -abs(stake)
    return 0.0


def render() -> None:
    st.header("Bet Log")

    log = _load_log()

    tab_log, tab_cal = st.tabs(["Bet Log", "Calibration Report"])

    # =========================================================================
    # TAB 1 — Bet Log
    # =========================================================================
    with tab_log:
        st.caption("Track bets placed and record outcomes manually.")

        # ---- Add new bet form ----
        with st.expander("+ Log a New Bet", expanded=False):
            with st.form("new_bet"):
                col1, col2 = st.columns(2)
                with col1:
                    date = st.date_input("Date")
                    matchup = st.text_input("Matchup (e.g. NYY @ BOS)")
                    bet_side = st.text_input("Bet side (team name)")
                    line = st.number_input("American line (e.g. -110, +130)", value=-110, step=5)
                    model_prob = st.number_input(
                        "Model win probability % (from Games page)",
                        min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                        help="Enter the model's predicted win % at the time of bet, e.g. 58.3",
                    )
                with col2:
                    stake = st.number_input("Stake ($)", min_value=1.0, value=50.0, step=10.0)
                    edge_pct = st.number_input("Edge % at time of bet", value=0.0, step=0.1)
                    signal_type = st.selectbox(
                        "Signal that triggered bet",
                        SIGNAL_TYPES,
                        help="Which regression signal (if any) supported this bet",
                    )
                    outcome = st.selectbox("Outcome", OUTCOMES)
                    notes = st.text_input("Notes (optional)")

                submitted = st.form_submit_button("Log Bet")

            if submitted:
                pnl = _compute_pnl(outcome, stake, line) if outcome != "Pending" else 0.0
                new_row = pd.DataFrame([{
                    "date": str(date),
                    "matchup": matchup,
                    "bet_side": bet_side,
                    "line": line,
                    "stake": stake,
                    "edge_pct": edge_pct,
                    "model_prob": round(model_prob / 100, 4) if model_prob > 0 else None,
                    "signal_type": signal_type,
                    "outcome": outcome,
                    "pnl": pnl,
                    "notes": notes,
                }])
                log = pd.concat([log, new_row], ignore_index=True)
                _save_log(log)
                st.success(f"Bet logged: {bet_side} ({matchup})")
                st.rerun()

        # ---- Summary stats ----
        if not log.empty:
            settled = log[log["outcome"].isin(["Win", "Loss"])]
            if not settled.empty:
                total_staked = settled["stake"].sum()
                total_pnl = settled["pnl"].sum()
                win_rate = (settled["outcome"] == "Win").mean() * 100
                roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Bets", len(settled))
                c2.metric("Win Rate", f"{win_rate:.1f}%")
                c3.metric("Total P&L", f"${total_pnl:+,.2f}")
                c4.metric("ROI", f"{roi:+.1f}%")

            # ---- Outcome editor ----
            st.subheader("Bet History")
            st.caption("Edit outcome in the table below, then click 'Save Changes'.")

            edited = st.data_editor(
                log,
                column_config={
                    "outcome": st.column_config.SelectboxColumn(
                        "Outcome", options=OUTCOMES
                    ),
                    "signal_type": st.column_config.SelectboxColumn(
                        "Signal", options=SIGNAL_TYPES
                    ),
                    "model_prob": st.column_config.NumberColumn(
                        "Model Prob", format="%.3f", help="Win probability (0-1)"
                    ),
                    "pnl": st.column_config.NumberColumn("P&L ($)", format="$%.2f"),
                    "stake": st.column_config.NumberColumn("Stake ($)", format="$%.2f"),
                },
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
            )

            if st.button("Save Changes"):
                edited["pnl"] = edited.apply(
                    lambda r: _compute_pnl(r["outcome"], r["stake"], r["line"])
                    if r["outcome"] != "Pending" else 0.0,
                    axis=1,
                )
                _save_log(edited)
                st.success("Bet log saved.")
                st.rerun()
        else:
            st.info("No bets logged yet. Use the form above to add your first bet.")

    # =========================================================================
    # TAB 2 — Calibration Report
    # =========================================================================
    with tab_cal:
        st.subheader("Model Calibration Report")
        st.caption(
            "Measures how well the model's predicted win probabilities match real outcomes. "
            "Requires settled bets (Win/Loss) with model_prob filled in."
        )

        settled = log[log["outcome"].isin(["Win", "Loss"])].copy()

        if settled.empty or settled["model_prob"].isna().all():
            st.info(
                "No calibration data yet. Log bets with **Model Probability** filled in "
                "and mark their outcomes as Win or Loss to see this report."
            )
        else:
            settled_with_prob = settled.dropna(subset=["model_prob"])

            # ---- Calibration by probability bucket ----
            st.markdown("#### Predicted vs Actual Win Rate by Probability Bucket")
            cal_df = compute_calibration_table(settled_with_prob)
            if not cal_df.empty:
                st.dataframe(
                    cal_df.rename(columns={
                        "bucket": "Prob Bucket",
                        "bets": "Bets",
                        "expected_win_rate": "Expected Win%",
                        "actual_win_rate": "Actual Win%",
                        "calibration_error": "Error (Actual−Expected)",
                        "total_staked": "Staked ($)",
                        "total_pnl": "P&L ($)",
                        "roi_pct": "ROI%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Tuning Suggestions")
                for suggestion in recommend_threshold_adjustments(cal_df):
                    st.write(f"- {suggestion}")

            # ---- ROI by signal type ----
            if "signal_type" in settled.columns and not settled["signal_type"].isna().all():
                st.markdown("#### ROI by Signal Type")
                sig_df = compute_signal_roi(settled.dropna(subset=["signal_type"]))
                if not sig_df.empty:
                    st.dataframe(
                        sig_df.rename(columns={
                            "signal_type": "Signal",
                            "bets": "Bets",
                            "win_rate": "Win Rate",
                            "total_staked": "Staked ($)",
                            "total_pnl": "P&L ($)",
                            "roi_pct": "ROI%",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            # ---- Edge bucket analysis ----
            st.markdown("#### Win Rate vs Edge % (Does Higher Edge = More Wins?)")
            edge_df = compute_edge_vs_outcome(settled)
            if not edge_df.empty:
                st.dataframe(
                    edge_df.rename(columns={
                        "edge_bucket": "Edge Bucket",
                        "bets": "Bets",
                        "avg_edge_pct": "Avg Edge%",
                        "actual_win_rate": "Actual Win%",
                        "roi_pct": "ROI%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Need more settled bets to compute edge bucket analysis.")

            st.divider()
            st.caption(
                f"Based on {len(settled_with_prob)} settled bets with model probability recorded "
                f"(out of {len(settled)} total settled bets)."
            )
