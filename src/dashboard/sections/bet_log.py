"""Bet log page — SQLite-backed bet tracking with model calibration reporting."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.bet_log_db import (
    COLUMNS, OUTCOMES, SIGNAL_TYPES,
    load_bets, insert_bet, save_all,
)
from src.models.calibration import (
    compute_calibration_table,
    compute_edge_vs_outcome,
    compute_signal_roi,
    recommend_threshold_adjustments,
)


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


def _render_dashboard(log: pd.DataFrame) -> None:
    """Render the analytics dashboard above the bet history table."""
    settled = log[log["outcome"].isin(["Win", "Loss"])].copy()
    pending = log[log["outcome"] == "Pending"]

    if settled.empty and pending.empty:
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_settled  = len(settled)
    wins           = (settled["outcome"] == "Win").sum() if not settled.empty else 0
    win_rate       = wins / total_settled * 100 if total_settled > 0 else 0.0
    total_pnl      = settled["pnl"].sum() if not settled.empty else 0.0
    total_staked   = settled["stake"].sum() if not settled.empty else 0.0
    roi            = total_pnl / total_staked * 100 if total_staked > 0 else 0.0
    avg_stake      = total_staked / total_settled if total_settled > 0 else 0.0
    n_pending      = len(pending)

    # Current streak
    streak_str = "—"
    if not settled.empty:
        outcomes_sorted = settled.sort_values("date")["outcome"].tolist()
        last = outcomes_sorted[-1]
        count = 0
        for o in reversed(outcomes_sorted):
            if o == last:
                count += 1
            else:
                break
        icon = "🔥" if last == "Win" else "❄️"
        streak_str = f"{icon} {count}{'W' if last == 'Win' else 'L'}"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Settled Bets", total_settled)
    k2.metric("Win Rate", f"{win_rate:.1f}%")
    k3.metric("Total P&L", f"${total_pnl:+,.2f}")
    k4.metric("ROI", f"{roi:+.1f}%")
    k5.metric("Avg Stake", f"${avg_stake:,.0f}")
    k6.metric("Streak / Pending", f"{streak_str}  ·  {n_pending}⏳")

    if settled.empty:
        return

    st.divider()

    # ── Row 1: Cumulative P&L + Bet Type Breakdown ────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("#### Cumulative P&L")
        cum = (
            settled.sort_values("date")
            .assign(cumulative_pnl=lambda d: d["pnl"].cumsum())
            .reset_index(drop=True)
        )
        cum["label"] = cum["date"].astype(str) + " · " + cum["matchup"].fillna("")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cum))),
            y=cum["cumulative_pnl"],
            mode="lines+markers",
            line=dict(color="#2ecc71" if total_pnl >= 0 else "#e74c3c", width=2),
            marker=dict(
                color=cum["outcome"].map({"Win": "#2ecc71", "Loss": "#e74c3c"}),
                size=7,
            ),
            hovertext=cum["label"],
            hoverinfo="text+y",
            name="Cumulative P&L",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#555")
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(tickprefix="$", gridcolor="#2a2a2a"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("#### P&L by Bet Type")
        if "bet_type" in settled.columns and settled["bet_type"].notna().any():
            type_df = (
                settled.groupby("bet_type")
                .agg(pnl=("pnl", "sum"), bets=("pnl", "count"))
                .reset_index()
            )
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in type_df["pnl"]]
            fig2 = go.Figure(go.Bar(
                x=type_df["bet_type"],
                y=type_df["pnl"],
                marker_color=colors,
                text=[f"${v:+,.0f}" for v in type_df["pnl"]],
                textposition="outside",
                customdata=type_df["bets"],
                hovertemplate="%{x}<br>P&L: $%{y:+,.2f}<br>Bets: %{customdata}<extra></extra>",
            ))
            fig2.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(tickprefix="$", gridcolor="#2a2a2a"),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No bet type data.")

    # ── Row 2: Monthly P&L + Win/Loss distribution ────────────────────────────
    row2_col1, row2_col2 = st.columns([3, 2])

    with row2_col1:
        st.markdown("#### Monthly P&L")
        settled["month"] = pd.to_datetime(settled["date"], errors="coerce").dt.to_period("M").astype(str)
        monthly = (
            settled.groupby("month")
            .agg(pnl=("pnl", "sum"), bets=("pnl", "count"), wins=("outcome", lambda x: (x == "Win").sum()))
            .reset_index()
        )
        monthly["win_rate"] = monthly["wins"] / monthly["bets"] * 100
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly["pnl"]]
        fig3 = go.Figure(go.Bar(
            x=monthly["month"],
            y=monthly["pnl"],
            marker_color=bar_colors,
            text=[f"${v:+,.0f}" for v in monthly["pnl"]],
            textposition="outside",
            customdata=list(zip(monthly["bets"], monthly["win_rate"])),
            hovertemplate="%{x}<br>P&L: $%{y:+,.2f}<br>Bets: %{customdata[0]}<br>Win rate: %{customdata[1]:.1f}%<extra></extra>",
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="#555")
        fig3.update_layout(
            height=220,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(tickprefix="$", gridcolor="#2a2a2a"),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        st.markdown("#### Outcome Distribution")
        outcome_counts = settled["outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig4 = go.Figure(go.Pie(
            labels=outcome_counts["Outcome"],
            values=outcome_counts["Count"],
            hole=0.55,
            marker_colors=["#2ecc71" if o == "Win" else "#e74c3c" for o in outcome_counts["Outcome"]],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} bets (%{percent})<extra></extra>",
        ))
        fig4.update_layout(
            height=220,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Recent form strip (last 10 settled bets) ──────────────────────────────
    st.markdown("#### Recent Form")
    recent = settled.sort_values("date").tail(10)
    cols = st.columns(len(recent))
    for col, (_, row) in zip(cols, recent.iterrows()):
        color = "#2ecc71" if row["outcome"] == "Win" else "#e74c3c"
        pnl_sign = "+" if row["pnl"] >= 0 else ""
        col.markdown(
            f'<div style="background:{color};border-radius:6px;padding:6px 4px;text-align:center">'
            f'<div style="font-weight:bold;font-size:13px;color:#000">{row["outcome"][0]}</div>'
            f'<div style="font-size:10px;color:#000">{pnl_sign}${row["pnl"]:.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()


def render() -> None:
    st.header("📒 Bet Log")
    st.caption("Stored in SQLite (data/bet_log.db) — free, local, no limits.")

    # Auto-settle any pending bets whose game date has passed
    try:
        from src.data.bet_log_db import settle_pending_bets as _settle
        n_settled = _settle()
        if n_settled > 0:
            st.toast(f"Auto-settled {n_settled} bet(s) from completed games.", icon="✅")
    except Exception:
        pass

    log = load_bets()

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
                insert_bet({
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
                })
                st.success(f"Bet logged: {bet_side} ({matchup})")
                st.rerun()

        # ---- Analytics dashboard ----
        if not log.empty:
            _render_dashboard(log)

            # ---- Bet History (styled, read-only) ----
            st.subheader("Bet History")

            def _row_color(row: pd.Series) -> list[str]:
                if row["outcome"] == "Win":
                    return ["background-color: #1a3d2b; color: #2ecc71"] * len(row)
                elif row["outcome"] == "Loss":
                    return ["background-color: #3d1a1a; color: #e74c3c"] * len(row)
                return [""] * len(row)

            display_df = log.copy()
            display_df["pnl"] = display_df["pnl"].map(lambda v: f"${v:+,.2f}")
            display_df["stake"] = display_df["stake"].map(lambda v: f"${v:,.2f}")
            display_df["model_prob"] = display_df["model_prob"].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else ""
            )

            styled = display_df.style.apply(_row_color, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ---- Outcome editor ----
            with st.expander("Edit Outcomes / Save Changes"):
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
                    save_all(edited)
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
