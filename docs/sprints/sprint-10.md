# Sprint 10 — Run Line and Total Edge Calculations

## Goal
Extend edge calculations from moneyline-only to all three main bet types (ML, run line, game total) using a Poisson run-distribution model.

## Context
The model currently computes win probability (→ ML edge) and projected runs per team, but edge on the ±1.5 run line and game total is never computed. Both can be derived from the same projected run outputs already produced by `win_probability.py`.

## Architecture Notes
**Approach chosen: Poisson joint distribution (no scipy)**
- `proj_home_runs` and `proj_away_runs` (λ) already exist as model outputs.
- Run line P(cover): treat each team's score as independent Poisson. Compute P(home − away > 1.5) via a 36×36 joint PMF outer product (0–35 runs each covers >99.9% of the probability mass).
- Total P(over): X+Y ~ Poisson(λ_home + λ_away). P(sum > total_line) is a direct CDF tail.
- Both use `math.lgamma` for log-space PMF computation — stable, no extra dependencies.

**Alternative considered: scipy.stats.skellam / poisson.cdf**
Rejected — scipy is not in requirements.txt. Adding it for two functions is not justified.

**Edge semantics match ML exactly**: `edge = model_prob − implied_prob`. Same `MIN_EDGE` threshold, same half-Kelly sizing logic available if needed.

## Tickets

### Ticket 10-01: Poisson helpers and edge functions in kelly.py
**Type:** Feature
**Files in scope:** `src/models/kelly.py`
**Description:** Add `_poisson_pmf`, `_p_home_covers_rl`, `_p_over_total`, `compute_rl_edge`, `compute_total_edge`.
**Acceptance Criteria:**
- [ ] `_p_home_covers_rl(5.0, 5.0)` returns value near 0.28 (symmetric teams, ~28% chance of 2+ run win)
- [ ] `_p_over_total(4.5, 4.0, 8.5)` returns plausible over probability (~50% range)
- [ ] `compute_rl_edge` returns `best_rl_side="PASS"` when no odds provided
- [ ] `compute_total_edge` returns correct direction when proj_total clearly exceeds total_line
**Edge Cases:**
- `proj_home_runs` or `proj_away_runs` is None or NaN → return neutral dict, no crash
- Odds missing (None) → edge_pct is None, best side is PASS

### Ticket 10-02: Wire RL and Total edges into compute_kelly_for_games
**Type:** Feature
**Files in scope:** `src/models/kelly.py`
**Description:** Extend `compute_kelly_for_games` to call `compute_rl_edge` and `compute_total_edge` per row and attach results as new columns.
**Acceptance Criteria:**
- [ ] Output DataFrame includes `home_rl_edge_pct`, `away_rl_edge_pct`, `best_rl_side`, `best_rl_edge_pct`
- [ ] Output DataFrame includes `over_edge_pct`, `under_edge_pct`, `best_total_direction`, `best_total_edge_pct`
- [ ] Rows missing projected runs gracefully produce null/PASS values
**Edge Cases:**
- `proj_home_runs` column absent from input → columns added with null values, no crash
- `total_line` is NaN → total edge columns are null

### Ticket 10-03: RL Edge% and Total Edge% columns in Main Line Game Bets table
**Type:** Feature
**Files in scope:** `src/dashboard/pages/games.py`
**Description:** Add `_rl_edge_pct` and `_total_edge_pct` derived columns; show them in the table. Rename existing `Edge%` header to `ML Edge%`.
**Acceptance Criteria:**
- [ ] Table shows `ML Edge%`, `RL Edge%`, `Total Edge%` as distinct columns
- [ ] Positive edge values are highlighted in green; negative in muted red
- [ ] Columns show `—` when edge data is unavailable (no odds, no proj runs)
**Edge Cases:**
- New columns absent from DataFrame (old cache) → shown as `—`
