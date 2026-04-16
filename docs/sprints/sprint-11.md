# Sprint 11 — Complete Peta + Sabermetrics Model

## Goal
Replace the approximate win probability model with the full Bill James / Joe Peta methodology: Log5 head-to-head probability, shrinkage-weighted RS/RA regression to the mean, and a lineup quality matchup adjustment on starting pitchers. Formula state is auto-detected from games played and displayed as a live banner in the app.

## Context
The model was producing sub-50% accuracy in early April because:
1. Pythagorean W% was being normalized (added + divided), not compared head-to-head — Log5 fixes this.
2. RS/RA from 10 games of data was being treated as reliable — regression to mean fixes this.
3. Pitcher FIP was not adjusted for the quality of the opposing lineup — lineup matchup fixes this.

## Architecture Notes

**Log5**: `P(A beats B) = (A - A*B) / (A + B - 2*A*B)`. Replaces `normalize(home_pyth + away_pyth)` as the base win probability. FIP, home field, and bullpen deltas are applied on top, then renormalized.

**Regression to mean**: `weight = G / (G + 30)`. Activates per-team once that team reaches `MIN_GAMES_FOR_REGRESSION = 20` games. League avg RS/G and RA/G are computed live from the current `team_stats_df`, not hardcoded. At 20 games: 40% actual, 60% prior. At 81 games: 73% actual. At 162 games: 84% actual.

**Lineup matchup**: Uses OPS (OBP + SLG) as proxy for lineup quality (wOBA unavailable from MLB Stats API; OPS correlates at r≈0.97). `effective_fip = starter_fip + (lineup_avg_ops - 0.720) * 3.0`. Graceful no-op when lineup not posted.

**Formula state banner**: Two states — `EARLY_SEASON` (any team < 20 G, blue info banner) and `REGRESSION_ACTIVE` (all teams ≥ 20 G, green success banner). Shown on Games page and Game Analysis page. Message explains what changed and why in plain language. Switches automatically — no manual trigger needed.

**Alternative considered and ruled out**: wRC+ / wOBA for lineup quality — not available from MLB Stats API. OPS used instead (same predictive power at this scale).

## Tickets

### Ticket 11-01: Log5 head-to-head probability ✅
**Type:** Feature
**Files:** `src/models/pythagorean.py`, `src/models/win_probability.py`
**Acceptance Criteria:**
- [x] Two equal teams produce exactly 0.5
- [x] .600 vs .400 team produces 0.6923 (Bill James reference)
- [x] FIP, home field, bullpen adjustments still applied after Log5 base
- [x] Tests pass

### Ticket 11-02: RS/RA regression to mean ✅
**Type:** Feature
**Files:** `src/models/pythagorean.py`, `src/models/win_probability.py`
**Acceptance Criteria:**
- [x] Team with < 20 games gets raw RS/RA (no regression)
- [x] Team with exactly 20 games: weight = 20/(20+30) = 0.40
- [x] Team with 162 games: weight ≈ 0.844
- [x] League avg RS/G and RA/G derived from passed team_stats_df
- [x] Tests pass

### Ticket 11-03: Lineup quality matchup adjustment ✅
**Type:** Feature
**Files:** `src/models/win_probability.py`
**Acceptance Criteria:**
- [x] League-average lineup (OPS 0.720) produces 0.0 adjustment
- [x] Strong lineup (OPS 0.770) produces +0.15 FIP penalty on opposing pitcher
- [x] Empty lineup DataFrame returns 0.0
- [x] Tests pass

### Ticket 11-04: Seasonal formula state + in-app banner ✅
**Type:** Feature
**Files:** `src/models/win_probability.py`, `src/dashboard/pages/games.py`, `src/dashboard/pages/game_analysis.py`
**Acceptance Criteria:**
- [x] Returns EARLY_SEASON when any team has < 20 games played
- [x] Returns REGRESSION_ACTIVE when all teams have ≥ 20 games played
- [x] Banner visible on Games page and Game Analysis page
- [x] Banner text explains the formula in plain language
- [x] Tests pass

### Ticket 11-05: Tests ✅
**Type:** Chore
**Files:** `tests/test_pythagorean.py`, `tests/test_win_probability.py`
**Result:** 59/59 tests passing
