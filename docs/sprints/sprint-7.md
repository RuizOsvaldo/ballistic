# Sprint 7 — NFL Data, Models, and Agent

**Goal:** Full NFL analytical layer — team EPA models, weather model, rest/schedule, player prop projections, and Groq agent integration.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| NFL-001 | requirements.txt — add nfl_data_py | [x] |
| NFL-002 | src/sports/football/data/nfl_stats.py — team EPA stats, player weekly stats, schedule | [x] |
| NFL-003 | src/sports/football/models/epa.py — EPA/play deviation signal, composite efficiency, spread model | [x] |
| NFL-004 | src/sports/football/models/weather.py — wind/temp/precip total adjustments | [x] |
| NFL-005 | src/sports/football/models/rest_schedule.py — bye week, short week, travel adjustments | [x] |
| NFL-006 | src/sports/football/models/player_props.py — QB/RB/WR/TE prop projections | [x] |
| NFL-007 | src/sports/football/models/win_probability.py — composite NFL win probability + spread | [x] |
| NFL-008 | src/sports/football/agent/groq_prompts.py — NFL game, prop, season total prompts | [x] |
| NFL-009 | tests/test_nfl_epa.py | [x] |
| NFL-010 | tests/test_nfl_player_props.py | [x] |
| NFL-011 | tests/test_nfl_weather.py | [x] |

## Sprint 7 Complete

**Data layer (`nfl_stats.py`):**
- `get_nfl_team_epa(season)` — offensive and defensive EPA/play per team
- `get_nfl_player_stats(season)` — weekly stats for QB/RB/WR/TE
- `get_nfl_schedule(season)` — game schedule with spread lines, totals, weather context

**Models:**
- `epa.py`: EPA composite efficiency, deviation from implied W%, regression signal. Spread equivalent = EPA diff × 14.
- `weather.py`: Wind/temp/precip → total line adjustment. Dome detection.
- `rest_schedule.py`: Bye week (+2.5 pts), short week/Thursday (-2.0 pts), home field (+2.5 pts), West Coast penalty.
- `player_props.py`: QB (pass yds, completions, TDs, INTs), RB (rush yds, rec yds, TDs), WR/TE (rec yds, receptions, TDs, air yards)
- `win_probability.py`: EPA base + rest + home court → win prob and implied spread

**Agent (`groq_prompts.py`):**
- `analyze_nfl_game()` — game matchup with EPA/DVOA context
- `analyze_nfl_prop()` — player prop with usage and matchup signals
- `analyze_nfl_season_total()` — preseason win total projection

**Tests:** 76 test cases across 3 test files — all passing.

## Key Constants
- EPA/play to spread factor: ×14
- Win prob logistic scale: /7 (7-point spread = ~73% win prob)
- Home field advantage: +2.5 pts
- Bye week rest bonus: +2.5 pts
- Short week penalty: -2.0 pts
- Weather: wind >15mph = -3pts off total; >25mph = -7pts; temp <30F = -2pts; rain = -2pts
- Preseason edge threshold: ±1.5 wins vs Vegas line
