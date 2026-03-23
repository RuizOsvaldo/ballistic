# Sprint 2 — Models, Agent, and Extended Stats

**Goal:** All statistical models implemented, Groq agent integrated, player props and preseason projections added.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| MLB-006 | Pythagorean model | [x] |
| MLB-007 | Regression signals (BABIP, FIP-ERA) | [x] |
| MLB-008 | Win probability model | [x] |
| MLB-009 | Kelly criterion | [x] |
| MLB-010 | Groq agent (src/shared/groq_agent.py) | [x] |
| MLB-011 | Extended baseball_stats.py (xFIP, SIERA, wRC+, barrel%, whiff%) | [x] |
| MLB-012 | Player props model (src/models/player_props.py) | [x] |
| MLB-013 | Preseason win total projections (src/models/preseason.py) | [x] |
| MLB-014 | Extend odds.py for totals and player props | [x] |

## Sprint 2 Complete

All models and the agent are in place:

**Models:**
- `pythagorean.py`: Pythagorean W%, deviation, and signal labels
- `regression_signals.py`: FIP-ERA gap and BABIP deviation with severity levels
- `win_probability.py`: Composite game win probability (Pythagorean + FIP + home field)
- `kelly.py`: Half-Kelly bet sizing with 3% minimum edge threshold
- `player_props.py`: Pitcher K projections, batter hit/TB/HR projections, prop edge calculation
- `preseason.py`: Peta-methodology win total projection with regression to mean and Vegas line comparison

**Data:**
- `baseball_stats.py`: Extended with xFIP, SIERA, wRC+, barrel%, exit velocity, whiff rate
- `odds.py`: Extended with game totals (O/U) and player props endpoints

**Agent:**
- `groq_agent.py`: Llama 3.3 70B via Groq API. Game, prop, and preseason reasoning functions.

## Notes
- Pythagorean formula: RS^2 / (RS^2 + RA^2)
- BABIP thresholds: >0.320 = high (ERA likely UP), <0.275 = low (ERA likely DOWN)
- FIP-ERA gap threshold: >0.75 = pitcher regression risk
- Preseason edge threshold: ±2 wins vs. Vegas line
- Kelly: half-Kelly, minimum 3% edge, max 5% of bankroll per bet
- Groq model: llama-3.3-70b-versatile, response_format=json_object
