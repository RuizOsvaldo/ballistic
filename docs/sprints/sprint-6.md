# Sprint 6 — NBA Dashboard

**Goal:** Replace the basketball stub with a full multi-section NBA dashboard — Games, Teams, Players, Props, and Season Totals.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| NBA-011 | src/dashboard/pages/basketball.py — full multi-section NBA page | [x] |
| NBA-017 | Updated app.py to pass bankroll to basketball page | [x] |
| NBA-012 | NBA Games section — today's games, spread/moneyline edge, filters, AI reasoning | [x] |
| NBA-013 | NBA Teams section — net rating chart, four factors table, regression signals | [x] |
| NBA-014 | NBA Players section — stat leaders, rolling averages, prop-relevant metrics | [x] |
| NBA-015 | NBA Props section — pts/reb/ast/PRA projections vs. prop lines, ranked by edge | [x] |
| NBA-016 | NBA Season Totals section — projected wins vs. Vegas O/U | [x] |

## Sprint 6 Complete

**Dashboard pages (all inside basketball.py with sub-navigation tabs):**

- **Games**: Today's NBA games sorted by edge %. Filters: team, min edge, BET only. Each game shows net rating differential, rest mismatch flags, model win prob, market prob, edge %, Kelly stake. AI reasoning on demand.
- **Teams**: All 30 teams ranked by net rating. Bar chart. Four factors table. Regression signal cards (teams with net rating >> actual W% highlighted).
- **Players**: Rolling 10-game stat averages. Usage rate, pace-adjusted per-36 numbers. Prop-relevant context columns.
- **Props**: Enter upcoming game → select player → model projects pts/reb/ast/PRA/3PM → compare to manual or live prop line → edge %. AI reasoning per prop.
- **Season Totals**: Prior season net rating → projected wins → Vegas O/U comparison chart. Flag OVER/UNDER bets.

## Notes
- Sub-navigation uses `st.tabs()` for clean in-page navigation (no sidebar reload)
- Props page: live prop lines fetched from The Odds API (SPORT_NBA already defined in odds.py)
- Kelly calculator in sidebar applies to NBA bets (bankroll already available from app.py)
- Signal severity consistent with baseball: High/Medium/Low color scheme
