# Sprint 8 — NFL Dashboard

**Goal:** Replace the football stub with a full multi-section NFL dashboard — Games, Teams, Players, Props, and Season Totals.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| NFL-012 | src/dashboard/pages/football.py — full multi-section NFL page with 5 tabs | [x] |
| NFL-013 | NFL Games tab — today's games, spread/moneyline edge, weather flags, rest alerts, AI reasoning | [x] |
| NFL-014 | NFL Teams tab — EPA chart, efficiency table, regression signal cards | [x] |
| NFL-015 | NFL Players tab — QB/RB/WR/TE leaders, air yards, usage, target share | [x] |
| NFL-016 | NFL Props tab — projection vs. line, edge %, AI reasoning per prop | [x] |
| NFL-017 | NFL Season Totals tab — projected wins vs. Vegas O/U, OVER/UNDER flags | [x] |
| NFL-018 | Updated app.py to pass bankroll to football page | [x] |

## Sprint 8 Complete

**Dashboard tabs (all inside football.py):**

- **Games**: Upcoming NFL games sorted by edge %. EPA differential, weather flag, rest mismatch alerts. Model win prob vs. market implied. Kelly stake. AI reasoning on demand.
- **Teams**: All 32 teams ranked by EPA composite. Offense vs. Defense scatter. Regression signal cards for teams where EPA diverges from record.
- **Players**: Filtered by position. Sortable by key stat. Usage rate, air yards share, snap count context.
- **Props**: Select player + opponent → model projects stat → compare to prop line → edge %. AI reasoning for BET signals.
- **Season Totals**: Prior season EPA → projected wins → Vegas O/U comparison. OVER/UNDER flags with 1.5-win threshold.

## Notes
- Weather tab section flags games with wind > 15mph or temp < 30F automatically
- Rest mismatch: Thursday short week vs. rested opponent is the strongest NFL betting edge
- Props page supports QB, RB, and WR/TE in one unified flow
- Signal severity consistent: High/Medium/Low color scheme matches baseball and basketball
