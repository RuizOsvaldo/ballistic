# Sprint 3 — Dashboard (Baseball)

**Goal:** Full Streamlit app with all baseball pages, multi-sport navigation, and AI reasoning.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| MLB-015 | Multi-sport app.py with sport selector sidebar | [x] |
| MLB-016 | Games page with edge table and AI reasoning button | [x] |
| MLB-017 | Teams page with Pythagorean chart and signal cards | [x] |
| MLB-018 | Pitchers page with FIP vs ERA scatter and regression table | [x] |
| MLB-019 | Player props page (pitcher K props + batter hit props) | [x] |
| MLB-020 | Preseason projections page with Vegas line input + chart | [x] |
| MLB-021 | Bet log page with P&L tracking | [x] |
| MLB-022 | Basketball page stub | [x] |
| MLB-023 | Football page stub | [x] |

## Sprint 3 Complete

Full dashboard is running with all baseball sections:

**Pages:**
- `games.py`: Today's games sorted by edge %. Filters (team, min edge, BET only). Each recommended bet shows model prob, market prob, edge %, Kelly stake, and an AI reasoning button (on-demand Groq call).
- `teams.py`: 30-team Pythagorean deviation bar chart + sortable stats table + regression signal cards.
- `pitchers.py`: FIP vs ERA scatter plot (diagonal = ERA equals FIP). Filterable stats table. Active signal cards.
- `props.py`: Pitcher K prop projections with manual line input → edge calculation → AI analysis. Batter BABIP regression table with histogram.
- `preseason.py`: Prior year stats → projected wins → Vegas line comparison chart + table. AI reasoning per team on demand.
- `bet_log.py`: Log bets manually, record outcomes, compute P&L and ROI automatically.
- `basketball.py`: Stub with planned Sprint 5 features described.
- `football.py`: Stub with planned Sprint 7 features described.

**Navigation:**
- Sidebar sport selector: Baseball / Basketball / Football
- Baseball sub-pages: Games, Teams, Pitchers, Player Props, Preseason Projections, Bet Log
- Kelly calculator in sidebar (baseball only)

## Notes
- Run: `streamlit run src/dashboard/app.py`
- AI reasoning is on-demand (button) to preserve Groq free tier quota
- Edge table: green = positive edge, red = negative
- Signal severity: High (red), Medium (orange), Low (yellow)
- Groq responses are structured JSON: { reasoning, confidence, key_risk }
