# Ballistic — Master Overview

## Purpose

Ballistic is a multi-sport betting analytics dashboard that applies quantitative, data-driven methods to identify market inefficiencies in sports betting. The core philosophy is borrowed from professional trading: find games where your model's estimated win probability diverges meaningfully from the implied probability embedded in the betting market, then size bets accordingly.

The approach is not about picking winners. It is about finding **positive expected value (+EV)** — situations where the market is wrong often enough, and by enough, that betting systematically produces profit over a large sample.

---

## Sports Covered

| Sport | Status | Primary Bet Types |
|---|---|---|
| Baseball (MLB) | Active — Sprints 1-4 | Moneylines, totals, run lines, player props, preseason win totals |
| Basketball (NBA) | Planned — Sprints 5-6 | Spreads, moneylines, totals, player props (PRA, pts, reb, ast) |
| Football (NFL) | Planned — Sprints 7-8 | Spreads, moneylines, totals, player props (pass yds, rush yds, TDs) |

Each sport has its own analytical framework, stat models, data sources, and documentation. They share a common dashboard shell, odds client, bet sizing engine, and AI reasoning agent.

---

## Core Philosophy

### 1. True Probability vs. Implied Probability

Every betting line contains an implied probability. A moneyline of -150 implies the favorite wins 60% of the time. If your model says they win 67% of the time, you have a 7% edge. Bet consistently with edges like that and you profit over time regardless of individual outcomes.

### 2. Regression to the Mean

Teams and players performing significantly above or below their underlying statistical true talent will move back toward that talent level over time. Identifying where luck is inflating or deflating performance is the primary source of edge across all three sports.

### 3. Defense-Independent, Context-Adjusted Stats

Raw stats like ERA, points per game, and passing yards are polluted by luck, defense, park effects, and pace. This system uses context-adjusted, defense-independent metrics wherever possible to isolate true talent from noise.

### 4. Player Props as a Primary Market

Player props are often the most mispriced lines in sports betting. Sportsbooks set prop lines quickly and move them based on betting volume rather than deep analysis. This creates consistent +EV opportunities when a player's underlying performance metrics diverge from what their prop line implies. Each sport module includes a dedicated prop analysis layer.

### 5. Kelly Criterion Bet Sizing

The Kelly Criterion calculates the mathematically optimal fraction of bankroll to wager given your edge and the odds. The system uses half-Kelly by default to reduce variance.

### 6. AI-Assisted Reasoning

After the quantitative models compute all signals, a Groq-hosted Llama 3.3 70B agent synthesizes those signals into a human-readable recommendation with a confidence score and plain-language reasoning. The models make the call — the agent explains why.

---

## System Architecture

```
Data Layer              Model Layer                  Agent Layer        Dashboard
----------              -----------                  -----------        ---------
pybaseball    -->   Pythagorean W%                                  --> Edge Table
The Odds API  -->   FIP / ERA Gap           -->   Groq Agent        --> Signal Cards
NBA API (TBD) -->   BABIP / xFIP / SIERA   -->   Llama 3.3 70B     --> Bet Recs
NFL API (TBD) -->   Win Probability         -->   Per-game          --> Player Props
              -->   Player Prop Models      -->   reasoning +       --> Charts
              -->   Kelly Criterion         -->   confidence score  --> Filters
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Dashboard | Streamlit | Interactive, filterable, chart-native |
| Data (Baseball) | pybaseball | FanGraphs, Baseball Savant, BBRef |
| Odds + Props | The Odds API (free tier) | All sports, live lines across books |
| AI Agent | Groq API — Llama 3.3 70B | Free tier, 6,000 req/day |
| Data (Basketball) | NBA API | Planned, Sprint 5 |
| Data (Football) | ESPN unofficial API | Planned, Sprint 7 |
| Language | Python 3.11+ | |
| Caching | Local parquet cache | Preserves API quotas |
| Secrets | python-dotenv + .env | Never committed to git |
| Secret scanning | detect-secrets | Pre-commit hook |

---

## Project Structure

```
MLB_Analysis/
├── docs/
│   ├── OVERVIEW.md              <- this file
│   ├── ARCHITECTURE.md          <- system design, data flow, module map
│   ├── SECURITY.md              <- secrets model and practices
│   ├── baseball/
│   │   ├── OVERVIEW.md          <- baseball logic, models, props approach
│   │   └── APPROACH.md          <- model-by-model reasoning detail
│   ├── basketball/
│   │   └── OVERVIEW.md          <- basketball logic, models, props approach
│   └── football/
│       └── OVERVIEW.md          <- football logic, models, props approach
├── sprints/
│   ├── sprint-01.md             <- complete
│   ├── sprint-02.md             <- models + Groq agent + props
│   ├── sprint-03.md             <- baseball dashboard
│   └── sprint-04.md             <- polish + multi-sport stubs
├── src/
│   ├── sports/
│   │   ├── baseball/
│   │   │   ├── data/
│   │   │   ├── models/
│   │   │   └── agent/
│   │   ├── basketball/          <- stub, Sprint 5
│   │   └── football/            <- stub, Sprint 7
│   ├── shared/
│   │   ├── odds/
│   │   ├── kelly.py
│   │   └── groq_agent.py
│   └── dashboard/
│       ├── app.py
│       ├── pages/
│       │   ├── baseball/
│       │   ├── basketball/      <- stub
│       │   └── football/        <- stub
│       └── components/
├── tests/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dashboard Pages Per Sport

| Page | Description |
|---|---|
| Today's Games | Upcoming games with edge %, signal flags, AI recommendation, Kelly stake |
| Teams | True-talent metrics vs. actual record, regression flags, trend charts |
| Players | Key stat leaders with context-adjusted metrics |
| Player Props | Prop lines vs. model projections, best value props ranked by edge % |
| Best Bets | Top-ranked +EV opportunities across all bet types |
| Preseason Projections | Season win total projections vs. Vegas O/U lines (MLB + NFL) |
| Bet History | Past recommendations with outcomes for tracking model accuracy |

---

## Every Bet Recommendation Includes

1. The bet — team or player, line type, odds, sportsbook
2. Model edge — estimated probability vs. implied probability
3. Kelly stake — recommended bet size as % of bankroll
4. Signal summary — which models flagged this and why
5. AI reasoning — Groq agent explanation in plain language
6. Confidence level — Low / Medium / High based on signal convergence

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ODDS_API_KEY` | Yes | The Odds API — the-odds-api.com |
| `GROQ_API_KEY` | Yes | Groq API — console.groq.com |
| `CACHE_TTL_HOURS` | No (default 24) | Hours before cached data expires |
| `CACHE_DIR` | No (default data/cache) | Local cache directory |

---

## Sprint Roadmap

| Sprint | Focus | Status |
|---|---|---|
| 1 | Foundation: structure, security, data ingestion, odds client | Complete |
| 2 | Baseball models, player props model, Groq agent, preseason projections | In progress |
| 3 | Baseball dashboard: all pages including props and AI recommendations | Pending |
| 4 | Polish: history tracker, tests, final docs, sport stubs | Pending |
| 5-6 | Basketball — separate planning session | Planned |
| 7-8 | Football — separate planning session | Planned |
