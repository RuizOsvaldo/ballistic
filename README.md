# Ballistic — Multi-Sport Betting Analytics Dashboard

A quantitative sports betting dashboard applying Joe Peta's *Trading Bases* methodology and advanced analytics to identify market inefficiencies across MLB, NBA, and NFL. Surfaces edge by comparing model-derived probabilities against Vegas implied odds, with AI-powered reasoning from Llama 3.3 70B via Groq.

---

## Sports Coverage

| Sport | Status | Features |
|---|---|---|
| Baseball (MLB) | Active | Games, Teams, Pitchers, Player Props, Preseason Projections, Bet Log |
| Basketball (NBA) | Planned — Sprint 5 | See docs/basketball/OVERVIEW.md |
| Football (NFL) | Planned — Sprint 7 | See docs/football/OVERVIEW.md |

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd MLB_Analysis
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your real keys (never commit this file):

```
ODDS_API_KEY=your_key_here     # the-odds-api.com — free tier
GROQ_API_KEY=your_key_here     # console.groq.com — free tier
```

### 3. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Security setup (recommended)

```bash
detect-secrets scan > .secrets.baseline
pre-commit install
```

This installs a pre-commit hook that blocks accidental API key commits.

---

## API Keys

| Key | Where to get it | Free tier |
|---|---|---|
| `ODDS_API_KEY` | the-odds-api.com | 500 requests/month |
| `GROQ_API_KEY` | console.groq.com | 6,000 requests/day |

---

## Features

### Baseball

**Games page**
- Today's matchups ranked by edge %
- Filters: team, minimum edge, BET only
- Each recommendation shows model prob, market prob, edge %, Kelly stake
- On-demand AI reasoning button (Llama 3.3 70B explains the edge in plain language)

**Teams page**
- Pythagorean deviation bar chart for all 30 teams
- Sortable stats table (W-L, Pythagorean W%, deviation, run differential)
- Regression signal cards (High/Medium/Low severity)

**Pitchers page**
- FIP vs ERA scatter plot — points above the diagonal are regression risks
- Filterable table by team and signal severity
- Active signal cards with direction indicators

**Player Props page**
- Pitcher strikeout projections vs. manual sportsbook lines
- Batter hit projections using BABIP regression
- Edge calculation with AI analysis for top props

**Preseason Projections page**
- Enter each team's Vegas win total O/U
- Model projects wins from prior Pythagorean W% + regression to mean
- Bar chart showing projection vs. Vegas line gap
- AI reasoning for each OVER/UNDER bet recommendation

**Bet Log**
- Manually log bets with line, stake, and edge at time of bet
- Record outcomes (Win/Loss/Push/Pending)
- Automatic P&L calculation and ROI summary

---

## Key Signals (Trading Bases methodology)

| Signal | Interpretation | Bet Use |
|---|---|---|
| Pythagorean deviation > +5% | Team overperforming run differential — luck, not skill | Fade on moneyline |
| Pythagorean deviation < -5% | Team underperforming — due for improvement | Back on moneyline |
| FIP - ERA > 0.75 | Pitcher ERA artificially low — will rise | Fade team/back opponent |
| BABIP > .320 (pitcher) | Pitcher getting lucky on contact | ERA will rise |
| BABIP < .275 (pitcher) | Pitcher being unlucky | ERA will fall |
| Edge % > 3% | Model win prob > market implied prob | BET at half-Kelly stake |
| Preseason edge > 2 wins | Model diverges from Vegas win total by 2+ wins | Season win total bet |

---

## Project Structure

```
MLB_Analysis/
├── docs/
│   ├── OVERVIEW.md              # Master app overview
│   ├── ARCHITECTURE.md          # System design and data flow
│   ├── SECURITY.md              # Secrets model and practices
│   ├── baseball/OVERVIEW.md     # Full baseball analysis logic
│   ├── basketball/OVERVIEW.md   # NBA framework (Sprint 5)
│   └── football/OVERVIEW.md     # NFL framework (Sprint 7)
├── sprints/
│   └── sprint-01 to 04          # CLASI sprint tracking
├── src/
│   ├── data/
│   │   ├── cache.py             # Parquet cache with TTL
│   │   ├── baseball_stats.py    # pybaseball: team, pitcher, batter stats
│   │   └── odds.py              # The Odds API: moneylines, totals, props
│   ├── models/
│   │   ├── pythagorean.py       # Pythagorean win expectation
│   │   ├── regression_signals.py# BABIP + FIP-ERA signal detection
│   │   ├── win_probability.py   # Composite game win probability
│   │   ├── kelly.py             # Half-Kelly bet sizing
│   │   ├── player_props.py      # Pitcher K and batter hit projections
│   │   └── preseason.py         # Season win total projections
│   ├── shared/
│   │   └── groq_agent.py        # Llama 3.3 70B reasoning agent
│   ├── sports/
│   │   ├── basketball/          # NBA stub — Sprint 5
│   │   └── football/            # NFL stub — Sprint 7
│   └── dashboard/
│       ├── app.py               # Streamlit entrypoint (multi-sport)
│       ├── pages/               # games, teams, pitchers, props, preseason, bet_log
│       │                        # basketball (stub), football (stub)
│       └── components/          # edge_table, signal_cards
├── tests/                       # pytest unit tests for all models
├── .env                         # secrets — never committed
├── .env.example                 # template — safe to commit
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: Pythagorean model, Kelly criterion, regression signals, win probability, player props, and preseason projections.

---

## Data Sources

| Source | Access | Data |
|---|---|---|
| FanGraphs | pybaseball | FIP, xFIP, SIERA, wRC+, BABIP, K%, BB%, barrel%, whiff% |
| Baseball Savant | pybaseball (Statcast) | Exit velocity, barrel%, sprint speed |
| Baseball Reference | pybaseball | W-L records, standings, run differential |
| The Odds API | REST (free tier) | Moneylines, totals, player props |
| Groq API | REST (free tier) | Llama 3.3 70B reasoning |

Stats cached to `data/cache/` as parquet files (24-hour TTL).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ODDS_API_KEY` | required | The Odds API key |
| `GROQ_API_KEY` | required | Groq API key |
| `CACHE_TTL_HOURS` | `24` | Hours before cached data expires |
| `CACHE_DIR` | `data/cache` | Local cache directory |
