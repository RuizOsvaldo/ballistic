# Ballistic -- CLAUDE.md

This file is the single source of truth for Claude Code sessions on this project.
Read it fully before writing any code, creating any file, or making any plan.

---

## Project Identity

**Ballistic** is a quantitative sports betting analytics dashboard built in Python and Streamlit.
It applies Joe Peta's *Trading Bases* methodology and advanced analytics to identify market
inefficiencies across MLB, NBA, and NFL.

**Active sport: MLB only.**
NBA and NFL work is planned for a future phase but is not in scope for any session until
explicitly authorized. Do not touch any file under `src/sports/basketball/` or
`src/sports/football/`, or `src/dashboard/pages/basketball.py` or
`src/dashboard/pages/football.py`, unless the user specifically says so.

---

## Stack

| Layer | Tool |
|---|---|
| Dashboard | Streamlit |
| Language | Python 3.11+ |
| Team stats (batting/pitching/records) | MLB Stats API (free, no key) — FanGraphs returns 403 for team endpoints |
| Individual pitcher/batter stats | pybaseball (FanGraphs leaderboards) |
| MLB schedule / starters / lineups / live | MLB Stats API (free, no key) |
| Odds / props | The Odds API (free tier, 500 req/month) |
| AI reasoning | Groq API -- Llama 3.3 70B |
| Caching | Local parquet cache with TTL (`src/data/cache.py`) |
| Bet tracking | SQLite (`data/bet_log.db`) |
| Secrets | python-dotenv + `.env` (never committed) |
| Secret scanning | detect-secrets + pre-commit hook |
| Container | Docker + docker-compose |
| Tests | pytest (`tests/`) |

---

## Directory Map

```
ballistic/
├── docs/
│   ├── OVERVIEW.md              -- master app overview
│   ├── ARCHITECTURE.md          -- system design and data flow
│   ├── SECURITY.md              -- secrets model
│   ├── baseball/
│   │   ├── OVERVIEW.md          -- full baseball analysis logic
│   │   └── SPECS.md             -- formula-level technical specs
│   ├── basketball/OVERVIEW.md   -- future scope, do not touch
│   └── football/OVERVIEW.md     -- future scope, do not touch
├── docs/sprints/                -- sprint files live here
│   └── sprint-NN.md
├── src/
│   ├── data/
│   │   ├── cache.py             -- parquet TTL cache, all I/O goes through cached()
│   │   ├── baseball_stats.py    -- team stats via MLB Stats API; pitcher/batter via pybaseball
│   │   │                           get_team_batting/pitching/records → MLB Stats API (team_id merge)
│   │   │                           get_pitcher_stats / get_batter_stats → pybaseball leaderboards
│   │   ├── odds.py              -- The Odds API client (moneylines, totals, props)
│   │   ├── game_results.py      -- MLB scores, starters, lineups, live games
│   │   ├── ballpark.py          -- park factors (static dict) + bullpen stats via pybaseball
│   │   ├── bet_log_db.py        -- SQLite bet log (singles + parlays)
│   │   └── predictions_db.py    -- SQLite prediction tracking
│   ├── models/
│   │   ├── pythagorean.py       -- Pythagorean win expectation
│   │   ├── regression_signals.py-- FIP-ERA gap, BABIP signal detection
│   │   ├── win_probability.py   -- composite game win probability
│   │   ├── kelly.py             -- half-Kelly bet sizing
│   │   ├── player_props.py      -- pitcher K and batter hit projections
│   │   ├── preseason.py         -- season win total projections
│   │   └── calibration.py       -- model accuracy and ROI analytics
│   ├── shared/
│   │   └── groq_agent.py        -- Groq Llama 3.3 70B reasoning agent
│   ├── sports/
│   │   ├── basketball/          -- DO NOT TOUCH (needs full audit)
│   │   └── football/            -- DO NOT TOUCH (needs full audit)
│   ├── dashboard/
│   │   ├── app.py               -- Streamlit entrypoint, navigation (3 top-level pages)
│   │   │                           Home (Baseball/Basketball/Football sport selector)
│   │   │                           Bet Log, Analysis
│   │   ├── pages/               -- Baseball sections (selected via sidebar dropdown):
│   │   │                           games, teams, pitchers, props, preseason,
│   │   │                           game_analysis, player_analysis
│   │   │                           basketball (placeholder), football (placeholder)
│   │   │                           bet_log, analysis
│   │   └── components/          -- edge_table, signal_cards
│   └── jobs/
│       ├── daily_predictions.py -- 8am email job (launchd)
│       └── verify_results.py    -- midnight result verification job
├── tests/                       -- pytest unit tests, all models covered
├── scripts/
│   ├── run_daily.sh             -- cron/launchd wrapper for daily job
│   └── verify_results.sh        -- cron/launchd wrapper for verification
├── data/
│   ├── cache/                   -- parquet cache files (gitignored)
│   ├── bet_log.db               -- SQLite (gitignored)
│   └── logs/                    -- daily job logs
├── .env                         -- secrets (gitignored, never touch)
├── .env.example                 -- safe template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Active Sprint State

Sprints 1-11 are complete and documented in `docs/sprints/`.
Sprint 9 key items: FanGraphs 403 fix (MLB Stats API), Game Analysis page, Player Analysis
page, bet slip, bullpen + park factors wired into win probability, live game feed.
Sprint 10 key items: Poisson run-line and game-total edge (±1.5 RL, O/U).
Sprint 11 key items: Log5 head-to-head probability, RS/RA regression to mean (20-game
threshold, automatic seasonal switch), lineup quality matchup FIP adjustment, vig-free
implied probability, Pythagorean exponent 1.83, in-app formula state banner.

Sprint files live at `docs/sprints/sprint-NN.md` using zero-padded numbering.

---

## Core Model Logic (MLB)

Read `docs/baseball/SPECS.md` for full formula-level detail before touching any model file.
The short version:

**Win probability pipeline:**
```
Step 1:  Pythagorean W% for each team (RS^1.83 / (RS^1.83 + RA^1.83))
         -- if team has >= 20 games: RS/RA blended with league mean
            weight = G / (G + 30); before 20 games: raw RS/RA used

Step 2:  Log5 head-to-head prob  P = (A - A*B) / (A + B - 2*A*B)
         -- gives true matchup probability, not just normalize(A+B)

Step 3:  Starter FIP adjustment
         home_fip_adj = (league_avg_fip - home_starter_fip) * 0.03
         -- effective_fip adjusted for opposing lineup OPS vs league avg (0.720)
            lineup_adj = (lineup_avg_ops - 0.720) * 3.0

Step 4:  Home field +0.04

Step 5:  Bullpen FIP adjustment  (opp_bullpen_fip - league_avg) * 0.33 * 0.30

Step 6:  Renormalize + clamp [0.30, 0.70]
```

**Edge and Kelly:**
```
edge = model_prob - vig_free_implied_prob
     vig_free: raw / (raw_home + raw_away)  -- vig removed before edge calc
flag when edge >= 3%
kelly_stake = (b*p - q) / b * 0.5   (half-Kelly, capped at 5% of bankroll)
```

**Caching TTLs:**
- Odds: 2 hours
- Team / pitcher / batter stats: 6 hours
- Bullpen stats: 6 hours
- Starters / lineups: 1 hour
- Completed game results: 24 hours (past dates only)

---

## Code Standards

These are non-negotiable. Every ticket must pass these before being declared done.

- One function = one responsibility. No multi-purpose functions.
- No fallback paths. One correct way to do each thing.
- Fail fast: raise explicit errors when preconditions are not met. Do not silently recover.
- No dead code. No commented-out blocks left behind.
- Variable and function names must be self-explanatory.
- Every non-obvious function gets a one-line docstring explaining its purpose.
- All expensive I/O goes through `cached()` in `src/data/cache.py`. Never call pybaseball,
  the MLB API, or The Odds API directly without caching.
- Secrets are loaded via `os.getenv()` and `load_dotenv()`. Never hardcode keys.
- Tests must be added or updated for any change to a model file under `src/models/`.

---

## Engineering Process -- Four Phases

Every coding task in this repo follows this process. No exceptions.

---

### Phase 1: Requirements Gathering

Before writing any code or plan, ask ALL questions below in one batch.
Do not proceed to Phase 2 until all are answered.

**For any task:**
1. What is the exact behavior being added, changed, or fixed?
2. Which files are in scope? (If not obvious, ask before assuming.)
3. What is the expected input and output?
4. What does "done" look like? What are the acceptance criteria?
5. Are there any constraints? (API quota, cache TTL, UI layout, performance, etc.)
6. Are there any known edge cases or failure scenarios?

**For bug fixes, also ask:**
- What exact behavior are you seeing?
- What behavior did you expect?
- When did it start? Did anything change before it broke?
- Can you share the full error message or log output?

**For new features, also ask:**
- Does this touch any existing cached data keys? If so, does the cache need to be cleared?
- Does this require a new sprint, or does it belong in the current sprint?
- Does this need a new test, or does an existing test need to be updated?

---

### Phase 2: Sprint Planning

Once all questions are answered, produce or update a sprint file before writing any code.

**Sprint file location:** `docs/sprints/sprint-NN.md`

**Sprint file format:**
```
# Sprint NN -- [Short Title]

## Goal
One sentence describing what this sprint delivers.

## Context
Brief summary of requirements gathered in Phase 1.

## Architecture Notes
Key technical decisions and why they were made.
Include at least one alternative that was considered and ruled out, with the reason.

## Tickets

### Ticket NN-01: [Title]
**Type:** Feature | Bug | Refactor | Chore
**Files in scope:** list the files
**Description:** What needs to be done and why.
**Acceptance Criteria:**
- [ ] Specific, verifiable condition 1
- [ ] Specific, verifiable condition 2
**Edge Cases:**
- What happens if X
- What happens if Y

### Ticket NN-02: [Title]
...
```

**Rules:**
- One ticket = one concern. Do not combine unrelated work.
- Every ticket needs at least two acceptance criteria.
- Every ticket needs at least one edge case documented.
- Show the sprint plan to the user and get confirmation before executing.

---

### Phase 3: Execution

Execute tickets in order. Do not skip. Do not combine.

For each ticket:
1. State which ticket you are starting.
2. Write code for that ticket only.
3. Run Phase 4 verification on that ticket before moving to the next.

**File creation rules:**
- New data fetching functions go in the appropriate `src/data/` module.
- New model logic goes in `src/models/`.
- New dashboard pages go in `src/dashboard/pages/`.
- New reusable UI components go in `src/dashboard/components/`.
- Sprint documentation goes in `docs/sprints/`.
- Tests go in `tests/` and must match the module they test.

---

### Phase 4: Verification

Run this before declaring any ticket done.

**Logic audit:**
- [ ] Does the code meet every acceptance criterion listed in the ticket?
- [ ] Does the code handle every edge case listed in the ticket?
- [ ] Are there any new edge cases that emerged during implementation?

**Code quality audit:**
- [ ] Does each function have a single responsibility?
- [ ] Are errors raised explicitly when preconditions are not met?
- [ ] Is there any dead code or unused imports?
- [ ] Are all expensive I/O calls going through `cached()`?
- [ ] Are any secrets hardcoded? (Must be zero.)

**Integration audit:**
- [ ] Does this ticket's output connect correctly to the previous ticket's output?
- [ ] If a model file changed, is there a corresponding test update?
- [ ] If a cache key changed, is the TTL correct and documented?

Only after every item passes: state "Ticket NN-XX is complete." Then move to the next.

If any item fails, fix it now. Do not note it as a future improvement.

---

### Mid-Task Interruption Protocol

If a new requirement emerges during execution:
1. Stop immediately.
2. State: "A new requirement has emerged: [describe it]."
3. Ask the user to clarify it.
4. If it fits in the current sprint, add a ticket and continue.
5. If it is out of scope, document it as a future sprint and continue with the current plan.

Never silently absorb new requirements.

---

## Anti-Patterns -- Never Do These

| Anti-Pattern | Why It Fails |
|---|---|
| Calling pybaseball or any API without `cached()` | Burns quota, breaks rate limits |
| Touching basketball/ or football/ files mid-MLB work | Scope creep, untested breakage |
| Combining multiple concerns in one ticket | Makes verification impossible |
| Declaring a ticket done without checking acceptance criteria | Passes the problem downstream |
| Making assumptions instead of asking | Silent mismatch between intent and code |
| Adding try/except that silently swallows errors | Hides bugs, makes debugging impossible |
| Using pybaseball for team batting/pitching/records | FanGraphs returns 403; use MLB Stats API instead |
| Hardcoding team names instead of using the mapping dicts | Breaks matching between data sources |
| Merging team stats DataFrames on name strings | Use team_id (int) — MLB API standings returns short names, stats returns full names |
| Skipping tests when changing model files | Breaks calibration tracking silently |

---

## Environment Setup Reference

```bash
# Local development
cp .env.example .env      # fill in ODDS_API_KEY and GROQ_API_KEY
pip install -r requirements.txt
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v

# Daily job (manual trigger)
venv/bin/python3 -m src.jobs.daily_predictions

# Docker
docker compose up --build
```

Required `.env` keys:
- `ODDS_API_KEY` -- the-odds-api.com (500 req/month free)
- `GROQ_API_KEY` -- console.groq.com (6,000 req/day free)
- `EMAIL_SENDER` -- Gmail address for daily job
- `EMAIL_APP_PASSWORD` -- Gmail App Password (not regular password)
- `EMAIL_RECIPIENT` -- delivery address for daily report
