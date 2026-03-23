# Ballistic — Architecture

## Data Flow

```
pybaseball (FanGraphs/BBRef)          The Odds API
        |                                    |
        v                                    v
 baseball_stats.py                       odds.py
 (team, pitcher, batter stats)     (moneylines, totals, props)
        |                                    |
        +-----------> cache.py <------------+
                           |
          +----------------+-----------------+
          |                |                 |
   pythagorean.py   regression_signals.py   player_props.py
          |                |                 |
          +----------------+-----------------+
                           |
                   win_probability.py
                           |
                       kelly.py
                           |
                   groq_agent.py  <-- Llama 3.3 70B via Groq API
                           |
          +----------------+-------------------+
          |                |                   |
      games.py         teams.py           pitchers.py
      props.py       preseason.py          bet_log.py
          |                |                   |
          +----------------+-------------------+
                           |
                        app.py  (Streamlit)
                           |
                         User
```

---

## Module Responsibilities

### `src/data/`

| Module | Responsibility |
|---|---|
| `cache.py` | Disk-backed parquet cache with configurable TTL. All expensive I/O goes through `cached()`. |
| `baseball_stats.py` | pybaseball wrappers: team stats (batting, pitching, records), pitcher stats (FIP, xFIP, SIERA, BABIP, K%, whiff%), batter stats (wRC+, BABIP, barrel%, exit velocity) |
| `odds.py` | Odds API client: moneylines (h2h), game totals, player props. Sport-agnostic helpers for MLB now; NBA/NFL planned. |

### `src/models/`

| Module | Responsibility |
|---|---|
| `pythagorean.py` | `compute_pythagorean(df)` — adds pyth_win_pct, pyth_deviation, pyth_signal |
| `regression_signals.py` | `compute_pitcher_signals(df)` and `compute_team_signals(df)` — FIP-ERA gap, BABIP deviation severity |
| `win_probability.py` | `compute_win_probabilities(games, teams, pitchers)` — Pythagorean base + FIP adj + home field |
| `kelly.py` | `compute_kelly(model_prob, odds)` and `compute_kelly_for_games(df)` — half-Kelly bet sizing |
| `player_props.py` | Pitcher K projections, batter hit/total base/HR projections, prop edge calculation |
| `preseason.py` | `compute_preseason_projections(prior_stats, vegas_lines)` — Peta methodology win total projections |

### `src/shared/`

| Module | Responsibility |
|---|---|
| `groq_agent.py` | Groq API client (Llama 3.3 70B). Three entry points: `analyze_mlb_game()`, `analyze_mlb_prop()`, `analyze_preseason_projection()`. Returns JSON with reasoning, confidence, key_risk. |

### `src/sports/` (stubs)

| Module | Responsibility |
|---|---|
| `baseball/` | Placeholder for baseball-specific models as they grow beyond shared `src/models/` |
| `basketball/` | NBA module — Sprint 5. Net Rating, Four Factors, pace, player props. |
| `football/` | NFL module — Sprint 7. EPA/play, DVOA, weather model, player props. |

### `src/dashboard/`

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit entry point. Sport selector (Baseball/Basketball/Football), Kelly sidebar, data loading with `@st.cache_data`. |
| `pages/games.py` | Today's games: edge table, filters, bet recommendation cards with AI reasoning button. |
| `pages/teams.py` | All 30 teams: Pythagorean deviation chart, stats table, regression signal cards. |
| `pages/pitchers.py` | Qualified starters: FIP vs ERA scatter, BABIP table, signal cards. |
| `pages/props.py` | Pitcher K props and batter hit props. Manual line input → edge calculation → AI analysis. |
| `pages/preseason.py` | Preseason win total projections vs. Vegas lines. Chart, table, AI reasoning per team. |
| `pages/bet_log.py` | CSV-backed bet tracker: log bets, record outcomes, view P&L and ROI. |
| `pages/basketball.py` | NBA stub — shows planned features for Sprint 5. |
| `pages/football.py` | NFL stub — shows planned features for Sprint 7. |
| `components/edge_table.py` | Styled DataFrame with green/red conditional formatting on edge %. |
| `components/signal_cards.py` | HTML signal cards with severity color coding and direction arrows. |

---

## Win Probability Model

```
base_prob          = team_pythagorean_win_pct
pitcher_adj        = (league_avg_fip - starter_fip) * 0.03   # ~3% per FIP point
home_field_adj     = +0.04 if home else 0.0
win_prob           = clamp(base_prob + pitcher_adj + home_field_adj, 0.05, 0.95)
```

League average FIP is computed dynamically as IP-weighted mean of all qualified starters.

---

## Edge & Kelly Calculation

```
implied_prob       = vig-removed probability from odds.py
edge               = win_prob - implied_prob
# Bets flagged when edge > 3% (configurable MIN_EDGE in kelly.py)

kelly_fraction     = (b * p - q) / b    # b = decimal odds - 1
stake_pct          = kelly_fraction * 0.5  # half-Kelly
dollar_stake       = bankroll * stake_pct
```

---

## Preseason Win Projection Model

```
pyth_pct           = RS² / (RS² + RA²)  [prior season]
regressed_pct      = pyth_pct * 0.80 + 0.500 * 0.20  [regression to mean]
war_win_adj        = net_war_change      [roster moves]
projected_wins     = regressed_pct * 162 + war_win_adj
edge_wins          = projected_wins - vegas_line
# Flag as bet when abs(edge_wins) >= 2.0 wins
```

---

## Groq Agent

The agent is invoked on-demand (button click) to avoid burning API quota on every page load.

```
Input:  statistical signals dict + odds context
Model:  llama-3.3-70b-versatile (Groq)
Output: JSON { reasoning, confidence, key_risk }
```

Three specialized functions: game analysis, player prop analysis, preseason projection analysis.
Each uses a tailored system prompt focused on that bet type.

---

## Caching Strategy

| Data | TTL | Notes |
|---|---|---|
| Team stats (pybaseball) | 24 hours | Refreshed once per day at first request |
| Pitcher stats (pybaseball) | 24 hours | Same |
| Batter stats (pybaseball) | 24 hours | Same |
| Odds (moneyline) | 1 hour | More frequent — lines move |
| Player props | 1 hour | Per game_id, sparse refresh |
| Streamlit layer | 6 hours (stats), 1 hour (odds) | Additional Streamlit `@st.cache_data` on top |

Cache files stored at `data/cache/*.parquet`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ODDS_API_KEY` | (required) | API key from the-odds-api.com |
| `GROQ_API_KEY` | (required) | API key from console.groq.com |
| `CACHE_TTL_HOURS` | `24` | Hours before cached data expires |
| `CACHE_DIR` | `data/cache` | Directory for parquet cache files |

---

## Container Architecture

The app ships as a single Docker container. Secrets are injected at runtime, never baked into the image.

```
Host machine
├── .env                    (secrets — never in image)
├── docker-compose.yml
│
└── Docker container
    ├── /app/               (source code)
    ├── /app/data/cache/    (volume-mounted — cache persists across restarts)
    │
    └── Streamlit :8501
```

### Build and run

```bash
# Development (local Python)
streamlit run src/dashboard/app.py

# Container (single container)
docker build -t sports-edge .
docker run -p 8501:8501 --env-file .env sports-edge

# Container (compose — recommended, handles volume + restart)
docker compose up --build
```

### Environment injection

Secrets are passed via `--env-file .env` or `docker compose` `env_file` directive. The image contains zero secrets. The `.dockerignore` explicitly excludes `.env` from the build context.

### Volume

`cache_data` volume mounts to `/app/data/cache` inside the container. This means parsed pybaseball and odds data persists between container restarts — the 24-hour TTL cache works correctly across `docker compose restart`.

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | >=1.35 | Dashboard framework |
| `pybaseball` | >=2.2.7 | Baseball stats (FanGraphs, BBRef) |
| `pandas` | >=2.2 | Data manipulation |
| `plotly` | >=5.20 | Interactive charts |
| `groq` | >=0.9 | Groq API client for Llama 3.3 70B |
| `requests` | >=2.31 | Odds API HTTP client |
| `python-dotenv` | >=1.0 | Environment variable loading |
| `pyarrow` | >=15.0 | Parquet read/write for cache |
| `detect-secrets` | >=1.4 | Pre-commit secret scanning |
