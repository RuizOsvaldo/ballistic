# Sprint 5 — NBA Data, Models, and Agent

**Goal:** Full NBA analytical layer — team models, player prop models, Groq agent integration, and data ingestion from nba_api.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| NBA-001 | requirements.txt — add nba_api | [x] |
| NBA-002 | src/sports/basketball/data/nba_stats.py — team stats, player stats, schedule | [x] |
| NBA-003 | src/sports/basketball/models/net_rating.py — net rating deviation, win probability | [x] |
| NBA-004 | src/sports/basketball/models/four_factors.py — four factors matchup and totals model | [x] |
| NBA-005 | src/sports/basketball/models/rest_schedule.py — back-to-back and rest adjustment | [x] |
| NBA-006 | src/sports/basketball/models/player_props.py — pts/reb/ast/PRA/3PM projections | [x] |
| NBA-007 | src/sports/basketball/models/win_probability.py — composite NBA win probability | [x] |
| NBA-008 | src/sports/basketball/agent/groq_prompts.py — NBA-specific Groq prompt builders | [x] |
| NBA-009 | tests/test_nba_net_rating.py | [x] |
| NBA-010 | tests/test_nba_player_props.py | [x] |
| NBA-011 | Docker container (Dockerfile, docker-compose.yml, .dockerignore) | [x] |

## Sprint 5 Complete

**Data layer:**
- `nba_stats.py`: nba_api wrappers for team stats (net rating, four factors, pace, W-L), player stats (pts/reb/ast/usg/3PM rolling), and today's schedule. All calls go through `cached()`.

**Models:**
- `net_rating.py`: Net rating deviation from implied W%, regression signal (±5 net rating points = flag)
- `four_factors.py`: Per-game four factors matchup, pace-adjusted total projection
- `rest_schedule.py`: Back-to-back detection, rest days calculation, point adjustment per situation
- `player_props.py`: Points (usage + pace + matchup), rebounds (rate-based + matchup), assists, PRA, 3PM
- `win_probability.py`: Net rating base + rest adjustment + home court — outputs home_win_prob

**Agent:**
- `groq_prompts.py`: NBA game analysis prompt, NBA prop analysis prompt

**Tests:**
- `test_nba_net_rating.py`: win probability bounds, regression signal thresholds
- `test_nba_player_props.py`: projection direction tests, prop edge calculation

## Notes
- nba_api rate limit: 600 req/min — add 0.6s sleep between calls to be safe
- Net rating → win probability: logistic transform, each point ≈ 2.7% win prob
- Home court advantage: +3.0 net rating points
- Back-to-back second night: -2.5 net rating adjustment
- Player prop regression: 3P% regressed 30% toward career avg; usage rate treated as stable
- NBA season: October through April (playoffs through June)
