# Sprint 1 — Foundation

**Goal:** Repo scaffolded, data ingestion working, docs written.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| MLB-001 | Repo scaffold | [x] |
| MLB-002 | pybaseball ingestion | [x] |
| MLB-003 | Odds API client | [x] |
| MLB-004 | Data cache layer | [x] |
| MLB-005 | Write OVERVIEW.md + ARCHITECTURE.md | [x] |

## Sprint 1 Complete

All foundation pieces are in place:
- Full directory structure scaffolded
- `src/data/cache.py` provides a disk-backed parquet cache with configurable TTL
- `src/data/baseball_stats.py` pulls team and pitcher stats via pybaseball
- `src/data/odds.py` fetches MLB moneylines from The Odds API and computes implied probabilities
- `docs/OVERVIEW.md` and `docs/ARCHITECTURE.md` written
