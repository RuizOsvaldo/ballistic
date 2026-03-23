# Sprint 4 — Polish, Security, and Tests

**Goal:** Security hardened, tests passing, README complete, multi-sport stubs documented.

## Tickets

| ID | Title | Status |
|----|-------|--------|
| MLB-024 | .gitignore — comprehensive coverage | [x] |
| MLB-025 | SECURITY.md — secrets model and practices | [x] |
| MLB-026 | requirements.txt updated (groq, detect-secrets, pre-commit) | [x] |
| MLB-027 | .env.example — placeholder values only, no real keys | [x] |
| MLB-028 | ARCHITECTURE.md updated for multi-sport | [x] |
| MLB-029 | docs/baseball/OVERVIEW.md | [x] |
| MLB-030 | docs/basketball/OVERVIEW.md | [x] |
| MLB-031 | docs/football/OVERVIEW.md | [x] |
| MLB-032 | Unit tests for models | [x] |
| MLB-033 | README.md | [x] |
| MLB-034 | Docker container (Dockerfile, docker-compose.yml, .dockerignore) | [x] |

## Sprint 4 Complete

All security, documentation, and test tickets are done. The baseball module is fully built and tested. Basketball and football stubs are in place with full analytical framework documentation.

**Security checklist final status:**
- [x] .env in .gitignore
- [x] .env.example has placeholder values only
- [x] SECURITY.md written
- [x] .pre-commit-config.yaml created (detect-secrets + pre-commit-hooks)
- [x] README.md complete
- [ ] Run `detect-secrets scan > .secrets.baseline` (one-time setup, user runs this)
- [ ] Run `pre-commit install` (one-time setup, user runs this)
- [ ] Verify GitHub repo is set to private

---

## Archived Work Items

### MLB-032 — Unit Tests

Tests to write in `tests/`:

```
tests/
├── test_pythagorean.py     # pythagorean_win_pct edge cases, compute_pythagorean output
├── test_regression.py      # FIP-ERA gap, BABIP signal severity thresholds
├── test_kelly.py           # Kelly formula, half-Kelly cap, edge threshold
├── test_win_probability.py # game_win_probability bounds and adjustments
├── test_player_props.py    # project_pitcher_strikeouts, project_batter_hits
└── test_preseason.py       # project_team_wins, edge_wins calculation
```

Run: `pytest tests/ -v`

### MLB-033 — README.md

Contents:
1. Project overview and philosophy (Trading Bases)
2. Setup: clone, pip install -r requirements.txt, cp .env.example .env
3. Configuration: ODDS_API_KEY, GROQ_API_KEY
4. Running: `streamlit run src/dashboard/app.py`
5. Data sources
6. Sport roadmap (Basketball Sprint 5, Football Sprint 7)

## Security Checklist

- [x] .env in .gitignore
- [x] .env.example has placeholder values only
- [x] SECURITY.md written
- [ ] detect-secrets baseline created: `detect-secrets scan > .secrets.baseline`
- [ ] pre-commit hook installed: `pre-commit install`
- [ ] Verify GitHub repo is set to private

## Notes
- Sprint 5-6: Basketball module (see docs/basketball/OVERVIEW.md for architecture)
- Sprint 7-8: Football module (see docs/football/OVERVIEW.md for architecture)
