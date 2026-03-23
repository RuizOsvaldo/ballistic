# Security Model

## Secrets Management

All API keys and sensitive configuration are stored in a `.env` file at the project root. This file is **never committed to git**.

### Setup

1. Copy the template: `cp .env.example .env`
2. Fill in your real keys in `.env`
3. Never share, commit, or log the `.env` file

### Keys Required

| Variable | Where to get it | Notes |
|---|---|---|
| `ODDS_API_KEY` | the-odds-api.com — free tier | 500 req/month free |
| `GROQ_API_KEY` | console.groq.com — free account | 6,000 req/day free |

### How Secrets Are Loaded

All modules load secrets via `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY", "")
```

Keys are never hardcoded, printed, or logged.

---

## Git Protection

`.gitignore` blocks the following from ever being staged:

- `.env` and all `.env.*` variants
- `data/cache/` — cached API responses
- `data/bet_log.csv` — personal bet history
- `__pycache__/` and compiled Python files
- `.streamlit/secrets.toml`

---

## Pre-commit Secret Scanning

`detect-secrets` scans every commit for accidentally included secrets before they reach git history.

### Install

```bash
pip install detect-secrets
detect-secrets scan > .secrets.baseline
pre-commit install
```

### How it works

Any string that looks like an API key, password, or token triggers a block. The `.secrets.baseline` file tracks known false positives so they don't block legitimate commits.

---

## Runtime Safety

- API keys are never included in Streamlit UI output, error messages, or logs
- The Odds API and Groq clients validate that keys are set and raise a clear `EnvironmentError` if missing — they do not fall back silently
- Cache files store only stats data — no credentials are ever written to disk

---

## Repository

Keep this repository **private** on GitHub. The `.env.example` file committed to the repo contains only placeholder values and is safe to share publicly, but the full repo should remain private while it contains personal bet tracking data.

---

## Checklist Before First Commit

- [ ] `.env` exists locally and is NOT staged (`git status` should not show `.env`)
- [ ] `.env.example` has placeholder values only
- [ ] `detect-secrets` baseline created
- [ ] Pre-commit hook installed
- [ ] Repository set to private on GitHub
