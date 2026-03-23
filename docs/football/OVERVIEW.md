# Football (NFL) — Analysis Overview

## Status

**Planned — Sprint 7-8.** This document defines the analytical framework and model architecture for the NFL module. No code is active yet. Implementation begins after the basketball module is complete.

---

## Philosophy

NFL betting is the most widely bet sport in the United States and consequently the most efficiently priced at the game level. The public pours money into every game, which keeps spreads sharp on high-profile matchups. However, significant edge exists in:

1. **Play-level efficiency metrics** that the public ignores in favor of box score stats
2. **Player props** where sportsbook lines lag behind role and usage changes
3. **Game totals** where weather and pace-of-play are systematically underpriced
4. **Preseason win totals** where roster construction is imperfectly priced before the season

The core analytical language of NFL modeling is **Expected Points Added (EPA)** and **DVOA** — metrics that measure what actually happened on each play relative to what was expected, stripping out garbage time, opponent adjustments, and scoring environment noise.

---

## Bet Types Covered

| Bet Type | Description |
|---|---|
| Moneyline | Win outright. Less common primary bet due to spreads. |
| Spread | Point spread. The primary NFL bet type. |
| Game Total (O/U) | Combined points. Weather-sensitive. |
| First Half | Spread or total for just the first half. |
| Player Props | Passing yards, rushing yards, receiving yards, TDs, receptions, completions |
| Team Props | Team total points, first team to score, rushing/passing yards |
| Season Win Totals | Preseason over/under on team wins — one of the most exploitable NFL markets |
| Division / Conference Futures | Season-long futures on division or conference winners |

---

## Team-Level Models

### 1. EPA Per Play (Primary Signal)

**Expected Points Added (EPA)** measures the value of every play relative to the expected points in that situation (down, distance, field position). A 5-yard gain on 3rd and 4 adds more expected points than a 5-yard gain on 1st and 10.

- **Offensive EPA/play:** How efficiently a team moves the ball relative to situation
- **Defensive EPA/play allowed:** How well a team suppresses opponent efficiency
- **Pass EPA vs. Rush EPA:** Passing is dramatically more efficient; teams that pass more win more

**Regression signal:** Teams winning games but posting negative or mediocre EPA/play differentials are outperforming their process. Teams with strong EPA/play differentials but average records are undervalued.

This is the NFL equivalent of Pythagorean win expectation — efficiency predicts future performance better than results.

---

### 2. DVOA (Defense-adjusted Value Over Average)

**DVOA** from Football Outsiders measures efficiency on every play compared to league average, adjusted for opponent quality. It is the most comprehensive single-number team quality metric in football.

| Metric | Description |
|---|---|
| Offensive DVOA | How efficiently a team's offense performs vs. league average |
| Defensive DVOA | How efficiently a team's defense performs (negative = better) |
| Special Teams DVOA | ST unit efficiency |
| Total DVOA | Weighted combination of all three |

**Bet use:** Teams with high Total DVOA priced as underdogs on the spread due to a poor record are prime back candidates. The market prices on record; DVOA prices on process.

---

### 3. Success Rate

**Success rate** measures the percentage of plays that are considered successful by down:
- 1st down: gain of 40% of yards needed
- 2nd down: gain of 60% of yards needed
- 3rd/4th down: conversion

Success rate is more predictive than yards per play because it captures play-by-play consistency rather than being skewed by big plays.

**Bet use:** Teams with high success rates but average yards per game are often undervalued on totals and spreads.

---

### 4. Situational and Contextual Factors

| Factor | Effect on Model |
|---|---|
| Rest advantage (bye week) | ~2-3 point edge vs. opponent coming off a short week |
| Short week (Thursday game) | -1.5 to -2.5 points for the team on short rest |
| Home field advantage | ~2.5 points on average; stronger in dome stadiums and cold-weather road games |
| Divisional games | Historically tighter spreads; familiarity reduces edge of better team |
| Weather | Wind > 15 mph: significant suppression of passing game and totals |
| Travel/time zone | West Coast teams playing at 1pm ET perform measurably worse |
| Coaching tendencies | Aggressive 4th-down coaches increase value in close games |

---

### 5. Preseason Win Total Projections

One of the highest-edge NFL markets. Lines are set in the offseason and the public bets on brand names and media narratives.

**Model inputs:**
- Prior season DVOA and EPA/play (most predictive carry-forward metrics)
- Offseason roster changes via approximate value / contract analysis
- Coaching changes and scheme adjustments
- Strength of schedule
- Division regression (divisions with three strong teams depress win totals for all)
- Historical variance: NFL team win totals regress strongly toward 8-8

**Output:** Projected wins with confidence interval. Flag when projection diverges from Vegas line by 1.5+ wins in either direction.

---

## Player Props

NFL player props are among the highest-edge markets in all of sports. Sportsbooks set lines quickly each week and public money, not sharp analysis, moves them.

### Quarterback Props

| Prop | Model Approach |
|---|---|
| Passing Yards O/U | EPA/dropback × projected pace × opponent pass DVOA |
| Completions O/U | Completion % Over Expected (CPOE) vs. opponent coverage |
| Touchdowns O/U | Red zone usage + opportunity rate + opponent red zone DVOA |
| Interceptions O/U | Turnover-worthy play rate vs. opponent ball-hawk tendency |

**Key signal:** CPOE (Completion Percentage Over Expected) measures how much better or worse a QB completes passes than the model expects given receiver separation, depth of target, and pressure. A QB with negative CPOE is due for regression on completion and yards props.

### Running Back Props

| Prop | Model Approach |
|---|---|
| Rushing Yards O/U | Carries projection × yards per carry vs. opponent run DVOA |
| Receptions O/U | Target share × opponent RB coverage grade |
| Rushing Touchdowns | Goal-line carry share + red zone usage rate |

**Key signal:** Snap count and route participation are stronger leading indicators than touch count for receiving backs. A RB whose snap count has spiked due to teammate injury is underpriced on receiving props.

### Wide Receiver / Tight End Props

| Prop | Model Approach |
|---|---|
| Receiving Yards O/U | Air yards share × target rate × opponent coverage DVOA by position |
| Receptions O/U | Target share × catch rate vs. coverage grade |
| Touchdowns O/U | End zone target share + red zone alignment tendency |
| Longest Reception | Air yards average + deep target rate |

**Key signal:** Air yards share is the most stable predictor of receiving props. A receiver running deep routes consistently accumulates air yards even in low catch-rate weeks — and eventually converts them. The market underprices players with high air yards but recent low catch rates.

---

## Advanced Stats

| Stat | Description | Bet Use |
|---|---|---|
| EPA/play (off/def) | Efficiency per play vs. situation expectation | Primary team quality signal |
| DVOA | Opponent-adjusted efficiency | Team spread and total modeling |
| CPOE | QB completion rate vs. expected | QB true talent, regression signal |
| Air Yards Share | % of team air yards targeted to a player | WR/TE prop projections |
| Target Share | % of team targets going to a player | Volume-based prop model |
| Pressure Rate | % of QB dropbacks under pressure | QB performance context |
| Yards After Contact | RB contact balance | Sustainable rushing performance |
| Separation (avg) | Average cushion from defender at catch | WR true talent vs. box score |
| Coverage Grade | PFF coverage grades by corner/safety | Matchup context for WR props |

---

## Game Totals Model

The NFL game total is one of the most weather-sensitive markets in sports:

| Condition | Effect on Total |
|---|---|
| Wind > 15 mph | -3 to -7 points — suppresses passing |
| Wind > 25 mph | -6 to -12 points |
| Temperature < 30°F | -1 to -3 points |
| Precipitation | -1 to -3 points |
| Dome stadium | No weather effect; historical totals run higher |

The market accounts for weather but frequently underadjusts on borderline conditions (10-15 mph wind, 30-40°F). The model applies quantified weather adjustments and flags when the posted total appears off.

---

## Groq Agent — Football Reasoning

For each game or prop, the agent receives:
- DVOA matchup (offense vs. defense by unit)
- EPA/play differential and implied spread comparison
- Rest and schedule context
- Weather projection for the game
- Key player availability flags
- Pace and total projection vs. posted line

The agent returns:
- Plain-language explanation of the edge
- Confidence level (Low / Medium / High)
- Key risk factors (injury uncertainty, line movement, weather variance)

---

## Data Sources (Planned)

| Source | Access Method | Data |
|---|---|---|
| ESPN unofficial API | requests (existing pattern) | Schedules, scores, basic player stats |
| Pro Football Reference | pfr_scraper / requests | Historical DVOA-equivalent, splits |
| Football Outsiders | Manual or scraper | DVOA, DVOA by week |
| nflfastR (R/Python) | nfl_data_py Python library | EPA/play, CPOE, play-by-play |
| The Odds API | REST API (existing client) | NFL spreads, totals, player props |

---

## Limitations and Risks

- **Injury and lineup:** NFL rosters are the most injury-affected in pro sports. A key injury (QB, LT, WR1) can swing spreads by 7+ points. Always verify injury report before game time.
- **Weather uncertainty:** Weather models 3-7 days out are unreliable. Recheck totals the morning of the game.
- **Week 1 uncertainty:** No in-season data. Rely on prior-year DVOA and preseason signals only.
- **Coaching schemes:** New offensive or defensive coordinators can render prior-year efficiency metrics less predictive.
- **Sharp line movement:** NFL spreads are the most heavily bet markets. Sharp money can move lines quickly and significantly.
