# Basketball (NBA) — Analysis Overview

## Status

**Planned — Sprint 5-6.** This document defines the analytical framework and model architecture for the NBA module. No code is active yet. Implementation begins after the baseball module is complete.

---

## Philosophy

NBA betting is a pace-and-efficiency market. The public bets on names, narratives, and last-night's box score. The edge comes from understanding **true team and player quality** measured by efficiency metrics, identifying **schedule and rest mismatches** the market underprices, and finding **player prop lines** set on seasonal averages that ignore recent role changes, matchup, and pace context.

The NBA is also a high-volume sport — 82 games per team, multiple games per night. This creates a large sample for model validation and frequent opportunities.

---

## Bet Types Covered

| Bet Type | Description |
|---|---|
| Moneyline | Win outright. More predictable than MLB due to star player impact. |
| Spread | Point spread. Primary NBA bet type. |
| Game Total (O/U) | Combined points. Heavily pace and defense dependent. |
| First Half / First Quarter | Isolates early-game tendencies and starting lineups. |
| Player Props | Points, rebounds, assists, PRA (combined), steals, blocks, 3-pointers |
| Season Win Totals | Preseason over/under on team wins (similar to MLB approach) |

---

## Team-Level Models

### 1. Net Rating (Primary Team Signal)

**Net Rating = Offensive Rating - Defensive Rating**

- **Offensive Rating (ORTG):** Points scored per 100 possessions
- **Defensive Rating (DRTG):** Points allowed per 100 possessions

Net Rating is the NBA equivalent of run differential in baseball. It is a far more stable predictor of future outcomes than win-loss record. Teams with a high Net Rating but mediocre record are undervalued. Teams with a strong record but average Net Rating are due for regression.

**Regression signal:** Net Rating diverging from implied W% by more than 5 points is a meaningful fade/back signal — same logic as Pythagorean deviation in baseball.

---

### 2. The Four Factors (Dean Oliver)

The four factors explain approximately 95% of the variance in team offensive and defensive efficiency:

| Factor | Formula | Weight |
|---|---|---|
| Effective Field Goal % (eFG%) | (FGM + 0.5 * 3PM) / FGA | ~40% |
| Turnover Rate (TOV%) | TOV / (FGA + 0.44*FTA + TOV) | ~25% |
| Offensive Rebound % (ORB%) | ORB / (ORB + Opp DRB) | ~20% |
| Free Throw Rate (FT/FGA) | FTA / FGA | ~15% |

**Bet use:** Teams with elite eFG% suppression are underrated defensively. Teams winning on ORB% are prone to regression against elite defensive rebounding teams. Matchup-specific Four Factor analysis is the core of spread and total modeling.

---

### 3. Pace Adjustment

All efficiency stats must be pace-adjusted. A team scoring 115 points per game in a 105-possession game is very different from one scoring 115 in a 98-possession game. When a fast team plays a slow team, the pace tends to average out — affecting the total significantly.

**Game total model:** `projected_total = (team1_ORTG + team2_ORTG) / 2 * ((team1_pace + team2_pace) / 2) / 100`

Compare to the posted total for over/under edge.

---

### 4. Rest and Schedule Factors

Rest is one of the most consistently mispriced factors in NBA betting:

| Situation | Effect |
|---|---|
| Back-to-back (second night) | -2 to -3 points vs. rested opponent |
| 3-in-4 nights | -1 to -2 points |
| 7+ days rest | Slight underperformance (rust factor) |
| Home/away split | ~3 point home advantage |

**Schedule spot analysis:** A team on their second back-to-back road game facing a well-rested home team is a prime fade opportunity. The market adjusts for this, but often underadjusts.

---

### 5. Lineup and Injury Adjustment

Star player impact in the NBA is larger than in any other team sport. A team missing their primary ball-handler or rim protector can swing 5-8 points in efficiency. The model flags:
- Games where a top-3 player is listed questionable or out
- Lineup combinations and their historical net ratings (via NBA lineup data)

---

## Player Props

Player props are the highest-edge market in NBA betting. Sportsbooks set lines based on seasonal averages. The model uses rolling windows, matchup data, and role context.

### Points Props

| Signal | Description |
|---|---|
| Rolling 10-game scoring average | More predictive than season average for hot/cold streaks |
| Usage rate | % of team plays ending with that player while on court |
| Opponent defensive rating vs. position | How well the opponent guards that player's position |
| Pace of game | Faster pace = more possessions = more scoring opportunities |
| Home/away split | Some players perform significantly better at home |

**Key insight:** When a player's usage rate has spiked due to a teammate's injury, their props are still priced on their pre-injury average. This is a consistent source of over value.

### Rebounds Props

| Signal | Description |
|---|---|
| ORB% and DRB% | Rebounding rate is more stable than raw rebound counts |
| Minutes projection | Fewer minutes = fewer rebound opportunities |
| Opponent ORB% | Aggressive offensive rebounding teams suppress opponent DRB |
| Matchup size | Center playing against a small-ball lineup gets more opportunities |

### Assists Props

| Signal | Description |
|---|---|
| Assist rate | AST / (minutes / 48 * team possessions while player is on court) |
| Teammate shooting context | Playmakers get fewer assists when shooters are slumping |
| Pace | Assists correlate with pace more than most stats |
| On/off splits | Some playmakers only facilitate when certain teammates play |

### PRA (Points + Rebounds + Assists) Props

PRA is the most popular NBA prop. The model projects each component independently then sums them with a small correlation adjustment (players who score more tend to get more assists but fewer rebounds).

### 3-Pointers Made Props

| Signal | Description |
|---|---|
| 3PA rate | Attempts per game. Volume drives makes. |
| 3P% vs. opponent defense | Some defenses allow significantly more open threes |
| Hot/cold streak BABIP equivalent | 3P% is volatile, regresses to career average |

---

## Advanced Stats

| Stat | Description | Bet Use |
|---|---|---|
| True Shooting % (TS%) | Points per shooting possession, includes FTs | Overall offensive efficiency |
| Box Plus/Minus (BPM) | Player impact estimate from box score | Player quality for props and matchups |
| RAPTOR / EPM | Tracking-based impact metrics | Best player true talent estimates |
| On/Off Net Rating | Team net rating with vs. without each player | Lineup adjustment and injury impact |
| Clutch Net Rating | Net rating in games within 5 points last 5 min | Late-game reliability |
| Second Chance Points | Points from offensive rebounds | Four Factor ORB% game application |
| Points in the Paint | Interior scoring | Matchup vs. opponent rim protection |

---

## Groq Agent — Basketball Reasoning

For each game or prop, the agent receives:
- Net Rating differential and implied spread comparison
- Four Factors matchup breakdown
- Rest and schedule context
- Relevant player availability flags
- Pace projection and total comparison

The agent returns:
- Plain-language reasoning for the bet
- Confidence level (Low / Medium / High)
- Key risk factors (injury uncertainty, line movement, etc.)

---

## Data Sources (Planned)

| Source | Access Method | Data |
|---|---|---|
| NBA API (stats.nba.com) | nba_api Python library | Team/player stats, lineups, advanced metrics |
| Basketball Reference | basketball_reference_web_scraper | Historical splits, four factors |
| The Odds API | REST API (existing client) | NBA spreads, totals, player props |

---

## Limitations and Risks

- **Load management:** Star players sitting out with no notice is the biggest source of model error in the NBA. Always check injury reports 1-2 hours before tip-off.
- **In-season trades:** Roster construction changes mid-season can invalidate historical averages.
- **Early season instability:** First 15-20 games of the season have high variance in efficiency metrics.
- **Prop availability:** Not all player props are available on every game via The Odds API free tier.
