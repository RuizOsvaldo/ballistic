# Baseball (MLB) — Analysis Overview

## Philosophy

MLB betting is the most analytically friendly major sport. Games are played 162 times per season, creating a large sample that rewards statistical models and punishes gut-feel betting. The foundational work comes from Joe Peta's *Trading Bases*, which applies Wall Street quantitative trading principles to the baseball betting market.

The core insight: the MLB betting market prices on **narrative and recent results**. Our models price on **underlying true talent and regression signals**. The gap between those two things is where edge lives.

---

## Bet Types Covered

| Bet Type | Description |
|---|---|
| Moneyline | Pick the winner outright. Primary game-level bet. |
| Run Line | Spread equivalent. Favorite -1.5 runs, underdog +1.5 runs. |
| Game Total (O/U) | Combined runs scored over or under the line. |
| First 5 Innings | Moneyline or total for just the first half — isolates starting pitchers. |
| Player Props | Individual performance lines: hits, strikeouts, home runs, RBI, bases, etc. |
| Preseason Win Totals | Season-long bet: does a team finish over or under their projected win total? |

---

## Game-Level Models

### 1. Pythagorean Win Expectation (Peta Core)

**Formula:** `W% = RS² / (RS² + RA²)`

A team's actual win-loss record contains significant luck. Close games, bullpen randomness, and sequencing cause teams to over or underperform their run differential. The Pythagorean formula predicts the win percentage a team *should* have based on runs scored and runs allowed.

**Signal:** When a team's actual W% diverges from their Pythagorean W% by more than 4-5%, they are a regression candidate.
- Actual W% > Pythagorean W% → team has been lucky → fade them
- Actual W% < Pythagorean W% → team has been unlucky → back them

**Bet use:** Moneyline and run line fade/back signals on overperforming and underperforming teams.

---

### 2. FIP vs ERA Gap (Peta Core)

**FIP (Fielding Independent Pitching):** `(13*HR + 3*BB - 2*K) / IP + constant`

ERA is polluted by defense quality and batted ball luck. FIP strips ERA down to only what a pitcher controls: home runs, walks, and strikeouts. When ERA and FIP diverge significantly, ERA is almost always wrong.

**Thresholds:**
- FIP - ERA > +0.75 → pitcher has been getting lucky, ERA will rise → fade their team
- FIP - ERA < -0.75 → pitcher has been unlucky, ERA will fall → back their team

**Extended:** xFIP replaces actual HR with expected HR based on fly ball rate. SIERA further adjusts for ground ball tendencies. All three are computed and displayed.

**Bet use:** Starting pitcher-level signal for moneyline, run line, and first-5-innings bets.

---

### 3. BABIP Deviation (Peta Core)

**BABIP (Batting Average on Balls in Play):** `(H - HR) / (AB - K - HR + SF)`

League average BABIP is approximately .300. It is largely outside a pitcher's control — defense, park, and sequencing luck drive most variation. Extreme BABIP values regress hard toward .300.

**Thresholds:**
- Pitcher BABIP > .320 → has been allowing too many hits to fall in → ERA will drop, back their team
- Pitcher BABIP < .275 → has been artificially suppressing hits → ERA will rise, fade their team

**Bat side:** Team BABIP works in reverse — a team getting lucky on BABIP will regress offensively.

**Bet use:** Pitcher and team regression signals feeding into win probability and game totals.

---

### 4. Win Probability Model

Combines all signals into a single estimated win probability per game using the full Peta + sabermetrics pipeline:

```
Step 1: Pythagorean W% per team — RS^1.83 / (RS^1.83 + RA^1.83)
        Early season (< 20 games): raw RS/RA used
        After 20 games: RS/RA blended toward league mean (weight = G / (G + 30))

Step 2: Log5 head-to-head probability — P = (A - A*B) / (A + B - 2*A*B)
        Converts two independent Pythagorean W%s into a true matchup probability

Step 3: Starter FIP adjustment (+/- 3% per FIP point vs. league avg)
        Effective FIP adjusted for opposing lineup quality: (lineup_avg_OPS - 0.720) * 3.0

Step 4: Home field +0.04

Step 5: Bullpen FIP adjustment

Step 6: Renormalize + clamp [0.30, 0.70]
```

**Edge:** `edge = win_prob - vig_free_implied_prob` (both sides normalized to remove the book's juice)

Bets are only flagged when edge ≥ 3%.

---

### 5. Game Total Model (Over/Under)

Inputs:
- Both teams' runs-per-game averages (park-adjusted)
- Both starting pitchers' FIP and xFIP
- Ballpark run factor
- Weather: temperature and wind speed/direction
- Bullpen fatigue: rolling appearances over last 3 days

Output: projected total runs with confidence interval. Compare to the posted total to find over/under edge.

**Key insight:** Cold weather and wind-in significantly suppress totals. Wind-out and heat inflate them. The market prices this imperfectly.

---

### 6. Preseason Win Total Projections (Peta)

Before the season, Vegas posts over/under lines on each team's win total (e.g., Yankees over/under 91.5 wins). This is one of the most exploitable markets because:
- Lines are set months before the season
- Roster construction and offseason moves are incompletely priced
- The public bets on big-market teams, inflating their lines

**Model inputs:**
- Prior season Pythagorean win percentage (better predictor than actual W-L)
- Prior season run differential
- Projected roster quality via aggregated WAR
- Park factors
- Division strength adjustment

**Output:** Projected win total with a confidence range. When the projection diverges from the Vegas line by 3+ wins, flag as a bet.

---

## Extended Stats (Beyond Peta)

| Stat | What It Measures | Bet Use |
|---|---|---|
| wRC+ | Weighted Runs Created Plus — park/era adjusted offensive value per batter. 100 = league avg. | Lineup quality assessment for totals and moneylines |
| xFIP | Expected FIP using expected HR rate from fly balls | More stable pitcher true talent than FIP |
| SIERA | Skill-Interactive ERA — adjusts for batted ball mix | Best single ERA estimator for starters |
| Barrel % | % of batted balls hit with elite exit velocity + launch angle | Batter quality, pitcher vulnerability |
| Exit Velocity (avg) | Average speed off the bat | Sustainable contact quality signal |
| Sprint Speed | Player speed percentile | Affects stolen base props, infield hit rate |
| Chase Rate | % of pitches outside the zone swung at | Batter discipline, strikeout prop signal |
| Whiff Rate | Swings and misses per swing | Pitcher strikeout sustainability |
| Park Factor | How much a ballpark inflates or deflates runs, HR | Normalize all counting stats |
| Umpire Tendencies | Historical strike zone size per umpire | Affects K and BB rates for that game |

---

## Player Props

Player props are set quickly by sportsbooks and moved by public action, not analytical models. This creates consistent value opportunities.

### Pitcher Props

| Prop | Model Approach |
|---|---|
| Strikeouts O/U | Pitcher whiff rate + chase rate vs. opposing lineup K% + umpire tendencies |
| Earned Runs O/U | FIP-based projection + park factor + opposing wRC+ |
| Outs Recorded O/U | Historical pitch count efficiency + opposing contact rate |
| First Strikeout | First batter faced strikeout rate vs. batter's K% |

**Key signal:** A pitcher with a high whiff rate facing a high-strikeout lineup in a pitcher's park with an ump who calls a wide zone is a strong strikeout over candidate. The market rarely prices all four factors simultaneously.

### Batter Props

| Prop | Model Approach |
|---|---|
| Hits O/U | BABIP-adjusted hit rate + hard contact rate vs. pitcher BABIP allowed |
| Home Runs | Barrel% + park HR factor + pitcher HR/FB rate |
| RBI | Run environment (lineup context) + RISP performance |
| Stolen Bases | Sprint speed + pitcher pickoff tendency + catcher pop time |
| Total Bases O/U | Contact quality metrics + park factor |
| Strikeouts O/U | Batter K% vs. pitcher whiff rate + umpire zone size |

**Key signal:** BABIP regression is the biggest prop edge source. A batter with a .240 BABIP over the last 30 days is getting unlucky — their hits prop is underpriced.

---

## Groq Agent — Baseball Reasoning

For each game or prop recommendation, the Groq agent (Llama 3.3 70B) receives:
- All computed model signals (Pythagorean gap, FIP-ERA, BABIP, xFIP, park factor, weather)
- The Vegas odds and implied probability
- The computed edge and Kelly stake

The agent returns:
- A 2-4 sentence plain-language explanation of why there is edge
- A confidence level (Low / Medium / High) based on how many signals converge
- Any caveats or risk factors to consider (injury reports, short rest, etc.)

---

## Data Sources

| Source | Access Method | Data |
|---|---|---|
| FanGraphs | pybaseball | FIP, xFIP, SIERA, wRC+, BABIP, WAR |
| Baseball Savant | pybaseball (statcast) | Exit velocity, barrel%, sprint speed, whiff rate |
| Baseball Reference | pybaseball | W-L records, run differential, historical splits |
| The Odds API | REST API | Moneylines, run lines, totals, player props |

---

## Signal Severity Levels

| Level | Criteria |
|---|---|
| Low | One signal present, edge 3-6% |
| Medium | Two signals converging, edge 6-10% |
| High | Three or more signals converging, edge > 10% |

Only Medium and High signals trigger a bet recommendation. Low signals are displayed for informational purposes.

---

## Limitations and Risks

- **Sample size:** Early-season models use raw RS/RA before 20 games. The app automatically switches to full regression-to-mean mode and displays a banner once all teams cross 20 games.
- **Lineup matchup uses OPS, not wOBA:** The MLB Stats API does not expose wOBA or wRC+. OPS (OBP + SLG) is used as a proxy (r ≈ 0.97 with wOBA).
- **Injury information:** The model does not automatically ingest injury reports. Always verify lineup before betting.
- **Bullpen volatility:** Starter models do not account for when a starter gets knocked out early.
- **Line movement:** Odds are fetched at a point in time. Sharp money can move lines significantly before game time.
- **Weather:** Wind direction and temperature affect game totals significantly. Not yet wired into the model.

## Data Sources

| Source | Access Method | Data |
|---|---|---|
| MLB Stats API | REST API (free, no key) | Team stats (batting/pitching/standings), schedule, starters, lineups, live game state |
| FanGraphs | pybaseball | Pitcher FIP, xFIP, SIERA, BABIP, K%, whiff% |
| The Odds API | REST API (key required) | Moneylines, run lines, totals, player props |
