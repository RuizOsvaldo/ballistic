"""About page — methodology, concepts, and how predictions are made."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About Ballistic")
    st.caption(
        "Ballistic is a quantitative sports betting analytics dashboard built on Joe Peta's "
        "*Trading Bases* methodology. This page explains every concept and formula behind the predictions."
    )

    st.info(
        "**How to read this page:** Each section covers one layer of the prediction pipeline, "
        "from raw data all the way to a bet recommendation. They build on each other in order.",
        icon="📖",
    )

    st.divider()

    # ── Section 1: Philosophy ─────────────────────────────────────────────────
    with st.expander("1 · The Philosophy — Why Markets Misprice Teams", expanded=True):
        st.markdown("""
The core insight from Joe Peta's *Trading Bases* (2013) is that the sports betting market
behaves like a **stock market reacting to headlines** rather than fundamentals.

When a team wins 8 of their last 10 games, books and bettors alike price them as a strong
favorite — even if those wins were driven by an unsustainably high BABIP against them, or
by runners stranding luck in close games. The market is pricing **narrative**. This model
prices **true talent**.

**The arbitrage opportunity:**
> Vegas sets lines off public perception. The model sets lines off math.
> When those two numbers diverge by more than 3%, there is edge.

The model uses three foundational tools to strip narrative away from talent:

| Tool | What it strips away |
|---|---|
| **Pythagorean Win %** | Wins and losses driven by sequencing luck |
| **FIP (Fielding Independent Pitching)** | ERA driven by defense, BABIP, and bullpen |
| **BABIP regression** | Batting average driven by where balls happen to land |

Each tool is described in full below.
        """)

    # ── Section 2: Pythagorean Win Expectation ────────────────────────────────
    with st.expander("2 · Pythagorean Win Expectation — True Team Quality"):
        st.markdown("""
**What problem it solves:** A team's W-L record contains noise. One-run games, walk-off
wins, and late-inning implosions can make a mediocre team look good or a good team look bad.
Runs scored and runs allowed over a full season are far more stable indicators of true quality.

**The formula (Pythagenpat — Bill James / Michael Wolverton):**
```
Pythagorean Win % = RS^1.83 / (RS^1.83 + RA^1.83)
```
where RS = season runs scored, RA = season runs allowed.

The exponent **1.83** (Pythagenpat) is more accurate than the classic 2.0 for actual MLB
run environments. It better handles extreme offensive or defensive seasons.

**Regression to the mean (Peta core):**

Early in the season, RS and RA from a handful of games are too noisy to trust. Once a team
reaches **20 games played**, the model automatically blends the team's actual RS/RA with the
current league average:

```
weight = G / (G + 30)

adj_RS/G = weight × actual_RS/G + (1 - weight) × league_avg_RS/G
```

At 20 games: 40% actual, 60% prior. At 81 games: 73% actual. At 162 games: 84% actual.
The league average is computed live from the current season's data, not hardcoded.

A banner in the app switches automatically from **Early Season** to **Regression Active**
once all 30 teams have crossed 20 games played.

**How to read it:**
- A team scoring 500 runs and allowing 450 has a Pythagorean W% of ~55%.
- If their actual record is only 48%, they are **underperforming** their true talent — and
  likely to win more games going forward. This is a **back signal**.
- If their actual record is 62%, they are **overperforming** — likely to regress. This is a
  **fade signal**.

**Deviation thresholds used in this app:**

| Deviation | Signal | Meaning |
|---|---|---|
| > +10% | 🔴 High overperformance | Strong fade signal |
| +5% to +10% | 🟡 Low overperformance | Mild fade |
| ±5% | ⚪ On track | No signal |
| -5% to -10% | 🟡 Low underperformance | Mild back signal |
| < -10% | 🟢 High underperformance | Strong back signal |

**Where it appears:** Teams page, Game Analysis page, and as the base layer of every win
probability calculation.
        """)

    # ── Section 3: FIP and BABIP ──────────────────────────────────────────────
    with st.expander("3 · FIP & BABIP — Separating Pitcher Luck from Skill"):
        st.markdown("""
ERA is the most widely cited pitching stat. It is also one of the most misleading over
short windows.

---

### FIP — Fielding Independent Pitching

**What it is:** FIP strips ERA down to only the three outcomes a pitcher **directly controls**:
home runs allowed, walks, and strikeouts. Everything else — grounders that sneak through,
diving catches, wind-blown pop-ups — is removed.

```
FIP = (13 × HR + 3 × BB - 2 × K) / IP + constant
```

The constant (~3.10) is calibrated so that league-average FIP equals league-average ERA,
making FIP directly comparable to ERA on the same scale.

**The FIP-ERA gap signal:**
```
FIP-ERA gap = FIP - ERA
```

| Gap | Signal | Meaning |
|---|---|---|
| ≥ +1.12 | 🔴 High | ERA is unsustainably low — expect ERA to rise sharply |
| +0.75 to +1.12 | 🟡 Low | ERA mildly low — some regression likely |
| -0.75 to +0.75 | ⚪ None | ERA reflects true talent |
| -0.75 to -1.12 | 🟡 Low | ERA mildly high — pitcher unlucky |
| ≤ -1.12 | 🟢 High | ERA is unsustainably high — expect ERA to drop |

**Practical example:** If a pitcher has a 2.80 ERA but a 4.10 FIP, the market is pricing him
as an ace. The model sees a league-average pitcher who has been very lucky on balls in play.
Backing the opponent is the play.

---

### BABIP — Batting Average on Balls in Play

**What it is:** BABIP measures what fraction of balls put in play (excluding home runs) fall
for hits. League average is approximately **.300**. Pitchers have very little control over
this number — it is driven primarily by defense, park dimensions, and sequencing randomness.

```
BABIP = (H - HR) / (AB - K - HR + SF)
```

**The BABIP signal:**

| BABIP | Signal | Meaning |
|---|---|---|
| > 0.340 | 🔴 High luck | Pitcher has been victimized by bad defense / bad luck — ERA will drop |
| 0.320–0.340 | 🟡 Mild luck | Some positive regression likely |
| 0.275–0.320 | ⚪ None | Normal range |
| 0.255–0.275 | 🟡 Mild suppression | ERA will likely rise slightly |
| < 0.255 | 🔴 Strong suppression | Pitcher has been unsustainably good — ERA will rise |

**When both FIP and BABIP agree:** The signal is escalated to High severity. A pitcher
with a wide FIP-ERA gap AND a BABIP above .340 is almost certainly due for a bad outing —
his ERA has nowhere to go but up.

**Where it appears:** Pitchers page, Player Analysis page, and as a FIP adjustment layer
inside the win probability model.
        """)

    # ── Section 4: Win Probability ────────────────────────────────────────────
    with st.expander("4 · Win Probability Model — How Game Picks Are Made"):
        st.markdown("""
The win probability model implements the full Joe Peta + Bill James sabermetrics pipeline.
It does not simply average the two teams' Pythagorean win percentages — it uses **Log5**
to compute the true head-to-head matchup probability, then stacks pitcher and park adjustments
on top.

### The Pipeline

```
Step 1:  Pythagorean W% per team  (RS^1.83 / (RS^1.83 + RA^1.83))
         If team has >= 20 games played: RS/RA blended toward league mean
           weight = G / (G + 30)
         If team has < 20 games played: raw RS/RA used

Step 2:  Log5 head-to-head probability (Bill James)
           P(A beats B) = (A - A×B) / (A + B - 2×A×B)

Step 3:  Starting pitcher FIP adjustment (with lineup matchup)
Step 4:  Home field advantage  (+4%)
Step 5:  Bullpen FIP adjustment
Step 6:  Renormalize + clamp [30%, 70%]
```

---

### Step 2: Why Log5 Instead of Averaging

The old approach (normalize home_pyth + away_pyth) consistently over-predicted favorites
and under-predicted close games. Log5 solves this mathematically:

```
P(A beats B) = (A - A×B) / (A + B - 2×A×B)
```

- Two equal .500 teams → exactly 50%
- .600 vs .400 team → 69.2% (Bill James reference value)
- Both teams at .700 → 50% (they're equally good relative to each other)

---

### Step 3: Starting Pitcher FIP Adjustment with Lineup Matchup

The effective FIP faced by a starter is adjusted for the quality of the opposing lineup:

```
lineup_matchup_adj = (lineup_avg_OPS - 0.720) × 3.0
effective_fip      = starter_fip + lineup_matchup_adj

pitch_adj          = (league_avg_fip - effective_fip) × 0.03
```

A strong lineup (OPS 0.770) adds +0.15 to the opposing starter's effective FIP — making
them look worse than their raw FIP suggests. A weak lineup subtracts.

League avg OPS = 0.720. League avg FIP is computed live as the IP-weighted mean of all
qualified starters. OPS is used as a lineup quality proxy because wOBA and wRC+ are not
available from the MLB Stats API (OPS correlates at r≈0.97 with wOBA).

When lineups have not been posted, the adjustment is 0.0 — graceful no-op.

**Example:** League avg FIP = 4.00. Home starter FIP = 3.00 (elite) vs. lineup OPS 0.770.
- Lineup adj: (0.770 - 0.720) × 3.0 = **+0.15**
- Effective FIP: 3.15
- Pitch adj: (4.00 - 3.15) × 0.03 = **+2.55%** to home win prob

---

### Step 5: Bullpen FIP Adjustment

The bullpen faces roughly **3 of every 9 innings**. Its FIP relative to league average
is scaled proportionally.

```
bullpen_adj = (opp_bullpen_fip - league_avg_bullpen_fip) × (3/9) × 0.30
```

A weak opponent bullpen (high FIP) means your team scores more late in the game.

---

### Lineup Status Confidence

| Status | Data quality | FIP adjustment applied? |
|---|---|---|
| ✅ Both starters confirmed | Full | Yes — both sides with lineup matchup |
| ⚠️ One starter confirmed | Partial | Yes — confirmed side only |
| ⏳ No starters posted | None | No — Log5 + home field only |

---

### Edge Calculation — Vig-Free Implied Probability

The edge is computed against the **vig-free** implied probability, not the raw line:

```
raw_home      = 1 / decimal_odds(home_american_odds)
raw_away      = 1 / decimal_odds(away_american_odds)
implied_prob  = raw_home / (raw_home + raw_away)   # vig stripped out

edge          = model_prob - implied_prob
```

This ensures the comparison is fair — the vig (typically 4–5%) is removed before edge
is measured.
        """)

    # ── Section 5: Run Line Edge ──────────────────────────────────────────────
    with st.expander("5 · Run Line Edge — The ±1.5 Spread"):
        st.markdown("""
The run line is MLB's version of a point spread. The favored team is typically priced at
**−1.5 runs**, meaning they must win by 2 or more runs to cover. The underdog is at **+1.5**,
meaning they cover if they win outright or lose by exactly 1 run.

### How the Edge is Calculated

The model treats each team's score as an independent **Poisson random variable** with mean
equal to the projected runs output by the win probability model.

```
home_score ~ Poisson(proj_home_runs)
away_score ~ Poisson(proj_away_runs)

P(home covers -1.5) = P(home_score - away_score > 1.5)
                    = computed via full joint Poisson PMF table (0–35 runs per team)

P(away covers +1.5) = 1 - P(home covers -1.5)
```

**Why Poisson?** Baseball run scoring closely follows a Poisson distribution. Games produce
a known average number of runs, the events (hits, walks, errors leading to runs) are
approximately independent, and run totals cluster around the projected mean.

**Edge:**
```
implied_prob = 1 / decimal_odds(rl_american_odds)
edge = P(cover) - implied_prob
```

A positive RL edge means the model believes the cover probability is higher than what the
market is pricing. The same ≥3% threshold applies as for the moneyline.

**Key difference from ML edge:** ML edge captures *who wins*. RL edge captures *by how much*.
A strong offensive home team vs. a weak bullpen is more likely to produce a blowout than a
close win — that distinction shows up here but not in the moneyline.
        """)

    # ── Section 6: Total Edge ─────────────────────────────────────────────────
    with st.expander("6 · Game Total Edge — Over / Under"):
        st.markdown("""
The game total (over/under) is the combined runs scored by both teams. The sportsbook sets
a line (e.g., 8.5) and bettors choose whether the final score will go over or under.

### How the Edge is Calculated

When two Poisson random variables are added together, their sum is also Poisson:
```
total_score = home_score + away_score
            ~ Poisson(proj_home_runs + proj_away_runs)

P(over total_line) = P(total_score > total_line)
                   = computed from Poisson CDF tail

P(under total_line) = 1 - P(over total_line)
```

**Edge:**
```
over_edge  = P(over)  - (1 / decimal_odds(over_american_odds))
under_edge = P(under) - (1 / decimal_odds(under_american_odds))
```

The direction with the higher positive edge is flagged as the recommendation. If neither
direction clears 3%, it is shown as a directional signal only (no BET flag).

### What drives total edge?

| Factor | Effect on total |
|---|---|
| Elite starters (low FIP) | Pushes projected total down → lean Under |
| Weak bullpens (high FIP) | Pushes projected total up → lean Over |
| Hitter-friendly park (PF > 105) | Adds to both projected scores → lean Over |
| Pitcher-friendly park (PF < 95) | Subtracts from both projected scores → lean Under |
| High projected run totals vs. low line | Strong Over edge |
        """)

    # ── Section 7: Kelly Criterion ────────────────────────────────────────────
    with st.expander("7 · Kelly Criterion — How Bet Sizes Are Calculated"):
        st.markdown("""
Finding an edge is only half the job. Sizing the bet correctly is the other half.
Bet too small and you leave money on the table. Bet too large and a bad run of variance
wipes out your bankroll even if the model is right.

### The Kelly Formula

The Kelly Criterion computes the mathematically optimal fraction of your bankroll to wager
to maximize long-run growth.

```
b = decimal_odds - 1.0          # profit per $1 risked
p = model_prob                  # model's estimated win probability
q = 1 - p                       # estimated loss probability

Kelly fraction = (b × p - q) / b
```

**Example:** Model prob = 60%, line = +100 (decimal 2.0), b = 1.0
```
Kelly = (1.0 × 0.60 - 0.40) / 1.0 = 0.20 → 20% of bankroll
```

That's extremely aggressive. Most professionals use **half-Kelly** to reduce variance:

### Half-Kelly (What This App Uses)

```
stake = kelly_full × 0.5
stake = min(stake, 0.05)   # hard cap at 5% of bankroll per game
```

**Why half-Kelly?** Full Kelly maximizes expected log wealth in theory, but in practice
the model has estimation error. Half-Kelly sacrifices roughly 25% of expected growth to
cut variance (and max drawdown) by about 50%.

**Why cap at 5%?** Even when edge is enormous, no single MLB game should represent a large
share of your risk. The 5% cap protects against correlated bad outcomes (rainouts, injuries,
umpire effects).

### Minimum Edge Threshold

```
Minimum edge to recommend a bet: 3%
```

Below 3%, the transaction costs (vig), variance, and model uncertainty consume the edge.
The number is based on the typical vig structure of -110/-110 markets (~4.5% vig).
        """)

    # ── Section 8: Player Props ───────────────────────────────────────────────
    with st.expander("8 · Player Props — Strikeouts, Hits, Home Runs, Earned Runs"):
        st.markdown("""
Player props are bets on individual player performance in a single game. Each prop type
uses a different formula, but all follow the same final step:

```
edge_pct = (model_projection - prop_line) / prop_line × 100

BET if |edge_pct| >= 5%
Direction: OVER if model > line, UNDER if model < line
```

*(Props use a 5% threshold vs. 3% for ML — lower sample sizes make props noisier.)*

---

### Pitcher Strikeouts

The 60/40 blend below is a Peta-derived approach: pitcher skill dominates, but the
opposing lineup's strikeout tendency is a meaningful second factor.

```
blended_k_rate = pitcher_k_pct × 0.60 + opponent_k_pct × 0.40
batters_faced  = innings_per_start × 4.3   (league avg batters per inning)
projected_K    = blended_k_rate × batters_faced
```

**What to watch:** `whiff_pct` is shown as a sustainability signal. A pitcher with a high
K% but low whiff rate is getting strikeouts on weak contact, not swing-and-miss stuff —
that K rate is more fragile.

---

### Pitcher Earned Runs

xFIP gets more weight because it normalizes for ballpark HR effects and fly-ball luck.
FIP is still included because it captures actual HR allowed (relevant for single-game props).

```
blended_era    = pitcher_fip × 0.40 + pitcher_xfip × 0.60
projected_ER   = (blended_era / 9.0) × projected_innings
```

---

### Batter Hits

BABIP regression is the core of the hits model. A hitter on a hot streak (.350 BABIP)
has their hits prop priced off that hot streak. The model regresses it 40% back toward .300.

```
regressed_babip = batter_babip × 0.60 + 0.300 × 0.40

balls_in_play   = at_bats_projected × (1 - batter_k_pct) × 0.96
projected_hits  = regressed_babip × balls_in_play
```

If the opposing pitcher's BABIP-allowed is available, it is blended in at 30% weight.

---

### Batter Total Bases & Home Runs

Total bases use slugging percentage adjusted for park HR factors.
Home runs use barrel rate as the core input — barrels are the most reliable contact-quality signal.

```
# Total bases
projected_TB = adjusted_slg × at_bats_projected

# Home runs
hr_per_fb     = barrel_pct × 0.55
projected_HR  = at_bats_projected × fb_pct × hr_per_fb × park_hr_factor
```
        """)

    # ── Section 9: Preseason Win Totals ───────────────────────────────────────
    with st.expander("9 · Preseason Win Totals — Season Betting Lines"):
        st.markdown("""
Before the season starts, sportsbooks post win total lines (e.g., "Yankees Over/Under 91.5
wins"). These are among the most beatable markets because books set them in the offseason
when data is thin and public opinion is noisy.

### The Formula

```
Step 1:  pyth_win_pct     = RS² / (RS² + RA²)    (prior season)
Step 2:  regressed_win_pct = pyth_win_pct × 0.80 + 0.500 × 0.20
Step 3:  projected_wins    = regressed_win_pct × 162 + war_adjustment
Step 4:  edge_vs_vegas     = projected_wins - vegas_line
```

**Recommendation rules:**
- edge ≥ +2.0 wins → **BET OVER**
- edge ≤ −2.0 wins → **BET UNDER**

### Why 20% regression to .500?

No team fully repeats its run environment. Roster turnover, injury luck, and schedule
strength all push teams toward the mean. Peta found that regressing Pythagorean W%
(not actual W%) by 20% toward .500 is the most accurate single-formula preseason predictor
across the modern era.

**The WAR adjustment:** If a team made major roster moves (signed a 5-WAR pitcher, traded
an outfielder), a manual WAR adjustment can be applied. Each win of WAR adds directly to
projected wins.
        """)

    # ── Section 10: AI Reasoning ──────────────────────────────────────────────
    with st.expander("10 · AI Reasoning Layer — What Llama 3.3 70B Does"):
        st.markdown("""
After the quantitative model produces a win probability, edge percent, and Kelly stake,
the Groq-hosted **Llama 3.3 70B** model receives the full signal packet and returns a
plain-language explanation.

### What is sent to the model

- Home and away team names
- Model win probabilities vs. market implied probabilities
- Edge percent
- Pythagorean deviation for each team (if available)
- Starting pitcher FIP, xFIP, SIERA, BABIP, FIP-ERA gap
- Signal severity and direction

### What comes back

| Field | Description |
|---|---|
| **Reasoning** | 2–4 sentence explanation of why the edge exists |
| **Confidence** | Low / Medium / High, based on how many signals converge |
| **Key Risk** | One caveat to consider before placing the bet |

### Confidence levels

| Level | Meaning |
|---|---|
| 🟢 High | 3+ signals agree — Pythagorean, FIP, and BABIP all point the same direction |
| 🟡 Medium | 2 signals agree |
| 🔴 Low | 1 signal, or signals conflict |

**Important:** The AI layer is an explanation tool, not a decision layer. The model's
math determines whether a bet is recommended. The AI tells you *why*.
        """)

    # ── Section 11: Data Sources ──────────────────────────────────────────────
    with st.expander("11 · Data Sources & Caching"):
        st.markdown("""
### Data Sources

| Data | Source | Auth required? |
|---|---|---|
| Team batting / pitching stats | MLB Stats API | No |
| Team standings / records | MLB Stats API | No |
| Individual pitcher / batter stats | pybaseball → FanGraphs | No |
| Today's schedule, starters, lineups | MLB Stats API | No |
| Live game state (score, count, bases) | ESPN Scoreboard API | No |
| Moneylines, run lines, totals | ESPN Odds (via event API) | No |
| Player prop lines | The Odds API | API key (free tier: 500 req/month) |
| AI reasoning | Groq API (Llama 3.3 70B) | API key (free tier: 6,000 req/day) |
| Park factors | Static lookup table (hardcoded) | — |

### Caching

All external calls are cached locally as **Parquet files** to minimize API usage.

| Data type | Cache TTL |
|---|---|
| Odds & lines | 6 hours |
| Team batting / pitching stats | 6 hours |
| Individual pitcher / batter stats | 6 hours |
| Bullpen stats | 6 hours |
| Probable starters | 1 hour |
| Completed game results (past dates) | 24 hours |

Click **🔄 Refresh All Data** in the sidebar to force-clear all caches and fetch fresh data.
        """)

    # ── Section 12: Known Limitations ────────────────────────────────────────
    with st.expander("12 · Known Limitations"):
        st.markdown("""
This model is quantitative and systematic — that means its gaps are predictable and worth
knowing before relying on it for real bets.

| Limitation | Impact | Notes |
|---|---|---|
| **No weather adjustment** | Game totals can be off significantly | Wind direction / temp affect scoring. Currently shown as context only |
| **No umpire adjustment** | Strikeout props lack ump-zone factor | Some umpires suppress Ks by ~10%, others inflate |
| **Static park factors** | HR and total projections are park-neutral | Hardcoded table; not adjusted for weather or roof status |
| **No bullpen fatigue** | Heavy bullpen usage in prior days is ignored | Affects late-game run scoring and run line |
| **Injury data is manual** | A star player scratched from the lineup is not detected | Always verify the lineup before betting |
| **OPS as lineup proxy** | Lineup matchup uses OPS instead of wOBA/wRC+ | MLB Stats API does not expose wOBA/wRC+; OPS correlates at r≈0.97 |
| **Odds API free tier** | 500 requests/month; player props may be stale | Monitor quota carefully |

### Seasonal formula behavior

The model automatically adapts to where we are in the season:

- **Before 20 games (any team):** Raw RS/RA used — no regression blend. The app shows a
  blue **Early Season** banner. Predictions are directionally useful but carry more variance.
- **After 20 games (all teams):** Regression to mean activates automatically — no manual
  switch needed. The app shows a green **Regression Active** banner. Model accuracy improves
  significantly once all teams have 20+ games of data.

### When to trust the model most

- **After all teams reach 20 games** — regression to mean is active, signals are stabilized
- **Both starters confirmed** — full FIP + lineup matchup adjustment is applied
- **Multiple signals agree** — Pythagorean + FIP + BABIP all pointing the same direction

### When to use caution

- Doubleheaders and bullpen games (no true starter — FIP adjustment is not applied)
- Teams with major recent roster moves not yet reflected in season stats
- Games with meaningful weather events (rain, high wind)
- Line movement close to game time (sharp money may be reacting to news the model doesn't have)
        """)

    st.divider()
    st.caption(
        "Ballistic is built for educational and analytical purposes. "
        "Always bet responsibly. Past model performance does not guarantee future results."
    )
