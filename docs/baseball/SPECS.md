# Baseball (MLB) — Technical Specifications

This document describes exactly what data the app pulls, how it maps to Joe Peta's *Trading Bases* methodology, and how every number flows into the formulas used to generate game winner predictions and player prop recommendations.

---

## 1. Data Sources and What Is Fetched

Team-level stats (batting, pitching, standings) come from the **MLB Stats API** (free, no key).
Individual pitcher and batter stats come from **pybaseball** (FanGraphs leaderboards).
Schedule, starters, lineups, and live game state come from the **MLB Stats API**.
Odds and props come from **The Odds API**. All data is disk-cached in Parquet files.

| Source | API / Call | Cache TTL | Columns Used |
|---|---|---|---|
| MLB Stats API — team batting | `GET /stats?group=hitting&stats=season` | 6 hours | `team_id`, `runs`, `hits`, `homeRuns` |
| MLB Stats API — team pitching | `GET /stats?group=pitching&stats=season` | 6 hours | `team_id`, `earnedRuns`, `era` |
| MLB Stats API — standings | `GET /standings` | 6 hours | `team_id`, `wins`, `losses`, `winningPercentage` |
| pybaseball → FanGraphs (pitchers) | `pitching_stats(season, qual=1)` | 6 hours | `name`, `team`, `ip`, `era`, `fip`, `xfip`, `siera`, `babip`, `k_pct`, `bb_pct`, `hr9`, `whiff_pct`, `gb_pct`, `fb_pct` |
| MLB Stats API — batter stats | `GET /stats?group=hitting&stats=season` (player-level) | 6 hours | `name`, `team`, `pa`, `ab`, `avg`, `obp`, `slg`, `babip`, `k_pct`, `bb_pct`, `hr`, `hits`, `sb` |
| MLB Stats API — schedule / starters | `GET /schedule?hydrate=probablePitcher` | 1 hour | `home_team`, `away_team`, `home_starter`, `away_starter` |
| MLB Stats API — lineups | `GET /schedule?hydrate=lineups` | 0.5 hours | `team`, `player_name`, `batting_position` |
| MLB Stats API — live game state | `GET /game/{gamePk}/feed/live` | real-time | score, inning, current pitcher |
| The Odds API | `GET /v4/sports/baseball_mlb/odds` | 2 hours | moneylines (H2H), run lines, totals, player props |

The team stats DataFrame is built by merging MLB Stats API batting + pitching + records on `team_id` (integer):

```
team_stats = records ⋈ batting ⋈ pitching   (joined on team_id, not team name)
team_stats["run_diff"] = runs_scored - runs_allowed
```

**Why team_id instead of team name:** The MLB Stats API standings endpoint returns short names (e.g., "Yankees") while the stats endpoint returns full names (e.g., "New York Yankees"). Merging on the integer `team_id` avoids all name-format mismatches.

---

## 2. Joe Peta's Core Framework

Joe Peta's *Trading Bases* argues that the MLB market misprices teams because it reacts to **narrative and recent results** instead of **underlying true talent**. The two main tools he uses to identify mispriced teams are:

1. **Pythagorean Win Expectation** — separates lucky win-loss records from true quality
2. **FIP and BABIP regression** — separates lucky pitcher stats from true talent

The app implements both, then combines them into a single win probability that is compared against the Vegas implied probability to find edge.

---

## 3. Game Winner Model — Step by Step

### Step 1: Pythagorean Win Expectation + Regression to the Mean

**Source file:** [src/models/pythagorean.py](../../src/models/pythagorean.py)

**What it is:** Derived from Bill James (Pythagenpat variant), used by Peta as the foundational team quality signal. A team's actual W-L record contains noise from one-run games, walk-off luck, and bullpen sequencing. Runs scored and runs allowed measure true offensive and defensive quality more reliably over a full season.

**Formula:**
```
pyth_win_pct = RS^1.83 / (RS^1.83 + RA^1.83)
```
Exponent `1.83` (Pythagenpat) is empirically more accurate than the original 2.0 across modern MLB data.

**Regression to the mean (early-season adjustment):**
```
MIN_GAMES = 20     # Threshold before regression activates, per team
SHRINKAGE_K = 30   # Regresses 50% at 30 games, 73% at 81 games, 84% at 162 games

weight = G / (G + 30)

adj_RS_per_game = weight * (RS/G) + (1 - weight) * league_avg_RS_per_game
adj_RA_per_game = weight * (RA/G) + (1 - weight) * league_avg_RA_per_game
```

When a team has fewer than 20 games, raw RS/RA is used (small sample is too noisy to regress reliably). Once 20 games are played, the shrinkage blend activates automatically and the app displays a formula-change banner.

**Data plugged in:**
- `RS` = `team_stats["runs_scored"]` — season total runs scored (MLB Stats API)
- `RA` = `team_stats["runs_allowed"]` — season total runs allowed (MLB Stats API)
- `G` = `wins + losses` — games played (MLB Stats API standings)
- `league_avg_RS_per_game`, `league_avg_RA_per_game` — computed live as mean across all teams

**Signal derived:**
```
pyth_deviation = actual_win_pct - pyth_win_pct

pyth_deviation > +0.05  →  "Overperforming"  →  likely to regress, fade on moneyline
pyth_deviation < -0.05  →  "Underperforming" →  likely to improve, back on moneyline
```

**Severity:**
- Low: deviation 0.05–0.10
- High: deviation > 0.10

---

### Step 2: Pitcher Regression Signals

**Source file:** [src/models/regression_signals.py](../../src/models/regression_signals.py)

These are Peta's pitcher-level signals. ERA is a noisy stat. FIP and BABIP reveal when ERA is lying.

#### FIP-ERA Gap

**What it is:** FIP (Fielding Independent Pitching) strips ERA down to only the three things a pitcher directly controls — home runs, walks, and strikeouts. When ERA and FIP diverge, ERA is almost always the variable that is wrong.

**Formula:**
```
FIP = (13 * HR + 3 * BB - 2 * K) / IP + constant
     (constant calibrated so league-avg FIP ≈ league-avg ERA, typically ~3.10)

fip_era_gap = fip - era
```

**Data plugged in:**
- `fip` = `pitcher_stats["fip"]` — FanGraphs FIP (season to date)
- `era` = `pitcher_stats["era"]` — FanGraphs ERA (season to date)

**Signal:**
```
fip_era_gap >= +0.75  →  ERA is artificially low, expect ERA to rise  →  fade this pitcher
fip_era_gap <= -0.75  →  ERA is artificially high, expect ERA to fall  →  back this pitcher

Severity:
  Low:    gap 0.75–1.12
  High:   gap >= 1.12  (1.5× threshold)
```

#### BABIP Deviation

**What it is:** BABIP (Batting Average on Balls in Play) is largely outside a pitcher's control. Defense, park dimensions, and sequencing randomness drive most of the variation. League average BABIP is approximately .300.

**Formula:**
```
BABIP = (H - HR) / (AB - K - HR + SF)

babip_deviation = pitcher_babip - 0.300
```

**Data plugged in:**
- `babip` = `pitcher_stats["babip"]` — FanGraphs BABIP (season to date)

**Signal:**
```
babip > 0.320  →  pitcher allowing too many balls to drop, defense/luck issue
                →  ERA will drop toward FIP (back this pitcher)

babip < 0.275  →  pitcher has been artificially suppressing hits
                →  ERA will rise as BABIP normalizes (fade this pitcher)

Severity mapping:
  babip_deviation >= 0.020  →  Low
  babip_deviation >= 0.040  →  High
```

When both FIP-ERA and BABIP signal the same direction, the combined severity is escalated and the `signal_notes` field captures both.

---

### Step 3: Win Probability Composite

**Source file:** [src/models/win_probability.py](../../src/models/win_probability.py)

This function combines all signals into a per-game win probability using the full Peta + sabermetrics pipeline.

**Full formula:**
```
# --- League avg FIP (IP-weighted) ---
league_avg_fip = Σ(pitcher_fip × pitcher_ip) / Σ(pitcher_ip)
               (fallback: 4.00 if data unavailable)

# --- Step A: Log5 head-to-head base probability ---
# Bill James Log5: true matchup probability given each team's overall quality
home_pyth = pythagorean_win_pct(home_RS, home_RA)   # after regression if G >= 20
away_pyth = pythagorean_win_pct(away_RS, away_RA)

home_log5 = (home_pyth - home_pyth * away_pyth) / (home_pyth + away_pyth - 2 * home_pyth * away_pyth)

# --- Step B: Lineup quality matchup adjustment ---
# Opposing lineup's avg OPS vs league avg (0.720) adjusts starter's effective FIP
# Only applied when lineups are posted; otherwise effective_fip = actual fip
lineup_adj_home_fip = (away_lineup_avg_ops - 0.720) * 3.0   # added to home starter FIP
lineup_adj_away_fip = (home_lineup_avg_ops - 0.720) * 3.0   # added to away starter FIP

effective_home_fip = home_starter_fip + lineup_adj_home_fip
effective_away_fip = away_starter_fip + lineup_adj_away_fip

# --- Step C: FIP pitcher adjustments ---
home_pitch_adj = (league_avg_fip - effective_home_fip) × 0.03
away_pitch_adj = (league_avg_fip - effective_away_fip) × 0.03

# --- Step D: Home field and stack ---
home_prob_raw = home_log5 + home_pitch_adj + 0.04 - away_pitch_adj
away_prob_raw = (1 - home_log5) + away_pitch_adj - home_pitch_adj

# --- Step E: Renormalize and clamp ---
home_prob = home_prob_raw / (home_prob_raw + away_prob_raw)
away_prob = away_prob_raw / (home_prob_raw + away_prob_raw)

home_model_prob = clamp(home_prob, 0.30, 0.70)
away_model_prob = clamp(away_prob, 0.30, 0.70)
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `home_RS`, `home_RA` | MLB Stats API (regression-adjusted if G ≥ 20) | `runs_scored`, `runs_allowed` |
| `home_starter_fip` | pybaseball FanGraphs leaderboard | `pitcher_stats["fip"]` matched by starter name |
| `away_starter_fip` | pybaseball FanGraphs leaderboard | `pitcher_stats["fip"]` matched by starter name |
| `home_lineup_avg_ops` | MLB Stats API lineups + batter stats | `obp + slg` averaged across posted lineup |
| `away_lineup_avg_ops` | MLB Stats API lineups + batter stats | `obp + slg` averaged across posted lineup |
| `league_avg_fip` | Computed from all pitchers | IP-weighted mean of `pitcher_stats["fip"]` |

**Constants:**
- `HOME_FIELD_ADJ = 0.04` — 4% home field advantage
- `FIP_ADJ_PER_POINT = 0.03` — each FIP point from league average shifts win prob by 3%
- `LINEUP_OPS_SCALE = 3.0` — each 0.010 OPS above average adds 0.030 to opposing pitcher's effective FIP
- `LEAGUE_AVG_OPS = 0.720` — MLB baseline; OPS used as proxy for wOBA (not available from MLB Stats API)

**Why Log5 instead of normalize(A+B):**
The old approach of adding two independent Pythagorean W%s and normalizing does not account for the *relative* quality of the two teams. Log5 is the correct Bill James formula for deriving matchup probability from two teams' overall quality levels. A .600 team vs. a .400 team produces 0.692 (Log5) vs. 0.600 (old normalize), which is the empirically correct value.

---

### Step 4: Edge Calculation Against Vegas

**Source file:** [src/data/odds.py](../../src/data/odds.py), [src/models/kelly.py](../../src/models/kelly.py)

**Vig removal:**
```
home_decimal = american_to_decimal(home_american_odds)
away_decimal = american_to_decimal(away_american_odds)

raw_home = 1 / home_decimal
raw_away = 1 / away_decimal

# Normalise both sides to remove the book's juice
home_implied_prob = raw_home / (raw_home + raw_away)
away_implied_prob = raw_away / (raw_home + raw_away)
```

This is computed inside `compute_kelly()` when `opponent_american_odds` is passed. Without vig removal, both sides of a -110/-110 market imply 52.4% — impossible since probabilities must sum to 100%. After removal each side correctly implies 50.0%.

**Edge:**
```
home_edge = home_model_prob - home_implied_prob
away_edge = away_model_prob - away_implied_prob
```

**Kelly sizing (half-Kelly):**
```
b = decimal_odds - 1.0          # profit per $1 risked
p = model_prob                  # model's estimated win probability
q = 1 - p

kelly_full = (b × p - q) / b
kelly_half = kelly_full × 0.5   # half-Kelly to reduce variance
stake = min(kelly_half, 0.05)   # cap at 5% of bankroll

Only recommend if: edge >= 0.03 (3%)
```

---

## 4. Player Props — Step by Step

**Source file:** [src/models/player_props.py](../../src/models/player_props.py)

All props follow the same final step: compare model projection to the sportsbook line, compute edge percent, and flag bets where edge >= 5%.

```
distance = model_projection - prop_line

edge_pct (OVER) = (distance / prop_line) × 100
edge_pct (UNDER) = (-distance / prop_line) × 100

recommendation = "BET" if edge_pct >= 5.0 else "PASS"
```

---

### Pitcher Strikeouts

**Formula:**
```
blended_k_rate = pitcher_k_pct × 0.60 + opponent_k_pct × 0.40
batters_faced  = ip_per_start × 4.3   (league avg batters per inning)
projected_k    = blended_k_rate × batters_faced × umpire_zone_factor
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `pitcher_k_pct` | FanGraphs pitcher leaderboard | `pitcher_stats["k_pct"]` |
| `ip_per_start` | FanGraphs pitcher leaderboard | `pitcher_stats["ip"] / 30` |
| `opponent_k_pct` | FanGraphs batter leaderboard | team-level `k_pct` (batter side) |
| `umpire_zone_factor` | Not yet wired; defaults to `1.0` | — |

**Blending logic (Peta-derived):** The 60/40 split reflects that pitcher skill is the dominant factor in strikeouts (~60%), but the opposing lineup's strikeout tendency is a meaningful adjustment (~40%). A soft-contact lineup facing a high-whiff pitcher will produce more Ks than either stat alone implies.

**Additional signal:** `whiff_pct` from Baseball Savant (via pybaseball) is displayed alongside the K% as a sustainability check — a pitcher with high K% but low whiff% is more likely to regress.

---

### Pitcher Earned Runs

**Formula:**
```
blended_era = pitcher_fip × 0.40 + pitcher_xfip × 0.60
projected_er = (blended_era / 9.0) × innings_projected × park_factor
```

If `xfip` is unavailable, `blended_era = pitcher_fip`.

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `pitcher_fip` | FanGraphs pitcher leaderboard | `pitcher_stats["fip"]` |
| `pitcher_xfip` | FanGraphs pitcher leaderboard | `pitcher_stats["xfip"]` |
| `innings_projected` | Passed from UI or default | typically 5.0–6.0 |
| `park_factor` | Not yet wired; defaults to `1.0` | — |

**Why xFIP gets more weight (60%):** xFIP replaces the pitcher's actual HR total with the expected HR based on their fly ball rate. This normalizes for ballpark HR factors and sequences of lucky/unlucky HR allowed. xFIP is the more stable of the two estimators for forward-looking projection.

---

### Batter Hits

**Formula:**
```
regressed_babip = batter_babip × 0.60 + 0.300 × 0.40

# Optional pitcher adjustment
if pitcher_babip_allowed is known:
    regressed_babip = regressed_babip × 0.70 + pitcher_babip_allowed × 0.30

balls_in_play  = at_bats_projected × (1 - batter_k_pct) × 0.96
projected_hits = regressed_babip × balls_in_play
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `batter_babip` | FanGraphs batter leaderboard | `batter_stats["babip"]` |
| `at_bats_projected` | FanGraphs batter leaderboard | `batter_stats["ab"] / 140` (per game estimate) |
| `batter_k_pct` | FanGraphs batter leaderboard | `batter_stats["k_pct"]` |
| `pitcher_babip_allowed` | FanGraphs pitcher leaderboard | `pitcher_stats["babip"]` (optional) |

**Core Peta insight:** A batter's BABIP over any rolling window contains significant luck. Regressing 40% toward .300 corrects for that variance. If a batter is hitting .350 BABIP over the last month, their hits prop is likely priced off that inflated BABIP. The model projects off the regressed estimate, which will be lower — favoring UNDER.

---

### Batter Total Bases

**Formula:**
```
adjusted_slg = batter_slg × (1 + (park_hr_factor - 1) × 0.30)
projected_tb = adjusted_slg × at_bats_projected
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `batter_slg` | FanGraphs batter leaderboard | `batter_stats["slg"]` |
| `at_bats_projected` | Derived | `batter_stats["ab"] / 140` |
| `park_hr_factor` | Not yet wired; defaults to `1.0` | — |

The park factor adjustment only applies to the HR component (~30% of SLG), leaving singles, doubles, and triples unaffected.

---

### Batter Home Runs

**Formula:**
```
hr_per_fb     = barrel_pct × 0.55    (empirical: ~55% of barrels become HRs)
fly_balls     = at_bats_projected × fb_pct
projected_hr  = fly_balls × hr_per_fb × park_hr_factor
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `barrel_pct` | FanGraphs batter leaderboard | `batter_stats["barrel_pct"]` |
| `fb_pct` | FanGraphs batter leaderboard | Not in batter fetch; partially through SLG proxy |
| `park_hr_factor` | Not yet wired; defaults to `1.0` | — |

---

## 5. Preseason Win Totals

**Source file:** [src/models/preseason.py](../../src/models/preseason.py)

**Formula:**
```
pyth_win_pct     = RS² / (RS² + RA²)   (prior season)
regressed_win_pct = pyth_win_pct × 0.80 + 0.500 × 0.20
projected_wins   = regressed_win_pct × 162 + war_adjustment

edge_vs_vegas    = projected_wins - vegas_line
recommendation:
  edge >= +2.0  →  BET OVER
  edge <= -2.0  →  BET UNDER
```

**Data plugged in:**
- Prior season `runs_scored` and `runs_allowed` from FanGraphs team batting/pitching
- `war_adjustment` — projected net WAR change from roster moves (entered manually if available)
- `vegas_line` — entered manually from the sportsbook

**Why 20% regression toward .500 (Peta):** No team repeats its exact run environment. Roster turnover, health variance, and schedule strength all push teams toward the mean. Peta's research found that 20% regression to .500 applied to Pythagorean W% (not actual W%) is the most accurate single-formula preseason predictor.

---

## 6. AI Reasoning Layer

**Source file:** [src/shared/groq_agent.py](../../src/shared/groq_agent.py)

After the model computes edge and Kelly stake, the Groq agent (Llama 3.3 70B) receives the full signal packet and returns a plain-language explanation.

**What is sent to the model:**
- Home and away team names
- Model win probabilities (home_model_prob, away_model_prob)
- Vegas implied probabilities (vig-removed)
- Edge percent
- Pythagorean deviation for each team
- Starter FIP, xFIP, SIERA, BABIP for each pitcher
- FIP-ERA gap and BABIP signal (if present)
- Signal severity and direction

**What is returned:**
- 2–4 sentence explanation of why the edge exists
- Confidence: Low / Medium / High (based on how many signals converge)
- Key risk: one caveat to consider (injury, bullpen, etc.)

---

## 7. Signal Severity Summary

| Signals Present | Min Edge | Severity | Recommendation |
|---|---|---|---|
| 1 signal | 3–6% | Low | Display only, no bet flag |
| 2 signals converging | 6–10% | Medium | BET flag |
| 3+ signals converging | > 10% | High | Strong BET flag |

---

## 8. Known Gaps (Things Not Yet Wired)

These appear in the code as constants or defaults and represent planned improvements:

| Gap | Current State | Impact |
|---|---|---|
| Park factors | Static lookup table per ballpark | HR/total models use park factor; not dynamically adjusted for roof/weather |
| Umpire zone factor | Hardcoded `1.0` | K prop projections don't adjust for ump tendencies |
| Weather (MLB) | Not yet implemented | Wind and temp affect totals significantly |
| Bullpen fatigue | Not yet implemented | Heavy prior-day usage not captured |
| Injury data | Manual only | Scratched starters not auto-detected |
| Sprint speed / stolen base props | Data fetched, model not built | SB props not available |
| Opponent k_pct for pitcher Ks | Falls back to league avg (0.228) if not passed | Reduces precision on K props |
| wOBA / wRC+ for lineup matchup | OPS used as proxy (r≈0.97 with wOBA) | MLB Stats API does not expose wOBA; FanGraphs batter endpoint returns 403 |
