# Baseball (MLB) — Technical Specifications

This document describes exactly what data the app pulls, how it maps to Joe Peta's *Trading Bases* methodology, and how every number flows into the formulas used to generate game winner predictions and player prop recommendations.

---

## 1. Data Sources and What Is Fetched

All baseball data comes from **pybaseball**, which wraps FanGraphs, Baseball Savant, and Baseball Reference. Odds data comes from **The Odds API**. All data is disk-cached in Parquet files to reduce API calls.

| Source | pybaseball Call | Cache TTL | Columns Used |
|---|---|---|---|
| FanGraphs (team batting) | `team_batting(season)` | 6 hours | `team`, `games`, `runs_scored` |
| FanGraphs (team pitching) | `team_pitching(season)` | 6 hours | `team`, `runs_allowed`, `era`, `fip` |
| Baseball Reference (standings) | `standings(season)` | 6 hours | `team`, `wins`, `losses`, `win_pct` |
| FanGraphs (pitcher leaderboard) | `pitching_stats(season, qual=1)` | 6 hours | `name`, `team`, `ip`, `era`, `fip`, `xfip`, `siera`, `babip`, `k_pct`, `bb_pct`, `hr9`, `whiff_pct`, `gb_pct`, `fb_pct` |
| FanGraphs (batter leaderboard) | `batting_stats(season, qual=100)` | 6 hours | `name`, `team`, `pa`, `avg`, `obp`, `slg`, `wrc_plus`, `babip`, `k_pct`, `bb_pct`, `hard_hit_pct`, `barrel_pct`, `exit_velocity`, `hr`, `hits`, `ab`, `sb` |
| The Odds API | `GET /v4/sports/baseball_mlb/odds` | 2 hours | moneylines (H2H), run lines, totals, player props |

The team stats DataFrame is built by merging batting + pitching + records on `team`:

```
team_stats = records ⋈ batting ⋈ pitching
team_stats["run_diff"] = runs_scored - runs_allowed
```

---

## 2. Joe Peta's Core Framework

Joe Peta's *Trading Bases* argues that the MLB market misprices teams because it reacts to **narrative and recent results** instead of **underlying true talent**. The two main tools he uses to identify mispriced teams are:

1. **Pythagorean Win Expectation** — separates lucky win-loss records from true quality
2. **FIP and BABIP regression** — separates lucky pitcher stats from true talent

The app implements both, then combines them into a single win probability that is compared against the Vegas implied probability to find edge.

---

## 3. Game Winner Model — Step by Step

### Step 1: Pythagorean Win Expectation

**Source file:** [src/models/pythagorean.py](../../src/models/pythagorean.py)

**What it is:** Derived from Bill James, used by Peta as the foundational team quality signal. A team's actual W-L record contains noise from one-run games, walk-off luck, and bullpen sequencing. Runs scored and runs allowed measure true offensive and defensive quality more reliably over a full season.

**Formula:**
```
pyth_win_pct = RS² / (RS² + RA²)
```

**Data plugged in:**
- `RS` = `team_stats["runs_scored"]` — season total runs scored (from FanGraphs team batting)
- `RA` = `team_stats["runs_allowed"]` — season total runs allowed (from FanGraphs team pitching)
- Exponent = `2.0` (Bill James canonical; 1.83 is empirically slightly better but 2.0 is used)

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

This function takes the Pythagorean base and layers in the starting pitcher adjustment and home field advantage to produce a single per-game win probability.

**Full formula:**
```
league_avg_fip = IP-weighted mean of all pitchers' FIP
               = Σ(pitcher_fip × pitcher_ip) / Σ(pitcher_ip)
               (fallback: 4.00 if data unavailable)

home_pitch_adj = (league_avg_fip - home_starter_fip) × 0.03
away_pitch_adj = (league_avg_fip - away_starter_fip) × 0.03

home_prob_raw = home_pyth_win_pct + home_pitch_adj + 0.04 - away_pitch_adj
away_prob_raw = away_pyth_win_pct + away_pitch_adj - home_pitch_adj

# Normalize to sum to 1.0
home_prob = home_prob_raw / (home_prob_raw + away_prob_raw)
away_prob = away_prob_raw / (home_prob_raw + away_prob_raw)

# Clamp to avoid extreme probabilities
home_model_prob = clamp(home_prob, 0.05, 0.95)
away_model_prob = clamp(away_prob, 0.05, 0.95)
```

**Data plugged in:**

| Variable | Source | Column |
|---|---|---|
| `home_pyth_win_pct` | Pythagorean model (Step 1) | `pyth_win_pct` for home team |
| `away_pyth_win_pct` | Pythagorean model (Step 1) | `pyth_win_pct` for away team |
| `home_starter_fip` | FanGraphs pitcher leaderboard | `pitcher_stats["fip"]` matched by starter name |
| `away_starter_fip` | FanGraphs pitcher leaderboard | `pitcher_stats["fip"]` matched by starter name |
| `league_avg_fip` | Computed from all pitchers | IP-weighted mean of `pitcher_stats["fip"]` |

**Constants:**
- `HOME_FIELD_ADJ = 0.04` — 4% home field advantage
- `FIP_ADJ_PER_POINT = 0.03` — each FIP point from league average shifts win prob by 3%

**Interpretation:**
- A pitcher with FIP of 3.00 against a league average of 4.00 adds `(4.00 - 3.00) × 0.03 = +3%` win probability to their team.
- A pitcher with FIP of 5.00 subtracts `(4.00 - 5.00) × 0.03 = -3%` from their team's win probability.

---

### Step 4: Edge Calculation Against Vegas

**Source file:** [src/data/odds.py](../../src/data/odds.py), [src/models/kelly.py](../../src/models/kelly.py)

**Vig removal:**
```
raw_home_implied = |home_american_odds| / (|home_american_odds| + 100)  # if negative odds
raw_away_implied = |away_american_odds| / (|away_american_odds| + 100)

vig = raw_home_implied + raw_away_implied - 1.0
home_implied_prob = raw_home_implied / (raw_home_implied + raw_away_implied)
away_implied_prob = raw_away_implied / (raw_home_implied + raw_away_implied)
```

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
| Park factors | Hardcoded `1.0` | HR and total models are park-neutral |
| Umpire zone factor | Hardcoded `1.0` | K prop projections don't adjust for ump |
| Weather (MLB) | Not yet implemented | Wind and temp affect totals significantly |
| Bullpen fatigue | Not yet implemented | Affects game totals, run line |
| Injury/lineup data | Manual only | Starter must be verified before betting |
| Sprint speed / stolen base props | Data fetched, model not built | SB props not available |
| Opponent k_pct (team-level) for pitcher Ks | Falls back to league avg (0.228) if not passed | Reduces precision on K props |
