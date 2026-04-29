"""Fetch completed MLB game scores from the MLB Stats API (free, no key required)."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import requests

from src.data.cache import cached

MLB_API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 10

# Odds API full name → MLB Stats API team name mapping
TEAM_NAME_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Athletics": "Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
}


def get_games_for_date(date: datetime.date) -> pd.DataFrame:
    """
    Return completed game scores for a given date.

    Returns DataFrame with columns:
      game_date, home_team, away_team, home_score, away_score, winner, status
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_results_{date_str}"
    # Past dates: cache 24h (scores don't change). Today: cache 5 min so finalized games are caught quickly.
    ttl = 24.0 if date < datetime.date.today() else 0.083

    def fetch() -> pd.DataFrame:
        url = f"{MLB_API}/schedule"
        params = {
            "sportId": 1,
            "date": date_str,
            "hydrate": "linescore",
        }
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    continue
                teams = game.get("teams", {})
                home = teams.get("home", {})
                away = teams.get("away", {})
                home_name = home.get("team", {}).get("name", "")
                away_name = away.get("team", {}).get("name", "")
                home_score = home.get("score")
                away_score = away.get("score")
                if home_score is None or away_score is None:
                    continue
                winner = home_name if home_score > away_score else away_name
                rows.append({
                    "game_date": date_str,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "winner": winner,
                    "status": status,
                })
        return pd.DataFrame(rows)

    return cached(cache_key, fetch, ttl_hours=ttl)


def get_probable_starters(date: datetime.date) -> pd.DataFrame:
    """
    Fetch probable starting pitchers for a given date from the MLB Stats API.

    Returns DataFrame with columns:
      home_team, away_team, home_starter, away_starter,
      home_starter_announced, away_starter_announced
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_starters_{date_str}"

    def fetch() -> pd.DataFrame:
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                home_info = teams.get("home", {})
                away_info = teams.get("away", {})
                home_team = home_info.get("team", {}).get("name", "")
                away_team = away_info.get("team", {}).get("name", "")
                home_pitcher = home_info.get("probablePitcher", {})
                away_pitcher = away_info.get("probablePitcher", {})
                rows.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_starter": home_pitcher.get("fullName") if home_pitcher else None,
                    "away_starter": away_pitcher.get("fullName") if away_pitcher else None,
                    "home_starter_announced": bool(home_pitcher),
                    "away_starter_announced": bool(away_pitcher),
                })
        return pd.DataFrame(rows)

    # Short TTL — starters get announced throughout the day
    return cached(cache_key, fetch, ttl_hours=1.0)


def get_today_lineups(date: datetime.date) -> pd.DataFrame:
    """
    Fetch confirmed starting lineups from the MLB Stats API.

    Returns DataFrame with columns: team, player_name, batting_position
    Returns empty DataFrame if lineups have not been posted yet.
    Lineups are typically posted 1–2 hours before first pitch.
    """
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"mlb_lineups_{date_str}"

    def fetch() -> pd.DataFrame:
        try:
            resp = requests.get(
                f"{MLB_API}/schedule",
                params={"sportId": 1, "date": date_str, "hydrate": "lineups"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        rows = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                teams = game.get("teams", {})
                home_name = teams.get("home", {}).get("team", {}).get("name", "")
                away_name = teams.get("away", {}).get("team", {}).get("name", "")

                # Try game-level lineups (newer API response format)
                game_lineups = game.get("lineups", {})
                if game_lineups:
                    for side_key, team_name in [("homePlayers", home_name), ("awayPlayers", away_name)]:
                        for i, player in enumerate(game_lineups.get(side_key, []), 1):
                            name = player.get("fullName") or player.get("name", "")
                            if name:
                                rows.append({"team": team_name, "player_name": name, "batting_position": i})
                else:
                    # Try team-level lineups (older format)
                    for side, team_name in [("home", home_name), ("away", away_name)]:
                        team_info = teams.get(side, {})
                        batting_order = team_info.get("lineups", {}).get("battingOrder", [])
                        for i, player in enumerate(batting_order, 1):
                            name = player.get("fullName") or player.get("name", "")
                            if name:
                                rows.append({"team": team_name, "player_name": name, "batting_position": i})

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return cached(cache_key, fetch, ttl_hours=0.5)


def get_live_games() -> pd.DataFrame:
    """
    Return any MLB games currently in progress with their current pitcher.

    Returns DataFrame with columns:
      game_pk, home_team, away_team, inning, inning_half,
      home_current_pitcher, away_current_pitcher,
      home_score, away_score, status
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={"sportId": 1, "date": today, "hydrate": "linescore"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Live":
                continue
            teams  = game.get("teams", {})
            ls     = game.get("linescore", {})
            gid    = game["gamePk"]
            home   = teams.get("home", {}).get("team", {}).get("name", "")
            away   = teams.get("away", {}).get("team", {}).get("name", "")
            rows.append({
                "game_pk":   gid,
                "home_team": home,
                "away_team": away,
                "inning":    ls.get("currentInning"),
                "inning_half": ls.get("inningHalf", ""),
                "home_score": ls.get("teams", {}).get("home", {}).get("runs"),
                "away_score": ls.get("teams", {}).get("away", {}).get("runs"),
                "status":    status,
            })

    if not rows:
        return pd.DataFrame()

    live_df = pd.DataFrame(rows)

    # Fetch current pitcher for each live game from the live feed
    home_pitchers, away_pitchers = [], []
    for _, row in live_df.iterrows():
        try:
            feed = requests.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{row['game_pk']}/feed/live",
                timeout=TIMEOUT,
            ).json()
            box    = feed["liveData"]["boxscore"]["teams"]
            def _current(side: str) -> str | None:
                pitchers = box[side]["pitchers"]
                if not pitchers:
                    return None
                players = box[side]["players"]
                pid = pitchers[-1]
                return players.get(f"ID{pid}", {}).get("person", {}).get("fullName")
            home_pitchers.append(_current("home"))
            away_pitchers.append(_current("away"))
        except Exception:
            home_pitchers.append(None)
            away_pitchers.append(None)

    live_df["home_current_pitcher"] = home_pitchers
    live_df["away_current_pitcher"] = away_pitchers
    return live_df


def get_live_game_state(game_pk: int) -> dict | None:
    """
    Fetch rich live state for a single in-progress game.

    Returns dict with:
      inning, inning_half, outs, balls, strikes,
      home_score, away_score,
      bases: {first, second, third}  — True if occupied
      home_starter, away_starter,
      home_pitcher: {name, pc, h, k, er, ip, is_starter}
      away_pitcher: {name, pc, h, k, er, ip, is_starter}
      pitching_side: 'home' | 'away'   (who is currently pitching)
      batter: {name, summary}           (current batter)
      on_deck: str | None
    Returns None on any fetch failure.
    """
    try:
        feed = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=TIMEOUT,
        ).json()
        ld  = feed["liveData"]
        ls  = ld["linescore"]
        box = ld["boxscore"]["teams"]

        offense = ls.get("offense", {})
        defense = ls.get("defense", {})

        # Inning half tells us who is pitching
        inning_half = ls.get("inningHalf", "Top")
        pitching_side  = "home" if inning_half.lower() == "top"  else "away"
        batting_side   = "away" if pitching_side == "home" else "home"

        def _pitcher_info(side: str) -> dict:
            pitchers = box[side]["pitchers"]
            players  = box[side]["players"]
            if not pitchers:
                return {}
            starter_id  = pitchers[0]
            current_id  = pitchers[-1]
            p = players.get(f"ID{current_id}", {})
            st = players.get(f"ID{starter_id}", {})
            stats = p.get("stats", {}).get("pitching", {})
            return {
                "name":       p.get("person", {}).get("fullName", ""),
                "starter":    st.get("person", {}).get("fullName", ""),
                "is_starter": starter_id == current_id,
                "pc":         stats.get("pitchesThrown", 0),
                "h":          stats.get("hits", 0),
                "k":          stats.get("strikeOuts", 0),
                "er":         stats.get("earnedRuns", 0),
                "ip":         stats.get("inningsPitched", "0.0"),
            }

        def _batter_info() -> dict:
            batter_obj = offense.get("batter", {})
            batter_id  = batter_obj.get("id")
            name       = batter_obj.get("fullName", "")
            summary    = ""
            if batter_id:
                for side in ("home", "away"):
                    key = f"ID{batter_id}"
                    if key in box[side]["players"]:
                        summary = (box[side]["players"][key]
                                   .get("stats", {}).get("batting", {})
                                   .get("summary", ""))
                        break
            return {"name": name, "summary": summary}

        gd      = feed.get("gameData", {})
        weather = gd.get("weather", {})
        venue   = gd.get("venue", {})
        fi      = venue.get("fieldInfo", {})

        def _team_hitting(side: str) -> dict:
            ts = box[side].get("teamStats", {}).get("batting", {})
            return {
                "hits": ts.get("hits", 0),
                "runs": ts.get("runs", 0),
                "rbi":  ts.get("rbi", 0),
            }

        def _batter_info() -> dict:
            batter_obj = offense.get("batter", {})
            batter_id  = batter_obj.get("id")
            name       = batter_obj.get("fullName", "")
            stats: dict = {}
            if batter_id:
                for side in ("home", "away"):
                    key = f"ID{batter_id}"
                    if key in box[side]["players"]:
                        bs = box[side]["players"][key].get("stats", {}).get("batting", {})
                        stats = {
                            "summary":  bs.get("summary", ""),
                            "hits":     bs.get("hits", 0),
                            "doubles":  bs.get("doubles", 0),
                            "triples":  bs.get("triples", 0),
                            "hr":       bs.get("homeRuns", 0),
                            "rbi":      bs.get("rbi", 0),
                            "runs":     bs.get("runs", 0),
                            "k":        bs.get("strikeOuts", 0),
                            "ab":       bs.get("atBats", 0),
                        }
                        break
            return {"name": name, **stats}

        return {
            "inning":       ls.get("currentInning", "?"),
            "inning_half":  inning_half,
            "outs":         ls.get("outs", 0),
            "balls":        ls.get("balls", 0),
            "strikes":      ls.get("strikes", 0),
            "home_score":   ls["teams"]["home"]["runs"],
            "away_score":   ls["teams"]["away"]["runs"],
            "home_hits":    ls["teams"]["home"].get("hits", 0),
            "away_hits":    ls["teams"]["away"].get("hits", 0),
            "home_team_batting": _team_hitting("home"),
            "away_team_batting": _team_hitting("away"),
            "bases": {
                "first":  bool(offense.get("first")),
                "second": bool(offense.get("second")),
                "third":  bool(offense.get("third")),
            },
            "home_pitcher":  _pitcher_info("home"),
            "away_pitcher":  _pitcher_info("away"),
            "pitching_side": pitching_side,
            "batting_side":  batting_side,
            "batter":        _batter_info(),
            "on_deck":       offense.get("onDeck", {}).get("fullName"),
            "weather": {
                "condition": weather.get("condition", ""),
                "temp":      weather.get("temp", ""),
                "wind":      weather.get("wind", ""),
                "roof":      fi.get("roofType", ""),
            },
            "venue_name":    venue.get("name", ""),
        }
    except Exception:
        return None


def get_results_for_date_range(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Fetch and combine results for a range of dates."""
    all_rows = []
    current = start
    while current <= end:
        df = get_games_for_date(current)
        if not df.empty:
            all_rows.append(df)
        current += datetime.timedelta(days=1)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def verify_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each unverified prediction, fetch the actual game result and mark correct/incorrect.
    Returns updated predictions_df with actual_winner and correct columns filled in.
    """
    if predictions_df.empty:
        return predictions_df

    df = predictions_df.copy()
    today = datetime.date.today()

    # Include rows needing ML verification, missing scores, or RL/O/U re-verification.
    # The scores condition catches rows verified before actual_home/away_score columns existed.
    scores_missing = (
        df["actual_winner"].notna() &
        (df["actual_winner"] != "Postponed") &
        (df["actual_home_score"].isna() | df["actual_away_score"].isna())
    )
    needs_verify = df[
        df["actual_winner"].isna() |
        scores_missing |
        (df["rl_side"].notna() & df["rl_correct"].isna()) |
        (df["total_direction"].notna() & df["total_correct"].isna())
    ].copy()
    if needs_verify.empty:
        return df

    # Only verify past games (not today's)
    needs_verify["_date"] = pd.to_datetime(needs_verify["prediction_date"]).dt.date
    past = needs_verify[needs_verify["_date"] < today]

    if past.empty:
        return df

    # Fetch results for all relevant dates
    min_date = past["_date"].min()
    results = get_results_for_date_range(min_date, today - datetime.timedelta(days=1))

    if results.empty:
        return df

    verified_at = datetime.datetime.now().isoformat()

    for idx, pred_row in past.iterrows():
        game_date = str(pred_row["prediction_date"])
        home = pred_row["home_team"]
        away = pred_row["away_team"]
        home_kw = home.split()[-1]
        away_kw = away.split()[-1]

        # Both teams must appear somewhere in the result row (either column)
        match = results[
            (results["game_date"] == game_date) &
            (
                results["home_team"].str.contains(home_kw, case=False, na=False) |
                results["away_team"].str.contains(home_kw, case=False, na=False)
            ) &
            (
                results["home_team"].str.contains(away_kw, case=False, na=False) |
                results["away_team"].str.contains(away_kw, case=False, na=False)
            )
        ]

        if match.empty:
            # No Final result exists — game was postponed or cancelled
            df.at[idx, "actual_winner"] = "Postponed"
            df.at[idx, "correct"] = -1
            df.at[idx, "verified_at"] = verified_at
            rl_mark = -1 if pd.notna(pred_row.get("rl_side")) else None
            total_mark = -1 if pd.notna(pred_row.get("total_direction")) else None
            if rl_mark is not None:
                df.at[idx, "rl_correct"] = rl_mark
            if total_mark is not None:
                df.at[idx, "total_correct"] = total_mark
            from src.data.predictions_db import update_result
            update_result(
                game_date, home, away, "Postponed", -1, verified_at,
                rl_correct=rl_mark,
                total_correct=total_mark,
            )
            continue

        result_row = match.iloc[0]
        actual_winner = result_row["winner"]
        predicted_winner = pred_row["predicted_winner"]
        correct = int(predicted_winner in actual_winner or actual_winner in predicted_winner)

        home_score = int(result_row["home_score"])
        away_score = int(result_row["away_score"])
        run_diff = home_score - away_score  # positive = home won

        # ── Run line result (±1.5) ────────────────────────────────────────────
        rl_side = pred_row.get("rl_side")
        rl_correct: int | None = None
        if rl_side == "HOME":
            # Home -1.5: home must win by 2+
            rl_correct = 1 if run_diff >= 2 else 0
        elif rl_side == "AWAY":
            # Away +1.5: away wins OR home wins by exactly 1
            rl_correct = 1 if run_diff <= 1 else 0

        # ── Game total result (O/U) ───────────────────────────────────────────
        total_direction = pred_row.get("total_direction")
        total_line = pred_row.get("total_line")
        total_correct: int | None = None
        if total_direction is not None and not pd.isna(total_direction) and total_line is not None and not pd.isna(total_line):
            try:
                actual_total = home_score + away_score
                tl = float(total_line)
                direction_upper = str(total_direction).upper()
                if direction_upper == "OVER":
                    total_correct = 1 if actual_total > tl else (None if actual_total == tl else 0)
                elif direction_upper == "UNDER":
                    total_correct = 1 if actual_total < tl else (None if actual_total == tl else 0)
            except (TypeError, ValueError):
                pass

        df.at[idx, "actual_winner"] = actual_winner
        df.at[idx, "correct"] = correct
        df.at[idx, "verified_at"] = verified_at
        if rl_correct is not None:
            df.at[idx, "rl_correct"] = rl_correct
        if total_correct is not None:
            df.at[idx, "total_correct"] = total_correct

        # Persist to DB
        from src.data.predictions_db import update_result
        update_result(
            game_date, home, away, actual_winner, correct, verified_at,
            actual_home_score=home_score,
            actual_away_score=away_score,
            rl_correct=rl_correct,
            total_correct=total_correct,
        )

    df = df.drop(columns=["_date"], errors="ignore")
    return df
