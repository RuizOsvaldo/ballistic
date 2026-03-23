"""NFL-specific Groq prompt builders — game analysis, prop analysis, season totals."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 450
_TEMPERATURE = 0.3


def _get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
            _client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    return _client


_NFL_GAME_SYSTEM = (
    "You are a quantitative NFL betting analyst. You receive EPA efficiency data, rest context, "
    "and weather signals for an NFL game and provide concise, data-driven bet reasoning. "
    "Be direct and specific. NFL spreads are sharp — only highlight genuine process-based edges. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence about the main risk)."
)

_NFL_PROP_SYSTEM = (
    "You are a quantitative NFL prop betting analyst specializing in player usage and matchup metrics. "
    "You receive EPA matchup data, player efficiency stats, and usage signals. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence)."
)

_NFL_SEASON_SYSTEM = (
    "You are a quantitative NFL analyst evaluating preseason win total bets. "
    "You receive EPA-based projected wins versus the Vegas line. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence)."
)


def analyze_nfl_game(signals: dict) -> dict:
    """
    Analyze an NFL game matchup and return structured reasoning.

    signals keys (all optional except home_team, away_team):
      home_team, away_team,
      home_epa_composite, away_epa_composite, epa_diff,
      spread_equivalent, posted_spread,
      home_rest_type, away_rest_type, rest_mismatch,
      weather_summary, weather_flag,
      home_model_prob, away_model_prob,
      home_implied_prob, away_implied_prob,
      home_edge_pct, away_edge_pct,
      best_bet_side, best_bet_edge
    """
    home = signals.get("home_team", "Home")
    away = signals.get("away_team", "Away")

    rest_section = ""
    if signals.get("home_rest_type") or signals.get("away_rest_type"):
        rest_section = f"""
Rest Context:
- {home}: {signals.get('home_rest_type', 'Normal')}
- {away}: {signals.get('away_rest_type', 'Normal')}
- Rest mismatch: {signals.get('rest_mismatch', False)}"""

    weather_section = ""
    if signals.get("weather_summary"):
        weather_section = f"\nWeather: {signals.get('weather_summary')}"

    prompt = f"""NFL Game: {away} @ {home}

EPA Efficiency:
- {home} composite EPA/play: {signals.get('home_epa_composite', 'N/A')}
- {away} composite EPA/play: {signals.get('away_epa_composite', 'N/A')}
- EPA differential: {signals.get('epa_diff', 'N/A')}
- Model spread equivalent: {signals.get('spread_equivalent', 'N/A')} pts
- Posted spread: {signals.get('posted_spread', 'N/A')} pts
{rest_section}{weather_section}

Recommended bet: {signals.get('best_bet_side', 'N/A')}
Model win probability: {signals.get('home_model_prob', 'N/A')} (home) / {signals.get('away_model_prob', 'N/A')} (away)
Market implied probability: {signals.get('home_implied_prob', 'N/A')} (home)
Edge: {signals.get('best_bet_edge', 'N/A')}%

Provide your reasoning in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _NFL_GAME_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }


def analyze_nfl_prop(signals: dict) -> dict:
    """
    Analyze an NFL player prop bet.

    signals keys (all optional except player_name, team, prop_type):
      player_name, team, position, opponent,
      prop_type (e.g. "Passing Yards"), line, projection, edge_pct, bet_direction,
      opp_def_epa, air_yards_share, target_share,
      season_avg, games_played
    """
    prop_type = signals.get("prop_type", "stat")
    player = signals.get("player_name", "Player")
    team = signals.get("team", "")
    position = signals.get("position", "")
    opp = signals.get("opponent", "opponent")

    usage_lines = []
    if signals.get("air_yards_share") is not None:
        usage_lines.append(f"  - Air yards share: {signals['air_yards_share']:.1%}")
    if signals.get("target_share") is not None:
        usage_lines.append(f"  - Target share: {signals['target_share']:.1%}")
    if signals.get("opp_def_epa") is not None:
        usage_lines.append(f"  - Opponent def EPA/play: {signals['opp_def_epa']:.3f}")
    if signals.get("season_avg") is not None:
        usage_lines.append(f"  - Season average: {signals['season_avg']:.1f} ({signals.get('games_played', '?')} games)")

    usage_text = "\n".join(usage_lines) if usage_lines else "  - No additional signals"

    prompt = f"""NFL Player Prop: {player} ({position}, {team}) vs {opp}
Prop: {prop_type} — Line: {signals.get('line', 'N/A')}
Model projection: {signals.get('projection', 'N/A')}
Bet: {signals.get('bet_direction', 'N/A')} | Edge: {signals.get('edge_pct', 'N/A')}%

Signals:
{usage_text}

Provide your reasoning in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _NFL_PROP_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }


def analyze_nfl_season_total(signals: dict) -> dict:
    """
    Analyze an NFL preseason win total bet.

    signals keys:
      team, projected_wins, vegas_line, bet_direction, edge_wins,
      prior_epa_composite, prior_win_pct, strength_of_schedule (optional)
    """
    team = signals.get("team", "Team")
    proj = signals.get("projected_wins", "N/A")
    line = signals.get("vegas_line", "N/A")
    direction = signals.get("bet_direction", "N/A")
    edge = signals.get("edge_wins", "N/A")

    signal_lines = []
    if signals.get("prior_epa_composite") is not None:
        signal_lines.append(f"  - Prior season EPA composite: {signals['prior_epa_composite']:.3f}")
    if signals.get("prior_win_pct") is not None:
        signal_lines.append(f"  - Prior season win%: {signals['prior_win_pct']:.1%}")
    if signals.get("strength_of_schedule") is not None:
        signal_lines.append(f"  - Strength of schedule note: {signals['strength_of_schedule']}")

    signals_text = "\n".join(signal_lines) if signal_lines else "  - No additional signals"

    prompt = f"""NFL Preseason Win Total: {team}
Model projection: {proj} wins
Vegas line: {line} wins
Bet: {direction} | Edge: {edge:+.1f} wins

Signals:
{signals_text}

Provide your reasoning in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _NFL_SEASON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }
