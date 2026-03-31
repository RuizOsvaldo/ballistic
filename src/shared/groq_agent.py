"""Groq/Llama 3.3 70B agent for bet reasoning and recommendation explanations."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 450
_TEMPERATURE = 0.3

# Lazy-init client so import doesn't fail if groq isn't installed yet
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
            if not _GROQ_API_KEY:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _client = Groq(api_key=_GROQ_API_KEY)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")
    return _client


# ---------------------------------------------------------------------------
# MLB game reasoning
# ---------------------------------------------------------------------------

_MLB_GAME_SYSTEM = (
    "You are a quantitative sports betting analyst. You receive statistical signals "
    "about an MLB game and provide concise, data-driven bet reasoning. "
    "Be direct and specific. Do not hedge excessively. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence about the main risk to this bet)."
)


def analyze_mlb_game(
    home_team: str,
    away_team: str,
    model_prob: float,
    implied_prob: float,
    edge_pct: float,
    bet_side: str,
    signals: dict,
) -> dict:
    """
    Generate AI reasoning for an MLB game bet recommendation.

    Parameters
    ----------
    signals : dict with optional keys:
        home_pyth_deviation, away_pyth_deviation,
        home_fip_era_gap, away_fip_era_gap,
        home_babip, away_babip,
        home_starter, away_starter,
        home_starter_fip, away_starter_fip,
        park_factor (optional)
    """
    signal_lines = []

    if signals.get("home_pyth_deviation") is not None:
        dev = signals["home_pyth_deviation"]
        signal_lines.append(f"  - {home_team} Pythagorean deviation: {dev:+.1%} ({'overperforming' if dev > 0 else 'underperforming'})")

    if signals.get("away_pyth_deviation") is not None:
        dev = signals["away_pyth_deviation"]
        signal_lines.append(f"  - {away_team} Pythagorean deviation: {dev:+.1%} ({'overperforming' if dev > 0 else 'underperforming'})")

    if signals.get("home_starter"):
        fip = signals.get("home_starter_fip", "N/A")
        era = signals.get("home_starter_era", "N/A")
        gap = signals.get("home_fip_era_gap")
        gap_str = f", FIP-ERA gap: {gap:+.2f}" if gap is not None else ""
        signal_lines.append(f"  - {home_team} starter ({signals['home_starter']}): FIP {fip}, ERA {era}{gap_str}")

    if signals.get("away_starter"):
        fip = signals.get("away_starter_fip", "N/A")
        era = signals.get("away_starter_era", "N/A")
        gap = signals.get("away_fip_era_gap")
        gap_str = f", FIP-ERA gap: {gap:+.2f}" if gap is not None else ""
        signal_lines.append(f"  - {away_team} starter ({signals['away_starter']}): FIP {fip}, ERA {era}{gap_str}")

    if signals.get("home_babip") is not None:
        signal_lines.append(f"  - {home_team} rotation BABIP: {signals['home_babip']:.3f} (league avg .300)")

    if signals.get("away_babip") is not None:
        signal_lines.append(f"  - {away_team} rotation BABIP: {signals['away_babip']:.3f} (league avg .300)")

    signals_text = "\n".join(signal_lines) if signal_lines else "  - No additional signals available"

    prompt = f"""MLB Game: {away_team} @ {home_team}

Recommended bet: {bet_side}
Model win probability: {model_prob:.1%}
Market implied probability: {implied_prob:.1%}
Edge: {edge_pct:+.1f}%

Statistical signals:
{signals_text}

Provide your reasoning for this bet in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _MLB_GAME_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }


# ---------------------------------------------------------------------------
# MLB player prop reasoning
# ---------------------------------------------------------------------------

_MLB_PROP_SYSTEM = (
    "You are a quantitative sports betting analyst specializing in MLB player props. "
    "You receive stat-based signals about a player prop and provide concise reasoning. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence)."
)


def analyze_mlb_prop(
    player_name: str,
    team: str,
    prop_type: str,
    line: float,
    model_projection: float,
    edge_pct: float,
    bet_direction: str,
    signals: dict,
) -> dict:
    """
    Generate AI reasoning for an MLB player prop bet.

    signals dict may include:
        babip, babip_deviation, k_pct, bb_pct, whiff_rate,
        opponent_pitcher, opponent_fip, rolling_avg
    """
    signal_lines = []
    if signals.get("babip") is not None:
        signal_lines.append(f"  - BABIP: {signals['babip']:.3f} (deviation from .300: {signals.get('babip_deviation', 0):+.3f})")
    if signals.get("rolling_avg") is not None:
        signal_lines.append(f"  - Rolling 10-game average: {signals['rolling_avg']:.2f}")
    if signals.get("k_pct") is not None:
        signal_lines.append(f"  - K%: {signals['k_pct']:.1%}")
    if signals.get("whiff_rate") is not None:
        signal_lines.append(f"  - Whiff rate: {signals['whiff_rate']:.1%}")
    if signals.get("opponent_pitcher"):
        opp_fip = signals.get("opponent_fip", "N/A")
        signal_lines.append(f"  - Opposing pitcher: {signals['opponent_pitcher']} (FIP: {opp_fip})")

    signals_text = "\n".join(signal_lines) if signal_lines else "  - No additional signals"

    prompt = f"""MLB Player Prop: {player_name} ({team})
Prop: {prop_type} — Line: {line}
Model projection: {model_projection:.2f}
Bet: {bet_direction} | Edge: {edge_pct:+.1f}%

Signals:
{signals_text}

Provide your reasoning in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _MLB_PROP_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }


# ---------------------------------------------------------------------------
# Preseason projection reasoning
# ---------------------------------------------------------------------------

_PRESEASON_SYSTEM = (
    "You are a quantitative sports analyst. You receive a team's preseason win projection "
    "versus Vegas win total line and provide reasoning on whether to bet over or under. "
    "Respond in JSON with keys: reasoning (2-3 sentences), confidence (Low/Medium/High), "
    "key_risk (one sentence)."
)


def analyze_preseason_projection(
    team: str,
    projected_wins: float,
    vegas_line: float,
    bet_direction: str,
    edge_wins: float,
    signals: dict,
) -> dict:
    """
    Generate AI reasoning for a preseason win total bet.

    signals dict may include:
        prior_pyth_win_pct, prior_run_diff, war_total,
        division_strength, park_factor
    """
    signal_lines = []
    if signals.get("prior_pyth_win_pct") is not None:
        signal_lines.append(f"  - Prior season Pythagorean W%: {signals['prior_pyth_win_pct']:.1%}")
    if signals.get("prior_run_diff") is not None:
        signal_lines.append(f"  - Prior season run differential: {signals['prior_run_diff']:+d}")
    if signals.get("war_total") is not None:
        signal_lines.append(f"  - Projected roster WAR: {signals['war_total']:.1f}")
    if signals.get("division_strength"):
        signal_lines.append(f"  - Division strength note: {signals['division_strength']}")

    signals_text = "\n".join(signal_lines) if signal_lines else "  - No additional signals"

    prompt = f"""Preseason Win Total: {team}
Model projection: {projected_wins:.1f} wins
Vegas line: {vegas_line:.1f} wins
Bet: {bet_direction} | Edge: {edge_wins:+.1f} wins

Signals:
{signals_text}

Provide your reasoning in JSON format."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _PRESEASON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "reasoning": f"Agent unavailable: {e}",
            "confidence": "Low",
            "key_risk": "Could not generate AI analysis.",
        }
