"""NBA-specific Groq prompt builders — game analysis and prop analysis."""

from __future__ import annotations

import json
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client: Groq | None = None
MODEL = "llama-3.3-70b-versatile"


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def analyze_nba_game(signals: dict) -> dict:
    """
    Analyze an NBA game matchup and return structured reasoning.

    signals keys:
      home_team, away_team,
      home_net_rtg, away_net_rtg, net_rtg_diff,
      home_rest_type, away_rest_type, rest_mismatch,
      home_efg_pct, away_efg_pct, home_drtg, away_drtg,
      home_model_prob, away_model_prob,
      home_implied_prob, away_implied_prob,
      home_edge_pct, away_edge_pct,
      best_bet_side, best_bet_edge

    Returns: { reasoning, confidence, key_risk }
    """
    prompt = f"""You are an NBA betting analyst using advanced statistics.

Game: {signals.get('away_team')} @ {signals.get('home_team')}

Team Quality (Net Rating):
- {signals.get('home_team')}: {signals.get('home_net_rtg')} net rating
- {signals.get('away_team')}: {signals.get('away_net_rtg')} net rating
- Net rating differential (home advantage included): {signals.get('net_rtg_diff')}

Rest Context:
- {signals.get('home_team')}: {signals.get('home_rest_type')}
- {signals.get('away_team')}: {signals.get('away_rest_type')}
- Rest mismatch flagged: {signals.get('rest_mismatch')}

Efficiency Matchup:
- {signals.get('home_team')} eFG%: {signals.get('home_efg_pct')} | Def RTG: {signals.get('home_drtg')}
- {signals.get('away_team')} eFG%: {signals.get('away_efg_pct')} | Def RTG: {signals.get('away_drtg')}

Model vs. Market:
- {signals.get('home_team')}: model {signals.get('home_model_prob'):.1%} vs market {signals.get('home_implied_prob'):.1%} | edge {signals.get('home_edge_pct')}%
- {signals.get('away_team')}: model {signals.get('away_model_prob'):.1%} vs market {signals.get('away_implied_prob'):.1%} | edge {signals.get('away_edge_pct')}%
- Recommended bet: {signals.get('best_bet_side')} ({signals.get('best_bet_edge')}% edge)

Analyze this matchup. Explain why there is or isn't edge, what the key factors are, and what risks could invalidate the bet.

Respond ONLY with valid JSON in this exact format:
{{"reasoning": "2-4 sentences explaining the edge", "confidence": "Low|Medium|High", "key_risk": "primary risk factor"}}"""

    try:
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"reasoning": f"Analysis unavailable: {e}", "confidence": "Low", "key_risk": "API error"}


def analyze_nba_prop(signals: dict) -> dict:
    """
    Analyze an NBA player prop and return structured reasoning.

    signals keys:
      player_name, team, stat_type (pts/reb/ast/pra/3pm),
      prop_line, model_projection, bet_direction, edge_pct,
      season_avg, usg_pct, opp_drtg, game_pace, rest_type

    Returns: { reasoning, confidence, key_risk }
    """
    stat = signals.get("stat_type", "stat").upper()
    prompt = f"""You are an NBA player prop analyst.

Prop: {signals.get('player_name')} ({signals.get('team')}) — {stat} O/U {signals.get('prop_line')}

Model vs. Line:
- Model projection: {signals.get('model_projection')}
- Sportsbook line: {signals.get('prop_line')}
- Direction: {signals.get('bet_direction')} | Edge: {signals.get('edge_pct')}%

Context:
- Season average {stat}: {signals.get('season_avg')}
- Usage rate: {signals.get('usg_pct', 'N/A')}
- Opponent defensive rating: {signals.get('opp_drtg', 'N/A')}
- Game pace: {signals.get('game_pace', 'N/A')}
- Player rest: {signals.get('rest_type', 'normal')}

Explain why this prop has value, what the key driver is, and the main risk.

Respond ONLY with valid JSON:
{{"reasoning": "2-3 sentences on the prop edge", "confidence": "Low|Medium|High", "key_risk": "main risk"}}"""

    try:
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=250,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"reasoning": f"Analysis unavailable: {e}", "confidence": "Low", "key_risk": "API error"}


def analyze_nba_season_total(signals: dict) -> dict:
    """
    Analyze a preseason NBA win total projection.

    signals keys:
      team, projected_wins, vegas_line, edge_wins, bet_direction,
      prior_net_rtg, prior_win_pct, key_offseason_moves
    """
    prompt = f"""You are an NBA season win total analyst.

Team: {signals.get('team')}
Model projection: {signals.get('projected_wins')} wins
Vegas line: {signals.get('vegas_line')} wins
Edge: {signals.get('edge_wins')} wins ({signals.get('bet_direction')})

Prior season: Net Rating {signals.get('prior_net_rtg')}, Win% {signals.get('prior_win_pct')}
Key offseason context: {signals.get('key_offseason_moves', 'No major changes noted')}

Explain the win total edge and key risks for this season projection.

Respond ONLY with valid JSON:
{{"reasoning": "2-3 sentences", "confidence": "Low|Medium|High", "key_risk": "main risk"}}"""

    try:
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=250,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"reasoning": f"Analysis unavailable: {e}", "confidence": "Low", "key_risk": "API error"}
