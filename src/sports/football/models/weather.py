"""NFL weather model — wind, temperature, and precipitation adjustments to game totals."""

from __future__ import annotations

import pandas as pd

# Total-line adjustments in points
WIND_THRESHOLDS = [
    (25, -7.0),   # wind >= 25 mph → -7 pts
    (20, -5.0),   # wind >= 20 mph → -5 pts
    (15, -3.0),   # wind >= 15 mph → -3 pts
    (10, -1.5),   # wind >= 10 mph → -1.5 pts
]

TEMP_THRESHOLDS = [
    (20, -3.0),   # temp <= 20°F → -3 pts
    (30, -2.0),   # temp <= 30°F → -2 pts
    (40, -1.0),   # temp <= 40°F → -1 pt
]

PRECIPITATION_ADJ = -2.0    # Rain or snow adds -2 pts

DOME_ROOFS = {"dome", "closed"}        # No weather effect
RETRACTABLE_CLOSED = {"retractable"}   # Treated as dome when listed as closed


def get_wind_adjustment(wind_mph: float | None) -> float:
    if wind_mph is None or wind_mph <= 0:
        return 0.0
    for threshold, adj in WIND_THRESHOLDS:
        if wind_mph >= threshold:
            return adj
    return 0.0


def get_temp_adjustment(temp_f: float | None) -> float:
    if temp_f is None:
        return 0.0
    for threshold, adj in TEMP_THRESHOLDS:
        if temp_f <= threshold:
            return adj
    return 0.0


def is_dome(roof: str | None) -> bool:
    """Return True if the game is played in a fully enclosed dome."""
    if not roof:
        return False
    return str(roof).lower().strip() in DOME_ROOFS


def compute_weather_adjustment(
    wind_mph: float | None,
    temp_f: float | None,
    precipitation: bool = False,
    roof: str | None = None,
) -> dict:
    """
    Compute total-line weather adjustment for a single game.

    Returns:
      - wind_adj    : points from wind
      - temp_adj    : points from temperature
      - precip_adj  : points from precipitation
      - total_adj   : combined adjustment (subtract from posted total for fair line)
      - is_dome     : True if weather has no effect
      - weather_flag: True if adjustment is meaningful (|adj| >= 2)
      - summary     : human-readable description
    """
    if is_dome(roof):
        return {
            "wind_adj": 0.0, "temp_adj": 0.0, "precip_adj": 0.0,
            "total_adj": 0.0, "is_dome": True,
            "weather_flag": False, "summary": "Dome — no weather effect",
        }

    wind_adj = get_wind_adjustment(wind_mph)
    temp_adj = get_temp_adjustment(temp_f)
    precip_adj = PRECIPITATION_ADJ if precipitation else 0.0
    total_adj = wind_adj + temp_adj + precip_adj

    parts = []
    if wind_adj < 0:
        parts.append(f"Wind {wind_mph}mph ({wind_adj:+.1f})")
    if temp_adj < 0:
        parts.append(f"Temp {temp_f}°F ({temp_adj:+.1f})")
    if precip_adj < 0:
        parts.append(f"Precipitation ({precip_adj:+.1f})")

    summary = " | ".join(parts) if parts else "No significant weather impact"
    flag = abs(total_adj) >= 2.0

    return {
        "wind_adj": wind_adj,
        "temp_adj": temp_adj,
        "precip_adj": precip_adj,
        "total_adj": round(total_adj, 1),
        "is_dome": False,
        "weather_flag": flag,
        "summary": summary,
    }


def add_weather_adjustments(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather adjustment columns to schedule DataFrame.

    Required columns (optional): wind, temp, roof
    Added columns: weather_total_adj, weather_flag, weather_summary,
                   adjusted_total (posted total + weather_adj)
    """
    df = schedule_df.copy()
    rows = []

    for _, row in df.iterrows():
        wind = row.get("wind")
        temp = row.get("temp")
        roof = row.get("roof")
        precip = False  # nfl_data_py doesn't provide precipitation directly

        try:
            wind_val = float(wind) if pd.notna(wind) else None
        except (ValueError, TypeError):
            wind_val = None

        try:
            temp_val = float(temp) if pd.notna(temp) else None
        except (ValueError, TypeError):
            temp_val = None

        ctx = compute_weather_adjustment(wind_val, temp_val, precip, roof)
        rows.append({
            "weather_total_adj": ctx["total_adj"],
            "weather_flag": ctx["weather_flag"],
            "weather_summary": ctx["summary"],
            "is_dome": ctx["is_dome"],
        })

    weather_df = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), weather_df], axis=1)

    if "total_line" in df.columns:
        df["adjusted_total"] = (df["total_line"] + df["weather_total_adj"]).round(1)

    return df
