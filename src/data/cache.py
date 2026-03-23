"""Disk-backed parquet cache with per-key TTL for API and data calls."""

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/cache"))

# Global fallback TTL — overridden per-call via the ttl_hours argument
_DEFAULT_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "6"))


def _cache_path(key: str) -> Path:
    safe_key = key.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_key}.parquet"


def is_fresh(key: str, ttl_hours: float | None = None) -> bool:
    path = _cache_path(key)
    if not path.exists():
        return False
    ttl = ttl_hours if ttl_hours is not None else _DEFAULT_TTL_HOURS
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < ttl


def read(key: str, ttl_hours: float | None = None) -> pd.DataFrame | None:
    if not is_fresh(key, ttl_hours):
        return None
    return pd.read_parquet(_cache_path(key))


def write(key: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(key), index=False)


def cached(key: str, fetch_fn, ttl_hours: float | None = None) -> pd.DataFrame:
    """
    Return cached DataFrame if fresh, otherwise call fetch_fn(), cache and return result.

    Parameters
    ----------
    key       : unique cache key string
    fetch_fn  : zero-argument callable that returns a DataFrame
    ttl_hours : how long the cache is valid. If None, uses CACHE_TTL_HOURS env var (default 6h).

    Common TTLs used in Ballistic:
      - Odds (any sport)           : 2.0  hours
      - Team / player stats        : 6.0  hours
      - NFL EPA / schedule         : 2.0  hours  (weather changes near game time)
      - NFL season play-by-play    : 12.0 hours
    """
    df = read(key, ttl_hours)
    if df is not None:
        return df
    df = fetch_fn()
    write(key, df)
    return df
