"""Live fetchers for ExoFOP and NASA Exoplanet Archive registries.

Two functions:
    fetch_exofop_tois(refresh=False, ttl_hours=24)   -> pd.DataFrame
    fetch_nea_ps(refresh=False, ttl_hours=24)        -> pd.DataFrame

Each writes a cached CSV under ``data/external/`` so repeat reads within the
TTL window don't hit the network. Force a refresh with ``refresh=True``.

Use this anywhere we previously read the static snapshots in
``data/external/exofop/``. The static snapshots are kept for reproducibility
of the May-2026 results; live fetches override them when current state is
desired.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT / "data" / "external"

EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
EXOFOP_TOI_CACHE = CACHE_DIR / "exofop_tess_tois_live.csv"

NEA_PS_CACHE = CACHE_DIR / "nea_ps_live.csv"

DEFAULT_TTL_HOURS = 24.0


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s < ttl_hours * 3600


def cache_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600


def fetch_exofop_tois(
    *,
    refresh: bool = False,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Current ExoFOP TOI catalog (TESS), one row per TOI.

    Columns include ``TIC ID``, ``TOI``, ``TFOPWG Disposition``,
    ``TESS Disposition``, period/depth/RA/Dec, etc.
    """
    if not refresh and _is_fresh(EXOFOP_TOI_CACHE, ttl_hours):
        return pd.read_csv(EXOFOP_TOI_CACHE)

    resp = requests.get(EXOFOP_TOI_URL, timeout=timeout)
    resp.raise_for_status()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXOFOP_TOI_CACHE.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content))


def fetch_nea_ps(
    *,
    refresh: bool = False,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    select: str = "pl_name,hostname,tic_id,ra,dec,disc_year,discoverymethod,disc_facility,pl_orbper,pl_rade",
    where: str = "default_flag=1",
    timeout: float = 180.0,
) -> pd.DataFrame:
    """Current NASA Exoplanet Archive PS (planetary systems) table.

    Defaults to confirmed planets only (``default_flag=1``). Adjust
    ``select`` / ``where`` to slice differently. Uses astroquery's TAP client.
    """
    if not refresh and _is_fresh(NEA_PS_CACHE, ttl_hours):
        return pd.read_csv(NEA_PS_CACHE)

    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    res = NasaExoplanetArchive.query_criteria(table="ps", select=select, where=where, cache=False)
    ps = res.to_pandas()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ps.to_csv(NEA_PS_CACHE, index=False)
    return ps


def status_summary() -> dict[str, dict[str, float | int | str | None]]:
    """Cache-state summary for the two live caches."""
    out: dict[str, dict[str, float | int | str | None]] = {}
    for label, path in (("exofop_tois", EXOFOP_TOI_CACHE), ("nea_ps", NEA_PS_CACHE)):
        age = cache_age_hours(path)
        out[label] = {
            "path": str(path.relative_to(PROJECT)),
            "exists": path.exists(),
            "age_hours": None if age is None else round(age, 2),
            "stale": False if age is None else age >= DEFAULT_TTL_HOURS,
        }
    return out
