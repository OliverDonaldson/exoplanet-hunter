"""Stellar parameter lookup from the TIC and Gaia DR3.

Used to enrich each target with auxiliary features (Teff, R*, log g, magnitude)
that feed the Wide path of the dual-view CNN. These are non-time-series
features; they help disambiguate transits (e.g. a 1% dip on a giant star
implies a stellar companion, not a planet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StellarParams:
    tic_id: int
    teff: float | None         # Effective temperature [K]
    radius: float | None       # Stellar radius [Rsun]
    logg: float | None         # log10(surface gravity / cm s^-2)
    tmag: float | None         # TESS magnitude
    gaia_id: int | None = None


def fetch_stellar_params(tic_id: int) -> StellarParams:
    """Look up stellar parameters for a TIC via the TIC v8 catalog (MAST).

    Falls back to NaN-filled values if the lookup fails — the model handles
    missing features through standardisation/imputation in the data module.
    """
    try:
        from astroquery.mast import Catalogs

        cat = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
        if cat is None or len(cat) == 0:
            return _empty(tic_id)
        row = cat[0]
        return StellarParams(
            tic_id=tic_id,
            teff=_safe_float(row.get("Teff")),
            radius=_safe_float(row.get("rad")),
            logg=_safe_float(row.get("logg")),
            tmag=_safe_float(row.get("Tmag")),
            gaia_id=_safe_int(row.get("GAIA")),
        )
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[stellar] TIC %d: %s", tic_id, exc)
        return _empty(tic_id)


def _empty(tic_id: int) -> StellarParams:
    return StellarParams(tic_id=tic_id, teff=None, radius=None, logg=None, tmag=None)


def _safe_float(x: object) -> float | None:
    try:
        v = float(x)                                          # type: ignore[arg-type]
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _safe_int(x: object) -> int | None:
    try:
        return int(x)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
