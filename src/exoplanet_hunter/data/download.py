"""Light-curve downloader for TESS and Kepler missions.

Wraps `lightkurve` to:

  * Resolve a target ID (TIC or KIC) to all available sectors/quarters.
  * Stitch into a single time series.
  * Cache to local disk and skip already-downloaded targets.
  * Report failures gracefully — many targets simply have no pipeline data.

Supports tiered storage: TESS and Kepler raw files can live in separate
directories (e.g. internal SSD vs external USB) via ``kepler_cache_dir``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Direct-archive Kepler download — bypasses MAST's CAOMv240 search backend by
# hitting the public archive.stsci.edu HTTP listing endpoint directly. This is
# the same FITS data lightkurve fetches (same files, same URLs ultimately) but
# without the CAOM search-layer indirection, which is what fails during MAST
# database outages (`Cannot open database "CAOMv240"` etc.).
# ----------------------------------------------------------------------------
_KEPLER_ARCHIVE_BASE = "https://archive.stsci.edu/pub/kepler/lightcurves"
_KEPLER_LLC_PATTERN = re.compile(r"kplr\d+-\d+_llc\.fits")

# Error substrings that indicate a TRANSIENT failure (MAST infrastructure
# blip, network drop, server restart). These must NOT be cached as permanent
# failures in the manifest — they should be retried on the next run.
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "CAOMv240", "SHUTDOWN", "Timeout", "network-related", "SQL Server",
    "Connection aborted", "HTTPError", "RemoteDisconnected", "500 Server",
)


def _is_transient_error(reason: str | None) -> bool:
    """True if `reason` looks like a transient infrastructure error.

    Used by ``_record_failure`` to decide whether to persist the failure to
    the manifest. Transient errors are returned to the caller but not cached.
    """
    if not reason:
        return False
    return any(marker in reason for marker in _TRANSIENT_ERROR_MARKERS)


@dataclass
class DownloadResult:
    target_id: int
    mission: str
    success: bool
    n_sectors: int
    n_points: int
    path: Path | None
    reason: str | None = None

    # Backward compat: existing code accesses .tic_id
    @property
    def tic_id(self) -> int:
        return self.target_id


class LightCurveDownloader:
    """Resumable bulk downloader for TESS and Kepler light curves.

    The downloader keeps a JSON manifest at ``cache_dir/manifest.json`` mapping
    target ID → DownloadResult metadata, so re-runs skip prior successes and
    don't repeatedly hammer MAST for known failures.

    Parameters
    ----------
    cache_dir : Where TESS raw FITS land (also the default for Kepler).
    kepler_cache_dir : If set, Kepler FITS go here instead (e.g. external USB).
    author : ``"SPOC"`` for TESS, ``"Kepler"`` for Kepler (auto-dispatched).
    cadence : 120 for 2-min TESS; None lets lightkurve pick the best.
    """

    _MISSION_CFG: dict[str, dict[str, Any]] = {
        "TESS":   {"prefix": "tic", "search": "TIC",  "author": "SPOC",   "mission": "TESS",   "cadence": 120},
        "Kepler": {"prefix": "kic", "search": "KIC",  "author": "Kepler", "mission": "Kepler", "cadence": None},
    }

    def __init__(
        self,
        cache_dir: Path,
        kepler_cache_dir: Path | None = None,
        author: str = "SPOC",
        cadence: int | None = 120,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kepler_cache_dir = Path(kepler_cache_dir) if kepler_cache_dir else None
        if self.kepler_cache_dir:
            self.kepler_cache_dir.mkdir(parents=True, exist_ok=True)
        self.author = author
        self.cadence = cadence
        self._manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()

    # ---------------------------------------------------------------- helpers

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text())
        except json.JSONDecodeError:
            log.warning("[download] corrupted manifest; starting fresh")
            return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2, default=str))

    def _target_path(self, target_id: int, mission: str = "TESS") -> Path:
        mcfg = self._MISSION_CFG[mission]
        prefix = mcfg["prefix"]
        if mission == "Kepler" and self.kepler_cache_dir:
            return self.kepler_cache_dir / f"{prefix}_{target_id}.fits"
        return self.cache_dir / f"{prefix}_{target_id}.fits"

    # ------------------------------------------------------------------ core

    def download_one(
        self, target_id: int, mission: str = "TESS", force: bool = False,
    ) -> DownloadResult:
        """Download all sectors/quarters for a single target and stitch them.

        Parameters
        ----------
        target_id : TIC ID (TESS) or KIC ID (Kepler).
        mission   : ``"TESS"`` or ``"Kepler"``.
        force     : ignore cache and re-download.
        """
        import lightkurve as lk

        mcfg = self._MISSION_CFG[mission]
        key = f"{mission}:{target_id}"
        target_path = self._target_path(target_id, mission)

        if not force and key in self._manifest:
            entry = self._manifest[key]
            path = Path(entry.get("path") or "")
            if entry.get("success") and path.exists():
                return DownloadResult(
                    target_id=target_id,
                    mission=mission,
                    success=True,
                    n_sectors=int(entry.get("n_sectors", 0)),
                    n_points=int(entry.get("n_points", 0)),
                    path=path,
                )
            if not entry.get("success"):
                return DownloadResult(
                    target_id=target_id,
                    mission=mission,
                    success=False,
                    n_sectors=0,
                    n_points=0,
                    path=None,
                    reason=entry.get("reason", "previously failed"),
                )

        # Also check legacy manifest keys (pre-Kepler: bare TIC ID as key)
        if not force and mission == "TESS":
            legacy_key = str(target_id)
            if legacy_key in self._manifest:
                entry = self._manifest[legacy_key]
                path = Path(entry.get("path") or "")
                if entry.get("success") and path.exists():
                    return DownloadResult(
                        target_id=target_id,
                        mission=mission,
                        success=True,
                        n_sectors=int(entry.get("n_sectors", 0)),
                        n_points=int(entry.get("n_points", 0)),
                        path=path,
                    )
                if not entry.get("success"):
                    return DownloadResult(
                        target_id=target_id,
                        mission=mission,
                        success=False,
                        n_sectors=0,
                        n_points=0,
                        path=None,
                        reason=entry.get("reason", "previously failed"),
                    )

        dl_dir = self._target_path(target_id, mission).parent / ".lightkurve"

        # --- Try direct archive first (Kepler only) ---
        # archive.stsci.edu serves the same Kepler LLC FITS files lightkurve
        # eventually downloads, but via a plain HTTP listing endpoint that is
        # on different infrastructure from MAST's CAOMv240 search backend.
        # When CAOM is down (which happens), this path still works.
        lc_collection = None
        if mission == "Kepler":
            try:
                lc_collection = self._fetch_kepler_via_direct_archive(target_id, dl_dir)
                log.debug(
                    "[download] KIC %d: %d quarters via direct archive",
                    target_id, len(lc_collection),
                )
            except FileNotFoundError as exc:
                # Permanent: archive listing has no LLC files for this KIC.
                return self._record_failure(target_id, mission, f"no archive data: {exc}")
            except Exception as exc:                          # noqa: BLE001
                log.warning(
                    "[download] direct archive failed for KIC %d (%s); "
                    "falling back to CAOM search",
                    target_id, exc,
                )
                # lc_collection stays None — fall through to lightkurve path.

        # --- Fallback / TESS path: lightkurve.search_lightcurve + download_all ---
        if lc_collection is None:
            search_str = f"{mcfg['search']} {target_id}"
            try:
                search = lk.search_lightcurve(
                    search_str,
                    mission=mcfg["mission"],
                    author=mcfg["author"],
                    cadence=mcfg["cadence"],
                )
            except Exception as exc:                          # noqa: BLE001
                return self._record_failure(target_id, mission, f"search error: {exc}")

            if len(search) == 0:
                return self._record_failure(target_id, mission, "no pipeline data")

            try:
                lc_collection = search.download_all(
                    download_dir=str(dl_dir),
                )
            except Exception as exc:                          # noqa: BLE001
                return self._record_failure(target_id, mission, f"download error: {exc}")

            if lc_collection is None or len(lc_collection) == 0:
                return self._record_failure(target_id, mission, "empty download")

        # --- Shared: stitch the per-quarter/sector LCs into one ---
        try:
            stitched = lc_collection.stitch()
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(target_id, mission, f"stitch error: {exc}")

        # Persist a compact FITS file (just time + flux + flux_err + centroids
        # if available — we don't need everything in the SPOC product).
        try:
            stitched.to_fits(target_path, overwrite=True)
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(target_id, mission, f"fits write error: {exc}")

        result = DownloadResult(
            target_id=target_id,
            mission=mission,
            success=True,
            n_sectors=len(lc_collection),
            n_points=len(stitched),
            path=target_path,
        )
        self._manifest[key] = {
            "success":   True,
            "n_sectors": result.n_sectors,
            "n_points":  result.n_points,
            "path":      str(target_path),
        }
        self._save_manifest()
        return result

    def _record_failure(self, target_id: int, mission: str, reason: str) -> DownloadResult:
        """Log + return a failure result.

        Transient failures (MAST outage, network drop) are NOT persisted to
        the manifest — they should be retried on the next run. Permanent
        failures (no pipeline data, no archive data, malformed FITS) are
        cached so re-runs don't hammer MAST for known-empty targets.
        """
        log.warning("[download] %s %d: %s", mission, target_id, reason)
        if not _is_transient_error(reason):
            key = f"{mission}:{target_id}"
            self._manifest[key] = {"success": False, "reason": reason}
            self._save_manifest()
        return DownloadResult(
            target_id=target_id,
            mission=mission,
            success=False,
            n_sectors=0,
            n_points=0,
            path=None,
            reason=reason,
        )

    # ----------------------------------------------- direct-archive (Kepler)

    def _direct_archive_kepler_fits(
        self, kic: int, dl_dir: Path, timeout: int = 30,
    ) -> list[Path]:
        """Download every Kepler LLC FITS for a KIC from archive.stsci.edu.

        Path scheme::

            https://archive.stsci.edu/pub/kepler/lightcurves/{KIC[:4]}/{KIC:09d}/

        Returns the list of local FITS paths (one per quarter).

        Raises
        ------
        FileNotFoundError
            Listing returned 404 or contains no LLC files (permanent gap —
            this KIC has no Kepler data).
        requests.RequestException
            Network / HTTP error against the archive (transient — the caller
            may fall back to the CAOM search path).
        """
        kic_padded = f"{kic:09d}"
        listing_url = f"{_KEPLER_ARCHIVE_BASE}/{kic_padded[:4]}/{kic_padded}/"

        r = requests.get(listing_url, timeout=timeout)
        if r.status_code == 404:
            raise FileNotFoundError(f"archive 404 for KIC {kic}")
        r.raise_for_status()

        filenames = sorted(set(_KEPLER_LLC_PATTERN.findall(r.text)))
        if not filenames:
            raise FileNotFoundError(f"no LLC FITS in archive listing for KIC {kic}")

        dl_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for fn in filenames:
            local = dl_dir / fn
            if not local.exists() or local.stat().st_size == 0:
                url = listing_url + fn
                with requests.get(url, stream=True, timeout=timeout * 2) as rr:
                    rr.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in rr.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                fh.write(chunk)
            paths.append(local)
        return paths

    def _fetch_kepler_via_direct_archive(
        self, kic: int, dl_dir: Path,
    ) -> Any:
        """Direct-archive Kepler downloader → ``lk.LightCurveCollection``.

        Wraps ``_direct_archive_kepler_fits`` by reading each downloaded FITS
        with ``lk.read`` and wrapping the result in a ``LightCurveCollection``
        compatible with the existing stitch + write code path.
        """
        import lightkurve as lk
        fits_paths = self._direct_archive_kepler_fits(kic, dl_dir)
        lcs = [lk.read(str(p)) for p in fits_paths]
        return lk.LightCurveCollection(lcs)

    # ----------------------------------------------------------------- batch

    def download_many(
        self,
        target_ids: list[int],
        missions: list[str] | None = None,
        force: bool = False,
    ) -> list[DownloadResult]:
        """Download a list of targets sequentially with progress logging.

        Parameters
        ----------
        target_ids : List of TIC/KIC IDs.
        missions   : Parallel list of mission strings ("TESS"/"Kepler").
                     If None, defaults to "TESS" for all.
        """
        from tqdm.auto import tqdm

        if missions is None:
            missions = ["TESS"] * len(target_ids)

        results: list[DownloadResult] = []
        for tid, mis in tqdm(
            zip(target_ids, missions),
            total=len(target_ids),
            desc="downloading",
            unit="target",
        ):
            results.append(self.download_one(int(tid), mission=mis, force=force))

        n_ok = sum(r.success for r in results)
        log.info("[download] complete — %d/%d succeeded", n_ok, len(results))
        return results
