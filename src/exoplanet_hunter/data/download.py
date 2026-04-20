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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


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

        search_str = f"{mcfg['search']} {target_id}"
        try:
            search = lk.search_lightcurve(
                search_str,
                mission=mcfg["mission"],
                author=mcfg["author"],
                cadence=mcfg["cadence"],
            )
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(target_id, mission, f"search error: {exc}")

        if len(search) == 0:
            return self._record_failure(target_id, mission, "no pipeline data")

        try:
            dl_dir = self._target_path(target_id, mission).parent / ".lightkurve"
            lc_collection = search.download_all(
                download_dir=str(dl_dir),
            )
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(target_id, mission, f"download error: {exc}")

        if lc_collection is None or len(lc_collection) == 0:
            return self._record_failure(target_id, mission, "empty download")

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
        log.warning("[download] %s %d: %s", mission, target_id, reason)
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
