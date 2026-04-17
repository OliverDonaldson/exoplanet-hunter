"""TESS light-curve downloader.

Wraps `lightkurve` to:

  * Resolve a TIC ID to all available SPOC sectors.
  * Stitch sectors into a single time series.
  * Cache to local disk and skip already-downloaded targets.
  * Report failures gracefully — many TICs simply have no SPOC data.
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
    tic_id: int
    success: bool
    n_sectors: int
    n_points: int
    path: Path | None
    reason: str | None = None


class LightCurveDownloader:
    """Resumable bulk downloader for TESS SPOC light curves.

    The downloader keeps a JSON manifest at `cache_dir/manifest.json` mapping
    TIC ID → DownloadResult metadata, so re-runs skip prior successes and
    don't repeatedly hammer MAST for known failures.
    """

    def __init__(
        self,
        cache_dir: Path,
        author: str = "SPOC",
        cadence: int | None = 120,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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

    def _target_path(self, tic_id: int) -> Path:
        return self.cache_dir / f"tic_{tic_id}.fits"

    # ------------------------------------------------------------------ core

    def download_one(self, tic_id: int, force: bool = False) -> DownloadResult:
        """Download all SPOC sectors for a single TIC and stitch them.

        Parameters
        ----------
        tic_id : TESS Input Catalogue ID.
        force  : ignore cache and re-download.
        """
        import lightkurve as lk

        key = str(tic_id)
        target_path = self._target_path(tic_id)

        if not force and key in self._manifest:
            entry = self._manifest[key]
            path = Path(entry.get("path") or "")
            if entry.get("success") and path.exists():
                return DownloadResult(
                    tic_id=tic_id,
                    success=True,
                    n_sectors=int(entry.get("n_sectors", 0)),
                    n_points=int(entry.get("n_points", 0)),
                    path=path,
                )
            if not entry.get("success"):
                # Don't retry persistent failures unless forced.
                return DownloadResult(
                    tic_id=tic_id,
                    success=False,
                    n_sectors=0,
                    n_points=0,
                    path=None,
                    reason=entry.get("reason", "previously failed"),
                )

        try:
            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                author=self.author,
                cadence=self.cadence,
            )
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(tic_id, f"search error: {exc}")

        if len(search) == 0:
            return self._record_failure(tic_id, "no SPOC data")

        try:
            lc_collection = search.download_all(
                download_dir=str(self.cache_dir / ".lightkurve"),
            )
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(tic_id, f"download error: {exc}")

        if lc_collection is None or len(lc_collection) == 0:
            return self._record_failure(tic_id, "empty download")

        try:
            stitched = lc_collection.stitch()
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(tic_id, f"stitch error: {exc}")

        # Persist a compact FITS file (just time + flux + flux_err + centroids
        # if available — we don't need everything in the SPOC product).
        try:
            stitched.to_fits(target_path, overwrite=True)
        except Exception as exc:                              # noqa: BLE001
            return self._record_failure(tic_id, f"fits write error: {exc}")

        result = DownloadResult(
            tic_id=tic_id,
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

    def _record_failure(self, tic_id: int, reason: str) -> DownloadResult:
        log.warning("[download] TIC %d: %s", tic_id, reason)
        self._manifest[str(tic_id)] = {"success": False, "reason": reason}
        self._save_manifest()
        return DownloadResult(
            tic_id=tic_id,
            success=False,
            n_sectors=0,
            n_points=0,
            path=None,
            reason=reason,
        )

    # ----------------------------------------------------------------- batch

    def download_many(self, tic_ids: list[int], force: bool = False) -> list[DownloadResult]:
        """Download a list of TICs sequentially with progress logging."""
        from tqdm.auto import tqdm

        results: list[DownloadResult] = []
        for tic in tqdm(tic_ids, desc="downloading", unit="target"):
            results.append(self.download_one(int(tic), force=force))

        n_ok = sum(r.success for r in results)
        log.info("[download] complete — %d/%d succeeded", n_ok, len(results))
        return results
