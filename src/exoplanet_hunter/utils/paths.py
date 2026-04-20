"""Centralised path resolution.

Hydra resolves paths from `conf/config.yaml` into an `omegaconf.DictConfig`;
this module wraps that into a typed convenience class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_raw_kepler: Path
    data_interim: Path
    data_processed: Path
    data_labels: Path
    models: Path
    results: Path

    @classmethod
    def from_cfg(cls, cfg: Any) -> ProjectPaths:
        """Build from a Hydra/OmegaConf `paths` group."""
        p = cfg.paths
        paths = cls(
            root=Path(p.root),
            data_raw=Path(p.data_raw),
            data_raw_kepler=Path(getattr(p, "data_raw_kepler", p.data_raw + "_kepler")),
            data_interim=Path(p.data_interim),
            data_processed=Path(p.data_processed),
            data_labels=Path(p.data_labels),
            models=Path(p.models),
            results=Path(p.results),
        )
        paths.ensure()
        return paths

    def ensure(self) -> None:
        """Create every directory if it doesn't exist."""
        for path in (
            self.data_raw,
            self.data_raw_kepler,
            self.data_interim,
            self.data_processed,
            self.data_labels,
            self.models,
            self.results,
        ):
            path.mkdir(parents=True, exist_ok=True)
