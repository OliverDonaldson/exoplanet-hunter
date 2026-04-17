"""Project-wide logger using rich for nice console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
    # Quieten noisy libraries.
    for noisy in ("matplotlib", "PIL", "urllib3", "h5py", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent."""
    _configure_root(level)
    return logging.getLogger(name)
