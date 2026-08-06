"""Logging helpers built on top of the ``rich`` console when available."""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

try:  # rich is optional; fall back to the stdlib formatter if missing.
    from rich.logging import RichHandler

    _HAVE_RICH = True
except Exception:  # pragma: no cover
    _HAVE_RICH = False


_CONFIGURED = False


def configure_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handlers: list[logging.Handler] = []
    if _HAVE_RICH:
        handlers.append(RichHandler(rich_tracebacks=True, show_path=False))
    else:  # pragma: no cover
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handlers.append(stream)
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fileh = logging.FileHandler(logfile)
        fileh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handlers.append(fileh)
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(message)s", handlers=handlers)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
