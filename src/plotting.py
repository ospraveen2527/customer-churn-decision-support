"""Tiny matplotlib helpers (object-oriented API, so no global backend is forced)."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def new_figure(figsize: tuple[float, float] = (6.5, 4.5)) -> tuple[Figure, Axes]:
    """Create a standalone Figure with one Axes."""
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    return fig, ax


def save_figure(fig: Figure, path: Path | str, dpi: int = 150) -> Path:
    """Save a figure (creating parent folders) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    logger.info("Saved figure: %s", out)
    return out
