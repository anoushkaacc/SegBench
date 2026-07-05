"""Shared type aliases and data containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    """Resolved dataset sample paths."""

    stem: str
    ground_truth_path: Path
    prediction_path: Path
    image_path: Path | None = None


@dataclass(frozen=True)
class LoadedSample:
    """In-memory sample arrays."""

    sample: Sample
    ground_truth: np.ndarray
    prediction: np.ndarray
    image: np.ndarray | None = None
