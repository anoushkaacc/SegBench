"""Input discovery and loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from segbench.exceptions import ValidationError
from segbench.types import LoadedSample, Sample

SUPPORTED_MASK_SUFFIXES = {".png", ".tif", ".tiff", ".npy", ".npz"}
SUPPORTED_IMAGE_SUFFIXES = SUPPORTED_MASK_SUFFIXES | {".jpg", ".jpeg", ".bmp"}


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        array = np.asarray(Image.open(path))
        if array.ndim == 3:
            array = array[..., 0]
        return array
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValidationError(f"Expected one array in {path}, found {len(data.files)}.")
            return data[data.files[0]]
    raise ValidationError(f"Unsupported file format: {path.suffix}")


def _discover_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.exists():
        raise ValidationError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValidationError(f"Expected directory path: {directory}")

    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            files[path.stem] = path
    return files


def collect_samples(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    images_dir: str | Path | None = None,
) -> list[Sample]:
    gt_dir = Path(ground_truth_dir)
    pred_dir = Path(prediction_dir)
    image_dir = Path(images_dir) if images_dir else None

    gt_files = _discover_files(gt_dir, SUPPORTED_MASK_SUFFIXES)
    pred_files = _discover_files(pred_dir, SUPPORTED_MASK_SUFFIXES)
    image_files = _discover_files(image_dir, SUPPORTED_IMAGE_SUFFIXES) if image_dir else {}

    if not gt_files:
        raise ValidationError(f"No ground-truth masks found in {gt_dir}")
    if not pred_files:
        raise ValidationError(f"No prediction masks found in {pred_dir}")

    missing_predictions = sorted(set(gt_files) - set(pred_files))
    extra_predictions = sorted(set(pred_files) - set(gt_files))
    if missing_predictions or extra_predictions:
        parts = []
        if missing_predictions:
            parts.append(f"missing predictions for: {', '.join(missing_predictions[:5])}")
        if extra_predictions:
            parts.append(f"extra predictions for: {', '.join(extra_predictions[:5])}")
        raise ValidationError("Filename mismatch between ground truth and predictions: " + "; ".join(parts))

    samples = [
        Sample(
            stem=stem,
            ground_truth_path=gt_files[stem],
            prediction_path=pred_files[stem],
            image_path=image_files.get(stem),
        )
        for stem in sorted(gt_files)
    ]
    return samples


def load_sample(sample: Sample) -> LoadedSample:
    ground_truth = _load_array(sample.ground_truth_path)
    prediction = _load_array(sample.prediction_path)
    image = _load_array(sample.image_path) if sample.image_path else None
    return LoadedSample(sample=sample, ground_truth=ground_truth, prediction=prediction, image=image)
