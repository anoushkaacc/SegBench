"""Metric computation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy import ndimage


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def confusion_matrix_from_arrays(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    encoded = ground_truth.astype(np.int64) * num_classes + prediction.astype(np.int64)
    counts = np.bincount(encoded.ravel(), minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _ensure_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _boundary_width(mask: np.ndarray, dilation_ratio: float) -> int:
    height, width = mask.shape
    diagonal = math.sqrt(height * height + width * width)
    return max(1, int(round(dilation_ratio * diagonal)))


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    binary_mask = _ensure_bool_mask(mask)
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    width = _boundary_width(binary_mask, dilation_ratio)
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(binary_mask, structure=structure, iterations=width, border_value=0)
    return binary_mask ^ eroded


def binary_boundary_iou(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    dilation_ratio: float = 0.02,
) -> float:
    gt_boundary = mask_to_boundary(ground_truth, dilation_ratio=dilation_ratio)
    pred_boundary = mask_to_boundary(prediction, dilation_ratio=dilation_ratio)
    union = np.logical_or(gt_boundary, pred_boundary).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    return float(intersection / union)


def _surface_distances(source_boundary: np.ndarray, target_boundary: np.ndarray) -> np.ndarray:
    if not np.any(source_boundary):
        return np.array([], dtype=float)
    if not np.any(target_boundary):
        return np.full(int(source_boundary.sum()), np.inf, dtype=float)
    target_distance = ndimage.distance_transform_edt(~target_boundary)
    return target_distance[source_boundary]


def binary_hd95(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt_boundary = mask_to_boundary(ground_truth, dilation_ratio=0.0)
    pred_boundary = mask_to_boundary(prediction, dilation_ratio=0.0)

    if not np.any(gt_boundary) and not np.any(pred_boundary):
        return 0.0
    if not np.any(gt_boundary) or not np.any(pred_boundary):
        return math.inf

    distances = np.concatenate(
        [
            _surface_distances(gt_boundary, pred_boundary),
            _surface_distances(pred_boundary, gt_boundary),
        ]
    )
    return float(np.percentile(distances, 95))


def aggregate_mean(values: Iterable[float]) -> float:
    numeric_values = np.asarray(list(values), dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        if numeric_values.size == 0:
            return math.nan
        if np.any(np.isinf(numeric_values)):
            return math.inf
        return math.nan
    return float(np.mean(finite_values))


def per_class_metrics(confusion_matrix: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(confusion_matrix).astype(float)
    support = confusion_matrix.sum(axis=1).astype(float)
    predicted = confusion_matrix.sum(axis=0).astype(float)
    fp = predicted - tp
    fn = support - tp
    union = tp + fp + fn

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    iou = safe_divide(tp, union)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    frequency = safe_divide(support, np.full_like(support, support.sum(), dtype=float))

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "frequency": frequency,
    }


def select_class_indices(
    confusion_matrix: np.ndarray,
    include_background: bool,
    background_label: int,
) -> np.ndarray:
    support = confusion_matrix.sum(axis=1)
    valid = support > 0
    if include_background:
        return valid
    if not 0 <= background_label < confusion_matrix.shape[0]:
        return valid
    selected = valid.copy()
    selected[background_label] = False
    return selected


def overall_metrics(
    confusion_matrix: np.ndarray,
    include_background: bool = True,
    background_label: int = 0,
) -> dict[str, float]:
    metrics = per_class_metrics(confusion_matrix)
    selected = select_class_indices(confusion_matrix, include_background, background_label)
    pixel_accuracy = safe_divide(
        np.array([np.trace(confusion_matrix, dtype=float)]),
        np.array([confusion_matrix.sum()], dtype=float),
    )[0]

    return {
        "mean_iou": float(np.mean(metrics["iou"][selected])) if np.any(selected) else 0.0,
        "mean_dice": float(np.mean(metrics["dice"][selected])) if np.any(selected) else 0.0,
        "mean_precision": float(np.mean(metrics["precision"][selected])) if np.any(selected) else 0.0,
        "mean_recall": float(np.mean(metrics["recall"][selected])) if np.any(selected) else 0.0,
        "mean_f1": float(np.mean(metrics["f1"][selected])) if np.any(selected) else 0.0,
        "pixel_accuracy": float(pixel_accuracy),
    }


def boundary_iou(*_: object, **__: object) -> float:
    return binary_boundary_iou(*_)


def hd95(*_: object, **__: object) -> float:
    return binary_hd95(*_)
