"""Metric computation helpers."""

from __future__ import annotations

import math

import numpy as np


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


def overall_metrics(confusion_matrix: np.ndarray) -> dict[str, float]:
    metrics = per_class_metrics(confusion_matrix)
    support = metrics["support"]
    valid = support > 0
    pixel_accuracy = safe_divide(
        np.array([np.trace(confusion_matrix, dtype=float)]),
        np.array([confusion_matrix.sum()], dtype=float),
    )[0]

    return {
        "mean_iou": float(np.mean(metrics["iou"][valid])) if np.any(valid) else 0.0,
        "mean_dice": float(np.mean(metrics["dice"][valid])) if np.any(valid) else 0.0,
        "mean_precision": float(np.mean(metrics["precision"][valid])) if np.any(valid) else 0.0,
        "mean_recall": float(np.mean(metrics["recall"][valid])) if np.any(valid) else 0.0,
        "mean_f1": float(np.mean(metrics["f1"][valid])) if np.any(valid) else 0.0,
        "pixel_accuracy": float(pixel_accuracy),
    }


def boundary_iou(*_: object, **__: object) -> float:
    return math.nan


def hd95(*_: object, **__: object) -> float:
    return math.nan
