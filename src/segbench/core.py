"""Top-level evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segbench.exceptions import ValidationError
from segbench.io import collect_samples, load_sample
from segbench.metrics import (
    aggregate_mean,
    boundary_iou,
    confusion_matrix_from_arrays,
    hd95,
    overall_metrics,
    per_class_metrics,
    select_class_indices,
)
from segbench.report import EvaluationReport


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValidationError(f"Expected 2D segmentation mask, got shape {mask.shape}")
    return mask.astype(np.int64, copy=False)


def _infer_num_classes(
    samples: list[tuple[np.ndarray, np.ndarray]],
    class_names: list[str] | None,
    ignore_index: int | None,
) -> int:
    if class_names:
        return len(class_names)
    max_label = 0
    for ground_truth, prediction in samples:
        for mask in (ground_truth, prediction):
            valid = mask != ignore_index if ignore_index is not None else np.ones(mask.shape, dtype=bool)
            if np.any(valid):
                max_label = max(max_label, int(mask[valid].max()))
    return max_label + 1


def _validate_labels(mask: np.ndarray, num_classes: int, ignore_index: int | None, path: Path) -> None:
    valid = mask != ignore_index if ignore_index is not None else np.ones(mask.shape, dtype=bool)
    if not np.any(valid):
        return
    min_value = int(mask[valid].min())
    max_value = int(mask[valid].max())
    if min_value < 0 or max_value >= num_classes:
        raise ValidationError(
            f"Invalid labels in {path}. Expected values in [0, {num_classes - 1}]"
            + (f" or ignore index {ignore_index}" if ignore_index is not None else "")
            + f", found range [{min_value}, {max_value}]."
        )


def evaluate(
    ground_truth: str | Path,
    predictions: str | Path,
    images: str | Path | None = None,
    class_names: list[str] | None = None,
    ignore_index: int | None = None,
    include_background: bool = False,
    background_label: int = 0,
    save_dir: str | Path | None = None,
) -> EvaluationReport:
    samples = collect_samples(ground_truth, predictions, images)

    loaded_masks: list[tuple[np.ndarray, np.ndarray]] = []
    image_shapes: list[tuple[int, int]] = []
    for sample in samples:
        loaded = load_sample(sample)
        gt = _normalize_mask(loaded.ground_truth)
        pred = _normalize_mask(loaded.prediction)
        if gt.shape != pred.shape:
            raise ValidationError(
                f"Shape mismatch for {sample.stem}: ground truth {gt.shape} vs prediction {pred.shape}"
            )
        loaded_masks.append((gt, pred))
        image_shapes.append(gt.shape)

    num_classes = _infer_num_classes(loaded_masks, class_names, ignore_index)
    if num_classes <= 0:
        raise ValidationError("Could not infer any classes from the provided masks.")

    class_names = class_names or [f"class_{index}" for index in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValidationError("Number of class names must match the number of classes.")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_scores: list[tuple[str, float]] = []
    boundary_scores: dict[int, list[float]] = {class_id: [] for class_id in range(num_classes)}
    hd95_scores: dict[int, list[float]] = {class_id: [] for class_id in range(num_classes)}

    for sample, (gt, pred) in zip(samples, loaded_masks, strict=True):
        _validate_labels(gt, num_classes, ignore_index, sample.ground_truth_path)
        _validate_labels(pred, num_classes, ignore_index, sample.prediction_path)
        valid = gt != ignore_index if ignore_index is not None else np.ones(gt.shape, dtype=bool)
        gt_valid = gt[valid]
        pred_valid = pred[valid]
        confusion += confusion_matrix_from_arrays(gt_valid, pred_valid, num_classes)
        image_confusion = confusion_matrix_from_arrays(gt_valid, pred_valid, num_classes)
        per_image_scores.append(
            (
                sample.stem,
                overall_metrics(
                    image_confusion,
                    include_background=include_background,
                    background_label=background_label,
                )["mean_iou"],
            )
        )
        for class_id in range(num_classes):
            gt_class = (gt == class_id) & valid
            pred_class = (pred == class_id) & valid
            if not np.any(gt_class) and not np.any(pred_class):
                continue
            boundary_scores[class_id].append(boundary_iou(gt_class, pred_class))
            hd95_scores[class_id].append(hd95(gt_class, pred_class))

    class_metrics = per_class_metrics(confusion)
    per_class_boundary_iou = np.array(
        [aggregate_mean(boundary_scores[class_id]) for class_id in range(num_classes)],
        dtype=float,
    )
    per_class_hd95 = np.array(
        [aggregate_mean(hd95_scores[class_id]) for class_id in range(num_classes)],
        dtype=float,
    )
    selected = select_class_indices(confusion, include_background, background_label)
    per_class_df = pd.DataFrame(
        {
            "class_id": np.arange(num_classes),
            "class_name": class_names,
            "iou": class_metrics["iou"],
            "dice": class_metrics["dice"],
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1": class_metrics["f1"],
            "support": class_metrics["support"].astype(int),
            "frequency": class_metrics["frequency"],
            "boundary_iou": per_class_boundary_iou,
            "hd95": per_class_hd95,
        }
    )
    per_class_df["is_background"] = per_class_df["class_id"] == background_label
    per_class_df["is_included"] = selected
    per_class_df = per_class_df[per_class_df["is_included"]].reset_index(drop=True)

    overall = overall_metrics(
        confusion,
        include_background=include_background,
        background_label=background_label,
    )
    overall["boundary_iou"] = aggregate_mean(per_class_boundary_iou[selected])
    overall["hd95"] = aggregate_mean(per_class_hd95[selected])
    overall["background_included"] = float(include_background)

    support_series = per_class_df.set_index("class_name")["support"]
    stats: dict[str, Any] = {
        "num_images": len(samples),
        "image_resolutions": sorted({f"{height}x{width}" for height, width in image_shapes}),
        "classes": per_class_df["class_name"].tolist(),
        "ignored_background_label": background_label if not include_background else None,
        "pixels_per_class": {name: int(value) for name, value in support_series.items()},
        "class_distribution": {
            name: float(freq)
            for name, freq in per_class_df.set_index("class_name")["frequency"].items()
        },
        "largest_class": str(support_series.idxmax()),
        "smallest_class": str(support_series.idxmin()),
    }

    ranked = sorted(per_image_scores, key=lambda item: item[1], reverse=True)
    qualitative = {
        "best": [name for name, _ in ranked[:3]],
        "worst": [name for name, _ in ranked[-3:]],
        "random": [name for name, _ in ranked[: min(3, len(ranked))]],
    }

    report = EvaluationReport(
        overall_metrics=overall,
        per_class_metrics=per_class_df,
        confusion_matrix=pd.DataFrame(
            confusion[np.ix_(selected, selected)],
            index=np.array(class_names)[selected].tolist(),
            columns=np.array(class_names)[selected].tolist(),
        ),
        dataset_stats=stats,
        qualitative_examples=qualitative,
    )
    if save_dir is not None:
        report.save(save_dir)
    return report
