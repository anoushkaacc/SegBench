from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from segbench import evaluate
from segbench.exceptions import ValidationError


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype("uint8")).save(path)


def test_evaluate_generates_basic_report(tmp_path: Path):
    gt_dir = tmp_path / "ground_truth"
    pred_dir = tmp_path / "predictions"
    gt_dir.mkdir()
    pred_dir.mkdir()

    _write_mask(gt_dir / "0001.png", np.array([[0, 1], [1, 1]], dtype=np.uint8))
    _write_mask(pred_dir / "0001.png", np.array([[0, 1], [0, 1]], dtype=np.uint8))

    report = evaluate(gt_dir, pred_dir, class_names=["background", "foreground"])

    assert report.mean_iou > 0
    assert list(report.per_class_metrics["class_name"]) == ["background", "foreground"]
    assert report.dataset_stats["num_images"] == 1


def test_evaluate_rejects_mismatched_shapes(tmp_path: Path):
    gt_dir = tmp_path / "ground_truth"
    pred_dir = tmp_path / "predictions"
    gt_dir.mkdir()
    pred_dir.mkdir()

    _write_mask(gt_dir / "0001.png", np.zeros((2, 2), dtype=np.uint8))
    _write_mask(pred_dir / "0001.png", np.zeros((3, 3), dtype=np.uint8))

    with pytest.raises(ValidationError):
        evaluate(gt_dir, pred_dir)
