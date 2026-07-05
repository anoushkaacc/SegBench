import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import torch
from torchmetrics.classification import MulticlassJaccardIndex

from segbench.metrics import (
    binary_boundary_iou,
    binary_hd95,
    confusion_matrix_from_arrays,
    mask_to_boundary,
    overall_metrics,
    per_class_metrics,
)


def test_binary_metrics_are_computed_from_confusion_matrix():
    ground_truth = np.array([[0, 1], [1, 1]])
    prediction = np.array([[0, 1], [0, 1]])
    confusion = confusion_matrix_from_arrays(ground_truth, prediction, num_classes=2)

    per_class = per_class_metrics(confusion)
    overall = overall_metrics(confusion, include_background=True)

    assert confusion.tolist() == [[1, 0], [1, 2]]
    assert np.allclose(per_class["iou"], [0.5, 2 / 3])
    assert np.allclose(per_class["dice"], [2 / 3, 0.8])
    assert overall["pixel_accuracy"] == 0.75
    assert np.isclose(overall["mean_iou"], (0.5 + (2 / 3)) / 2)


def test_missing_class_is_ignored_in_macro_average():
    confusion = np.array(
        [
            [5, 0, 0],
            [1, 4, 0],
            [0, 0, 0],
        ]
    )

    overall = overall_metrics(confusion, include_background=True)
    assert np.isclose(overall["mean_recall"], (1.0 + 0.8) / 2)


def test_overall_metrics_excludes_background_by_default_when_requested():
    confusion = np.array(
        [
            [50, 10],
            [20, 20],
        ]
    )

    overall = overall_metrics(confusion, include_background=False, background_label=0)
    assert np.isclose(overall["mean_iou"], 20 / (20 + 10 + 20))
    assert np.isclose(overall["mean_recall"], 20 / 40)


def test_multiclass_metrics_match_reference_libraries():
    ground_truth = np.array([[0, 1, 2], [1, 2, 2], [0, 1, 2]])
    prediction = np.array([[0, 2, 2], [1, 2, 1], [0, 1, 0]])

    confusion = confusion_matrix_from_arrays(ground_truth, prediction, num_classes=3)
    per_class = per_class_metrics(confusion)
    overall = overall_metrics(confusion, include_background=False, background_label=0)

    y_true = ground_truth.ravel()
    y_pred = prediction.ravel()
    labels = [1, 2]

    expected_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    expected_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    expected_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    expected_iou = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    torch_jaccard = MulticlassJaccardIndex(num_classes=3, average=None)
    torch_iou = torch_jaccard(
        torch.tensor(prediction, dtype=torch.int64),
        torch.tensor(ground_truth, dtype=torch.int64),
    ).numpy()

    assert np.allclose(per_class["precision"][1:], expected_precision)
    assert np.allclose(per_class["recall"][1:], expected_recall)
    assert np.allclose(per_class["f1"][1:], expected_f1)
    assert np.allclose(per_class["iou"][1:], expected_iou)
    assert np.allclose(per_class["iou"], torch_iou)
    assert np.isclose(overall["mean_iou"], np.mean(expected_iou))


def test_boundary_metrics_are_perfect_for_identical_masks():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True

    assert binary_boundary_iou(mask, mask) == 1.0
    assert binary_hd95(mask, mask) == 0.0


def test_hd95_matches_bruteforce_for_simple_shift():
    gt = np.zeros((6, 6), dtype=bool)
    pred = np.zeros((6, 6), dtype=bool)
    gt[2:4, 2:4] = True
    pred[2:4, 3:5] = True

    gt_boundary = mask_to_boundary(gt, dilation_ratio=0.0)
    pred_boundary = mask_to_boundary(pred, dilation_ratio=0.0)
    gt_points = np.argwhere(gt_boundary)
    pred_points = np.argwhere(pred_boundary)
    directed = np.concatenate([cdist(gt_points, pred_points).min(axis=1), cdist(pred_points, gt_points).min(axis=1)])
    expected_hd95 = np.percentile(directed, 95)

    assert np.isclose(binary_hd95(gt, pred), expected_hd95)
    assert 0.0 <= binary_boundary_iou(gt, pred) < 1.0
