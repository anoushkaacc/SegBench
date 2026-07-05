import numpy as np

from segbench.metrics import confusion_matrix_from_arrays, overall_metrics, per_class_metrics


def test_binary_metrics_are_computed_from_confusion_matrix():
    ground_truth = np.array([[0, 1], [1, 1]])
    prediction = np.array([[0, 1], [0, 1]])
    confusion = confusion_matrix_from_arrays(ground_truth, prediction, num_classes=2)

    per_class = per_class_metrics(confusion)
    overall = overall_metrics(confusion)

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

    overall = overall_metrics(confusion)
    assert np.isclose(overall["mean_recall"], (1.0 + 0.8) / 2)
