# SegBench

SegBench is a Python library for evaluating binary and multi-class image
segmentation predictions with a single `evaluate()` call.

## Quick start

```python
from segbench import evaluate

report = evaluate(
    ground_truth="ground_truth",
    predictions="predictions",
    images="images",
    class_names=["background", "road", "vehicle"],
    save_dir="results",
)

print(report.mean_iou)
report.save()
```

## CLI

```bash
segbench evaluate --gt ground_truth --pred predictions --images images --output results
```
