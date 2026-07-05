from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from segbench import evaluate


def main() -> None:
    root = Path("benchmark_workspace")
    if root.exists():
        shutil.rmtree(root)

    gt_dir = root / "ground_truth"
    pred_dir = root / "predictions"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    num_images = 200
    shape = (256, 256)

    for index in range(num_images):
        gt = rng.integers(0, 3, size=shape, dtype=np.uint8)
        noise = rng.random(shape) < 0.08
        pred = gt.copy()
        pred[noise] = rng.integers(0, 3, size=int(noise.sum()), dtype=np.uint8)
        Image.fromarray(gt).save(gt_dir / f"{index:04d}.png")
        Image.fromarray(pred).save(pred_dir / f"{index:04d}.png")

    start = time.perf_counter()
    report = evaluate(
        ground_truth=gt_dir,
        predictions=pred_dir,
        class_names=["background", "class_1", "class_2"],
        save_dir=root / "results",
    )
    elapsed = time.perf_counter() - start

    pixels = num_images * shape[0] * shape[1]
    print(f"images={num_images}")
    print(f"shape={shape[0]}x{shape[1]}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"images_per_second={num_images / elapsed:.2f}")
    print(f"pixels_per_second={pixels / elapsed:.0f}")
    print(f"mean_iou={report.mean_iou:.4f}")


if __name__ == "__main__":
    main()
