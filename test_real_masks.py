from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from segbench import evaluate


SOURCE_DIR = Path(r"D:\new base\base\originals\mask")

# Assumption:
# black is background, and the four foreground classes are green, blue, red, yellow.
COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (0, 255, 0): 1,
    (0, 0, 255): 2,
    (255, 0, 0): 3,
    (255, 255, 0): 4,
}

CLASS_NAMES = ["black", "green", "blue", "red", "yellow"]
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def rgb_to_label(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] < 3:
        raise ValueError(f"Expected an RGB mask, got shape {mask.shape}")

    rgb = mask[..., :3]
    labels = np.full(rgb.shape[:2], fill_value=-1, dtype=np.int64)

    for color, class_id in COLOR_TO_CLASS.items():
        matches = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
        labels[matches] = class_id

    unknown = labels == -1
    if np.any(unknown):
        unique_colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)
        preview = [tuple(int(channel) for channel in color) for color in unique_colors[:10]]
        raise ValueError(f"Found unmapped colors in mask: {preview}")

    return labels.astype(np.uint8)


def convert_directory(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        mask = np.asarray(Image.open(path).convert("RGB"))
        labels = rgb_to_label(mask)
        Image.fromarray(labels).save(target_dir / f"{path.stem}.png")
        count += 1

    if count == 0:
        raise ValueError(f"No supported mask files found in {source_dir}")
    return count


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Mask directory not found: {SOURCE_DIR}")

    workspace_tmp = Path("real_mask_test_output")
    if workspace_tmp.exists():
        shutil.rmtree(workspace_tmp)
    gt_dir = workspace_tmp / "ground_truth"
    pred_dir = workspace_tmp / "predictions"

    try:
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        count = convert_directory(SOURCE_DIR, gt_dir)
        shutil.copytree(gt_dir, pred_dir, dirs_exist_ok=True)

        report = evaluate(
            ground_truth=gt_dir,
            predictions=pred_dir,
            class_names=CLASS_NAMES,
            save_dir=workspace_tmp / "results",
        )

        print(f"Converted {count} masks from: {SOURCE_DIR}")
        print(f"Results saved to: {workspace_tmp / 'results'}")
        print("Overall metrics:")
        for name, value in report.overall_metrics.items():
            print(f"  {name}: {value}")
        print("\nPer-class metrics:")
        print(report.per_class_metrics.to_string(index=False))
    finally:
        print(f"\nTemporary working directory: {workspace_tmp}")


if __name__ == "__main__":
    main()
