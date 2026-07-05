"""Static visualization generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_bar(values: np.ndarray, labels: list[str], title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(labels))
    ax.barh(positions, values)
    ax.set_yticks(positions, labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_figures(report, save_dir: str | Path) -> None:
    figures_dir = Path(save_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_class = report.per_class_metrics
    labels = per_class["class_name"].tolist()

    _save_bar(per_class["support"].to_numpy(), labels, "Class Distribution", "Pixels", figures_dir / "class_distribution.png")
    _save_bar(per_class["iou"].to_numpy(), labels, "Per-Class IoU", "IoU", figures_dir / "iou_per_class.png")
    _save_bar(per_class["dice"].to_numpy(), labels, "Per-Class Dice", "Dice", figures_dir / "dice_per_class.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = report.confusion_matrix.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
