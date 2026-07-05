"""Evaluation report object and export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from segbench.visuals import render_figures


@dataclass
class EvaluationReport:
    overall_metrics: dict[str, float]
    per_class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    dataset_stats: dict[str, Any]
    qualitative_examples: dict[str, list[str]]

    @property
    def mean_iou(self) -> float:
        return self.overall_metrics["mean_iou"]

    @property
    def mean_dice(self) -> float:
        return self.overall_metrics["mean_dice"]

    def export_csv(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        metrics_dir = save_path / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([self.overall_metrics]).to_csv(metrics_dir / "overall.csv", index=False)
        self.per_class_metrics.to_csv(metrics_dir / "per_class.csv", index=False)
        self.confusion_matrix.to_csv(metrics_dir / "confusion_matrix.csv")

    def export_json(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        metrics_dir = save_path / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall_metrics": self.overall_metrics,
            "per_class_metrics": self.per_class_metrics.to_dict(orient="records"),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "dataset_stats": self.dataset_stats,
            "qualitative_examples": self.qualitative_examples,
        }
        (metrics_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_markdown(self, save_dir: str | Path) -> None:
        reports_dir = Path(save_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# SegBench Summary",
            "",
            "## Overall Metrics",
            "",
        ]
        for name, value in self.overall_metrics.items():
            lines.append(f"- {name}: {value:.4f}")
        lines.extend(
            [
                "",
                "## Dataset Stats",
                "",
            ]
        )
        for name, value in self.dataset_stats.items():
            lines.append(f"- {name}: {value}")
        (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def export_html(self, save_dir: str | Path) -> None:
        reports_dir = Path(save_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        rows = "".join(
            f"<tr><th>{key}</th><td>{value:.4f}</td></tr>" for key, value in self.overall_metrics.items()
        )
        html = (
            "<html><head><title>SegBench Report</title></head><body>"
            "<h1>SegBench Report</h1>"
            "<h2>Overall Metrics</h2>"
            f"<table>{rows}</table>"
            "<h2>Dataset Stats</h2>"
            f"{self.per_class_metrics.to_html(index=False)}"
            "</body></html>"
        )
        (reports_dir / "report.html").write_text(html, encoding="utf-8")

    def export_latex(self, save_dir: str | Path) -> None:
        reports_dir = Path(save_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        latex = "\\begin{tabular}{lr}\nMetric & Value \\\\\n\\hline\n"
        latex += "\n".join(f"{key} & {value:.4f} \\\\" for key, value in self.overall_metrics.items())
        latex += "\n\\end{tabular}\n"
        (reports_dir / "ieee_table.tex").write_text(latex, encoding="utf-8")

    def save(self, save_dir: str | Path | None = None) -> None:
        target_dir = Path(save_dir or "results")
        self.export_csv(target_dir)
        self.export_json(target_dir)
        self.export_markdown(target_dir)
        self.export_html(target_dir)
        self.export_latex(target_dir)
        render_figures(self, target_dir)
