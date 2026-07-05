"""Command line interface for SegBench."""

from __future__ import annotations

import argparse
from pathlib import Path

from segbench.core import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="segbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run segmentation evaluation.")
    evaluate_parser.add_argument("--gt", "--ground-truth", dest="ground_truth", required=True)
    evaluate_parser.add_argument("--pred", "--predictions", dest="predictions", required=True)
    evaluate_parser.add_argument("--images", dest="images")
    evaluate_parser.add_argument("--output", dest="save_dir", default="results")
    evaluate_parser.add_argument("--ignore-index", dest="ignore_index", type=int)
    evaluate_parser.add_argument("--class-names", nargs="*", dest="class_names")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        report = evaluate(
            ground_truth=args.ground_truth,
            predictions=args.predictions,
            images=args.images,
            class_names=args.class_names,
            ignore_index=args.ignore_index,
            save_dir=args.save_dir,
        )
        report.save(Path(args.save_dir))
        print(f"Saved evaluation to {args.save_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
