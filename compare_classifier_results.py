# -*- coding: utf-8 -*-
"""
Collect SVM and YOLO classification results into one comparison table.

This script does not retrain models. It only reads existing result files and
normalizes the available metrics into a single CSV for reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def read_svm_comparison(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"SVM comparison file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows: List[Dict] = []
    for _, row in df.iterrows():
        rows.append({
            "model": row["model"],
            "family": "SVM",
            "run": csv_path.parents[1].name,
            "eval_split": "holdout_test",
            "accuracy": row.get("accuracy"),
            "f1_macro": row.get("f1_macro"),
            "precision_macro": row.get("precision_macro"),
            "recall_macro": row.get("recall_macro"),
            "roc_auc": row.get("roc_auc"),
            "train_loss": None,
            "val_loss": None,
            "best_params": row.get("best_params"),
            "source_file": str(csv_path),
        })
    return rows


def read_yolo_results(results_csv: Path, model_name: str, run_name: Optional[str] = None) -> Optional[Dict]:
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    if df.empty:
        return None

    last = df.iloc[-1]
    return {
        "model": model_name,
        "family": "YOLO-CLS",
        "run": run_name or results_csv.parent.name,
        "eval_split": "val",
        "accuracy": float(last.get("metrics/accuracy_top1", 0)),
        "f1_macro": None,
        "precision_macro": None,
        "recall_macro": None,
        "roc_auc": None,
        "train_loss": float(last.get("train/loss", 0)),
        "val_loss": float(last.get("val/loss", 0)),
        "best_params": "see training script/results.yaml",
        "source_file": str(results_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing classifier results into one CSV.")
    parser.add_argument(
        "--svm_csv",
        type=str,
        default="svm_runs/svm_compare_20260511_224516/reports/svm_comparison.csv",
        help="Path to SVM comparison CSV.",
    )
    parser.add_argument("--output", type=str, default="outputs/classifier_model_comparison.csv")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    svm_csv = (base_dir / args.svm_csv).resolve() if not Path(args.svm_csv).is_absolute() else Path(args.svm_csv)
    output = (base_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = read_svm_comparison(svm_csv)

    yolo_runs = [
        ("yolov8n-cls", base_dir / "wheat_drought_runs" / "exp_augmented2" / "results.csv", "exp_augmented2"),
        ("yolov8s-cls", base_dir / "wheat_drought_runs" / "exp_augmented2_s" / "results.csv", "exp_augmented2_s"),
        ("yolov8n-cls-old", base_dir / "wheat_drought_runs" / "exp1_upsampled4" / "results.csv", "exp1_upsampled4"),
    ]
    for model_name, path, run_name in yolo_runs:
        row = read_yolo_results(path, model_name, run_name)
        if row is not None:
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Saved comparison table: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
