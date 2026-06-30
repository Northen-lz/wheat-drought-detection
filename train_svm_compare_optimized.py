# -*- coding: utf-8 -*-
"""
Train and compare SVM classifiers on the wheat drought 32-dimensional features.

The script compares:
1. Baseline SVM with default RBF kernel parameters.
2. Optimized SVM selected by cross-validated grid search.

Default input:
    data/raw/XyWheatCW.npy
    data/raw/YWheatCW.npy

Main outputs:
    svm_runs/<run_name>/reports/svm_comparison.csv
    svm_runs/<run_name>/reports/svm_grid_search_results.csv
    svm_runs/<run_name>/models/baseline_svm.joblib
    svm_runs/<run_name>/models/optimized_svm.joblib
    svm_runs/<run_name>/plots/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
CLASS_NAMES = ["control", "drought"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_from_csv(csv_path: Path, label_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    if label_col is None:
        label_col = df.columns[-1]
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy()
    return X, y, feature_cols


def load_from_npy(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D feature array, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
    return X.astype(np.float64), y, feature_names


def load_dataset(args: argparse.Namespace, base_dir: Path) -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, str]]:
    csv_path = Path(args.features_csv).resolve() if args.features_csv else base_dir / "data" / "wheat_32_features.csv"

    if args.features_csv or csv_path.exists():
        X, y, feature_names = load_from_csv(csv_path, args.label_col)
        source = {"type": "csv", "features_csv": str(csv_path)}
    else:
        x_path = Path(args.x_npy).resolve() if args.x_npy else base_dir / "data" / "raw" / "XyWheatCW.npy"
        y_path = Path(args.y_npy).resolve() if args.y_npy else base_dir / "data" / "raw" / "YWheatCW.npy"
        X, y, feature_names = load_from_npy(x_path, y_path)
        source = {"type": "npy", "x_npy": str(x_path), "y_npy": str(y_path)}

    return X, y, feature_names, source


def maybe_limit_samples(X: np.ndarray, y: np.ndarray, max_samples: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return X, y

    _, X_small, _, y_small = train_test_split(
        X,
        y,
        test_size=max_samples,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_small, y_small


def make_baseline_svm() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)),
    ])


def make_optimized_svm_search(cv_splits: int, quick: bool) -> GridSearchCV:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        # Keep probability disabled during grid search for speed; the final
        # optimized model is refit with probability=True for ROC/AUC output.
        ("svm", SVC(probability=False, random_state=RANDOM_STATE)),
    ])

    if quick:
        param_grid = [
            {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10], "svm__class_weight": [None, "balanced"]},
            {"svm__kernel": ["rbf"], "svm__C": [1, 10], "svm__gamma": ["scale", 0.01], "svm__class_weight": [None, "balanced"]},
        ]
    else:
        param_grid = [
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.01, 0.1, 1, 10, 100],
                "svm__class_weight": [None, "balanced"],
            },
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.1, 1, 10, 100],
                "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "svm__class_weight": [None, "balanced"],
            },
            {
                "svm__kernel": ["poly"],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01],
                "svm__degree": [2, 3],
                "svm__class_weight": [None, "balanced"],
            },
        ]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )


def make_final_optimized_svm(best_params: Dict) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
    model.set_params(**best_params)
    return model


def predict_probability(model: Pipeline, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_drought": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_drought": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_drought": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_auc"] = float(auc(fpr, tpr))
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary_df: pd.DataFrame, save_path: Path) -> None:
    metrics = ["accuracy", "f1_macro", "recall_drought", "roc_auc"]
    existing_metrics = [m for m in metrics if m in summary_df.columns]
    x = np.arange(len(summary_df))
    width = 0.8 / max(1, len(existing_metrics))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, metric in enumerate(existing_metrics):
        ax.bar(x + idx * width, summary_df[metric], width=width, label=metric)

    ax.set_xticks(x + width * (len(existing_metrics) - 1) / 2)
    ax.set_xticklabels(summary_df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("SVM baseline vs optimized comparison")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(
    model_name: str,
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reports_dir: Path,
    plots_dir: Path,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = predict_probability(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    (reports_dir / f"classification_report_{model_name}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, plots_dir / f"confusion_matrix_{model_name}.png", f"{model_name} confusion matrix")
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, plots_dir / f"roc_curve_{model_name}.png", f"{model_name} ROC curve")

    pred_df = pd.DataFrame({
        "true_label": y_test,
        "pred_label": y_pred,
    })
    if y_prob is not None:
        pred_df["prob_drought"] = y_prob
    pred_df.to_csv(reports_dir / f"predictions_{model_name}.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline SVM with optimized SVM for wheat drought classification.")
    parser.add_argument("--features_csv", type=str, default=None, help="Optional CSV feature table. Last column is label by default.")
    parser.add_argument("--label_col", type=str, default=None, help="Label column name when using --features_csv.")
    parser.add_argument("--x_npy", type=str, default=None, help="Feature npy path. Defaults to data/raw/XyWheatCW.npy.")
    parser.add_argument("--y_npy", type=str, default=None, help="Label npy path. Defaults to data/raw/YWheatCW.npy.")
    parser.add_argument("--output_root", type=str, default="svm_runs", help="Output root directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Run folder name. Defaults to timestamp.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out test ratio.")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for grid search.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for quick experiments.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional stratified sample limit for debugging.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_name = args.run_name or pd.Timestamp.now().strftime("svm_compare_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root).resolve() / run_name)
    reports_dir = ensure_dir(run_dir / "reports")
    plots_dir = ensure_dir(run_dir / "plots")
    models_dir = ensure_dir(run_dir / "models")

    X, y, feature_names, data_source = load_dataset(args, base_dir)
    X, y = maybe_limit_samples(X, y, args.max_samples)

    print("=" * 70)
    print("SVM comparison experiment")
    print(f"Data source: {data_source}")
    print(f"Samples: {len(y)}, features: {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    class_balance = {int(cls): int(count) for cls, count in zip(classes, counts)}
    print(f"Class balance: {class_balance}")
    print(f"Output directory: {run_dir}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    metadata = {
        "data_source": data_source,
        "samples": int(len(y)),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "test_size": args.test_size,
        "cv": args.cv,
        "quick": args.quick,
        "max_samples": args.max_samples,
        "random_state": RANDOM_STATE,
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTraining baseline SVM...")
    baseline_model = make_baseline_svm()
    baseline_model.fit(X_train, y_train)
    joblib.dump(baseline_model, models_dir / "baseline_svm.joblib")
    baseline_metrics = evaluate_and_save("baseline_svm", baseline_model, X_test, y_test, reports_dir, plots_dir)

    print("\nOptimizing SVM with GridSearchCV...")
    search = make_optimized_svm_search(cv_splits=args.cv, quick=args.quick)
    search.fit(X_train, y_train)
    optimized_model = make_final_optimized_svm(search.best_params_)
    optimized_model.fit(X_train, y_train)
    joblib.dump(optimized_model, models_dir / "optimized_svm.joblib")

    grid_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    grid_results.to_csv(reports_dir / "svm_grid_search_results.csv", index=False, encoding="utf-8-sig")

    optimized_metrics = evaluate_and_save("optimized_svm", optimized_model, X_test, y_test, reports_dir, plots_dir)

    summary_rows = [
        {"model": "baseline_svm", "best_params": json.dumps({"kernel": "rbf", "C": 1.0, "gamma": "scale"}), **baseline_metrics},
        {"model": "optimized_svm", "best_params": json.dumps(search.best_params_, ensure_ascii=False), **optimized_metrics},
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
    summary_df.to_csv(reports_dir / "svm_comparison.csv", index=False, encoding="utf-8-sig")
    plot_metric_comparison(summary_df, plots_dir / "svm_metric_comparison.png")

    best_info = {
        "best_model_by_test_f1_macro": summary_df.iloc[0]["model"],
        "grid_search_best_cv_f1_macro": float(search.best_score_),
        "grid_search_best_params": search.best_params_,
        "comparison_csv": str((reports_dir / "svm_comparison.csv").resolve()),
        "optimized_model": str((models_dir / "optimized_svm.joblib").resolve()),
    }
    (run_dir / "best_svm_summary.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBest GridSearchCV parameters:")
    print(search.best_params_)
    print("\nComparison results:")
    print(summary_df.to_string(index=False))
    print("\nFinished. Key files:")
    print(f"- {reports_dir / 'svm_comparison.csv'}")
    print(f"- {reports_dir / 'svm_grid_search_results.csv'}")
    print(f"- {models_dir / 'optimized_svm.joblib'}")
    print(f"- {plots_dir / 'svm_metric_comparison.png'}")


if __name__ == "__main__":
    main()
