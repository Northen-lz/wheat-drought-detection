# -*- coding: utf-8 -*-
"""
使用相同的 YOLO 验证/测试集评估 SVM 和 YOLO 分类器。
默认统一测试集：

data/wheat_drought_data_upsampled/val/control
data/wheat_drought_data_upsampled/val/drought

对于 SVM将每个 32，32 的特征图 resize 回 4×8 后展平，得到 32 维特征向量。这样可以保证 SVM 和 YOLO 在完全相同的图像文件上进行评估。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
CLASS_NAMES = ["control", "drought"]
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_images(split_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for image_path in sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTS):
            rows.append({
                "image_path": str(image_path),
                "label": label,
                "label_name": class_name,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No images found in {split_dir}")
    return df


def stratified_limit(df: pd.DataFrame, max_samples: Optional[int]) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)
    _, sample_df = train_test_split(
        df,
        test_size=max_samples,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    return sample_df.sort_values("image_path").reset_index(drop=True)


def image_to_32_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        gray = img.convert("L")
        small = gray.resize((8, 4), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def build_feature_matrix(records: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([image_to_32_features(Path(p)) for p in records["image_path"]])
    y = records["label"].to_numpy(dtype=int)
    return X, y


def make_svm_baseline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=RANDOM_STATE)),
    ])


def train_optimized_svm(X_train: np.ndarray, y_train: np.ndarray, cv: int) -> GridSearchCV:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=False, random_state=RANDOM_STATE)),
    ])
    param_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10], "svm__class_weight": [None, "balanced"]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1], "svm__class_weight": [None, "balanced"]},
    ]
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_drought": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_drought": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_drought": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def plot_confusion(cm: np.ndarray, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
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


def save_model_eval(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_records: pd.DataFrame,
    reports_dir: Path,
    plots_dir: Path,
    extra: Optional[Dict] = None,
) -> Dict:
    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pred_df = test_records[["image_path", "label", "label_name"]].copy()
    pred_df["pred_label"] = y_pred
    pred_df["pred_label_name"] = [CLASS_NAMES[int(i)] for i in y_pred]
    pred_df.to_csv(reports_dir / f"predictions_{model_name}.csv", index=False, encoding="utf-8-sig")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    (reports_dir / f"classification_report_{model_name}.txt").write_text(report, encoding="utf-8")
    plot_confusion(cm, plots_dir / f"confusion_matrix_{model_name}.png", f"{model_name} confusion matrix")

    row = {
        "model": model_name,
        **metrics,
        "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False),
    }
    if extra:
        row.update(extra)
    return row


def predict_yolo(model_path: Path, image_paths: Iterable[str], batch: int, imgsz: int) -> np.ndarray:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required to evaluate YOLO models") from exc

    model = YOLO(str(model_path))
    preds: List[int] = []
    results = model.predict(
        source=list(image_paths),
        imgsz=imgsz,
        batch=batch,
        verbose=False,
        stream=True,
    )
    for result in results:
        preds.append(int(result.probs.top1))
    return np.asarray(preds, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVM and YOLO models on the same YOLO validation/test set.")
    parser.add_argument("--data_dir", type=str, default="data/wheat_drought_data_upsampled", help="YOLO ImageFolder dataset root.")
    parser.add_argument("--test_split", type=str, default="val", help="Split used as unified test set, usually val or test.")
    parser.add_argument("--output_root", type=str, default="outputs/unified_eval", help="Output root directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Run folder name.")
    parser.add_argument("--svm_train_samples", type=int, default=20000, help="Stratified SVM training sample limit; <=0 means full train split.")
    parser.add_argument("--svm_cv", type=int, default=3, help="CV folds for optimized SVM.")
    parser.add_argument("--skip_optimized_svm", action="store_true", help="Only run baseline SVM.")
    parser.add_argument("--skip_yolo", action="store_true", help="Skip YOLO evaluation.")
    parser.add_argument("--yolo_batch", type=int, default=64, help="YOLO prediction batch size.")
    parser.add_argument("--imgsz", type=int, default=32, help="YOLO image size.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    run_name = args.run_name or pd.Timestamp.now().strftime("unified_yolo_val_%Y%m%d_%H%M%S")
    run_dir = ensure_dir((base_dir / args.output_root).resolve() / run_name)
    reports_dir = ensure_dir(run_dir / "reports")
    plots_dir = ensure_dir(run_dir / "plots")
    models_dir = ensure_dir(run_dir / "models")

    train_records_all = find_images(data_dir / "train")
    train_records = stratified_limit(train_records_all, None if args.svm_train_samples <= 0 else args.svm_train_samples)
    test_records = find_images(data_dir / args.test_split).reset_index(drop=True)

    print("=" * 70)
    print("Unified classifier evaluation")
    print(f"Dataset: {data_dir}")
    print(f"Unified test split: {args.test_split}, samples: {len(test_records)}")
    print(f"SVM train samples: {len(train_records)} / {len(train_records_all)}")
    print(f"Output directory: {run_dir}")

    X_train, y_train = build_feature_matrix(train_records)
    X_test, y_test = build_feature_matrix(test_records)

    rows: List[Dict] = []

    print("\nTraining baseline SVM on YOLO train images...")
    baseline_svm = make_svm_baseline()
    baseline_svm.fit(X_train, y_train)
    joblib.dump(baseline_svm, models_dir / "baseline_svm_from_yolo_train.joblib")
    y_pred = baseline_svm.predict(X_test)
    rows.append(save_model_eval(
        "baseline_svm",
        y_test,
        y_pred,
        test_records,
        reports_dir,
        plots_dir,
        {"family": "SVM", "train_source": "yolo_train_images", "best_params": json.dumps({"C": 1.0, "gamma": "scale", "kernel": "rbf"})},
    ))

    if not args.skip_optimized_svm:
        print("\nOptimizing SVM on YOLO train images...")
        search = train_optimized_svm(X_train, y_train, cv=args.svm_cv)
        optimized_svm = search.best_estimator_
        joblib.dump(optimized_svm, models_dir / "optimized_svm_from_yolo_train.joblib")
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            reports_dir / "optimized_svm_grid_search_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        y_pred = optimized_svm.predict(X_test)
        rows.append(save_model_eval(
            "optimized_svm",
            y_test,
            y_pred,
            test_records,
            reports_dir,
            plots_dir,
            {
                "family": "SVM",
                "train_source": "yolo_train_images",
                "cv_best_f1_macro": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
            },
        ))

    if not args.skip_yolo:
        yolo_models = [
            ("yolov8n_cls", base_dir / "wheat_drought_runs" / "exp_augmented2" / "weights" / "best.pt"),
            ("yolov8s_cls", base_dir / "wheat_drought_runs" / "exp_augmented2_s" / "weights" / "best.pt"),
        ]
        for model_name, model_path in yolo_models:
            if not model_path.exists():
                print(f"Skip {model_name}, weight not found: {model_path}")
                continue
            print(f"\nEvaluating {model_name} on unified test split...")
            y_pred = predict_yolo(model_path, test_records["image_path"], batch=args.yolo_batch, imgsz=args.imgsz)
            rows.append(save_model_eval(
                model_name,
                y_test,
                y_pred,
                test_records,
                reports_dir,
                plots_dir,
                {"family": "YOLO-CLS", "train_source": "yolo_train_images", "best_params": str(model_path)},
            ))

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(reports_dir / "unified_model_comparison.csv", index=False, encoding="utf-8-sig")

    config = {
        "data_dir": str(data_dir),
        "test_split": args.test_split,
        "test_samples": int(len(test_records)),
        "svm_train_samples": int(len(train_records)),
        "svm_train_total_available": int(len(train_records_all)),
        "class_names": CLASS_NAMES,
        "note": "All listed models are evaluated on the same YOLO split image files.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nUnified comparison:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {reports_dir / 'unified_model_comparison.csv'}")


if __name__ == "__main__":
    main()
