
"""
train_rgb_ml_baseline.py

对照实验方案：
单株 RGB 图像 -> 提取植被指数/颜色/纹理特征 -> 多机器学习模型对比 -> 网格搜索选最优参数 -> 保存模型与结果

支持两种数据组织方式：
1) ImageFolder 结构（不带显式划分）：
   data_dir/
      control/
      drought/

2) 带划分结构：
   data_dir/
      train/control
      train/drought
      val/control
      val/drought
      test/control
      test/drought

输出目录默认：
./ml_rgb_results/<时间戳>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ["control", "drought"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in ALLOWED_EXTS])


def detect_dataset_layout(data_dir: Path) -> str:
    split_dirs = [data_dir / "train", data_dir / "val", data_dir / "test"]
    if any(d.exists() for d in split_dirs):
        return "with_split"
    return "flat"


def load_dataset_records(data_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    layout = detect_dataset_layout(data_dir)

    if layout == "with_split":
        for split in ["train", "val", "test"]:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            for class_name in CLASS_NAMES:
                class_dir = split_dir / class_name
                for img_path in find_images(class_dir):
                    records.append({
                        "image_path": str(img_path),
                        "label_name": class_name,
                        "label": CLASS_NAMES.index(class_name),
                        "split": split,
                    })
    else:
        for class_name in CLASS_NAMES:
            class_dir = data_dir / class_name
            for img_path in find_images(class_dir):
                records.append({
                    "image_path": str(img_path),
                    "label_name": class_name,
                    "label": CLASS_NAMES.index(class_name),
                    "split": "unsplit",
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到可用图像。\n"
            f"请检查目录是否满足：control/drought 或 train|val|test/control|drought"
        )
    return df


def make_splits_if_needed(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    if set(df["split"].unique()) != {"unsplit"}:
        return df.copy()

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        stratify=train_val_df["label"],
        random_state=random_state,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return a / (b + eps)


def build_veg_mask(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]
    exg = 2.0 * g - r - b
    exg_u8 = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
    try:
        _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask > 0
    except Exception:
        mask = np.ones(exg.shape, dtype=bool)

    if mask.mean() < 0.03:
        mask = np.ones(exg.shape, dtype=bool)

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0
    if mask.mean() < 0.03:
        mask = np.ones(exg.shape, dtype=bool)
    return mask


def masked_stats(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    values = arr[mask]
    if values.size == 0:
        values = arr.reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def compute_color_and_index_features(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    rgb_f = rgb.astype(np.float32) / 255.0
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] /= 255.0
    lab[:, :, 1] /= 255.0
    lab[:, :, 2] /= 255.0

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    ngrdi = safe_divide(g - r, g + r)
    vari = safe_divide(g - r, g + r - b)
    gli = safe_divide(2 * g - r - b, 2 * g + r + b)
    rgbvi = safe_divide(g * g - r * b, g * g + r * b)
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745 / 255.0

    features: Dict[str, float] = {}
    for name, channel in [
        ("r", r), ("g", g), ("b", b),
        ("h", hsv[:, :, 0]), ("s", hsv[:, :, 1]), ("v", hsv[:, :, 2]),
        ("l", lab[:, :, 0]), ("a", lab[:, :, 1]), ("bb", lab[:, :, 2]),
        ("exg", exg), ("exr", exr), ("ngrdi", ngrdi), ("vari", vari),
        ("gli", gli), ("rgbvi", rgbvi), ("cive", cive),
    ]:
        features.update(masked_stats(channel, mask, name))

    features["veg_area_ratio"] = float(mask.mean())
    return features


def compute_texture_features(rgb: np.ndarray, levels: int = 32) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    quant = np.floor(gray_resized.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quant = np.clip(quant, 0, levels - 1)

    glcm = graycomatrix(
        quant,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    features: Dict[str, float] = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).reshape(-1)
        features[f"glcm_{prop}_mean"] = float(np.mean(vals))
        features[f"glcm_{prop}_std"] = float(np.std(vals))
    return features


def extract_features_from_image(image_path: Path, image_size: int = 224) -> Dict[str, float]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"无法读取图像：{image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    mask = build_veg_mask(rgb)
    features = {}
    features.update(compute_color_and_index_features(rgb, mask))
    features.update(compute_texture_features(rgb))
    return features


def build_feature_table(df: pd.DataFrame, image_size: int = 224) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        feats = extract_features_from_image(img_path, image_size=image_size)
        feats["image_path"] = str(img_path)
        feats["label"] = int(row["label"])
        feats["label_name"] = row["label_name"]
        feats["split"] = row["split"]
        rows.append(feats)
    return pd.DataFrame(rows)


def get_model_search_spaces(use_xgboost: bool = False) -> Dict[str, Tuple[Pipeline, Dict]]:
    models: Dict[str, Tuple[Pipeline, Dict]] = {
        "RandomForest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "ExtraTrees": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "SVM": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.001],
                "clf__kernel": ["rbf"],
            },
        ),
        "LogisticRegression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__solver": ["lbfgs"],
            },
        ),
        "GradientBoosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }

    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = (
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", XGBClassifier(
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_estimators=300,
                        n_jobs=-1,
                    )),
                ]),
                {
                    "clf__n_estimators": [200, 400],
                    "clf__max_depth": [3, 5],
                    "clf__learning_rate": [0.03, 0.1],
                    "clf__subsample": [0.8, 1.0],
                    "clf__colsample_bytree": [0.8, 1.0],
                },
            )
        except Exception:
            print("提示：未安装 xgboost，已跳过 XGBoost。")
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_binary": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_binary": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bar(summary_df: pd.DataFrame, metric: str, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order_df = summary_df.sort_values(metric, ascending=False)
    ax.bar(order_df["model_name"], order_df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticklabels(order_df["model_name"], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(best_model, feature_names: List[str], save_path: Path, top_k: int = 20) -> None:
    clf = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    if not hasattr(clf, "feature_importances_"):
        return
    importances = np.asarray(clf.feature_importances_)
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names))[::-1], vals)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_title("Top 特征重要性")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB 单株图像 + 植被指数 + 机器学习对照实验")
    parser.add_argument("--data_dir", type=str, required=True, help="分类数据集目录")
    parser.add_argument("--output_root", type=str, default="./ml_rgb_results", help="结果输出根目录")
    parser.add_argument("--run_name", type=str, default=None, help="本次实验名称；默认自动生成")
    parser.add_argument("--image_size", type=int, default=224, help="提取特征时统一缩放尺寸")
    parser.add_argument("--cv", type=int, default=5, help="GridSearchCV 折数")
    parser.add_argument("--use_xgboost", action="store_true", help="若已安装 xgboost，则加入对比")
    parser.add_argument("--val_size", type=float, default=0.15, help="无显式划分时的验证集比例")
    parser.add_argument("--test_size", type=float, default=0.15, help="无显式划分时的测试集比例")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = args.run_name or pd.Timestamp.now().strftime("rgb_ml_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / run_name)
    features_dir = ensure_dir(run_dir / "features")
    reports_dir = ensure_dir(run_dir / "reports")
    plots_dir = ensure_dir(run_dir / "plots")
    models_dir = ensure_dir(run_dir / "models")
    best_dir = ensure_dir(run_dir / "best_model")

    print(f"数据目录：{data_dir}")
    print(f"结果目录：{run_dir}")

    raw_df = load_dataset_records(data_dir)
    all_df = make_splits_if_needed(raw_df, val_size=args.val_size, test_size=args.test_size)
    all_df.to_csv(features_dir / "dataset_records.csv", index=False, encoding="utf-8-sig")

    print("开始提取 RGB 指数、颜色和纹理特征...")
    feat_df = build_feature_table(all_df, image_size=args.image_size)
    feat_df.to_csv(features_dir / "features_all.csv", index=False, encoding="utf-8-sig")

    feature_cols = [c for c in feat_df.columns if c not in ["image_path", "label", "label_name", "split"]]
    (features_dir / "feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"特征维度：{len(feature_cols)}")

    train_df = feat_df[feat_df["split"] == "train"].reset_index(drop=True)
    val_df = feat_df[feat_df["split"] == "val"].reset_index(drop=True)
    test_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = get_model_search_spaces(use_xgboost=args.use_xgboost)

    comparison_rows = []
    best_val_f1 = -1.0
    best_name = None
    best_model = None

    print("开始多模型网格搜索与对比...")
    for model_name, (pipe, param_grid) in search_spaces.items():
        print(f"\n>>> 训练模型：{model_name}")
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)

        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)

        joblib.dump(model, models_dir / f"{model_name}_best.joblib")

        (reports_dir / f"classification_report_val_{model_name}.txt").write_text(
            classification_report(y_val, y_val_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),
            encoding="utf-8"
        )
        (reports_dir / f"classification_report_test_{model_name}.txt").write_text(
            classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),
            encoding="utf-8"
        )

        pred_df = test_df[["image_path", "label_name"]].copy()
        pred_df["pred_label"] = y_test_pred
        pred_df["pred_label_name"] = [CLASS_NAMES[i] for i in y_test_pred]
        if y_test_prob is not None:
            pred_df["prob_drought"] = y_test_prob
        pred_df.to_csv(reports_dir / f"predictions_test_{model_name}.csv", index=False, encoding="utf-8-sig")

        cm = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm, CLASS_NAMES, plots_dir / f"confusion_matrix_{model_name}.png", f"{model_name} 测试集混淆矩阵")
        if y_test_prob is not None:
            plot_roc(y_test, y_test_prob, plots_dir / f"roc_{model_name}.png", f"{model_name} 测试集 ROC 曲线")

        row = {
            "model_name": model_name,
            "best_params": json.dumps(gs.best_params_, ensure_ascii=False),
            "cv_best_f1_macro": float(gs.best_score_),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
        }
        if "roc_auc" in val_metrics:
            row["val_roc_auc"] = val_metrics["roc_auc"]
        if "roc_auc" in test_metrics:
            row["test_roc_auc"] = test_metrics["roc_auc"]
        comparison_rows.append(row)

        print(f"{model_name} 最优参数：{gs.best_params_}")
        print(f"{model_name} 验证集 F1-macro：{val_metrics['f1_macro']:.4f}")
        print(f"{model_name} 测试集 F1-macro：{test_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_name = model_name
            best_model = model

    if best_model is None or best_name is None:
        raise RuntimeError("未成功训练任何模型。")

    summary_df = pd.DataFrame(comparison_rows).sort_values("val_f1_macro", ascending=False)
    summary_df.to_csv(reports_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    plot_bar(summary_df, "val_f1_macro", plots_dir / "compare_val_f1_macro.png", "各模型验证集 F1-macro 对比")
    plot_bar(summary_df, "test_f1_macro", plots_dir / "compare_test_f1_macro.png", "各模型测试集 F1-macro 对比")
    plot_bar(summary_df, "test_accuracy", plots_dir / "compare_test_accuracy.png", "各模型测试集 Accuracy 对比")

    joblib.dump(best_model, best_dir / "best_model.joblib")
    best_row = summary_df.iloc[0].to_dict()
    metadata = {
        "best_model_name": best_name,
        "feature_count": len(feature_cols),
        "feature_columns_file": str((features_dir / "feature_columns.txt").resolve()),
        "class_names": CLASS_NAMES,
        "data_dir": str(data_dir),
        "image_size": args.image_size,
        "best_summary": best_row,
    }
    save_json(metadata, best_dir / "best_model_meta.json")

    try:
        plot_feature_importance(best_model, feature_cols, plots_dir / "best_model_feature_importance.png", top_k=20)
    except Exception:
        pass

    report_txt = f"""
RGB 单株图像 + 植被指数 + 机器学习对照实验已完成。

【输入数据】
- 数据目录：{data_dir}
- 图像尺寸统一：{args.image_size}
- 类别：{CLASS_NAMES}

【数据划分】
- train：{len(train_df)}
- val：{len(val_df)}
- test：{len(test_df)}

【最优模型】
- 模型名称：{best_name}
- 验证集 F1-macro：{best_row.get('val_f1_macro')}
- 测试集 F1-macro：{best_row.get('test_f1_macro')}
- 测试集 Accuracy：{best_row.get('test_accuracy')}

【关键输出位置】
- 特征表：{features_dir / 'features_all.csv'}
- 模型对比：{reports_dir / 'model_comparison.csv'}
- 最优模型：{best_dir / 'best_model.joblib'}
- 最优模型元信息：{best_dir / 'best_model_meta.json'}
- 图表目录：{plots_dir}
"""
    (run_dir / "README_result.txt").write_text(report_txt.strip(), encoding="utf-8")

    print("\n================ 完成 ================")
    print(report_txt)


if __name__ == "__main__":
    main()
