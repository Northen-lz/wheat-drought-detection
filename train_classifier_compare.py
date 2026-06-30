from pathlib import Path
import json
import csv
import time
import pandas as pd
from ultralytics import YOLO


def read_cls_result_csv(csv_path: Path):
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    last = df.iloc[-1]

    return {
        "top1_acc": float(last.get("metrics/accuracy_top1", 0)),
        "top5_acc": float(last.get("metrics/accuracy_top5", 0)),
        "train_loss": float(last.get("train/loss", 0)),
        "val_loss": float(last.get("val/loss", 0)),
    }


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent

    # ===== 已有增强后8n结果 =====
    n_run_dir = base_dir / "wheat_drought_runs" / "exp_augmented2"

    # ===== 新训练8s =====
    s_model_path = "yolov8s-cls.pt"   # 直接用官方预训练权重名
    data_dir = base_dir / "data" / "wheat_drought_data_upsampled"
    project_dir = base_dir / "wheat_drought_runs"
    s_run_name = "exp_augmented2_s"

    # ===== 读取已有8n结果 =====
    n_csv = n_run_dir / "results.csv"
    n_metrics = read_cls_result_csv(n_csv)

    if n_metrics is None:
        print(f"未找到增强后 8n 的结果文件：{n_csv}")
        raise SystemExit

    print("已读取增强后 yolov8n-cls 结果：")
    print(n_metrics)

    # ===== 开始训练8s =====
    print("\n开始训练 yolov8s-cls（增强后数据）...")
    model = YOLO(s_model_path)

    start_time = time.time()

    model.train(
        data=str(data_dir),
        epochs=50,
        imgsz=32,
        batch=16,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.9,
        augment=True,
        project=str(project_dir),
        name=s_run_name,
        workers=2
    )

    train_time = round(time.time() - start_time, 2)

    # ===== 验证 =====
    model.val()

    # ===== 导出 ONNX =====
    model.export(format='onnx')

    # ===== 读取8s结果 =====
    s_run_dir = project_dir / s_run_name
    s_csv = s_run_dir / "results.csv"
    s_metrics = read_cls_result_csv(s_csv)

    if s_metrics is None:
        print(f"未找到 8s 的结果文件：{s_csv}")
        raise SystemExit

    # ===== 汇总对比 =====
    compare_rows = [
        {
            "model": "yolov8n-cls",
            "run_name": "exp_augmented2",
            "top1_acc": n_metrics["top1_acc"],
            "top5_acc": n_metrics["top5_acc"],
            "train_loss": n_metrics["train_loss"],
            "val_loss": n_metrics["val_loss"],
            "train_time_sec": ""
        },
        {
            "model": "yolov8s-cls",
            "run_name": s_run_name,
            "top1_acc": s_metrics["top1_acc"],
            "top5_acc": s_metrics["top5_acc"],
            "train_loss": s_metrics["train_loss"],
            "val_loss": s_metrics["val_loss"],
            "train_time_sec": train_time
        }
    ]

    out_csv = project_dir / "classifier_compare_augmented_n_vs_s.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=compare_rows[0].keys())
        writer.writeheader()
        writer.writerows(compare_rows)

    out_json = project_dir / "classifier_compare_augmented_n_vs_s.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(compare_rows, f, ensure_ascii=False, indent=2)

    print("\n=== 增强后分类模型对比结果 ===")
    for row in compare_rows:
        print(row)

    print(f"\n对比结果已保存：{out_csv}")
    print(f"对比结果已保存：{out_json}")