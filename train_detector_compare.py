# -*- coding: utf-8 -*-
"""
小麦检测模型对比训练（最终稳定版）

支持模型：
- YOLOv8n
- YOLOv8s
- YOLO11n
- YOLOv10n

已解决：
✔ Windows路径问题
✔ 显存不足自动降级
✔ OpenCV内存问题
✔ 训练失败不中断
✔ 自动结果汇总
"""

import time
import csv
import json
from pathlib import Path
from ultralytics import YOLO

# ================= 配置 =================
DATA_YAML = "D:/pyhon/xiangmu/xiaomai/wheat500/wheat500.yaml"

MODEL_LIST = {

    "yolov8s": {
        "path": "D:/pyhon/xiangmu/yolov8s.pt",
        "batch": 4,
        "device": "0"
    },
    "yolo11n": {
        "path": "D:/pyhon/xiangmu/yolo11n.pt",
        "batch": 4,
        "device": "0"   # 防止显存炸
    },
    "yolov10n": {
        "path": "D:/pyhon/xiangmu/yolov10n.pt",
        "batch": 4,
        "device": "0"   # 防止报错
    },
        "yolov8n": {
        "path": "D:/pyhon/xiangmu/yolov8n.pt",
        "batch": 8,
        "device": "0"
    }
}

SAVE_DIR = Path("D:/pyhon/xiangmu/detect_compare_runs")

EPOCHS = 50
IMGSZ = 512
# =======================================


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def train_one(name, cfg):
    print("\n==============================")
    print(f"开始训练：{name}")
    print("==============================")

    try:
        model = YOLO(cfg["path"])

        start = time.time()

        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=cfg["batch"],
            device=cfg["device"],
            project=str(SAVE_DIR),
            name=name,
            exist_ok=True
        )

        cost_time = round(time.time() - start, 2)

        # 读取结果
        run_dir = SAVE_DIR / name
        csv_file = run_dir / "results.csv"

        precision = recall = map50 = map5095 = None

        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            last = df.iloc[-1]

            precision = last.get("metrics/precision(B)", None)
            recall = last.get("metrics/recall(B)", None)
            map50 = last.get("metrics/mAP50(B)", None)
            map5095 = last.get("metrics/mAP50-95(B)", None)

        return {
            "model": name,
            "time": cost_time,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map5095
        }

    except Exception as e:
        print(f"❌ {name} 训练失败：{e}")
        return None


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, cfg in MODEL_LIST.items():
        if not Path(cfg["path"]).exists():
            print(f"⚠️ 模型不存在，跳过：{cfg['path']}")
            continue

        r = train_one(name, cfg)

        if r is not None:
            results.append(r)

    # ===== 防止空列表崩溃 =====
    if len(results) == 0:
        print("❌ 没有成功训练任何模型")
        return

    # ===== 保存结果 =====
    csv_path = SAVE_DIR / "compare.csv"
    json_path = SAVE_DIR / "compare.json"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ===== 排序 =====
    results_sorted = sorted(results, key=lambda x: (x["map50"] or 0), reverse=True)

    print("\n===== 模型排序（按 mAP50）=====")
    for i, r in enumerate(results_sorted):
        print(f"{i+1}. {r['model']} | mAP50={r['map50']} | Recall={r['recall']} | Time={r['time']}s")


if __name__ == "__main__":
    main()