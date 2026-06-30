from pathlib import Path

from ultralytics import YOLO

from utils.config import BASE_DIR, DATASET_DIR


def main():
    dataset_path = DATASET_DIR
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到增强后的分类数据集目录：{dataset_path}")

    print(f"分类数据集目录：{dataset_path}")

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=str(dataset_path),
        epochs=50,
        imgsz=32,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.9,
        augment=True,
        project=str(BASE_DIR / "wheat_drought_runs"),
        name="exp_augmented",
    )

    print("训练完成，开始验证...")
    metrics = model.val()
    print(metrics)

    print("开始导出 ONNX 模型...")
    export_result = model.export(format="onnx")
    print(f"ONNX 导出完成：{export_result}")


if __name__ == "__main__":
    main()