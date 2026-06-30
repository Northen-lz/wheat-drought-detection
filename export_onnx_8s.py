from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent

    pt_path = base_dir / "wheat_drought_runs" / "exp_augmented2_s" / "weights" / "best.pt"

    if not pt_path.exists():
        print(f"未找到权重文件：{pt_path}")
        raise SystemExit

    print(f"加载模型：{pt_path}")
    model = YOLO(str(pt_path))

    print("开始导出 ONNX...")
    out = model.export(
        format="onnx",
        imgsz=32,
        opset=19
    )

    print("导出完成：", out)