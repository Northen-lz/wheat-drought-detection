from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent

# ========= 目标目录结构 =========
DIRS = [
    BASE / "models" / "detector",
    BASE / "models" / "classifier",
    BASE / "data" / "raw",
    BASE / "data" / "wheat_drought_data_upsampled",
    BASE / "outputs" / "captured_photos",
    BASE / "outputs" / "logs",
    BASE / "outputs" / "train_results",
    BASE / "utils",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)

print("已创建目录结构。")

# ========= 文件重命名 / 搬运规则 =========
# 只在源文件存在且目标不存在时执行，避免重复覆盖
MOVE_RULES = [
    # 脚本文件重命名
    ("download.py", "prepare_dataset.py"),
    ("onnx.py", "test_onnx.py"),
    ("test.py", "test_infer.py"),
    ("train_wheat_drought.py", "train_classifier.py"),

    # 原始数据目录
    ("raw", "data/raw"),

    # 数据集目录
    ("wheat_drought_data_upsampled", "data/wheat_drought_data_upsampled"),

    # 输出目录
    ("captured_photos", "outputs/captured_photos"),
]

def safe_move(src_rel, dst_rel):
    src = BASE / src_rel
    dst = BASE / dst_rel

    if not src.exists():
        print(f"跳过，不存在: {src_rel}")
        return

    if dst.exists():
        print(f"跳过，目标已存在: {dst_rel}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"已移动: {src_rel} -> {dst_rel}")

for src, dst in MOVE_RULES:
    safe_move(src, dst)

# ========= 尝试整理模型文件 =========
# 你当前上传的基础模型文件
MODEL_RULES = [
    ("yolo11n.pt", "models/detector/yolo11n.pt"),
    ("yolov8n-cls.pt", "models/classifier/yolov8n-cls.pt"),
]

for src_rel, dst_rel in MODEL_RULES:
    safe_move(src_rel, dst_rel)

# ========= 尝试收集训练结果 =========
# 如果你根目录下已经有这些结果，就搬到 outputs/train_results/
TRAIN_RESULT_FILES = [
    "results.png",
    "confusion_matrix.png",
    "results.csv",
]

for name in TRAIN_RESULT_FILES:
    src = BASE / name
    dst = BASE / "outputs" / "train_results" / name
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"已移动训练结果: {name} -> outputs/train_results/{name}")

print("\n整理完成。")
print("请手动确认以下文件是否已经放到对应目录：")
print("1. 检测模型 best.pt -> models/detector/best.pt")
print("2. 分类模型 best.onnx -> models/classifier/best.onnx")
print("3. 训练结果 -> outputs/train_results/")