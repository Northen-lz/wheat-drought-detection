import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


# ======================
# 1. 路径配置
# ======================
DETECT_MODEL_PATH = "D:\\pyhon\\xiangmu\\detect_compare_runs\\yolov8s\\weights\\best.pt"          # YOLOv8 检测模型路径
CLASS_MODEL_PATH = "D:\\pyhon\\newA_vscode\\wheat_drought_runs\\exp_augmented2_s\\weights\\best.onnx"         # ONNX 分类模型路径
TEST_IMAGE_PATH = "D:\\pyhon\\newA_vscode\\test.png"           # 测试图片路径

CONF_THRESH = 0.5
RUN_TIMES = 10


# ======================
# 2. 加载模型
# ======================
detect_model = YOLO(DETECT_MODEL_PATH)

cls_session = ort.InferenceSession(
    CLASS_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

cls_input_name = cls_session.get_inputs()[0].name


# ======================
# 3. ONNX分类预处理
# ======================
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))

    arr = gray.astype(np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)     # HWC, 3通道
    arr = np.expand_dims(arr, axis=0)      # NHWC
    arr = np.transpose(arr, (0, 3, 1, 2))  # NCHW

    return arr


# ======================
# 4. 单张图片完整推理
# ======================
def process_image(image_path):
    time_record = {}

    # 图像读取
    t1 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    t2 = time.time()
    time_record["图像读取"] = t2 - t1

    # YOLO检测
    results = detect_model(img, conf=CONF_THRESH, verbose=False)
    t3 = time.time()
    time_record["YOLO检测"] = t3 - t2

    wheat_count = 0
    drought_count = 0

    # ONNX分类
    cls_start = time.time()

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            h, w = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            wheat_count += 1

            input_tensor = preprocess_crop(crop)
            output = cls_session.run(None, {cls_input_name: input_tensor})[0][0]

            label = "drought" if output[1] > output[0] else "control"

            if label == "drought":
                drought_count += 1

    cls_end = time.time()
    time_record["ONNX分类"] = cls_end - cls_start

    # 结果统计
    stat_start = time.time()
    drought_rate = drought_count / wheat_count if wheat_count > 0 else 0
    stat_end = time.time()
    time_record["结果统计"] = stat_end - stat_start

    total_time = sum(time_record.values())
    time_record["总耗时"] = total_time

    return time_record, wheat_count, drought_count, drought_rate


# ======================
# 5. 连续运行测试
# ======================
def main():
    all_records = []

    print("开始系统推理性能测试")
    print("-" * 40)

    for i in range(RUN_TIMES):
        record, wheat_count, drought_count, drought_rate = process_image(TEST_IMAGE_PATH)
        all_records.append(record)

        print(f"第 {i + 1} 次测试：")
        print(f"  图像读取: {record['图像读取']:.4f}s")
        print(f"  YOLO检测: {record['YOLO检测']:.4f}s")
        print(f"  ONNX分类: {record['ONNX分类']:.4f}s")
        print(f"  结果统计: {record['结果统计']:.4f}s")
        print(f"  总耗时: {record['总耗时']:.4f}s")
        print(f"  小麦株数: {wheat_count}")
        print(f"  干旱株数: {drought_count}")
        print(f"  干旱率: {drought_rate:.2%}")
        print("-" * 40)

    # 平均耗时
    avg_record = {}
    for key in all_records[0].keys():
        avg_record[key] = sum(r[key] for r in all_records) / RUN_TIMES

    print("平均测试结果：")
    print(f"平均图像读取时间: {avg_record['图像读取']:.4f}s")
    print(f"平均YOLO检测时间: {avg_record['YOLO检测']:.4f}s")
    print(f"平均ONNX分类时间: {avg_record['ONNX分类']:.4f}s")
    print(f"平均结果统计时间: {avg_record['结果统计']:.4f}s")
    print(f"平均总处理时间: {avg_record['总耗时']:.4f}s")

    fps = 1 / avg_record["总耗时"] if avg_record["总耗时"] > 0 else 0
    print(f"平均FPS: {fps:.2f}")

    # ONNX模型体积
    if os.path.exists(CLASS_MODEL_PATH):
        model_size = os.path.getsize(CLASS_MODEL_PATH) / 1024 / 1024
        print(f"ONNX模型体积: {model_size:.2f}MB")

    # 保存CSV结果
    with open("system_runtime_test.csv", "w", encoding="utf-8-sig") as f:
        f.write("测试次数,图像读取,YOLO检测,ONNX分类,结果统计,总耗时\n")
        for i, r in enumerate(all_records):
            f.write(
                f"{i + 1},"
                f"{r['图像读取']:.4f},"
                f"{r['YOLO检测']:.4f},"
                f"{r['ONNX分类']:.4f},"
                f"{r['结果统计']:.4f},"
                f"{r['总耗时']:.4f}\n"
            )

    print("测试结果已保存为 system_runtime_test.csv")


if __name__ == "__main__":
    main()