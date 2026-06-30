import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

from utils.config import DET_MODEL_PATH, CLS_MODEL_PATH


def load_models():
    if not DET_MODEL_PATH.exists():
        raise FileNotFoundError(f"检测模型不存在：{DET_MODEL_PATH}")

    if not CLS_MODEL_PATH.exists():
        raise FileNotFoundError(f"分类模型不存在：{CLS_MODEL_PATH}")

    det_model = YOLO(str(DET_MODEL_PATH))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        cls_session = ort.InferenceSession(str(CLS_MODEL_PATH), providers=providers)
    except Exception:
        cls_session = ort.InferenceSession(
            str(CLS_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    cls_input_name = cls_session.get_inputs()[0].name
    return det_model, cls_session, cls_input_name


det_model, cls_session, cls_input_name = load_models()


def preprocess_crop_for_cls(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    img_pil = Image.fromarray(gray).resize((32, 32)).convert("L")

    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = np.transpose(arr, (0, 3, 1, 2))
    return arr


def process_image_with_detection(frame_bgr, conf_thresh=0.5):
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("输入图像为空，无法进行检测。")

    results = det_model(frame_bgr, conf=conf_thresh, verbose=False)
    crops = []
    boxes = []

    for r in results:
        # 关键修复：没有检测框时直接跳过
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)

            # 防止非法框
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            crop_input = preprocess_crop_for_cls(crop)
            crops.append(crop_input)
            boxes.append((x1, y1, x2, y2, float(box.conf.item())))

    log_data = {
        "图像": "uploaded.png",
        "小麦株数": len(crops),
        "干旱株数": 0,
        "干旱率": "0%",
        "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not crops:
        return frame_bgr, log_data

    drought_count = 0

    for crop_input, (x1, y1, x2, y2, det_conf) in zip(crops, boxes):
        output = cls_session.run(None, {cls_input_name: crop_input})[0][0]

        label = "drought" if output[1] > output[0] else "control"
        conf = float(output[1] if label == "drought" else output[0])

        color = (0, 0, 255) if label == "drought" else (0, 255, 0)
        text_y = y1 - 10 if y1 - 10 > 15 else y1 + 20

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr,
            f"{label}:{conf:.1%}",
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if label == "drought":
            drought_count += 1

    log_data["干旱株数"] = drought_count
    log_data["干旱率"] = f"{drought_count / len(crops):.1%}" if len(crops) > 0 else "0%"

    return frame_bgr, log_data