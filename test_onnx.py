import numpy as np
import onnxruntime as ort
from PIL import Image

from utils.config import CLS_MODEL_PATH, DATASET_DIR


def main():
    if not CLS_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到 ONNX 模型：{CLS_MODEL_PATH}")

    img_path = DATASET_DIR / "val" / "control" / "control_0.png"
    if not img_path.exists():
        raise FileNotFoundError(f"未找到测试图片：{img_path}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(CLS_MODEL_PATH), providers=providers)
    except Exception:
        session = ort.InferenceSession(
            str(CLS_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    img = Image.open(str(img_path)).resize((32, 32)).convert("L")
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)   # [32, 32, 3]
    img_array = np.expand_dims(img_array, axis=0)    # [1, 32, 32, 3]
    img_array = np.transpose(img_array, (0, 3, 1, 2))  # [1, 3, 32, 32]

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})

    print(f"测试图片: {img_path.name}")
    print(f"ONNX Pred: {output[0][0]}")


if __name__ == "__main__":
    main()