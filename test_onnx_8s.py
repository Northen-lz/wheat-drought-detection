from pathlib import Path
import onnxruntime as ort
import numpy as np

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    onnx_path = base_dir / "wheat_drought_runs" / "exp_augmented2_s" / "weights" / "best.onnx"

    if not onnx_path.exists():
        print(f"未找到 ONNX 文件：{onnx_path}")
        raise SystemExit

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    y = session.run(None, {input_name: x})

    print("输入名：", input_name)
    print("输出结果：", y)
    print("输出 shape：", y[0].shape)