from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

from utils.config import RAW_DATA_DIR, DATASET_DIR


def main():
    x_path = RAW_DATA_DIR / "XyWheatCW.npy"
    y_path = RAW_DATA_DIR / "YWheatCW.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"未找到特征文件：{x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"未找到标签文件：{y_path}")

    X = np.load(x_path)
    Y = np.load(y_path)

    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
    print(f"Class balance: Control={np.sum(Y == 0)}, Drought={np.sum(Y == 1)}")

    # 重构为 (4, 8)
    h, w = 4, 8
    X_reshaped = X.reshape(-1, h, w)

    # 上采样至 32x32
    X_upsampled = np.array([
        resize(img, (32, 32), mode="constant", preserve_range=True)
        for img in X_reshaped
    ])

    # 预览图保存到项目根目录
    base_dir = Path(__file__).resolve().parent
    control_sample = X_upsampled[Y == 0][0]
    drought_sample = X_upsampled[Y == 1][0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(control_sample, cmap="viridis")
    axs[0].set_title("Control Sample")
    axs[0].axis("off")

    axs[1].imshow(drought_sample, cmap="viridis")
    axs[1].set_title("Drought Sample")
    axs[1].axis("off")

    preview_path = base_dir / "upsampled_samples.png"
    plt.tight_layout()
    plt.savefig(preview_path)
    plt.close(fig)
    print(f"预览图已保存：{preview_path}")

    # 创建目录
    for split in ["train", "val"]:
        for cls in ["control", "drought"]:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    control_idx = np.where(Y == 0)[0].tolist()
    drought_idx = np.where(Y == 1)[0].tolist()
    random.shuffle(control_idx)
    random.shuffle(drought_idx)

    split_ratio = 0.8

    for idx_list, cls in [(control_idx, "control"), (drought_idx, "drought")]:
        split_pt = int(len(idx_list) * split_ratio)
        train_idx = idx_list[:split_pt]
        val_idx = idx_list[split_pt:]

        for i, idx in enumerate(train_idx):
            img_array = X_upsampled[idx]
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img.save(DATASET_DIR / "train" / cls / f"{cls}_{i}.png")

        for i, idx in enumerate(val_idx):
            img_array = X_upsampled[idx]
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img.save(DATASET_DIR / "val" / cls / f"{cls}_{i}.png")

    print(f"数据集已生成：{DATASET_DIR}")


if __name__ == "__main__":
    main()