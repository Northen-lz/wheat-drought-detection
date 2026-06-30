from typing import Dict
from pathlib import Path
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.transform import resize

from utils.config import RAW_DATA_DIR, DATASET_DIR


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def adjust_brightness(img: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """亮度扰动：模拟整体响应偏强或偏弱"""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    factor = random.uniform(*factor_range)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def adjust_contrast(img: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """对比度扰动：模拟高低响应差异变化"""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    factor = random.uniform(*factor_range)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def add_gaussian_noise(img: np.ndarray, sigma_range=(0.01, 0.03)) -> np.ndarray:
    """高斯噪声：模拟仪器采集噪声"""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape)
    out = img + noise
    return np.clip(out, 0.0, 1.0)


def apply_slight_blur(img: np.ndarray, radius_range=(0.3, 0.8)) -> np.ndarray:
    """轻微模糊：模拟平滑误差或分辨率轻微退化"""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    radius = random.uniform(*radius_range)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil_img).astype(np.float32) / 255.0


def build_augmented_variants(img):
    """返回与论文一致的4种增强"""
    return {
        "brightness": adjust_brightness(img),
        "contrast": adjust_contrast(img),
        "noise": add_gaussian_noise(img),
        "blur": apply_slight_blur(img),
    }

def load_raw_data():
    x_path = RAW_DATA_DIR / "XYWheatCW.npy"
    y_path = RAW_DATA_DIR / "YWheatCW.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"未找到特征文件：{x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"未找到标签文件：{y_path}")

    X = np.load(x_path)
    Y = np.load(y_path)

    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")


    print(f"Class balance: Control={np.sum(Y == 0)}, Drought={np.sum(Y == 1)}")

    return X, Y
# =========================
# 保存32维特征CSV（供SVM使用）
# =========================
import pandas as pd

csv_save_path = BASE_DIR / "data" / "wheat_32_features.csv"

df = pd.DataFrame(X)
df["label"] = Y

df.to_csv(csv_save_path, index=False)

print(f"SVM特征CSV已保存：{csv_save_path}")


def upsample_features(X: np.ndarray) -> np.ndarray:
    """32维特征 -> 4x8 -> 32x32"""
    X_reshaped = X.reshape(-1, 4, 8)

    X_upsampled = np.array([
        resize(img, (32, 32), mode="constant", preserve_range=True)
        for img in X_reshaped
    ])

    min_v = X_upsampled.min()
    max_v = X_upsampled.max()
    X_upsampled = (X_upsampled - min_v) / (max_v - min_v + 1e-8)

    return X_upsampled


def save_basic_preview(X_upsampled: np.ndarray, Y: np.ndarray, save_dir: Path):
    control_sample = X_upsampled[Y == 0][0]
    drought_sample = X_upsampled[Y == 1][0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(control_sample, cmap="viridis")
    axs[0].set_title("Control Sample")
    axs[0].axis("off")

    axs[1].imshow(drought_sample, cmap="viridis")
    axs[1].set_title("Drought Sample")
    axs[1].axis("off")

    preview_path = save_dir / "upsampled_samples.png"
    plt.tight_layout()
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"基础预览图已保存：{preview_path}")


def save_augmentation_preview(X_upsampled: np.ndarray, Y: np.ndarray, save_dir: Path):
    control_img = X_upsampled[np.where(Y == 0)[0][0]]
    drought_img = X_upsampled[np.where(Y == 1)[0][0]]

    control_aug = build_augmented_variants(control_img)
    drought_aug = build_augmented_variants(drought_img)

    control_list = [
        control_img,
        control_aug["brightness"],
        control_aug["contrast"],
        control_aug["noise"],
        control_aug["blur"],
    ]

    drought_list = [
        drought_img,
        drought_aug["brightness"],
        drought_aug["contrast"],
        drought_aug["noise"],
        drought_aug["blur"],
    ]

    titles = ["原始样本", "亮度扰动", "对比度扰动", "高斯噪声", "轻微模糊"]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    for i in range(5):
        axes[0, i].imshow(control_list[i], cmap="viridis")
        axes[0, i].set_title(f"Control-{titles[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(drought_list[i], cmap="viridis")
        axes[1, i].set_title(f"Drought-{titles[i]}")
        axes[1, i].axis("off")

    aug_preview_path = save_dir / "augmentation_preview_fluorescence.png"
    plt.tight_layout()
    plt.savefig(aug_preview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"增强对比图已保存：{aug_preview_path}")


def save_image(img_array: np.ndarray, save_path: Path):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img.save(save_path)


def prepare_output_dirs():
    for split in ["train", "val"]:
        for cls in ["control", "drought"]:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def build_dataset(X_upsampled: np.ndarray, Y: np.ndarray, split_ratio=0.8):
    control_idx = np.where(Y == 0)[0].tolist()
    drought_idx = np.where(Y == 1)[0].tolist()

    random.shuffle(control_idx)
    random.shuffle(drought_idx)

    total_train = 0
    total_val = 0

    for idx_list, cls in [(control_idx, "control"), (drought_idx, "drought")]:
        split_pt = int(len(idx_list) * split_ratio)
        train_idx = idx_list[:split_pt]
        val_idx = idx_list[split_pt:]

        for i, idx in enumerate(train_idx):
            base_img = X_upsampled[idx]

            save_image(base_img, DATASET_DIR / "train" / cls / f"{cls}_{i}_orig.png")
            total_train += 1

            aug_dict = build_augmented_variants(base_img)
            for aug_name, aug_img in aug_dict.items():
                save_image(
                    aug_img,
                    DATASET_DIR / "train" / cls / f"{cls}_{i}_{aug_name}.png"
                )
                total_train += 1

        for i, idx in enumerate(val_idx):
            base_img = X_upsampled[idx]
            save_image(base_img, DATASET_DIR / "val" / cls / f"{cls}_{i}.png")
            total_val += 1

    print(f"训练集样本总数（含增强）: {total_train}")
    print(f"验证集样本总数（原始）: {total_val}")
    print(f"增强后数据集已生成：{DATASET_DIR}")


def main():
    random.seed(42)
    np.random.seed(42)

    base_dir = Path(__file__).resolve().parent

    X, Y = load_raw_data()
    X_upsampled = upsample_features(X)

    prepare_output_dirs()
    save_basic_preview(X_upsampled, Y, base_dir)
    save_augmentation_preview(X_upsampled, Y, base_dir)
    build_dataset(X_upsampled, Y, split_ratio=0.8)


if __name__ == "__main__":
    main()