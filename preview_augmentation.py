from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.transform import resize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

X_PATH = RAW_DATA_DIR / "XYWheatCW.npy"
Y_PATH = RAW_DATA_DIR / "YWheatCW.npy"

SAVE_PATH = BASE_DIR / "augmentation_preview_fluorescence.png"


def adjust_brightness(img: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    factor = random.uniform(*factor_range)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def adjust_contrast(img: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    factor = random.uniform(*factor_range)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)
    return np.array(pil_img).astype(np.float32) / 255.0


def add_gaussian_noise(img: np.ndarray, sigma_range=(0.01, 0.03)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape)
    out = img + noise
    return np.clip(out, 0.0, 1.0)


def apply_slight_blur(img: np.ndarray, radius_range=(0.3, 0.8)) -> np.ndarray:
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    radius = random.uniform(*radius_range)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil_img).astype(np.float32) / 255.0


def load_and_upsample():
    if not X_PATH.exists():
        raise FileNotFoundError(f"未找到特征文件：{X_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"未找到标签文件：{Y_PATH}")

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    X_reshaped = X.reshape(-1, 4, 8)
    X_upsampled = np.array([
        resize(img, (32, 32), mode="constant", preserve_range=True)
        for img in X_reshaped
    ])

    min_v = X_upsampled.min()
    max_v = X_upsampled.max()
    X_upsampled = (X_upsampled - min_v) / (max_v - min_v + 1e-8)

    return X_upsampled, Y


def plot_augmentation_comparison():
    X_upsampled, Y = load_and_upsample()

    control_indices = np.where(Y == 0)[0]
    drought_indices = np.where(Y == 1)[0]

    control_img = X_upsampled[random.choice(control_indices)]
    drought_img = X_upsampled[random.choice(drought_indices)]

    control_aug_list = [
        control_img,
        adjust_brightness(control_img),
        adjust_contrast(control_img),
        add_gaussian_noise(control_img),
        apply_slight_blur(control_img),
    ]

    drought_aug_list = [
        drought_img,
        adjust_brightness(drought_img),
        adjust_contrast(drought_img),
        add_gaussian_noise(drought_img),
        apply_slight_blur(drought_img),
    ]

    titles = ["原始样本", "亮度扰动", "对比度扰动", "高斯噪声", "轻微模糊"]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    for i in range(5):
        axes[0, i].imshow(control_aug_list[i], cmap="viridis")
        axes[0, i].set_title(f"Control-{titles[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(drought_aug_list[i], cmap="viridis")
        axes[1, i].set_title(f"Drought-{titles[i]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"增强对比图已保存：{SAVE_PATH}")


if __name__ == "__main__":
    plot_augmentation_comparison()