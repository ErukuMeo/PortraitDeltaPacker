"""
T1.2 — 块匹配差异检测算法。

将基准图与每个变体图按块逐一比较，使用 SSIM（结构相似性）指标
判断是否相同，为每个变体生成差异布尔位图。
"""

import math
from typing import Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _pad_to_block(image: np.ndarray, block_size: int) -> np.ndarray:
    """镜像填充图像，使其宽高均为 block_size 的整数倍。"""
    h, w = image.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h == 0 and pad_w == 0:
        return image
    return cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_REFLECT,
    )


def _block_variance(block: np.ndarray) -> float:
    """计算图像块的逐像素方差（多通道取标量值）。"""
    return float(np.var(block))


def _block_mse(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """计算两个图像块之间的均方误差。"""
    return float(np.mean((block_a.astype(np.float64) - block_b.astype(np.float64)) ** 2))


def _block_ssim(block_a: np.ndarray, block_b: np.ndarray) -> float:
    """计算单个图像块的 SSIM 值。

    使用 ``skimage.metrics.structural_similarity``，
    并自适应选择适合块比较的 ``win_size``。
    """
    bs = block_a.shape[0]
    # SSIM 要求 win_size 为奇数且 ≤ 最小边长
    win_size = min(7, bs) if bs >= 7 else bs
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        # 块太小 — 回退到 1 − 归一化 MSE
        mse = _block_mse(block_a, block_b)
        max_val = 255.0
        return 1.0 - mse / (max_val ** 2)

    # 多通道时，对各通道分别计算 SSIM 后取均值
    if block_a.ndim == 3:
        channel_ssims = []
        for c in range(block_a.shape[2]):
            s = ssim(
                block_a[:, :, c], block_b[:, :, c],
                data_range=255,
                win_size=win_size,
                channel_axis=None,
            )
            channel_ssims.append(s)
        return float(np.mean(channel_ssims))

    return float(ssim(
        block_a, block_b, data_range=255,
        win_size=win_size, channel_axis=None,
    ))


def detect_diffs(
    base_img: np.ndarray,
    variant_imgs: Dict[str, np.ndarray],
    block_size: int = 16,
    threshold: float = 0.98,
) -> Dict[str, np.ndarray]:
    """检测基准图与各变体之间的差异块。

    参数
    ----------
    base_img : np.ndarray
        基准图像，(H, W, 3) uint8 RGB 数组。
    variant_imgs : dict[str, np.ndarray]
        变体名 → (H, W, 3) uint8 RGB 数组。
    block_size : int
        网格块大小（像素），可选 8、16 或 32。
    threshold : float
        SSIM 阈值（0.0–1.0）。SSIM ≥ 阈值即判定为相同块。

    返回
    -------
    diff_masks : dict[str, np.ndarray]
        变体名 → 布尔数组，形状为 ``(grid_rows, grid_cols)``，
        ``True`` 表示差异块。
    """
    # 填充图像使尺寸为 block_size 的整数倍
    padded_base = _pad_to_block(base_img, block_size)
    h, w = padded_base.shape[:2]

    grid_rows = h // block_size
    grid_cols = w // block_size

    result: Dict[str, np.ndarray] = {}

    for vname, vimg in variant_imgs.items():
        padded_variant = _pad_to_block(vimg, block_size)
        diff_mask = np.zeros((grid_rows, grid_cols), dtype=bool)

        for r in range(grid_rows):
            for c in range(grid_cols):
                y0 = r * block_size
                y1 = y0 + block_size
                x0 = c * block_size
                x1 = x0 + block_size

                block_a = padded_base[y0:y1, x0:x1]
                block_b = padded_variant[y0:y1, x0:x1]

                # 低纹理区域 → 使用 MSE 判定，避免 SSIM 不稳定
                var_a = _block_variance(block_a)
                var_b = _block_variance(block_b)

                if var_a < 0.5 and var_b < 0.5:
                    mse = _block_mse(block_a, block_b)
                    same = mse <= 2.0
                else:
                    sim = _block_ssim(block_a, block_b)
                    same = sim >= threshold

                if not same:
                    diff_mask[r, c] = True

        result[vname] = diff_mask

    return result
