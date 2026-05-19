"""
T1.1 — 图像预处理与配准模块。

扫描目录中的 PNG 文件，校验尺寸合法性，将图像加载为 NumPy 数组，
并选定基准变体。
"""

import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def load_variants(
    input_dir: str,
    base_name: Optional[str] = None,
) -> Tuple[dict, dict, str, Tuple[int, int, int]]:
    """加载并校验目录中所有变体 PNG 图像。

    参数
    ----------
    input_dir : str
        包含变体 PNG 文件的目录路径。
    base_name : str, 可选
        指定基准变体名称。若为 ``None``，则取字母序最小的变体。

    返回
    -------
    variants : dict[str, np.ndarray]
        变体名 → RGB 图像数组 (H×W×3, uint8)。
    alpha_masks : dict[str, np.ndarray | None]
        变体名 → Alpha 通道 (H×W, uint8)，若无则为 ``None``。
    base_name : str
        选定的基准变体名称。
    image_shape : tuple[int, int, int]
        (高度, 宽度, 通道数) — 始终为 3 (RGB)。

    异常
    ------
    FileNotFoundError
        当 ``input_dir`` 目录不存在时触发。
    ValueError
        当目录中无 PNG 文件，或图像尺寸不一致时触发。
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"目录不存在: {input_dir}")

    # --- 扫描 PNG 文件 ---
    png_files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    )
    # 跳过非 PNG 文件并给出提示
    for f in os.listdir(input_dir):
        full = os.path.join(input_dir, f)
        if os.path.isfile(full) and not f.lower().endswith(".png"):
            print(f"警告: 跳过非 PNG 文件: {f}", file=sys.stderr)

    if not png_files:
        raise ValueError(f"目录 {input_dir} 中未找到 PNG 文件")

    # --- 确定基准变体 ---
    variant_names = [os.path.splitext(f)[0] for f in png_files]
    if base_name is not None:
        if base_name not in variant_names:
            raise ValueError(
                f"指定的基准变体 '{base_name}' 未找到。"
                f"可用变体: {variant_names}"
            )
    else:
        base_name = variant_names[0]  # 字母序第一

    # --- 加载图像 ---
    variants: dict[str, np.ndarray] = {}
    alpha_masks: dict[str, Optional[np.ndarray]] = {}
    reference_shape: Optional[Tuple[int, int]] = None

    for name, filename in zip(variant_names, png_files):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")

        h, w = img.shape[:2]
        c = 1 if img.ndim == 2 else img.shape[2]

        # 校验尺寸
        if w > 2048 or h > 2048:
            raise ValueError(
                f"图像 '{name}' 尺寸为 {w}×{h}，超过最大限制 2048×2048"
            )
        if reference_shape is None:
            reference_shape = (h, w)
        elif (h, w) != reference_shape:
            raise ValueError(
                f"图像尺寸不一致: '{variant_names[0]}' 为 "
                f"{reference_shape[1]}×{reference_shape[0]}, "
                f"但 '{name}' 为 {w}×{h}"
            )

        # 分离 Alpha 通道（若存在）
        if c == 4:
            # BGRA → RGB + A
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            alpha = img[:, :, 3]
        elif c == 3:
            # BGR → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = None
        elif c == 1:
            # 灰度 → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            alpha = None
        else:
            raise ValueError(f"图像 '{name}' 通道数异常: {c}")

        variants[name] = rgb
        alpha_masks[name] = alpha

    image_shape = (reference_shape[0], reference_shape[1], 3)
    return variants, alpha_masks, base_name, image_shape
