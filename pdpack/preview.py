"""
T3.3 — 预览与验证工具。

提供从 .pdpack 数据重建变体图像的功能，
并可使用 PSNR 与原图进行比对验证。
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pdpack.serializer import PDPackFile


def reconstruct(
    ppf: PDPackFile,
    variant_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """从 .pdpack 文件重建变体图像。

    将各变体的差异区域按其存储坐标覆盖到基础图上。

    参数
    ----------
    ppf : PDPackFile
        反序列化后的 .pdpack 数据。
    variant_name : str, 可选
        若指定，则仅重建此变体。否则重建所有变体。

    返回
    -------
    dict[str, np.ndarray]
        变体名 → 重建后的 (H, W, 3) uint8 图像。
    """
    base = ppf.base_image
    base_h, base_w = base.shape[:2]
    metadata_variants = ppf.metadata.get("variants", {})

    targets = [variant_name] if variant_name else list(metadata_variants.keys())
    result: Dict[str, np.ndarray] = {}

    for vname in targets:
        # 以基础图副本为画布
        canvas = base.copy()

        regions = ppf.variant_regions.get(vname, [])
        meta_regions = metadata_variants.get(vname, [])

        for i, region_img in enumerate(regions):
            if i >= len(meta_regions):
                continue
            mr = meta_regions[i]
            x, y, w, h = mr["x"], mr["y"], mr["w"], mr["h"]

            # 安全裁剪到画布边界
            x = max(0, min(x, base_w - 1))
            y = max(0, min(y, base_h - 1))
            w = max(1, min(w, base_w - x))
            h = max(1, min(h, base_h - y))

            # 若区域图像尺寸不匹配则缩放
            rh, rw = region_img.shape[:2]
            if rw != w or rh != h:
                region_img = cv2.resize(region_img, (w, h))

            # 若基础图含 Alpha 而差异区域不含，仅覆写 RGB 通道
            if canvas.ndim == 3 and canvas.shape[2] == 4 and region_img.ndim == 2 or (region_img.ndim == 3 and region_img.shape[2] == 3):
                canvas[y:y + h, x:x + w, :3] = region_img[:, :, :3] if region_img.ndim == 3 else np.stack([region_img]*3, axis=-1)
            else:
                canvas[y:y + h, x:x + w] = region_img

        result[vname] = canvas

    return result


def save_reconstructed(
    reconstructed: Dict[str, np.ndarray],
    output_dir: str,
) -> List[str]:
    """将重建图像保存为 PNG 文件。

    参数
    ----------
    reconstructed : dict[str, np.ndarray]
        :func:`reconstruct` 的输出。
    output_dir : str
        保存 PNG 的目标目录。

    返回
    -------
    list[str]
        已保存文件的路径列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for vname, img in reconstructed.items():
        path = os.path.join(output_dir, f"{vname}_reconstructed.png")
        pil_img = Image.fromarray(img)
        pil_img.save(path)
        saved.append(path)

    return saved


def verify(
    reconstructed: Dict[str, np.ndarray],
    original_dir: str,
) -> Dict[str, float]:
    """计算重建图像与原图之间的 PSNR。

    参数
    ----------
    reconstructed : dict[str, np.ndarray]
        :func:`reconstruct` 的输出。
    original_dir : str
        包含原始 ``{variant}.png`` 文件的目录。

    返回
    -------
    dict[str, float]
        变体名 → PSNR 值（单位 dB）。
        ``inf`` 表示两张图完全一致。
    """
    psnr_values: Dict[str, float] = {}

    for vname, recon in reconstructed.items():
        orig_path = os.path.join(original_dir, f"{vname}.png")
        if not os.path.isfile(orig_path):
            continue

        orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
        if orig is None:
            continue
        if orig.ndim == 3 and orig.shape[2] == 4:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2RGBA)
        elif orig.ndim == 3 and orig.shape[2] == 3:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        elif orig.ndim == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

        # 尺寸不匹配时缩放
        if orig.shape[:2] != recon.shape[:2]:
            orig = cv2.resize(orig, (recon.shape[1], recon.shape[0]))

        # 通道数不匹配时对齐（仅比较 RGB 通道）
        if orig.ndim == 3 and recon.ndim == 3 and orig.shape[2] != recon.shape[2]:
            channels = min(orig.shape[2], recon.shape[2])
            a = orig[:, :, :channels].astype(np.float64)
            b = recon[:, :, :channels].astype(np.float64)
        else:
            a = orig.astype(np.float64)
            b = recon.astype(np.float64)

        mse = np.mean((a - b) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))

        psnr_values[vname] = psnr

    return psnr_values
