"""
T1.4 — 差异块图集打包。

将基础图和各变体的差异区域组装为最终输出结构：
基础图 PNG + 各变体差异图 PNG + 元数据字典。
"""

import io
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


def assemble(
    base_img: np.ndarray,
    variant_regions: Dict[str, List[dict]],
    alpha_masks: Optional[Dict[str, Optional[np.ndarray]]] = None,
    has_alpha: bool = False,
    base_name: str = "",
) -> tuple:
    """将基础图与各变体差异区域组装为输出数据。

    参数
    ----------
    base_img : np.ndarray
        基础图 (H, W, 3) uint8 RGB。
    variant_regions : dict[str, list[dict]]
        变体名 → 差异区域字典列表。每个区域字典须含
        ``x, y, w, h, pixels`` 键。
    alpha_masks : dict[str, np.ndarray | None], 可选
        各变体的 Alpha 通道 (H, W) uint8，或 ``None``。
    has_alpha : bool
        基础图是否含 Alpha 通道。

    返回
    -------
    base_png : bytes
        基础图 PNG 字节流。
    variant_diff_pngs : dict[str, list[bytes]]
        变体名 → 差异区域 PNG 字节流列表，每项为一个差异区域。
    metadata : dict
        可 JSON 序列化的元数据字典。
    """
    h, w = base_img.shape[:2]

    # --- 基础图 PNG ---
    base_png = _ndarray_to_png(base_img, has_alpha)

    # --- 各变体差异图 PNG ---
    variant_diff_pngs: Dict[str, List[bytes]] = {}
    metadata_variants: Dict[str, list] = {}

    for vname, regions in variant_regions.items():
        diff_list: List[bytes] = []
        meta_list: list = []

        for i, region in enumerate(regions):
            px = int(region["x"])
            py = int(region["y"])
            pw = int(region["w"])
            ph = int(region["h"])
            pixels = region["pixels"]

            # 跳过尺寸无效的区域
            if pw <= 0 or ph <= 0:
                continue

            # 以左上角坐标命名区域
            region_name = f"r{px}_{py}"
            # 重名时追加索引
            if any(m["name"] == region_name for m in meta_list):
                region_name = f"r{px}_{py}_{i}"

            diff_png = _ndarray_to_png(pixels)
            diff_list.append(diff_png)
            meta_list.append({
                "name": region_name,
                "x": px,
                "y": py,
                "w": pw,
                "h": ph,
            })

        variant_diff_pngs[vname] = diff_list
        metadata_variants[vname] = meta_list

    # --- 元数据 ---
    metadata = {
        "base": {
            "width": int(w),
            "height": int(h),
            "has_alpha": has_alpha,
            "name": base_name,
        },
        "variants": metadata_variants,
    }

    return base_png, variant_diff_pngs, metadata


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _ndarray_to_png(arr: np.ndarray, has_alpha: bool = False) -> bytes:
    """将 NumPy 图像数组通过 Pillow 编码为 PNG 字节流。"""
    if arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mode = "RGBA"
    elif arr.ndim == 2:
        mode = "L"
        if has_alpha:
            mode = "RGBA"
    else:
        mode = "RGB"

    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
