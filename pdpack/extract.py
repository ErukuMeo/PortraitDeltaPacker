"""
T1.3 — 差异区域提取与轴对齐矩形合并。

将块级差异布尔位图转换为像素空间的轴对齐矩形列表，
再通过贪心合并减少碎片化。
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def extract_diff_regions(
    diff_mask: np.ndarray,
    block_size: int,
    image: np.ndarray,
    alpha_mask: Optional[np.ndarray] = None,
) -> List[dict]:
    """从单个变体的差异位图中提取差异区域。

    参数
    ----------
    diff_mask : np.ndarray
        布尔数组，形状 ``(grid_rows, grid_cols)``。
    block_size : int
        每个网格块的像素边长。
    image : np.ndarray
        原始变体图像 (H, W, 3) uint8 — 仅用于确定像素边界（裁剪）。

    返回
    -------
    list[dict]
        每个字典包含 ``{"x", "y", "w", "h", "pixels"}``，
        坐标为像素坐标，相对于图像左上角原点。
    """
    if not diff_mask.any():
        return []

    # 转为 uint8 供 OpenCV 连通域分析使用
    mask_u8 = diff_mask.astype(np.uint8)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=4,
    )

    img_h, img_w = image.shape[:2]

    rects: List[dict] = []
    for label_id in range(1, n_labels):  # 跳过背景 (0)
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]

        # 从块网格坐标转换为像素坐标
        px = left * block_size
        py = top * block_size
        pw = width * block_size
        ph = height * block_size

        # 裁剪到图像边界内
        px = max(0, px)
        py = max(0, py)
        pw = min(pw, img_w - px)
        ph = min(ph, img_h - py)

        if pw <= 0 or ph <= 0:
            continue

        pixels = image[py:py + ph, px:px + pw].copy()
        if alpha_mask is not None:
            alpha_region = alpha_mask[py:py + ph, px:px + pw].copy()
            pixels = np.dstack([pixels, alpha_region])

        rects.append({
            "x": px,
            "y": py,
            "w": pw,
            "h": ph,
            "pixels": pixels,
        })

    return rects


def merge_rectangles(
    rects: List[dict],
    diff_mask: np.ndarray,
    block_size: int,
    image: np.ndarray = None,
    alpha_mask: Optional[np.ndarray] = None,
    blank_tolerance: float = 0.25,
) -> List[dict]:
    """贪心合并轴对齐矩形，减少碎片化。

    两个矩形可合并的条件：
    * 水平或垂直相邻（或重叠）。
    * 沿另一轴的投影重叠 ≥ 50%。
    * 合并后的矩形中非差异块比例 ≤ *blank_tolerance*。

    参数
    ----------
    rects : list[dict]
        初始矩形列表（含像素坐标及 ``pixels`` 数据）。
    diff_mask : np.ndarray
        原始差异位图，用于验证空白比例。
    block_size : int
        网格块大小。
    image : np.ndarray, 可选
        原始变体图像。若提供，则从合并后的矩形中重新提取像素。
    blank_tolerance : float
        合并后矩形中允许的最大非差异块比例。

    返回
    -------
    list[dict]
        合并后的矩形列表，含坐标及像素数据。
    """
    if len(rects) <= 1:
        return rects

    # 使用可变副本，按面积降序排列
    working = [_rect_to_mutable(r) for r in rects]
    working.sort(key=lambda r: r["w"] * r["h"], reverse=True)

    changed = True
    while changed:
        changed = False
        for i in range(len(working)):
            if working[i] is None:
                continue
            for j in range(i + 1, len(working)):
                if working[j] is None:
                    continue
                merged = _try_merge(
                    working[i], working[j],
                    diff_mask, block_size, blank_tolerance,
                )
                if merged is not None:
                    working[i] = merged
                    working[j] = None
                    changed = True
        working = [r for r in working if r is not None]

    # 从原图中提取合并后矩形的像素数据
    if image is not None:
        for r in working:
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            img_h, img_w = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            r["x"], r["y"], r["w"], r["h"] = x, y, w, h
            r["pixels"] = image[y:y + h, x:x + w].copy()
            if alpha_mask is not None:
                alpha_region = alpha_mask[y:y + h, x:x + w].copy()
                r["pixels"] = np.dstack([r["pixels"], alpha_region])

    return working


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _rect_to_mutable(rect: dict) -> dict:
    """将矩形转为可修改的字典副本。"""
    return {
        "x": rect["x"],
        "y": rect["y"],
        "w": rect["w"],
        "h": rect["h"],
    }


def _try_merge(
    a: dict, b: dict,
    diff_mask: np.ndarray,
    block_size: int,
    blank_tolerance: float,
):
    """尝试合并两个矩形；成功返回合并后矩形，失败返回 None。"""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]

    # --- 检查水平相邻 ---
    h_overlap = min(ay2, by2) - max(ay1, by1)
    if h_overlap > 0:
        min_h = min(ay2 - ay1, by2 - by1)
        if h_overlap >= 0.5 * min_h:
            # 检查是否水平相邻或重叠
            if ax1 <= bx2 and bx1 <= ax2:
                merged = {
                    "x": min(ax1, bx1),
                    "y": min(ay1, by1),
                    "w": max(ax2, bx2) - min(ax1, bx1),
                    "h": max(ay2, by2) - min(ay1, by1),
                }
                if _blank_ratio_ok(merged, diff_mask, block_size, blank_tolerance):
                    return merged

    # --- 检查垂直相邻 ---
    v_overlap = min(ax2, bx2) - max(ax1, bx1)
    if v_overlap > 0:
        min_w = min(ax2 - ax1, bx2 - bx1)
        if v_overlap >= 0.5 * min_w:
            if ay1 <= by2 and by1 <= ay2:
                merged = {
                    "x": min(ax1, bx1),
                    "y": min(ay1, by1),
                    "w": max(ax2, bx2) - min(ax1, bx1),
                    "h": max(ay2, by2) - min(ay1, by1),
                }
                if _blank_ratio_ok(merged, diff_mask, block_size, blank_tolerance):
                    return merged

    return None


def _blank_ratio_ok(
    rect: dict,
    diff_mask: np.ndarray,
    block_size: int,
    tolerance: float,
) -> bool:
    """检查矩形中非差异块比例是否 ≤ 容忍度。"""
    # 将像素矩形转为块网格索引
    r0 = rect["y"] // block_size
    r1 = (rect["y"] + rect["h"] + block_size - 1) // block_size
    c0 = rect["x"] // block_size
    c1 = (rect["x"] + rect["w"] + block_size - 1) // block_size

    r0 = max(0, r0)
    r1 = min(diff_mask.shape[0], r1)
    c0 = max(0, c0)
    c1 = min(diff_mask.shape[1], c1)

    if r0 >= r1 or c0 >= c1:
        return False

    sub = diff_mask[r0:r1, c0:c1]
    total = sub.size
    if total == 0:
        return False
    diff_count = int(sub.sum())
    blank_ratio = 1.0 - (diff_count / total)
    return blank_ratio <= tolerance
