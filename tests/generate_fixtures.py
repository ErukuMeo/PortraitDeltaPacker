"""
生成 PDPack 测试套件所需的合成测试固件。

创建小尺寸 PNG 图像 (64×64)，基础图与各变体图之间存在已知差异区域。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def make_base() -> np.ndarray:
    """创建 64×64 RGB 基础图 — 绿色背景 + 白色圆形。"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = [34, 139, 34]  # 森林绿

    # 中央白色圆形
    cy, cx = 32, 32
    r = 20
    for y in range(64):
        for x in range(64):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[y, x] = [255, 255, 255]

    return img


def make_variant_happy(base: np.ndarray) -> np.ndarray:
    """开心变体: 在左下角添加黄色矩形（嘴）。"""
    img = base.copy()
    img[40:48, 20:44] = [255, 255, 0]  # 黄色矩形
    return img


def make_variant_angry(base: np.ndarray) -> np.ndarray:
    """愤怒变体: 在顶部添加两个红色圆点（眼睛）。"""
    img = base.copy()
    img[16:24, 16:24] = [255, 0, 0]    # 左眼 红色
    img[16:24, 40:48] = [255, 0, 0]    # 右眼 红色
    return img


def make_variant_sad(base: np.ndarray) -> np.ndarray:
    """悲伤变体: 添加蓝色泪滴。"""
    img = base.copy()
    img[50:56, 30:34] = [0, 100, 255]  # 蓝色泪滴
    return img


def main():
    """生成所有测试固件并保存。"""
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    base = make_base()

    Image.fromarray(base).save(os.path.join(FIXTURE_DIR, "neutral.png"))
    Image.fromarray(make_variant_happy(base)).save(os.path.join(FIXTURE_DIR, "happy.png"))
    Image.fromarray(make_variant_angry(base)).save(os.path.join(FIXTURE_DIR, "angry.png"))
    Image.fromarray(make_variant_sad(base)).save(os.path.join(FIXTURE_DIR, "sad.png"))

    # 完全相同的副本，用于边界情况测试
    Image.fromarray(base.copy()).save(os.path.join(FIXTURE_DIR, "identical.png"))

    print(f"已在 {FIXTURE_DIR} 生成 5 个测试固件图像")


if __name__ == "__main__":
    main()
