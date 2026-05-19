"""
T1.2 — diff 模块测试。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.diff import detect_diffs, _pad_to_block


class TestPadToBlock:
    """_pad_to_block 函数测试。"""

    def test_already_aligned(self):
        """尺寸已对齐时无需填充。"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        padded = _pad_to_block(img, 16)
        assert padded.shape == (64, 64, 3)

    def test_needs_padding(self):
        """尺寸不对齐时应填充至整数倍。"""
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        padded = _pad_to_block(img, 16)
        assert padded.shape[0] == 64  # 60 → 64
        assert padded.shape[1] == 64


class TestDetectDiffs:
    """detect_diffs 函数测试。"""

    def test_identical_images(self):
        """两张完全相同的图像不应产生差异块。"""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = detect_diffs(img, {"same": img.copy()}, block_size=16)
        assert result["same"].sum() == 0  # 无 True 条目

    def test_fully_different(self):
        """完全不同的图像应产生大量差异块。"""
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        variant = np.ones((64, 64, 3), dtype=np.uint8) * 255
        result = detect_diffs(base, {"white": variant}, block_size=16)
        assert result["white"].sum() > 0

    def test_partial_difference(self):
        """已知区域的差异应被检测到。"""
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        base[:, :] = [100, 100, 100]

        variant = base.copy()
        # 修改左上角 16×16 块
        variant[0:16, 0:16] = [200, 200, 200]

        result = detect_diffs(base, {"diff": variant}, block_size=16)
        # 块 (0,0) 应被标记
        assert result["diff"][0, 0] is True

    def test_block_size_32(self):
        """block_size=32 应正常工作。"""
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        base[:, :] = [100, 100, 100]
        variant = base.copy()
        variant[0:32, 0:32] = [200, 200, 200]
        result = detect_diffs(base, {"d": variant}, block_size=32)
        assert result["d"][0, 0] is True

    def test_multiple_variants(self):
        """多个变体应同时被处理。"""
        base = np.zeros((32, 32, 3), dtype=np.uint8)
        base[:, :] = [50, 50, 50]
        v1 = base.copy()
        v1[0:8, 0:8] = [200, 0, 0]
        v2 = base.copy()
        v2[16:24, 16:24] = [0, 200, 0]

        result = detect_diffs(base, {"v1": v1, "v2": v2}, block_size=8)
        assert "v1" in result
        assert "v2" in result
