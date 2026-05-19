"""
T1.3 — extract 模块测试（差异区域提取与矩形合并）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.extract import extract_diff_regions, merge_rectangles


class TestExtractDiffRegions:
    """extract_diff_regions 函数测试。"""

    def test_empty_mask(self):
        """全为 False 的掩码应返回空列表。"""
        mask = np.zeros((4, 4), dtype=bool)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        regions = extract_diff_regions(mask, 16, img)
        assert regions == []

    def test_single_block(self):
        """单个 True 块应产生一个区域。"""
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        regions = extract_diff_regions(mask, 16, img)
        assert len(regions) == 1
        r = regions[0]
        assert r["x"] == 16
        assert r["y"] == 16
        assert r["w"] == 16
        assert r["h"] == 16

    def test_two_adjacent_blocks(self):
        """两个相邻块应合并为一个区域。"""
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[1, 2] = True
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        regions = extract_diff_regions(mask, 16, img)
        assert len(regions) == 1
        assert regions[0]["w"] == 32  # 2 个块宽
        assert regions[0]["h"] == 16

    def test_pixels_content(self):
        """提取的像素应与源图像区域一致。"""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[2, 2] = True

        regions = extract_diff_regions(mask, 16, img)
        r = regions[0]
        expected = img[32:48, 32:48]
        np.testing.assert_array_equal(r["pixels"], expected)


class TestMergeRectangles:
    """merge_rectangles 函数测试。"""

    def test_no_rects(self):
        """空列表应返回空列表。"""
        assert merge_rectangles([], np.zeros((4, 4), dtype=bool), 16) == []

    def test_single_rect(self):
        """单个矩形保持不变。"""
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        rects = [{"x": 0, "y": 0, "w": 16, "h": 16}]
        result = merge_rectangles(rects, mask, 16)
        assert len(result) == 1

    def test_merge_horizontal_adjacent(self):
        """两个水平相邻矩形应合并。"""
        mask = np.ones((1, 2), dtype=bool)
        rects = [
            {"x": 0, "y": 0, "w": 16, "h": 16},
            {"x": 16, "y": 0, "w": 16, "h": 16},
        ]
        result = merge_rectangles(rects, mask, 16)
        assert len(result) == 1
        assert result[0]["w"] == 32
        assert result[0]["h"] == 16
