"""
T1.4 — assemble 模块测试。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.assemble import assemble


class TestAssemble:
    """assemble 函数测试。"""

    def test_basic_output(self):
        """组装后应生成基础图 PNG、差异图 PNG 和元数据。"""
        base_img = np.zeros((64, 64, 3), dtype=np.uint8)
        base_img[:, :] = [100, 150, 200]

        # 单个差异区域
        diff_pixels = np.ones((16, 16, 3), dtype=np.uint8) * 255
        variant_regions = {
            "happy": [
                {"x": 0, "y": 0, "w": 16, "h": 16, "pixels": diff_pixels},
            ],
        }

        base_png, diff_pngs, metadata = assemble(
            base_img, variant_regions, base_name="neutral",
        )

        assert isinstance(base_png, bytes)
        assert len(base_png) > 0
        assert "happy" in diff_pngs
        assert len(diff_pngs["happy"]) == 1
        assert isinstance(diff_pngs["happy"][0], bytes)

        assert metadata["base"]["width"] == 64
        assert metadata["base"]["height"] == 64
        assert metadata["base"]["name"] == "neutral"
        assert "neutral" in metadata["variants"]
        assert metadata["variants"]["neutral"] == []
        assert diff_pngs["neutral"] == []
        assert "happy" in metadata["variants"]
        assert len(metadata["variants"]["happy"]) == 1
        region = metadata["variants"]["happy"][0]
        assert region["x"] == 0
        assert region["y"] == 0
        assert region["w"] == 16
        assert region["h"] == 16

    def test_multiple_regions(self):
        """每个变体可有多个差异区域。"""
        base_img = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        regions = {
            "a": [
                {"x": 0, "y": 0, "w": 16, "h": 16, "pixels": np.zeros((16, 16, 3), dtype=np.uint8)},
                {"x": 32, "y": 32, "w": 16, "h": 16, "pixels": np.zeros((16, 16, 3), dtype=np.uint8)},
            ],
        }

        _base_png, diff_pngs, metadata = assemble(
            base_img, regions, base_name="neutral",
        )
        assert diff_pngs["neutral"] == []
        assert len(diff_pngs["a"]) == 2
        assert metadata["variants"]["neutral"] == []
        assert len(metadata["variants"]["a"]) == 2

    def test_has_alpha_flag(self):
        """设置 has_alpha=True 时元数据应反映此标志。"""
        base_img = np.zeros((32, 32, 3), dtype=np.uint8)
        _base_png, _diff_pngs, metadata = assemble(
            base_img, {}, has_alpha=True, base_name="default",
        )
        assert metadata["base"]["has_alpha"] is True
        assert metadata["variants"]["default"] == []
