"""
T2.2 — serializer 模块测试（序列化往返测试）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.assemble import assemble
from pdpack.serializer import (
    PDPackFile,
    serialize,
    deserialize,
    MAGIC,
)


class TestSerializeDeserialize:
    """序列化/反序列化往返测试。"""

    def test_round_trip(self):
        """打包 → 解包 → 验证数据完整性。"""
        base_img = np.zeros((64, 64, 3), dtype=np.uint8)
        base_img[:, :] = [34, 139, 34]

        diff1 = np.ones((16, 16, 3), dtype=np.uint8) * 255
        diff2 = np.zeros((8, 8, 3), dtype=np.uint8)

        variant_regions = {
            "happy": [
                {"x": 10, "y": 20, "w": 16, "h": 16, "pixels": diff1},
            ],
            "sad": [
                {"x": 30, "y": 40, "w": 8, "h": 8, "pixels": diff2},
            ],
        }

        base_png, diff_pngs, metadata = assemble(
            base_img, variant_regions, base_name="neutral",
        )

        data = serialize(base_png, metadata, diff_pngs)
        assert isinstance(data, bytes)
        assert len(data) > 0

        ppf = deserialize(data)
        assert isinstance(ppf, PDPackFile)
        assert ppf.header.magic == MAGIC
        assert ppf.header.version == 1
        assert ppf.header.variant_count == 3

        # 验证基础图
        np.testing.assert_array_equal(ppf.base_image, base_img)

        # 验证元数据
        assert ppf.metadata == metadata

        # 验证变体差异区域
        assert "neutral" in ppf.variant_regions
        assert ppf.variant_regions["neutral"] == []
        assert "happy" in ppf.variant_regions
        assert len(ppf.variant_regions["happy"]) == 1
        np.testing.assert_array_equal(
            ppf.variant_regions["happy"][0], diff1,
        )
        assert "sad" in ppf.variant_regions
        assert len(ppf.variant_regions["sad"]) == 1
        np.testing.assert_array_equal(
            ppf.variant_regions["sad"][0], diff2,
        )

    def test_variant_mapping_uses_sorted_names(self):
        """序列化和反序列化应使用同一变体顺序映射差异区域。"""
        base_img = np.zeros((32, 32, 3), dtype=np.uint8)
        diff_a = np.ones((8, 8, 3), dtype=np.uint8) * 10
        diff_z = np.ones((8, 8, 3), dtype=np.uint8) * 200
        variant_regions = {
            "z": [{"x": 0, "y": 0, "w": 8, "h": 8, "pixels": diff_z}],
            "a": [{"x": 8, "y": 8, "w": 8, "h": 8, "pixels": diff_a}],
        }

        base_png, diff_pngs, metadata = assemble(
            base_img, variant_regions, base_name="m",
        )
        ppf = deserialize(serialize(base_png, metadata, diff_pngs))

        np.testing.assert_array_equal(ppf.variant_regions["a"][0], diff_a)
        np.testing.assert_array_equal(ppf.variant_regions["z"][0], diff_z)
        assert ppf.variant_regions["m"] == []

    def test_bad_magic(self):
        """魔数错误的数据应触发 ValueError。"""
        import struct
        bad = struct.pack(">4s", b"BAD!") + b"\x00" * 100
        with __import__("pytest").raises(ValueError, match="无效魔数"):
            deserialize(bad)

    def test_too_small(self):
        """小于文件头大小的数据应触发异常。"""
        with __import__("pytest").raises(ValueError, match="过小"):
            deserialize(b"tiny")
