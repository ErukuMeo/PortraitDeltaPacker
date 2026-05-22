"""
T1.1 — loader 模块测试。
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

# 确保 pdpack 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.loader import load_variants


def _write_png(path: str, arr: np.ndarray) -> None:
    """将 NumPy 数组写入 PNG 文件。"""
    Image.fromarray(arr).save(path)


class TestLoadVariants:
    """load_variants 函数测试。"""

    def test_load_basic(self):
        """加载两个尺寸一致的合法 PNG 文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            img_a = np.zeros((64, 64, 3), dtype=np.uint8)
            img_a[:, :] = [100, 150, 200]
            img_b = np.zeros((64, 64, 3), dtype=np.uint8)
            img_b[:, :] = [50, 100, 150]

            _write_png(os.path.join(tmp, "a.png"), img_a)
            _write_png(os.path.join(tmp, "b.png"), img_b)

            variants, alphas, base, shape = load_variants(tmp)

            assert base == "a"  # 字母序第一
            assert len(variants) == 2
            assert shape == (64, 64, 3)
            assert variants["a"].shape == (64, 64, 3)
            assert variants["b"].shape == (64, 64, 3)
            assert alphas["a"] is None
            assert alphas["b"] is None

    def test_specify_base(self):
        """按名称指定默认变体。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(os.path.join(tmp, "z.png"), np.zeros((32, 32, 3), dtype=np.uint8))
            _write_png(os.path.join(tmp, "a.png"), np.zeros((32, 32, 3), dtype=np.uint8))

            _variants, _alphas, base, _shape = load_variants(tmp, base_name="z")
            assert base == "z"

    def test_specify_base_not_found(self):
        """指定不存在的默认变体应触发异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(os.path.join(tmp, "a.png"), np.zeros((32, 32, 3), dtype=np.uint8))
            with pytest.raises(ValueError, match="未找到"):
                load_variants(tmp, base_name="nonexistent")

    def test_no_png_files(self):
        """目录中无 PNG 文件应触发异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="未找到 PNG"):
                load_variants(tmp)

    def test_dir_not_found(self):
        """目录不存在应触发 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_variants("/nonexistent/dir/path")

    def test_size_mismatch(self):
        """图像尺寸不一致应触发异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(os.path.join(tmp, "a.png"), np.zeros((32, 32, 3), dtype=np.uint8))
            _write_png(os.path.join(tmp, "b.png"), np.zeros((64, 64, 3), dtype=np.uint8))
            with pytest.raises(ValueError, match="尺寸不一致"):
                load_variants(tmp)

    def test_exceeds_max_size(self):
        """超过 2048×2048 的图像应触发异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(os.path.join(tmp, "big.png"), np.zeros((2049, 100, 3), dtype=np.uint8))
            with pytest.raises(ValueError, match="超过最大限制"):
                load_variants(tmp)

    def test_rgba_image(self):
        """加载 RGBA 图像应分离 Alpha 通道。"""
        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((32, 32, 4), dtype=np.uint8)
            img[:, :, :3] = [100, 150, 200]
            img[:, :, 3] = 128
            _write_png(os.path.join(tmp, "rgba.png"), img)

            variants, alphas, base, shape = load_variants(tmp)

            assert variants["rgba"].shape == (32, 32, 3)
            assert alphas["rgba"].shape == (32, 32)
            assert alphas["rgba"][0, 0] == 128

    def test_skips_non_png(self):
        """非 PNG 文件应被跳过并输出警告。"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(os.path.join(tmp, "a.png"), np.zeros((32, 32, 3), dtype=np.uint8))
            with open(os.path.join(tmp, "readme.txt"), "w") as f:
                f.write("hello")
            # 不应报错 — txt 被跳过
            variants, _alphas, _base, _shape = load_variants(tmp)
            assert len(variants) == 1
