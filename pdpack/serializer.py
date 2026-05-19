"""
T2.2 — .pdpack 二进制序列化器与反序列化器。

实现 TaskSpec §T2.1 中描述的大端序二进制格式。
"""

import io
import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

MAGIC = b"PDPK"
HEADER_SIZE = 24

FLAG_HAS_ALPHA = 0x0001
FLAG_COMPRESSED = 0x0002  # 预留 — 始终为 0


@dataclass
class PDPackHeader:
    """解析后的 .pdpack 文件头。"""
    magic: bytes               # b"PDPK"
    version: int               # uint16
    flags: int                 # uint16 标志位
    variant_count: int         # uint16
    offset_table: int          # uint32 — 偏移表的字节偏移

    @property
    def has_alpha(self) -> bool:
        """是否含 Alpha 通道。"""
        return bool(self.flags & FLAG_HAS_ALPHA)

    @property
    def compressed(self) -> bool:
        """是否已压缩（预留）。"""
        return bool(self.flags & FLAG_COMPRESSED)


@dataclass
class PDPackFile:
    """.pdpack 文件的内存表示。"""
    header: PDPackHeader
    base_image: np.ndarray               # (H, W, 3) 或 (H, W, 4) uint8
    metadata: dict
    variant_regions: Dict[str, List[np.ndarray]]  # 变体名 → [区域图像, ...]


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def serialize(
    base_png: bytes,
    metadata: dict,
    variant_diff_pngs: Dict[str, List[bytes]],
    flags: int = 0,
    version: int = 1,
) -> bytes:
    """将数据打包为 .pdpack 字节流。

    参数
    ----------
    base_png : bytes
        基础图 PNG 字节流。
    metadata : dict
        元数据字典（将被 JSON 编码）。
    variant_diff_pngs : dict[str, list[bytes]]
        各变体的差异区域 PNG 字节流。
    flags : int
        文件头标志位（bit 0 = has_alpha, bit 1 = compressed）。
    version : int
        格式版本号。

    返回
    -------
    bytes
        完整的 .pdpack 文件内容。
    """
    metadata_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    variant_names = sorted(variant_diff_pngs.keys())
    variant_count = len(variant_names)

    # --- 计算偏移表大小 ---
    # base: offset(4) + size(4) = 8
    # metadata: offset(4) + size(4) = 8
    # 每个变体: region_count(2) + M * (offset(4) + size(4))
    table_size = 8 + 8  # base + metadata 条目
    for vname in variant_names:
        regions = variant_diff_pngs[vname]
        table_size += 2 + len(regions) * 8

    # --- 计算各段绝对偏移 ---
    offset_table = HEADER_SIZE
    base_offset = HEADER_SIZE + table_size
    metadata_offset = base_offset + len(base_png)

    # 变体区域数据从元数据之后开始
    data_offset = metadata_offset + len(metadata_json)

    # --- 构建二进制数据 ---
    buf = io.BytesIO()

    # 文件头
    buf.write(_pack_header(version, flags, variant_count, offset_table))

    # 偏移表
    buf.write(struct.pack(">II", base_offset, len(base_png)))
    buf.write(struct.pack(">II", metadata_offset, len(metadata_json)))

    region_offsets: Dict[str, List[tuple]] = {}
    for vname in variant_names:
        regions = variant_diff_pngs[vname]
        buf.write(struct.pack(">H", len(regions)))
        for ri, rpng in enumerate(regions):
            entry_offset = data_offset
            entry_size = len(rpng)
            buf.write(struct.pack(">II", entry_offset, entry_size))
            region_offsets.setdefault(vname, []).append((entry_offset, entry_size))
            data_offset += entry_size

    # 基础图 PNG
    buf.write(base_png)

    # 元数据 JSON
    buf.write(metadata_json)

    # 各变体差异区域 PNG
    for vname in variant_names:
        for rpng in variant_diff_pngs[vname]:
            buf.write(rpng)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# 反序列化
# ---------------------------------------------------------------------------

def deserialize(data: bytes) -> PDPackFile:
    """将 .pdpack 字节流解析为 :class:`PDPackFile`。

    参数
    ----------
    data : bytes
        原始 .pdpack 文件内容。

    返回
    -------
    PDPackFile

    异常
    ------
    ValueError
        当魔数不匹配或文件格式错误时触发。
    """
    if len(data) < HEADER_SIZE:
        raise ValueError("文件过小，无法包含有效的 .pdpack 文件头")

    # --- 文件头 ---
    magic = data[0:4]
    if magic != MAGIC:
        raise ValueError(f"无效魔数: 期望 {MAGIC!r}, 实际 {magic!r}")

    version, flags, variant_count, offset_table = struct.unpack_from(
        ">HHHI", data, 4,
    )

    header = PDPackHeader(
        magic=magic,
        version=version,
        flags=flags,
        variant_count=variant_count,
        offset_table=offset_table,
    )

    # --- 偏移表 ---
    pos = offset_table
    base_offset, base_size = struct.unpack_from(">II", data, pos)
    pos += 8
    metadata_offset, metadata_size = struct.unpack_from(">II", data, pos)
    pos += 8

    # 读取各变体的偏移信息
    variant_region_offsets: Dict[str, List[tuple]] = {}
    # 此时尚不知道变体名称 — 先用数字键代替
    for vi in range(variant_count):
        region_count = struct.unpack_from(">H", data, pos)[0]
        pos += 2
        offsets = []
        for _ in range(region_count):
            ro, rs = struct.unpack_from(">II", data, pos)
            pos += 8
            offsets.append((ro, rs))
        variant_region_offsets[str(vi)] = offsets

    # --- 基础图 ---
    base_img = _read_png(data, base_offset, base_size)

    # --- 元数据 ---
    meta_bytes = data[metadata_offset:metadata_offset + metadata_size]
    metadata = json.loads(meta_bytes.decode("utf-8"))

    # --- 将数字键映射为实际变体名称 ---
    variant_names = list(metadata.get("variants", {}).keys())
    named_region_offsets: Dict[str, List[tuple]] = {}
    for vi, vname in enumerate(variant_names):
        named_region_offsets[vname] = variant_region_offsets.get(str(vi), [])

    # --- 变体差异区域 ---
    variant_regions: Dict[str, List[np.ndarray]] = {}
    for vname, offsets in named_region_offsets.items():
        regions = []
        for ro, rs in offsets:
            region_img = _read_png(data, ro, rs)
            regions.append(region_img)
        variant_regions[vname] = regions

    return PDPackFile(
        header=header,
        base_image=base_img,
        metadata=metadata,
        variant_regions=variant_regions,
    )


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _pack_header(version: int, flags: int, variant_count: int,
                 offset_table: int) -> bytes:
    """打包 24 字节文件头。"""
    return struct.pack(
        ">4sHHHI10s",
        MAGIC,
        version,
        flags,
        variant_count,
        offset_table,
        b"\x00" * 10,  # 预留
    )


def _read_png(data: bytes, offset: int, size: int) -> np.ndarray:
    """从 *data* 中读取 PNG 段并返回 NumPy 数组。"""
    segment = data[offset:offset + size]
    img = Image.open(io.BytesIO(segment))
    return np.array(img)
