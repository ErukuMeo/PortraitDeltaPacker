"""
PDPack — 角色立绘差分打包工具。

离线打包工具，将角色的多个"表情"变体立绘通过块匹配差异分析，
检测公共区域与差异区域，提取基础层 + 每个变体的差异层，
打包为自定义 .pdpack 二进制格式。
"""

__version__ = "0.1.0"
__all__ = [
    "loader",
    "diff",
    "extract",
    "assemble",
    "serializer",
    "schema",
    "preview",
    "cli",
]
