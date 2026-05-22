"""
T2.3 — 元数据 JSON Schema 定义与校验。

提供 .pdpack 元数据的 JSON Schema (draft-07) 定义，
以及一个不依赖外部库的轻量级校验函数。
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Schema 定义
# ---------------------------------------------------------------------------

METADATA_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["base", "variants"],
    "properties": {
        "base": {
            "type": "object",
            "required": ["width", "height"],
            "properties": {
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "has_alpha": {"type": "boolean"},
                "name": {"type": "string"},
            },
        },
        "variants": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "x", "y", "w", "h"],
                    "properties": {
                        "name": {"type": "string"},
                        "x": {"type": "integer", "minimum": 0},
                        "y": {"type": "integer", "minimum": 0},
                        "w": {"type": "integer", "minimum": 1},
                        "h": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 轻量级校验器（无需外部 Schema 库）
# ---------------------------------------------------------------------------

def validate_metadata(metadata: dict) -> Tuple[bool, List[str]]:
    """按 .pdpack 元数据 Schema 校验 *metadata*。

    参数
    ----------
    metadata : dict
        待校验的元数据字典。

    返回
    -------
    (valid, errors) : tuple[bool, list[str]]
        ``valid`` 为 ``True`` 表示元数据符合 Schema。
        ``errors`` 为（可能为空的）人类可读的错误信息列表。
    """
    errors: List[str] = []

    if not isinstance(metadata, dict):
        return False, ["元数据必须为 JSON 对象"]

    # --- base 字段 ---
    base = metadata.get("base")
    if base is None:
        errors.append("缺少必填字段: 'base'")
    elif not isinstance(base, dict):
        errors.append("'base' 必须为对象")
    else:
        for field in ("width", "height"):
            val = base.get(field)
            if val is None:
                errors.append(f"'base.{field}' 为必填字段")
            elif not isinstance(val, int) or val < 1:
                errors.append(f"'base.{field}' 必须为正整数")

        has_alpha = base.get("has_alpha")
        if has_alpha is not None and not isinstance(has_alpha, bool):
            errors.append("'base.has_alpha' 必须为布尔值")

        name = base.get("name")
        if name is not None and not isinstance(name, str):
            errors.append("'base.name' 必须为字符串")

    # --- variants 字段 ---
    variants = metadata.get("variants")
    if variants is None:
        errors.append("缺少必填字段: 'variants'")
    elif not isinstance(variants, dict):
        errors.append("'variants' 必须为对象")
    else:
        for vname, regions in variants.items():
            if not isinstance(regions, list):
                errors.append(
                    f"'variants.{vname}' 必须为数组"
                )
                continue
            for i, region in enumerate(regions):
                if not isinstance(region, dict):
                    errors.append(
                        f"'variants.{vname}[{i}]' 必须为对象"
                    )
                    continue
                for field in ("name", "x", "y", "w", "h"):
                    if field not in region:
                        errors.append(
                            f"'variants.{vname}[{i}].{field}' 为必填字段"
                        )
                name_val = region.get("name")
                if name_val is not None and not isinstance(name_val, str):
                    errors.append(
                        f"'variants.{vname}[{i}].name' 必须为字符串"
                    )
                for coord in ("x", "y"):
                    val = region.get(coord)
                    if val is not None and (not isinstance(val, int) or val < 0):
                        errors.append(
                            f"'variants.{vname}[{i}].{coord}' 必须为"
                            f"非负整数"
                        )
                for dim in ("w", "h"):
                    val = region.get(dim)
                    if val is not None and (not isinstance(val, int) or val < 1):
                        errors.append(
                            f"'variants.{vname}[{i}].{dim}' 必须为"
                            f"正整数"
                        )

        base_name = base.get("name") if isinstance(base, dict) else None
        if isinstance(base_name, str) and base_name and base_name not in variants:
            errors.append(f"'variants' 必须包含默认变体 '{base_name}'")

    return len(errors) == 0, errors
