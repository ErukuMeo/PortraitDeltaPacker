"""
T2.3 — schema 模块测试。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdpack.schema import validate_metadata, METADATA_SCHEMA


class TestSchemaDefinition:
    """Schema 定义测试。"""

    def test_schema_has_required_fields(self):
        """Schema 应包含必需的顶层字段。"""
        assert "$schema" in METADATA_SCHEMA
        assert "required" in METADATA_SCHEMA
        assert "base" in METADATA_SCHEMA["required"]
        assert "variants" in METADATA_SCHEMA["required"]


class TestValidateMetadata:
    """validate_metadata 函数测试。"""

    def test_valid_minimal(self):
        """最小合法元数据应通过校验。"""
        md = {
            "base": {"width": 64, "height": 64},
            "variants": {},
        }
        valid, errors = validate_metadata(md)
        assert valid
        assert errors == []

    def test_valid_full(self):
        """完整合法元数据应通过校验。"""
        md = {
            "base": {"width": 2048, "height": 1024, "has_alpha": True},
            "variants": {
                "happy": [
                    {"name": "r10_20", "x": 10, "y": 20, "w": 32, "h": 32},
                ],
            },
        }
        valid, errors = validate_metadata(md)
        assert valid
        assert errors == []

    def test_missing_base(self):
        """缺少 base 字段应校验失败。"""
        md = {"variants": {}}
        valid, errors = validate_metadata(md)
        assert not valid
        assert any("base" in e for e in errors)

    def test_missing_variants(self):
        """缺少 variants 字段应校验失败。"""
        md = {"base": {"width": 64, "height": 64}}
        valid, errors = validate_metadata(md)
        assert not valid
        assert any("variants" in e for e in errors)

    def test_bad_base_width(self):
        """base.width 为 0 应校验失败。"""
        md = {
            "base": {"width": 0, "height": 64},
            "variants": {},
        }
        valid, errors = validate_metadata(md)
        assert not valid
        assert any("width" in e for e in errors)

    def test_bad_region_coords(self):
        """区域坐标为负应校验失败。"""
        md = {
            "base": {"width": 64, "height": 64},
            "variants": {
                "a": [
                    {"name": "bad", "x": -1, "y": 0, "w": 1, "h": 1},
                ],
            },
        }
        valid, errors = validate_metadata(md)
        assert not valid
        assert any("x" in e for e in errors)

    def test_bad_region_dims(self):
        """区域尺寸为 0 应校验失败。"""
        md = {
            "base": {"width": 64, "height": 64},
            "variants": {
                "a": [
                    {"name": "bad", "x": 0, "y": 0, "w": 0, "h": 1},
                ],
            },
        }
        valid, errors = validate_metadata(md)
        assert not valid
        assert any("w" in e for e in errors)

    def test_not_a_dict(self):
        """非字典输入应校验失败。"""
        valid, errors = validate_metadata("not a dict")
        assert not valid
