"""
T3.1 — 命令行接口。

提供 ``pdpack`` 命令，含 ``pack`` 和 ``preview`` 两个子命令，
以及全局选项。
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from pdpack import __version__
from pdpack.assemble import assemble
from pdpack.diff import detect_diffs
from pdpack.extract import extract_diff_regions, merge_rectangles
from pdpack.loader import load_variants
from pdpack.schema import validate_metadata
from pdpack.serializer import deserialize, serialize


def main(argv: Optional[List[str]] = None) -> None:
    """"pdpack" CLI 入口点。"""
    parser = argparse.ArgumentParser(
        prog="pdpack",
        description="角色立绘差分打包工具 — 将角色立绘变体打包为 .pdpack 格式",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"pdpack {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- pack 子命令 ----
    pack_parser = sub.add_parser("pack", help="将变体 PNG 打包为 .pdpack 文件")
    pack_parser.add_argument(
        "input_dir", help="包含变体 PNG 图像的目录路径",
    )
    pack_parser.add_argument(
        "-o", "--output", default="./output.pdpack",
        help="输出 .pdpack 文件路径 (默认: ./output.pdpack)",
    )
    pack_parser.add_argument(
        "--block-size", type=int, default=32,
        choices=[8, 16, 32, 64],
        help="块大小: 8, 16, 32 或 64 (默认: 32)",
    )
    pack_parser.add_argument(
        "--threshold", type=float, default=0.98,
        help="SSIM 相似度阈值 0.0–1.0 (默认: 0.98)",
    )
    pack_parser.add_argument(
        "--base", dest="base_name", default=None,
        help="指定基准变体名 (默认: 字母序第一)",
    )
    pack_parser.add_argument(
        "--json-metadata", dest="json_metadata", default=None,
        help="同时导出独立的 JSON 元数据文件",
    )
    pack_parser.add_argument(
        "--extract-diffs", dest="extract_diffs", default=None,
        help="将差异区域 PNG 导出到指定目录",
    )
    pack_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志输出",
    )
    pack_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式 — 仅输出错误",
    )
    pack_parser.set_defaults(func=_cmd_pack)

    # ---- preview 子命令 ----
    preview_parser = sub.add_parser("preview", help="从 .pdpack 文件重建图像")
    preview_parser.add_argument(
        "pdpack_file", help=".pdpack 文件路径",
    )
    preview_parser.add_argument(
        "-v", "--variant", dest="variant_name", default=None,
        help="指定要预览的变体名 (默认: 全部)",
    )
    preview_parser.add_argument(
        "-o", "--output", default=".",
        help="重建 PNG 的输出目录 (默认: .)",
    )
    preview_parser.add_argument(
        "--verify", dest="verify_dir", default=None,
        help="原始输入目录路径，用于 PSNR 验证",
    )
    preview_parser.add_argument(
        "--verbose", action="store_true",
        help="详细日志输出",
    )
    preview_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式 — 仅输出错误",
    )
    preview_parser.set_defaults(func=_cmd_preview)

    args = parser.parse_args(argv)
    args.func(args)


# ---------------------------------------------------------------------------
# pack 子命令
# ---------------------------------------------------------------------------

def _cmd_pack(args: argparse.Namespace) -> None:
    """执行 "pack" 子命令。"""
    _configure_logging(args)

    t_start = time.monotonic()

    # 第 1 步 — 加载
    t0 = time.monotonic()
    try:
        variants, alpha_masks, base_name, image_shape = load_variants(
            args.input_dir, base_name=args.base_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        _die(2, str(exc))
    if not args.quiet:
        _log(f"已加载 {len(variants)} 个变体, 基准='{base_name}', "
             f"尺寸={image_shape[1]}×{image_shape[0]}", args)
    if args.verbose:
        _log_timing("T1.1 加载", t0)

    # 判断是否有变体含 Alpha 通道
    has_alpha = any(
        mask is not None for mask in alpha_masks.values()
    )

    # 制作 diff 检测用图像：将 RGBA 预乘合成到黑色背景上（模拟游戏引擎渲染）
    # 透明像素 → 黑色，抗锯齿边缘的微弱差异自然被可见度加权抑制
    variants_for_detection = {}
    if has_alpha:
        for name, img in variants.items():
            mask = alpha_masks.get(name)
            if mask is not None:
                alpha_f = mask.astype(np.float32) / 255.0
                composited = (img.astype(np.float32) * alpha_f[:, :, np.newaxis]).astype(np.uint8)
                variants_for_detection[name] = composited
            else:
                variants_for_detection[name] = img
    else:
        variants_for_detection = variants

    # 第 2 步 — 差异检测
    t0 = time.monotonic()
    base_for_detection = variants_for_detection[base_name]
    other_for_detection = {k: v for k, v in variants_for_detection.items() if k != base_name}
    try:
        diff_masks = detect_diffs(
            base_for_detection, other_for_detection,
            block_size=args.block_size,
            threshold=args.threshold,
        )
    except Exception as exc:
        _die(3, f"差异检测失败: {exc}")
    if args.verbose:
        _log_timing("T1.2 差异检测", t0)

    # 第 3 步 — 区域提取 + 合并（使用原始 variants，非预乘版本）
    t0 = time.monotonic()
    variant_regions: Dict[str, List[dict]] = {}
    for vname, dmask in diff_masks.items():
        vimg = variants[vname]
        amask = alpha_masks.get(vname) if has_alpha else None
        raw_regions = extract_diff_regions(dmask, args.block_size, vimg, amask)
        merged_regions = merge_rectangles(raw_regions, dmask, args.block_size, vimg, amask)
        # 过滤几乎全透明的 diff 区域（抗锯齿边缘假阳性）
        if has_alpha and amask is not None:
            merged_regions = _filter_transparent_regions(merged_regions, min_visible_pct=5.0)
        variant_regions[vname] = merged_regions
    if args.verbose:
        _log_timing("T1.3 区域提取与合并", t0)

    # 第 4 步 — 组装（若含 Alpha，先合入基础图）
    base_img = variants[base_name]
    base_for_assembly = base_img
    if has_alpha and alpha_masks.get(base_name) is not None:
        alpha = alpha_masks[base_name]
        base_for_assembly = np.dstack([base_img, alpha])
    t0 = time.monotonic()
    try:
        base_png, variant_diff_pngs, metadata = assemble(
            base_for_assembly, variant_regions, alpha_masks, has_alpha, base_name,
        )
    except Exception as exc:
        _die(3, f"组装失败: {exc}")
    if args.verbose:
        _log_timing("T1.4 组装", t0)

    # 校验元数据
    valid, errors = validate_metadata(metadata)
    if not valid:
        for err in errors:
            print(f"元数据警告: {err}", file=sys.stderr)

    # 第 5 步 — 序列化
    t0 = time.monotonic()
    flags = 0
    if has_alpha:
        flags |= 1  # FLAG_HAS_ALPHA
    try:
        pdpack_data = serialize(base_png, metadata, variant_diff_pngs, flags=flags)
    except Exception as exc:
        _die(3, f"序列化失败: {exc}")
    if args.verbose:
        _log_timing("T2.2 序列化", t0)

    # 第 6 步 — 写入输出
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(pdpack_data)
    if args.verbose:
        _log_timing("写入输出", t0)

    # 可选导出
    if args.json_metadata:
        with open(args.json_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.extract_diffs:
        _clear_dir(args.extract_diffs)
        for vname, pngs in variant_diff_pngs.items():
            for i, png_bytes in enumerate(pngs):
                path = os.path.join(args.extract_diffs, f"{vname}_diff_{i}.png")
                with open(path, "wb") as f:
                    f.write(png_bytes)

    # 摘要
    t_total = time.monotonic() - t_start
    if not args.quiet:
        total_regions = sum(len(r) for r in variant_regions.values())
        _log(
            f"已打包 {len(variants)} 个变体, {total_regions} 个差异区域 "
            f"→ {args.output}  ({t_total:.2f}s)",
            args,
        )

    sys.exit(0)


# ---------------------------------------------------------------------------
# preview 子命令
# ---------------------------------------------------------------------------

def _cmd_preview(args: argparse.Namespace) -> None:
    """执行 "preview" 子命令。"""
    _configure_logging(args)

    # 读取 .pdpack 文件
    try:
        with open(args.pdpack_file, "rb") as f:
            data = f.read()
    except OSError as exc:
        _die(2, f"无法读取 {args.pdpack_file}: {exc}")

    try:
        ppf = deserialize(data)
    except ValueError as exc:
        _die(2, f"无效的 .pdpack 文件: {exc}")

    from pdpack.preview import reconstruct, save_reconstructed, verify

    variant_names = list(ppf.metadata.get("variants", {}).keys())
    target = args.variant_name

    if target and target not in variant_names:
        _die(1, f"变体 '{target}' 未找到。"
                f"可用变体: {variant_names}")

    # 保存基础图（基准变体）
    _clear_dir(args.output)
    from PIL import Image
    base_name = ppf.metadata.get("base", {}).get("name", "base")
    base_path = os.path.join(args.output, f"{base_name}_reconstructed.png")
    Image.fromarray(ppf.base_image).save(base_path)
    if not args.quiet:
        _log(f"已保存基础图(基准变体) {base_path}", args)

    # 重建各变体差异
    reconstructed = reconstruct(ppf, variant_name=target)
    saved = save_reconstructed(reconstructed, args.output)

    if not args.quiet:
        for path in saved:
            _log(f"已保存 {path}", args)

    # 可选 PSNR 验证
    if args.verify_dir:
        psnr_values = verify(reconstructed, args.verify_dir)
        for vname, psnr in psnr_values.items():
            status = "通过" if psnr > 40 else "警告"
            if not args.quiet:
                _log(f"  {vname}: PSNR={psnr:.2f} dB [{status}]", args)
            if psnr <= 40:
                print(f"警告: '{vname}' 的 PSNR 为 {psnr:.2f} dB "
                      f"(≤ 40 dB) — 可能存在差异检测遗漏",
                      file=sys.stderr)

    sys.exit(0)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _clear_dir(dirpath: str) -> None:
    """清空目标文件夹（若存在），用于 --extract-diffs / preview -o 等目录输出。"""
    import shutil
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok=True)


def _filter_transparent_regions(regions: List[dict], min_visible_pct: float = 5.0) -> List[dict]:
    """过滤几乎全透明的 diff 区域（抗锯齿边缘假阳性）。

    仅对 RGBA 区域生效；RGB 区域（无 Alpha）直接保留。
    """
    kept = []
    for r in regions:
        pixels = r.get("pixels")
        if pixels is None or pixels.ndim != 3:
            kept.append(r)
        elif pixels.shape[2] == 3:
            kept.append(r)  # 无 Alpha，所有像素视为可见
        elif pixels.shape[2] == 4:
            visible = 100 * (pixels[:, :, 3] > 10).sum() / pixels[:, :, 3].size
            if visible >= min_visible_pct:
                kept.append(r)
    return kept


def _configure_logging(args: argparse.Namespace) -> None:
    """确保 --verbose 与 --quiet 不同时设置。"""
    if args.verbose and args.quiet:
        _die(1, "不能同时使用 --verbose 和 --quiet")


def _log(msg: str, args: argparse.Namespace) -> None:
    """将消息输出到 stdout（静默模式下不输出）。"""
    if not args.quiet:
        print(msg)


def _log_timing(label: str, t0: float) -> None:
    """输出一行耗时信息到 stdout。"""
    elapsed = time.monotonic() - t0
    print(f"  [{label}] {elapsed:.3f}s")


def _die(exit_code: int, msg: str) -> None:
    """将错误消息输出到 stderr 并退出。"""
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
