# Portrait Delta Packer

角色立绘差分打包工具。将角色多个"表情"变体立绘经块匹配差异分析，提取公共**基础层**与各变体**差异层**，打包为 `.pdpack` 二进制格式，减少包体体积。

## 安装

```bash
pip install -e .
```

依赖 Python 3.9+。核心包：`opencv-python` `numpy` `Pillow` `scikit-image`。

## 快速开始

```bash
# 打包
python -m pdpack pack ./tests/fixtures -o test.pdpack

# 重建预览
python -m pdpack preview test.pdpack -o ./preview/

# 带 PSNR 验证预览（确认还原质量）
python -m pdpack preview test.pdpack -o ./preview/ --verify ./tests/fixtures
```

预览输出中包含基础图（基准变体）及各变体的重建图，命名格式 `{variant}_reconstructed.png`。PSNR ≥ 40 dB 为通过，实测通常 ≥ 60 dB（像素级近乎无损）。

## 命令参考

### pack — 打包

```
python -m pdpack pack <输入目录> [-o 输出文件] [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output` | `./output.pdpack` | 输出文件路径 |
| `--base` | 字母序首张 | 指定基准变体名称 |
| `--block-size` | `32` | 块大小：`8` `16` `32` `64` |
| `--threshold` | `0.98` | SSIM 相似度阈值 (0.0–1.0) |
| `--json-metadata` | — | 同时导出独立 JSON 元数据 |
| `--extract-diffs` | — | 导出差异区域 PNG 到指定目录 |
| `--verbose` `-v` | — | 详细日志（含每步耗时） |
| `--quiet` `-q` | — | 静默，仅错误输出 |

### preview — 预览重建

```
python -m pdpack preview <.pdpack 文件> [-o 输出目录] [选项]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output` | `.` | 输出目录 |
| `-v, --variant` | 全部 | 指定预览单个变体 |
| `--verify` | — | 指定原始目录路径，输出 PSNR |
| `--verbose` | — | 详细日志 |
| `--quiet` | — | 静默模式 |

## 工作原理

1. **加载校验** — 扫描 PNG，校验尺寸一致且 ≤ 2048×2048，分离 Alpha 通道
2. **差异检测** — 基准图与各变体逐块 SSIM 比对，低纹理区域回退 MSE
3. **区域提取** — 连通域分析 → 包围盒合并（贪心，容忍 ≤ 25% 空白）
4. **组装序列化** — 基础图 PNG + 差异 PNGs → 大端序 `.pdpack` 二进制
5. **预览验证** — 反序列化重建图像，可选 PSNR 比对原图

## 项目结构

```
PortraitDeltaPacker/
├── pdpack/
│   ├── cli.py               # 命令行 (argparse)
│   ├── loader.py            # 图像加载校验
│   ├── diff.py              # 块匹配 SSIM 差异检测
│   ├── extract.py           # 区域提取与矩形合并
│   ├── assemble.py          # 基础图 + 差异图组装
│   ├── serializer.py        # .pdpack 序列化/反序列化
│   ├── schema.py            # 元数据 JSON Schema 校验
│   └── preview.py           # 重建与 PSNR 验证
├── tests/
├── Docs/
│   ├── TaskBoard.md
│   └── TaskSpec.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## 退出码

| 值 | 含义 |
|----|------|
| 0 | 成功 |
| 1 | 参数错误 |
| 2 | 输入错误（目录不存在、尺寸不匹配等） |
| 3 | 处理异常（内存不足、写入失败等） |

## 批量处理

```bash
for dir in assets/characters/*/; do
  python -m pdpack pack "$dir" -o "output/$(basename $dir).pdpack" --quiet
done
```
