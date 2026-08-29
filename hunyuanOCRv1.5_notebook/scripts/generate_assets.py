#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "assets" / "images"
MANIFEST = ROOT / "assets" / "manifest.json"
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size: int, bold: bool = False):
    candidates = list(FONT_CANDIDATES)
    if bold:
        candidates.insert(0, "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        candidates.insert(1, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text(draw, xy, value, size=30, fill="#17212b", bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def bilingual_columns(path: Path):
    image = Image.new("RGB", (1600, 1120), "#f7f5ef")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 120), fill="#123c45")
    text(draw, (70, 55), "边缘智能实验记录  /  Edge AI Lab Note", 46, "white", True, "lm")
    text(draw, (70, 155), "2026-08-28   Document ID: HY-015", 23, "#53636b")
    draw.line((800, 205, 800, 1020), fill="#d2cec2", width=3)
    text(draw, (70, 225), "摘要", 36, "#0f5966", True)
    left = [
        "本实验在 AMD Radeon PRO W7900D 上验证", "HunyuanOCR-1.5 的文档解析能力。",
        "输入保持原始分辨率，推理温度设为 0。", "重点观察中英混排、公式与阅读顺序。",
        "", "核心配置", "• MIOPEN_FIND_MODE = 2", "• Context length = 131072",
        "• Model precision = BF16", "", "公式", "平均延迟  L = (1/n) Σ tᵢ", "吞吐率  R = n / Σ tᵢ",
    ]
    y = 290
    for line in left:
        text(draw, (75, y), line, 29 if not line.startswith("•") else 27, "#17212b", line in {"核心配置", "公式"})
        y += 53
    text(draw, (865, 225), "Executive Summary", 36, "#b8492e", True)
    right = [
        "This report validates HunyuanOCR 1.5 on", "an AMD Radeon PRO W7900D GPU.",
        "The source image remains at native resolution,", "with deterministic decoding at temperature 0.",
        "", "Observations", "1. Mixed Chinese and English text is preserved.",
        "2. Equations remain in reading order.", "3. Markdown is suitable for downstream search.",
        "", "Conclusion", "Correctness must be checked before throughput.", "HTTP 200 alone is not an OCR quality gate.",
    ]
    y = 290
    for line in right:
        text(draw, (865, y), line, 27, "#17212b", line in {"Observations", "Conclusion"})
        y += 53
    draw.rectangle((55, 1045, 1545, 1080), fill="#e3dfd4")
    text(draw, (800, 1062), "Synthetic sample generated for the HunyuanOCR notebook demo", 20, "#5d625e", False, "mm")
    image.save(path, optimize=True)


def invoice_table(path: Path):
    image = Image.new("RGB", (1500, 1080), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (70, 70), "智算服务费用清单", 48, "#17324d", True)
    text(draw, (1430, 82), "INVOICE 2026-0815", 28, "#335d7e", False, "ra")
    text(draw, (70, 145), "客户：示例研究院 / Example Research Lab", 28)
    text(draw, (70, 190), "日期：2026-08-28     币种：CNY", 26, "#4d5965")
    columns = [70, 700, 920, 1160, 1430]
    top, row_h = 280, 108
    headers = ["项目 / Item", "数量", "单价", "金额"]
    draw.rectangle((70, top, 1430, top + row_h), fill="#17324d")
    for index, header in enumerate(headers):
        text(draw, ((columns[index] + columns[index + 1]) // 2, top + row_h // 2), header, 27, "white", True, "mm")
    rows = [
        ("W7900D GPU 推理服务", "12 h", "¥ 38.00", "¥ 456.00"),
        ("文档 OCR 批处理", "1,250 页", "¥ 0.08", "¥ 100.00"),
        ("结果归档与校验", "1 项", "¥ 80.00", "¥ 80.00"),
        ("技术支持 / Support", "2 h", "¥ 120.00", "¥ 240.00"),
    ]
    for row_index, row in enumerate(rows, 1):
        y0 = top + row_index * row_h
        fill = "#eef4f6" if row_index % 2 else "#ffffff"
        draw.rectangle((70, y0, 1430, y0 + row_h), fill=fill)
        for x in columns:
            draw.line((x, y0, x, y0 + row_h), fill="#9aaab3", width=2)
        for index, value in enumerate(row):
            text(draw, ((columns[index] + columns[index + 1]) // 2, y0 + row_h // 2), value, 26, "#17212b", False, "mm")
    bottom = top + (len(rows) + 1) * row_h
    for x in columns:
        draw.line((x, top, x, bottom), fill="#718693", width=2)
    draw.rectangle((70, top, 1430, bottom), outline="#17324d", width=4)
    text(draw, (1020, bottom + 65), "小计", 29, "#354b5e", True)
    text(draw, (1400, bottom + 65), "¥ 876.00", 30, "#17324d", True, "ra")
    text(draw, (1020, bottom + 120), "税额 6%", 27, "#354b5e")
    text(draw, (1400, bottom + 120), "¥ 52.56", 28, "#17324d", False, "ra")
    draw.line((1010, bottom + 150, 1430, bottom + 150), fill="#17324d", width=3)
    text(draw, (1020, bottom + 190), "合计 / Total", 31, "#b8492e", True)
    text(draw, (1400, bottom + 190), "¥ 928.56", 34, "#b8492e", True, "ra")
    image.save(path, optimize=True)


def rotated_note(path: Path):
    canvas = Image.new("RGB", (1400, 900), "#ccd7d0")
    draw = ImageDraw.Draw(canvas)
    for x in range(0, 1400, 80):
        draw.line((x, 0, x + 280, 900), fill="#bdc8c1", width=2)
    note = Image.new("RGBA", (920, 560), "#fff2a8")
    nd = ImageDraw.Draw(note)
    nd.rectangle((0, 0, 920, 560), outline="#d8bf55", width=5)
    text(nd, (55, 45), "部署检查 / Deployment Check", 39, "#493e20", True)
    lines = [
        "[✓] 模型权重已校验", "[✓] MIOPEN_FIND_MODE=2", "[✓] GPU: gfx1100",
        "[ ] 完整 1651 页复测", "", "Reminder:", "Accuracy before speed.",
        "Port 8000  |  TP=1  |  BF16",
    ]
    y = 120
    for line in lines:
        text(nd, (62, y), line, 31 if not line.startswith("Reminder") else 34, "#493e20", line == "Reminder:")
        y += 55
    rotated = note.rotate(7, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    canvas.paste(rotated, (235, 135), rotated)
    low = canvas.resize((700, 450), Image.Resampling.LANCZOS)
    low.save(path, optimize=True)


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "01_bilingual_columns.png": bilingual_columns,
        "02_invoice_table.png": invoice_table,
        "03_rotated_low_resolution_note.png": rotated_note,
    }
    records = []
    for name, generator in generators.items():
        path = IMAGE_DIR / name
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        generator(temporary)
        temporary.replace(path)
        with Image.open(path) as image:
            records.append({
                "name": name,
                "bytes": path.stat().st_size,
                "size": list(image.size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "license": "Original synthetic asset generated by this project",
            })
    temporary_manifest = MANIFEST.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps({"schema_version": "1.0", "images": records}, indent=2) + "\n"
    )
    temporary_manifest.replace(MANIFEST)
    print(json.dumps({"status": "PASS", "images": records}, indent=2))


if __name__ == "__main__":
    main()
