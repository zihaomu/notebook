import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def find_font(size: int = 26):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_lines(draw: ImageDraw.ImageDraw, text: str, font, width: int):
    lines = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def visualize_result(image_path: str | Path, result: dict[str, Any], output_path: str | Path):
    image_path = Path(image_path)
    output_path = Path(output_path)
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    max_height = 1300
    if original.height > max_height:
        ratio = max_height / original.height
        original = original.resize((int(original.width * ratio), max_height), Image.Resampling.LANCZOS)
    panel_width = max(720, int(original.width * 0.75))
    canvas = Image.new("RGB", (original.width + panel_width, max(original.height, 900)), "#f5f3ec")
    canvas.paste(original, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(30)
    body_font = find_font(22)
    small_font = find_font(18)
    x = original.width + 32
    draw.text((x, 24), "HunyuanOCR 1.5", fill="#111827", font=title_font)
    quality = result.get("quality", {})
    status = "PASS" if quality.get("passed") else "CHECK OUTPUT"
    color = "#137333" if quality.get("passed") else "#b3261e"
    draw.text((x, 72), status, fill=color, font=body_font)
    y = 120
    for line in _fit_lines(draw, result.get("text", ""), body_font, panel_width - 64):
        if y > canvas.height - 46:
            draw.text((x, y), "...", fill="#374151", font=body_font)
            break
        draw.text((x, y), line, fill="#1f2937", font=body_font)
        y += 32
    draw.text(
        (x, canvas.height - 34),
        "doc_parse returns Markdown in reading order; no synthetic boxes are drawn.",
        fill="#6b7280",
        font=small_font,
    )
    spatial_blocks = [block for block in result.get("blocks", []) if block.get("bbox")]
    if spatial_blocks:
        overlay = ImageDraw.Draw(canvas)
        for block in spatial_blocks:
            bbox = block["bbox"]
            overlay.rectangle(tuple(bbox), outline="#e11d48", width=3)
            overlay.text((bbox[0], bbox[1]), str(block["index"]), fill="#e11d48", font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path
